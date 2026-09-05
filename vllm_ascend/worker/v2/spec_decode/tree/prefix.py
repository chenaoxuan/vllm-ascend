import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout


def _select_topb_nodes(
    path_scores: torch.Tensor,
    depths: torch.Tensor,
    budget: int,
    depth_bonus: float,
) -> torch.Tensor:
    """Select the global Top-B supertree nodes, batched over requests.

    Path scores are prefix-monotone and ``depth_bonus <= 0`` keeps the
    bonus-adjusted scores monotone, so the Top-B set is prefix-closed.
    ``budget`` is B (non-root nodes to keep).
    """
    node_count = int(depths.shape[1])
    k = min(int(budget), node_count)
    candidate_scores = path_scores.float() + float(depth_bonus) * depths.float()
    selected = torch.topk(
        candidate_scores, k=k, largest=True, sorted=False, dim=-1
    ).indices
    return torch.sort(selected + 1, dim=-1).values


class DominoCorrectionScorer:
    """Domino correction head scorer for prefix (DARTree) construction."""

    def __init__(self, draft_model, target_model, hidden_dim=None):
        fc1, self.middle, fc2 = self._unwrap_correction_mlp(draft_model.embed_proj)
        gru = draft_model.prefix_gru
        if gru.num_layers != 1 or gru.bidirectional:
            raise ValueError(
                "Domino correction requires a single-layer unidirectional prefix_gru"
            )
        self.gru_hidden_dim = int(gru.hidden_size)
        if hidden_dim is None:
            hidden_dim = fc1.in_features - self.gru_hidden_dim
        self.hidden_dim = int(hidden_dim)
        if fc1.in_features != self.hidden_dim + self.gru_hidden_dim:
            raise ValueError(
                "Domino correction MLP input dim mismatch: got "
                f"{fc1.in_features}, expected {self.hidden_dim + self.gru_hidden_dim}"
            )
        self.w_z = fc1.weight[:, : self.hidden_dim].detach().contiguous()
        self.w_s = fc1.weight[:, self.hidden_dim :].detach().contiguous()
        self.fc1_bias = fc1.bias.detach() if fc1.bias is not None else None
        self.fc2_weight = fc2.weight.detach().contiguous()
        self.fc2_bias = fc2.bias.detach() if fc2.bias is not None else None
        self.gru_w_ih = gru.weight_ih_l0.detach().contiguous()
        self.gru_w_hh = gru.weight_hh_l0.detach().contiguous()
        self.gru_b_ih = gru.bias_ih_l0.detach() if gru.bias else None
        self.gru_b_hh = gru.bias_hh_l0.detach() if gru.bias else None
        embed_weight = target_model.model.embed_tokens.weight
        self._gru_input_proj_table = F.linear(
            embed_weight, self.gru_w_ih, self.gru_b_ih
        ).contiguous()

    @staticmethod
    def _unwrap_correction_mlp(embed_proj):
        if not isinstance(embed_proj, nn.Sequential):
            raise TypeError(
                "Domino correction requires draft_model.embed_proj to be nn.Sequential"
            )
        modules = list(embed_proj.children())
        if len(modules) < 2:
            raise ValueError("embed_proj must contain at least two layers")
        if not isinstance(modules[0], nn.Linear) or not isinstance(
            modules[-1], nn.Linear
        ):
            raise ValueError("embed_proj first/last layers must be nn.Linear")
        fc1 = modules[0]
        fc2 = modules[-1]
        middle = (
            nn.Sequential(*modules[1:-1]) if len(modules) > 2 else nn.Identity()
        )
        return fc1, middle, fc2

    def project_z(self, parallel_hiddens: torch.Tensor) -> torch.Tensor:
        return F.linear(parallel_hiddens, self.w_z, self.fc1_bias)

    def update_hidden(
        self, token_ids: torch.Tensor, h_state: torch.Tensor
    ) -> torch.Tensor:
        token_ids = token_ids.reshape(-1)
        gi = self._gru_input_proj_table.index_select(0, token_ids)
        gh = F.linear(h_state, self.gru_w_hh, self.gru_b_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h_state


def _prefix_corrected_candidates(
    scorer,
    z: torch.Tensor,
    parent_hidden: torch.Tensor,
    candidate_vals: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_weight: torch.Tensor,
    candidate_bias: torch.Tensor | None,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefix-only Domino candidate correction for one depth (not Markov)."""
    s_proj = F.linear(parent_hidden, scorer.w_s, None)
    mid = scorer.middle(z.unsqueeze(1) + s_proj)
    bias = torch.einsum("rwm,rcm->rwc", mid, candidate_weight)
    if candidate_bias is not None:
        bias = bias + candidate_bias.unsqueeze(1)
    candidate_logits = candidate_vals.unsqueeze(1).to(bias.dtype) + bias
    candidate_logits = candidate_logits.float()
    top_vals, top_ids = torch.topk(candidate_logits, k=k, dim=-1)
    log_z = torch.logsumexp(candidate_logits, dim=-1, keepdim=True)
    top_scores = top_vals - log_z
    width = parent_hidden.size(1)
    cand_ids = candidate_ids.unsqueeze(1).expand(-1, width, -1)
    sel_ids = torch.gather(cand_ids, 2, top_ids)
    return top_scores, sel_ids


class PrefixTreeBuilder(TreeBuilder):
    """DARTree-style uniform-width supertree + prefix-closed Top-B prune."""

    required_backend = "dflash"
    method = "prefix"

    def __init__(
        self,
        budget: int,
        topk: int,
        *,
        correction_scorer: DominoCorrectionScorer | None = None,
        prefix_len: int = 0,
        depth_bonus: float = -0.2,
        supertree_width: int | None = None,
        pruned: bool = True,
    ):
        super().__init__(budget, topk)
        self.correction_scorer = correction_scorer
        self.prefix_len = prefix_len
        self.depth_bonus = depth_bonus
        self.supertree_width = supertree_width
        self.pruned = pruned

    def build(
        self,
        draft_logits: torch.Tensor,
        out: TreeLayout,
        *,
        root_token_ids: torch.Tensor | None = None,
        draft_hidden: torch.Tensor | None = None,
    ) -> TreeLayout:
        budget = self.budget
        topk = self.topk
        depth_bonus = self.depth_bonus
        prefix_len = self.prefix_len
        correction_scorer = self.correction_scorer
        num_reqs, spec_num, vocab = draft_logits.shape
        device = draft_logits.device
        k = min(topk, vocab, budget)

        if self.pruned:
            width = self.supertree_width if self.supertree_width is not None else k
            width = min(int(width), k)
        else:
            width = min(int(np.ceil(budget / spec_num)), k, budget)
        width = max(1, width)
        supertree_budget = width * spec_num

        with_correction = correction_scorer is not None
        if with_correction:
            candidate_count = k
            base_float = draft_logits.float()
            candidate_vals, candidate_ids = torch.topk(
                base_float, k=candidate_count, dim=-1
            )
            flat_cids = candidate_ids.reshape(-1)
            candidate_weight = correction_scorer.fc2_weight.index_select(
                0, flat_cids
            ).view(num_reqs, spec_num, candidate_count, -1)
            candidate_bias = None
            if correction_scorer.fc2_bias is not None:
                candidate_bias = correction_scorer.fc2_bias.index_select(
                    0, flat_cids
                ).view(num_reqs, spec_num, candidate_count)
            z_parts = correction_scorer.project_z(draft_hidden[:, :spec_num])
            gru_hidden_dim = correction_scorer.gru_hidden_dim
        else:
            log_probs = torch.log_softmax(draft_logits.float(), dim=-1)

        max_nodes = supertree_budget + 1
        tokens = torch.full(
            (num_reqs, supertree_budget), -1, dtype=torch.long, device=device
        )
        depths = torch.zeros(
            (num_reqs, supertree_budget), dtype=torch.long, device=device
        )
        parents = torch.zeros((num_reqs, max_nodes), dtype=torch.long, device=device)
        path_scores = torch.zeros(
            (num_reqs, max_nodes), dtype=torch.float32, device=device
        )
        hidden_states = None
        if with_correction:
            root_h0 = torch.zeros(
                (num_reqs, gru_hidden_dim), dtype=draft_logits.dtype, device=device
            )
            root_hidden = correction_scorer.update_hidden(
                root_token_ids.reshape(-1), root_h0
            )
            hidden_states = torch.zeros(
                (num_reqs, max_nodes, gru_hidden_dim),
                dtype=root_hidden.dtype,
                device=device,
            )
            hidden_states[:, 0] = root_hidden

        frontier = torch.zeros((num_reqs, width), dtype=torch.long, device=device)
        frontier_len = 1
        num_nodes = 0
        for child_depth in range(1, spec_num + 1):
            if num_nodes >= supertree_budget:
                break
            take = min(width, frontier_len * k, supertree_budget - num_nodes)
            if take <= 0:
                break
            depth_slot = child_depth - 1
            valid_parent = torch.arange(width, device=device)[None, :] < frontier_len

            if with_correction:
                parent_hidden = torch.gather(
                    hidden_states,
                    1,
                    frontier.unsqueeze(-1).expand(-1, -1, gru_hidden_dim),
                )
                if depth_slot < prefix_len:
                    logits = draft_logits[:, depth_slot].float()
                    top_vals, top_ids = torch.topk(logits, k=k, dim=-1)
                    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
                    top_scores = (top_vals - log_z).unsqueeze(1).expand(-1, width, -1)
                    cand_ids = top_ids.unsqueeze(1).expand(-1, width, -1)
                else:
                    top_scores, cand_ids = _prefix_corrected_candidates(
                        correction_scorer,
                        z_parts[:, depth_slot],
                        parent_hidden,
                        candidate_vals[:, depth_slot],
                        candidate_ids[:, depth_slot],
                        candidate_weight[:, depth_slot],
                        candidate_bias[:, depth_slot]
                        if candidate_bias is not None
                        else None,
                        k,
                    )
            else:
                top_vals, top_ids = torch.topk(log_probs[:, depth_slot, :], k=k, dim=-1)
                top_scores = top_vals.unsqueeze(1).expand(-1, width, -1)
                cand_ids = top_ids.unsqueeze(1).expand(-1, width, -1)

            top_scores = torch.where(
                valid_parent[:, :, None],
                top_scores,
                torch.full_like(top_scores, float("-inf")),
            )
            parent_scores = torch.gather(path_scores, 1, frontier)
            cand_scores = parent_scores.unsqueeze(-1) + top_scores
            flat = cand_scores.reshape(num_reqs, -1)
            vals, sel = torch.topk(flat, k=take, dim=-1)
            parent_pos = sel // k
            sel_tokens = torch.gather(cand_ids.reshape(num_reqs, -1), 1, sel)
            sel_parents = torch.gather(frontier, 1, parent_pos)

            start = num_nodes + 1
            tokens[:, num_nodes : num_nodes + take] = sel_tokens
            depths[:, num_nodes : num_nodes + take] = child_depth
            parents[:, start : start + take] = sel_parents
            path_scores[:, start : start + take] = vals

            if with_correction:
                parent_hidden_sel = parent_hidden.gather(
                    1, parent_pos.unsqueeze(-1).expand(-1, -1, gru_hidden_dim)
                )
                child_hidden = correction_scorer.update_hidden(
                    sel_tokens.reshape(-1),
                    parent_hidden_sel.reshape(-1, gru_hidden_dim),
                ).reshape(num_reqs, take, gru_hidden_dim)
                hidden_states[:, start : start + take] = child_hidden

            new_ids = (
                torch.arange(start, start + take, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(num_reqs, -1)
            )
            if take < width:
                new_ids = torch.cat(
                    [
                        new_ids,
                        torch.zeros(
                            (num_reqs, width - take), dtype=torch.long, device=device
                        ),
                    ],
                    dim=-1,
                )
            frontier = new_ids
            frontier_len = take
            num_nodes += take

        if num_nodes > budget:
            kept_ids = _select_topb_nodes(
                path_scores[:, 1 : num_nodes + 1],
                depths[:, :num_nodes],
                budget,
                depth_bonus,
            )
            tokens = torch.gather(tokens, 1, kept_ids - 1)
            depths = torch.gather(depths, 1, kept_ids - 1)
            old_parents = torch.gather(parents, 1, kept_ids)
            old_to_new = torch.zeros(
                (num_reqs, num_nodes + 1), dtype=torch.long, device=device
            )
            new_ids = torch.arange(
                1, budget + 1, dtype=torch.long, device=device
            ).unsqueeze(0)
            old_to_new.scatter_(1, kept_ids, new_ids.expand(num_reqs, budget))
            parent_ids = torch.gather(old_to_new, 1, old_parents)
            num_nodes = budget
        else:
            tokens = tokens[:, :num_nodes]
            depths = depths[:, :num_nodes]
            parent_ids = parents[:, 1 : num_nodes + 1]

        return finalize_tree_layout(out, tokens, depths, parent_ids, num_nodes)
