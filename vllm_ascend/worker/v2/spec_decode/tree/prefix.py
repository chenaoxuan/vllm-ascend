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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefix-only Domino candidate correction for one depth (not Markov).

    Returns ``(top_scores, sel_ids, candidate_logits)`` where
    ``candidate_logits`` is ``[R, width, C]`` over the base top-C set (the
    actual Domino proposal before the width×k top cut).
    """
    from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time_accum

    with tree_time_accum("build_score_linear"):
        s_proj = F.linear(parent_hidden, scorer.w_s, None)
    with tree_time_accum("build_score_middle"):
        mid = scorer.middle(z.unsqueeze(1) + s_proj)
    with tree_time_accum("build_score_einsum"):
        bias = torch.einsum("rwm,rcm->rwc", mid, candidate_weight)
        if candidate_bias is not None:
            bias = bias + candidate_bias.unsqueeze(1)
    with tree_time_accum("build_score_logits"):
        candidate_logits = candidate_vals.unsqueeze(1).to(bias.dtype) + bias
        candidate_logits = candidate_logits.float()
    with tree_time_accum("build_score_topk"):
        top_vals, top_ids = torch.topk(candidate_logits, k=k, dim=-1)
    with tree_time_accum("build_score_logsumexp"):
        log_z = torch.logsumexp(candidate_logits, dim=-1, keepdim=True)
        top_scores = top_vals - log_z
    with tree_time_accum("build_score_id_gather"):
        width = parent_hidden.size(1)
        cand_ids = candidate_ids.unsqueeze(1).expand(-1, width, -1)
        sel_ids = torch.gather(cand_ids, 2, top_ids)
    return top_scores, sel_ids, candidate_logits


def _scatter_parent_proposal(
    proposal_logits: torch.Tensor,
    parent_ids: torch.Tensor,
    valid_parent: torch.Tensor,
    full_logits: torch.Tensor,
    req_idx: torch.Tensor | None = None,
) -> None:
    """Write ``full_logits[r, j]`` into ``proposal_logits[r, parent_ids[r, j]]``.

    Proposal buffers are FP32 (NPU IndexPut does not accept BF16 selfRef).
    """
    num_reqs, width, _vocab = full_logits.shape
    device = full_logits.device
    full_logits = full_logits.to(dtype=proposal_logits.dtype)
    if req_idx is None:
        req_idx = torch.arange(num_reqs, device=device)
    else:
        req_idx = req_idx[:num_reqs]
    for j in range(width):
        write = valid_parent[:, j]
        pid = parent_ids[:, j].clamp(min=0, max=proposal_logits.shape[1] - 1)
        proposal_logits[req_idx, pid] = torch.where(
            write.unsqueeze(-1),
            full_logits[:, j],
            proposal_logits[req_idx, pid],
        )


class PrefixTreeBuilder(TreeBuilder):
    """DARTree-style uniform-width supertree + prefix-closed Top-B prune.

    Expansion **k** and beam **width** both come from ``topk``
    (``k = min(topk, vocab, budget)``). Domino shortlist **C** is
    ``params["candidate_size"]`` (missing/None → ``C = k``; else clamp
    ``C = max(k, min(int(C), vocab))``). If the expanded supertree
    exceeds ``budget``, ``_select_topb_nodes`` prunes it (hard-coded
    ``depth_bonus=-0.2``).
    """

    required_backend = "dflash"
    method = "prefix"

    def __init__(
        self,
        budget: int,
        topk: int,
        *,
        correction_scorer: DominoCorrectionScorer | None = None,
        prefix_len: int = 0,
        params: dict | None = None,
    ):
        super().__init__(budget, topk)
        params = params if params is not None else {}
        self.correction_scorer = correction_scorer
        self.prefix_len = prefix_len
        self.depth_bonus = -0.2
        self.candidate_size = params.get("candidate_size")
        # Growable scratch reused across propose() calls.
        self._scratch_reqs = 0
        self._scratch_vocab = 0
        self._scratch_spec = 0
        self._scratch_gru = 0
        self._tokens_buf: torch.Tensor | None = None
        self._depths_buf: torch.Tensor | None = None
        self._parents_buf: torch.Tensor | None = None
        self._path_scores_buf: torch.Tensor | None = None
        self._frontier_buf: torch.Tensor | None = None
        self._prop_temp_buf: torch.Tensor | None = None
        self._prop_layer_buf: torch.Tensor | None = None
        self._hidden_buf: torch.Tensor | None = None
        self._old_to_new_buf: torch.Tensor | None = None
        self._req_arange: torch.Tensor | None = None
        self._width_arange: torch.Tensor | None = None
        self._node_arange: torch.Tensor | None = None
        self._budget_ids: torch.Tensor | None = None
        self._frontier_pad: torch.Tensor | None = None

    def _ensure_scratch(
        self,
        num_reqs: int,
        vocab: int,
        spec_num: int,
        device: torch.device,
        *,
        need_proposal: bool,
        gru_hidden_dim: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        k = min(self.topk, vocab, self.budget)
        width = k
        supertree_budget = width * spec_num
        max_nodes = supertree_budget + 1
        grow = (
            self._tokens_buf is None
            or num_reqs > self._scratch_reqs
            or vocab > self._scratch_vocab
            or spec_num > self._scratch_spec
            or gru_hidden_dim > self._scratch_gru
            or self._tokens_buf.device != device
        )
        if grow:
            self._scratch_reqs = max(num_reqs, self._scratch_reqs)
            self._scratch_vocab = max(vocab, self._scratch_vocab)
            self._scratch_spec = max(spec_num, self._scratch_spec)
            self._scratch_gru = max(gru_hidden_dim, self._scratch_gru)
            r = self._scratch_reqs
            v = self._scratch_vocab
            s = self._scratch_spec
            w = min(self.topk, v, self.budget)
            sb = w * s
            mn = sb + 1
            self._tokens_buf = torch.empty((r, sb), dtype=torch.long, device=device)
            self._depths_buf = torch.empty((r, sb), dtype=torch.long, device=device)
            self._parents_buf = torch.empty((r, mn), dtype=torch.long, device=device)
            self._path_scores_buf = torch.empty(
                (r, mn), dtype=torch.float32, device=device
            )
            self._frontier_buf = torch.empty((r, w), dtype=torch.long, device=device)
            self._frontier_pad = torch.zeros((r, w), dtype=torch.long, device=device)
            self._old_to_new_buf = torch.empty((r, mn), dtype=torch.long, device=device)
            self._req_arange = torch.arange(r, device=device, dtype=torch.long)
            self._width_arange = torch.arange(w, device=device, dtype=torch.long)
            self._node_arange = torch.arange(mn + 1, device=device, dtype=torch.long)
            self._budget_ids = torch.arange(
                1, self.budget + 1, device=device, dtype=torch.long
            )
            if need_proposal:
                # Proposal buffers stay FP32 (NPU IndexPut rejects BF16 selfRef).
                self._prop_temp_buf = torch.empty(
                    (r, mn, v), dtype=torch.float32, device=device
                )
                self._prop_layer_buf = torch.empty(
                    (r, w, v), dtype=torch.float32, device=device
                )
            if gru_hidden_dim > 0:
                self._hidden_buf = torch.empty(
                    (r, mn, self._scratch_gru), dtype=dtype, device=device
                )
        else:
            if need_proposal and self._prop_temp_buf is None:
                r = self._scratch_reqs
                v = self._scratch_vocab
                w = min(self.topk, v, self.budget)
                sb = w * self._scratch_spec
                mn = sb + 1
                self._prop_temp_buf = torch.empty(
                    (r, mn, v), dtype=torch.float32, device=device
                )
                self._prop_layer_buf = torch.empty(
                    (r, w, v), dtype=torch.float32, device=device
                )

    def build(
        self,
        draft_logits: torch.Tensor,
        out: TreeLayout,
        *,
        root_token_ids: torch.Tensor | None = None,
        draft_hidden: torch.Tensor | None = None,
        proposal_logits: torch.Tensor | None = None,
    ) -> TreeLayout:
        from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time

        tokens, depths, parent_ids, num_nodes = self._build_impl(
            draft_logits,
            out,
            root_token_ids=root_token_ids,
            draft_hidden=draft_hidden,
            proposal_logits=proposal_logits,
        )
        with tree_time("build_finalize_layout"):
            return finalize_tree_layout(out, tokens, depths, parent_ids, num_nodes)

    def _build_impl(
        self,
        draft_logits: torch.Tensor,
        out: TreeLayout,
        *,
        root_token_ids: torch.Tensor | None = None,
        draft_hidden: torch.Tensor | None = None,
        proposal_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self._build_impl_torch(
            draft_logits,
            root_token_ids=root_token_ids,
            draft_hidden=draft_hidden,
            proposal_logits=proposal_logits,
        )

    def _finalize_nodes(
        self,
        tokens: torch.Tensor,
        depths: torch.Tensor,
        parents: torch.Tensor,
        path_scores: torch.Tensor,
        num_nodes: int,
        budget: int,
        depth_bonus: float,
        num_reqs: int,
        vocab: int,
        *,
        need_proposal: bool,
        prop_temp: torch.Tensor | None,
        proposal_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
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
            old_to_new = self._old_to_new_buf[:num_reqs, : num_nodes + 1]
            old_to_new.zero_()
            new_ids = self._budget_ids.unsqueeze(0)
            old_to_new.scatter_(1, kept_ids, new_ids.expand(num_reqs, budget))
            parent_ids = torch.gather(old_to_new, 1, old_parents)
            if need_proposal and prop_temp is not None:
                proposal_logits.fill_(float("-inf"))
                proposal_logits[:, 0] = prop_temp[:, 0]
                gathered = torch.gather(
                    prop_temp,
                    1,
                    kept_ids.unsqueeze(-1).expand(-1, -1, vocab),
                )
                proposal_logits[:, 1 : budget + 1] = gathered
            num_nodes = budget
        else:
            tokens = tokens[:, :num_nodes]
            depths = depths[:, :num_nodes]
            parent_ids = parents[:, 1 : num_nodes + 1]
            if need_proposal and prop_temp is not None:
                proposal_logits.fill_(float("-inf"))
                proposal_logits[:, : num_nodes + 1] = prop_temp[:, : num_nodes + 1]
        return tokens, depths, parent_ids, num_nodes

    def _build_impl_torch(
        self,
        draft_logits: torch.Tensor,
        *,
        root_token_ids: torch.Tensor | None = None,
        draft_hidden: torch.Tensor | None = None,
        proposal_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        from vllm_ascend.worker.v2.spec_decode.tree.timer import (
            tree_time,
            tree_time_accum,
            tree_time_accum_flush,
        )

        budget = self.budget
        topk = self.topk
        depth_bonus = self.depth_bonus
        prefix_len = self.prefix_len
        correction_scorer = self.correction_scorer
        num_reqs, spec_num, vocab = draft_logits.shape
        device = draft_logits.device
        k = min(topk, vocab, budget)
        width = k
        supertree_budget = width * spec_num
        need_proposal = proposal_logits is not None

        with_correction = correction_scorer is not None
        gru_hidden_dim = (
            correction_scorer.gru_hidden_dim if with_correction else 0
        )

        with tree_time("build_precompute"):
            self._ensure_scratch(
                num_reqs,
                vocab,
                spec_num,
                device,
                need_proposal=need_proposal,
                gru_hidden_dim=gru_hidden_dim,
                dtype=draft_logits.dtype,
            )

            if with_correction:
                raw_c = self.candidate_size
                if raw_c is None:
                    candidate_count = k
                else:
                    candidate_count = max(k, min(int(raw_c), vocab))
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
            else:
                log_probs = torch.log_softmax(draft_logits.float(), dim=-1)

            max_nodes = supertree_budget + 1
            tokens = self._tokens_buf[:num_reqs, :supertree_budget]
            tokens.fill_(-1)
            depths = self._depths_buf[:num_reqs, :supertree_budget]
            depths.zero_()
            parents = self._parents_buf[:num_reqs, :max_nodes]
            parents.zero_()
            path_scores = self._path_scores_buf[:num_reqs, :max_nodes]
            path_scores.zero_()
            prop_temp = None
            if need_proposal:
                prop_temp = self._prop_temp_buf[:num_reqs, :max_nodes, :vocab]
                prop_temp.fill_(float("-inf"))
            hidden_states = None
            if with_correction:
                hidden_states = self._hidden_buf[
                    :num_reqs, :max_nodes, :gru_hidden_dim
                ]
                hidden_states.zero_()
                root_hidden = correction_scorer.update_hidden(
                    root_token_ids.reshape(-1), hidden_states[:, 0]
                )
                hidden_states[:, 0] = root_hidden

            frontier = self._frontier_buf[:num_reqs, :width]
            frontier.zero_()
            frontier_len = 1
            num_nodes = 0
            width_arange = self._width_arange[:width]
            req_idx = self._req_arange[:num_reqs]

        for child_depth in range(1, spec_num + 1):
            if num_nodes >= supertree_budget:
                break
            take = min(width, frontier_len * k, supertree_budget - num_nodes)
            if take <= 0:
                break
            depth_slot = child_depth - 1
            valid_parent = width_arange[None, :] < frontier_len

            with tree_time_accum("build_expand_score"):
                if with_correction:
                    if depth_slot < prefix_len:
                        # Needed later by build_expand_gru; not Domino-scored.
                        parent_hidden = torch.gather(
                            hidden_states,
                            1,
                            frontier.unsqueeze(-1).expand(
                                -1, -1, gru_hidden_dim
                            ),
                        )
                        logits = draft_logits[:, depth_slot].float()
                        top_vals, top_ids = torch.topk(logits, k=k, dim=-1)
                        log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
                        top_scores = (top_vals - log_z).unsqueeze(1).expand(
                            -1, width, -1
                        )
                        cand_ids = top_ids.unsqueeze(1).expand(-1, width, -1)
                        if prop_temp is not None:
                            full = logits.unsqueeze(1).expand(-1, width, -1)
                            _scatter_parent_proposal(
                                prop_temp,
                                frontier,
                                valid_parent,
                                full,
                                req_idx=req_idx,
                            )
                    else:
                        with tree_time_accum("build_score_gather"):
                            parent_hidden = torch.gather(
                                hidden_states,
                                1,
                                frontier.unsqueeze(-1).expand(
                                    -1, -1, gru_hidden_dim
                                ),
                            )
                        top_scores, cand_ids, cand_logits = (
                            _prefix_corrected_candidates(
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
                        )
                        if prop_temp is not None:
                            full = self._prop_layer_buf[:num_reqs, :width, :vocab]
                            full.fill_(float("-inf"))
                            ids = candidate_ids[:, depth_slot].unsqueeze(1).expand(
                                -1, width, -1
                            )
                            full.scatter_(2, ids, cand_logits.to(full.dtype))
                            _scatter_parent_proposal(
                                prop_temp,
                                frontier,
                                valid_parent,
                                full,
                                req_idx=req_idx,
                            )
                else:
                    top_vals, top_ids = torch.topk(
                        log_probs[:, depth_slot, :], k=k, dim=-1
                    )
                    top_scores = top_vals.unsqueeze(1).expand(-1, width, -1)
                    cand_ids = top_ids.unsqueeze(1).expand(-1, width, -1)
                    if prop_temp is not None:
                        full = (
                            draft_logits[:, depth_slot]
                            .float()
                            .unsqueeze(1)
                            .expand(-1, width, -1)
                        )
                        _scatter_parent_proposal(
                            prop_temp,
                            frontier,
                            valid_parent,
                            full,
                            req_idx=req_idx,
                        )

                top_scores = torch.where(
                    valid_parent[:, :, None],
                    top_scores,
                    torch.full_like(top_scores, float("-inf")),
                )

            with tree_time_accum("build_expand_select"):
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

                new_ids = (
                    self._node_arange[start : start + take]
                    .unsqueeze(0)
                    .expand(num_reqs, -1)
                )
                if take < width:
                    pad = self._frontier_pad[:num_reqs, : width - take]
                    pad.zero_()
                    new_ids = torch.cat([new_ids, pad], dim=-1)
                frontier = new_ids

            if with_correction:
                with tree_time_accum("build_expand_gru"):
                    parent_hidden_sel = parent_hidden.gather(
                        1,
                        parent_pos.unsqueeze(-1).expand(-1, -1, gru_hidden_dim),
                    )
                    child_hidden = correction_scorer.update_hidden(
                        sel_tokens.reshape(-1),
                        parent_hidden_sel.reshape(-1, gru_hidden_dim),
                    ).reshape(num_reqs, take, gru_hidden_dim)
                    hidden_states[:, start : start + take] = child_hidden

            frontier_len = take
            num_nodes += take

        tree_time_accum_flush()
        with tree_time("build_prune"):
            return self._finalize_nodes(
                tokens,
                depths,
                parents,
                path_scores,
                num_nodes,
                budget,
                depth_bonus,
                num_reqs,
                vocab,
                need_proposal=need_proposal,
                prop_temp=prop_temp,
                proposal_logits=proposal_logits,
            )
