from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder, TreeBuildState


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
    """Prefix-only Domino candidate correction for one depth (not Markov).

    Returns the corrected top-k scores and token IDs over the base top-C set.
    """
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


def expand_prefix_layer(
    top_scores: torch.Tensor,
    cand_ids: torch.Tensor,
    path_scores: torch.Tensor,
    frontier: torch.Tensor,
    tokens: torch.Tensor,
    depths: torch.Tensor,
    parents: torch.Tensor,
    selection_scratch: torch.Tensor,
    *,
    live: int,
    num_nodes: int,
    child_depth: int,
    force_torch: bool = False,
) -> None:
    """Select ``width`` children for one expand layer and write node slices.

    ``live`` / ``num_nodes`` / ``child_depth`` are host ints (from depth).
    ``selection_scratch`` holds Triton's intermediate flat candidate indices.
    ``force_torch`` skips Triton (ACLGraph capture cannot use that kernel).
    """
    from vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch import (
        use_tree_triton,
    )

    if not force_torch and use_tree_triton():
        from vllm_ascend.ops.triton.spec_decode.tree.prefix_expand import (
            expand_prefix_layer_triton,
        )
        from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time_accum

        with tree_time_accum("build_expand_select"):
            expand_prefix_layer_triton(
                top_scores,
                cand_ids,
                path_scores,
                frontier,
                tokens,
                depths,
                parents,
                selection_scratch,
                live,
                num_nodes,
                child_depth,
            )
    else:
        _expand_prefix_layer_torch(
            top_scores,
            cand_ids,
            path_scores,
            frontier,
            tokens,
            depths,
            parents,
            live=live,
            num_nodes=num_nodes,
            child_depth=child_depth,
            timed=not force_torch,
        )


def _expand_prefix_layer_torch(
    top_scores: torch.Tensor,
    cand_ids: torch.Tensor,
    path_scores: torch.Tensor,
    frontier: torch.Tensor,
    tokens: torch.Tensor,
    depths: torch.Tensor,
    parents: torch.Tensor,
    *,
    live: int,
    num_nodes: int,
    child_depth: int,
    timed: bool = True,
) -> None:
    from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time_accum

    ctx = tree_time_accum("build_expand_select") if timed else nullcontext()
    with ctx:
        num_reqs, width, _ = top_scores.shape
        if live < width:
            valid_parent = torch.arange(width, device=top_scores.device) < live
            top_scores = torch.where(
                valid_parent[None, :, None],
                top_scores,
                torch.full_like(top_scores, float("-inf")),
            )
        parent_scores = torch.gather(path_scores, 1, frontier)
        cand_scores = parent_scores.unsqueeze(-1) + top_scores
        flat = cand_scores.reshape(num_reqs, -1)
        # Match Triton's score-descending, flat-index-ascending selection.
        # torch.topk does not preserve index order on ties, which can change
        # the retained parents (and Domino GRU states) between eager and replay.
        sel = torch.argsort(flat, dim=-1, descending=True, stable=True)[:, :width]
        vals = torch.gather(flat, 1, sel)
        pos = sel // width
        sel_tokens = torch.gather(cand_ids.reshape(num_reqs, -1), 1, sel)
        sel_parents = torch.gather(frontier, 1, pos)
        start = num_nodes + 1
        tokens[:, num_nodes : num_nodes + width] = sel_tokens
        depths[:, num_nodes : num_nodes + width] = child_depth
        parents[:, start : start + width] = sel_parents
        path_scores[:, start : start + width] = vals


@dataclass
class DFlashScoreContext:
    log_probs: torch.Tensor


@dataclass
class DominoScoreContext:
    scorer: DominoCorrectionScorer
    draft_logits: torch.Tensor
    hidden_states: torch.Tensor
    candidate_vals: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_weight: torch.Tensor
    z_parts: torch.Tensor
    candidate_bias: torch.Tensor | None


PrefixScoreContext = DFlashScoreContext | DominoScoreContext


class PrefixTreeBuilder(TreeBuilder[PrefixScoreContext]):
    """Uniform-width tree with optional Domino prefix correction."""

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
        self.gru_hidden_dim = (
            correction_scorer.gru_hidden_dim if correction_scorer is not None else 0
        )
        self._hidden_buf: torch.Tensor | None = None
        self._selection_buf: torch.Tensor | None = None

    def _beam_width(self, vocab: int) -> int:
        return min(self.topk, vocab, self.budget)

    def _ensure_scratch(
        self,
        num_reqs: int,
        vocab: int,
        spec_num: int,
        device: torch.device,
    ) -> None:
        super()._ensure_scratch(num_reqs, vocab, spec_num, device)
        if self._scratch_frozen:
            return
        selection_shape = self._frontier_buf.shape
        if (
            self._selection_buf is None
            or self._selection_buf.shape != selection_shape
            or self._selection_buf.device != device
        ):
            self._selection_buf = torch.empty(
                selection_shape, dtype=torch.long, device=device
            )
        scorer = self.correction_scorer
        if scorer is None:
            return
        shape = (*self._parents_buf.shape, self.gru_hidden_dim)
        # The LM head may emit FP32 logits while the GRU weights remain BF16.
        # Eager builds and graph preparation must use the same GRU state dtype.
        dtype = scorer.gru_w_hh.dtype
        if (
            self._hidden_buf is None
            or self._hidden_buf.shape != shape
            or self._hidden_buf.device != device
            or self._hidden_buf.dtype != dtype
        ):
            self._hidden_buf = torch.empty(shape, dtype=dtype, device=device)

    def _prepare_scores(
        self,
        state: TreeBuildState,
        draft_logits: torch.Tensor,
        root_token_ids: torch.Tensor,
        draft_hidden: torch.Tensor | None,
    ) -> PrefixScoreContext:
        num_reqs, spec_num, vocab = draft_logits.shape
        width = state.width
        scorer = self.correction_scorer
        if scorer is None:
            return DFlashScoreContext(torch.log_softmax(draft_logits.float(), dim=-1))

        raw_c = self.candidate_size
        candidate_count = (
            width if raw_c is None else max(width, min(int(raw_c), vocab))
        )
        candidate_vals, candidate_ids = torch.topk(
            draft_logits.float(), k=candidate_count, dim=-1
        )
        flat_cids = candidate_ids.reshape(-1)
        candidate_weight = scorer.fc2_weight.index_select(0, flat_cids).view(
            num_reqs, spec_num, candidate_count, -1
        )
        candidate_bias = None
        if scorer.fc2_bias is not None:
            candidate_bias = scorer.fc2_bias.index_select(0, flat_cids).view(
                num_reqs, spec_num, candidate_count
            )
        z_parts = scorer.project_z(draft_hidden[:, :spec_num])
        hidden_states = self._hidden_buf[
            :num_reqs, : state.parents.shape[1], : self.gru_hidden_dim
        ]
        hidden_states.zero_()
        hidden_states[:, 0] = scorer.update_hidden(
            root_token_ids.reshape(-1), hidden_states[:, 0]
        )
        return DominoScoreContext(
            scorer=scorer,
            draft_logits=draft_logits,
            hidden_states=hidden_states,
            candidate_vals=candidate_vals,
            candidate_ids=candidate_ids,
            candidate_weight=candidate_weight,
            z_parts=z_parts,
            candidate_bias=candidate_bias,
        )

    def _score_layer(
        self, state: TreeBuildState, ctx: PrefixScoreContext, depth_slot: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = state.width
        if isinstance(ctx, DFlashScoreContext):
            top_vals, top_ids = torch.topk(
                ctx.log_probs[:, depth_slot, :], k=width, dim=-1
            )
            return (
                top_vals.unsqueeze(1).expand(-1, width, -1),
                top_ids.unsqueeze(1).expand(-1, width, -1),
            )

        if depth_slot < self.prefix_len:
            logits = ctx.draft_logits[:, depth_slot].float()
            top_vals, top_ids = torch.topk(logits, k=width, dim=-1)
            log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
            return (
                (top_vals - log_z).unsqueeze(1).expand(-1, width, -1),
                top_ids.unsqueeze(1).expand(-1, width, -1),
            )
        parent_hidden = torch.gather(
            ctx.hidden_states,
            1,
            state.frontier.unsqueeze(-1).expand(-1, -1, self.gru_hidden_dim),
        )
        return _prefix_corrected_candidates(
            ctx.scorer,
            ctx.z_parts[:, depth_slot],
            parent_hidden,
            ctx.candidate_vals[:, depth_slot],
            ctx.candidate_ids[:, depth_slot],
            ctx.candidate_weight[:, depth_slot],
            ctx.candidate_bias[:, depth_slot]
            if ctx.candidate_bias is not None
            else None,
            width,
        )

    def _expand_layer(
        self,
        state: TreeBuildState,
        ctx: PrefixScoreContext,
        top_scores: torch.Tensor,
        cand_ids: torch.Tensor,
        child_depth: int,
        graph_safe: bool,
    ) -> tuple[torch.Tensor, int]:
        width = state.width
        expand_prefix_layer(
            top_scores,
            cand_ids,
            state.path_scores,
            state.frontier,
            state.tokens,
            state.depths,
            state.parents,
            self._selection_buf[: state.tokens.shape[0], :width],
            live=1 if child_depth == 1 else width,
            num_nodes=state.num_nodes,
            child_depth=child_depth,
            force_torch=graph_safe,
        )
        start = state.num_nodes + 1
        frontier = self._node_arange[start : start + width].unsqueeze(0).expand(
            state.tokens.shape[0], -1
        )
        return frontier, width

    def _update_state(
        self,
        state: TreeBuildState,
        ctx: PrefixScoreContext,
        n_add: int,
        *,
        graph_safe: bool,
    ) -> None:
        if isinstance(ctx, DFlashScoreContext):
            return
        from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time_accum

        with nullcontext() if graph_safe else tree_time_accum("build_expand_gru"):
            start = state.num_nodes + 1
            # Selected parent IDs also cover layers before prefix correction
            # starts, so every retained branch advances its GRU state.
            selected_parents = state.parents[:, start : start + n_add]
            parent_hidden = torch.gather(
                ctx.hidden_states,
                1,
                selected_parents.unsqueeze(-1).expand(-1, -1, self.gru_hidden_dim),
            )
            child_hidden = ctx.scorer.update_hidden(
                state.tokens[:, state.num_nodes : state.num_nodes + n_add].reshape(-1),
                parent_hidden.reshape(-1, self.gru_hidden_dim),
            ).reshape(state.tokens.shape[0], n_add, self.gru_hidden_dim)
            ctx.hidden_states[:, start : start + n_add] = child_hidden

    def _select_nodes(
        self, state: TreeBuildState, ctx: PrefixScoreContext
    ) -> torch.Tensor | None:
        if state.num_nodes <= self.budget:
            return None
        return _select_topb_nodes(
            state.path_scores[:, 1 : state.num_nodes + 1],
            state.depths[:, : state.num_nodes],
            self.budget,
            self.depth_bonus,
        )
