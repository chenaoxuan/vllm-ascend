import os
from dataclasses import dataclass, field

import torch

from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder, TreeBuildState


def _markov_correct_logits(
    draft_model,
    depth_logits: torch.Tensor,
    frontier_tokens: torch.Tensor,
) -> torch.Tensor:
    """Add DSpark bias in draft-vocab space from target-vocab frontier tokens."""
    base = depth_logits.unsqueeze(1)
    if draft_model is None:
        return base.expand(-1, frontier_tokens.size(1), -1)
    markov_emb = draft_model.markov_embed(frontier_tokens)
    # NPU unquantized gemm is 2D-only; restore the frontier shape afterward.
    bias = draft_model.markov_bias(markov_emb.reshape(-1, markov_emb.shape[-1]))
    return base + bias.view(*markov_emb.shape[:-1], -1)


@dataclass
class BeamScoreContext:
    """Per-build device tensors; the greedy-pool list holds device node IDs."""

    draft_logits: torch.Tensor
    frontier_tokens: torch.Tensor
    req_ids: torch.Tensor | None = None
    greedy_frontier: torch.Tensor | None = None
    greedy_candidate: torch.Tensor | None = None
    greedy_pools: list[torch.Tensor] = field(default_factory=list)


class BeamTreeBuilder(TreeBuilder[BeamScoreContext]):
    """Level-wise beam expansion with optional DSpark Markov bias."""

    required_backend = "dspark"
    method = "beam"

    def __init__(self, budget: int, topk: int, draft_model=None):
        super().__init__(budget, topk)
        self.draft_model = draft_model
        self.keep_greedy = os.getenv("VLLM_ASCEND_TREE_KEEP_GREEDY") == "1"

    def _beam_width(self, vocab: int) -> int:
        return min(self.topk, vocab)

    def _prepare_scores(
        self,
        state: TreeBuildState,
        draft_logits: torch.Tensor,
        root_token_ids: torch.Tensor,
        draft_hidden: torch.Tensor | None,
    ) -> BeamScoreContext:
        state.frontier = state.frontier[:, :1]
        ctx = BeamScoreContext(draft_logits, root_token_ids.unsqueeze(1))
        if self.keep_greedy:
            num_reqs, spec_num, _ = draft_logits.shape
            if self.budget < spec_num:
                raise ValueError("Keeping the complete greedy chain requires budget >= depth")
            ctx.req_ids = torch.arange(num_reqs, device=draft_logits.device)
            ctx.greedy_frontier = torch.zeros(
                num_reqs, dtype=torch.long, device=draft_logits.device
            )
        return ctx

    def _score_layer(
        self,
        state: TreeBuildState,
        ctx: BeamScoreContext,
        depth_slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_logits = _markov_correct_logits(
            self.draft_model, ctx.draft_logits[:, depth_slot], ctx.frontier_tokens
        )
        log_probs = torch.log_softmax(step_logits.float(), dim=-1)
        top_vals, top_ids = log_probs.topk(state.width, dim=-1)
        if self.keep_greedy:
            # topk can omit argmax on exact-logit ties. Keep the sequential edge.
            greedy_token = step_logits[ctx.req_ids, ctx.greedy_frontier].argmax(-1)
            local_ids = top_ids[ctx.req_ids, ctx.greedy_frontier]
            present = (local_ids == greedy_token[:, None]).any(-1)
            last_id = torch.where(present, local_ids[:, -1], greedy_token)
            top_ids[ctx.req_ids, ctx.greedy_frontier, -1] = last_id
            top_vals[ctx.req_ids, ctx.greedy_frontier, -1] = log_probs[
                ctx.req_ids, ctx.greedy_frontier, last_id
            ]
            greedy_child = (
                top_ids[ctx.req_ids, ctx.greedy_frontier] == greedy_token[:, None]
            ).long().argmax(-1)
            ctx.greedy_candidate = ctx.greedy_frontier * state.width + greedy_child

        if self.draft_model is not None:
            top_ids = self.draft_model.map_draft_to_target(top_ids)
        return top_vals, top_ids

    def _expand_layer(
        self,
        state: TreeBuildState,
        ctx: BeamScoreContext,
        top_scores: torch.Tensor,
        cand_ids: torch.Tensor,
        child_depth: int,
        graph_safe: bool,
    ) -> tuple[torch.Tensor, int]:
        # Shape values and slice bounds are host ints; selections stay on device.
        num_reqs, batch, k = top_scores.shape
        frontier_scores = torch.gather(state.path_scores, 1, state.frontier)
        candidate_scores = frontier_scores.unsqueeze(-1) + top_scores
        num_candidates = batch * k
        cand_flat = candidate_scores.reshape(num_reqs, -1)
        top_ids_flat = cand_ids.reshape(num_reqs, -1)
        parent_flat = state.frontier.repeat_interleave(k, dim=-1)
        selected = cand_flat.topk(min(k, num_candidates), dim=-1).indices
        if self.keep_greedy:
            present = (selected == ctx.greedy_candidate[:, None]).any(-1)
            selected[:, -1] = torch.where(
                present, selected[:, -1], ctx.greedy_candidate
            )
            ctx.greedy_frontier = (
                selected == ctx.greedy_candidate[:, None]
            ).long().argmax(-1)

        sel_tokens = torch.gather(top_ids_flat, 1, selected)
        sel_scores = torch.gather(cand_flat, 1, selected)
        sel_parents = torch.gather(parent_flat, 1, selected)
        n_add = selected.shape[1]
        start = state.num_nodes + 1
        state.tokens[:, state.num_nodes : state.num_nodes + n_add] = sel_tokens
        state.depths[:, state.num_nodes : state.num_nodes + n_add] = child_depth
        state.parents[:, start : start + n_add] = sel_parents
        state.path_scores[:, start : start + n_add] = sel_scores
        ctx.frontier_tokens = sel_tokens
        frontier = self._node_arange[start : start + n_add].unsqueeze(0).expand(
            num_reqs, -1
        )
        if self.keep_greedy:
            ctx.greedy_pools.append(frontier[ctx.req_ids, ctx.greedy_frontier])
        return frontier, n_add

    def _select_nodes(
        self, state: TreeBuildState, ctx: BeamScoreContext
    ) -> torch.Tensor:
        num_pool = state.num_nodes
        num_nodes = min(self.budget, num_pool)
        pool_scores = state.path_scores[:, 1 : num_pool + 1]
        # Stable score order puts ancestors first, including exact-score ties.
        if self.keep_greedy:
            priority = pool_scores.clone()
            greedy_slots = torch.stack(ctx.greedy_pools, dim=1) - 1
            priority.scatter_(1, greedy_slots, float("inf"))
        else:
            priority = pool_scores
        packed = priority.argsort(dim=-1, descending=True, stable=True)[:, :num_nodes]
        packed_depth = torch.gather(state.depths[:, :num_pool], 1, packed)
        packed_rank = packed_depth * num_pool + packed
        packed = torch.gather(packed, 1, packed_rank.argsort(dim=-1))
        return packed + 1
