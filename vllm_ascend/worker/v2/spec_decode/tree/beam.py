from dataclasses import dataclass

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
    """Per-build logits and target-vocabulary frontier tokens stay on device."""

    draft_logits: torch.Tensor
    frontier_tokens: torch.Tensor


class BeamTreeBuilder(TreeBuilder[BeamScoreContext]):
    """Level-wise beam expansion with optional DSpark Markov bias."""

    required_backend = "dspark"
    method = "beam"

    def __init__(self, budget: int, topk: int, draft_model=None):
        super().__init__(budget, topk)
        self.draft_model = draft_model

    def _beam_width(self, vocab: int) -> int:
        return min(self.topk, vocab)

    def _pool_capacity(self, width: int, spec_num: int) -> int:
        # Root expands width children; later frontiers expand width squared.
        return width + (spec_num - 1) * width * width

    def _prepare_scores(
        self,
        state: TreeBuildState,
        draft_logits: torch.Tensor,
        root_token_ids: torch.Tensor,
        draft_hidden: torch.Tensor | None,
    ) -> BeamScoreContext:
        state.frontier = state.frontier[:, :1]
        return BeamScoreContext(draft_logits, root_token_ids.unsqueeze(1))

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
        start = state.num_nodes + 1
        # Keep all children in the pool; only the selected beam expands again.
        state.tokens[:, state.num_nodes : state.num_nodes + num_candidates] = top_ids_flat
        state.depths[:, state.num_nodes : state.num_nodes + num_candidates] = child_depth
        state.parents[:, start : start + num_candidates] = parent_flat
        state.path_scores[:, start : start + num_candidates] = cand_flat

        selected = cand_flat.topk(min(k, num_candidates), dim=-1).indices
        ctx.frontier_tokens = torch.gather(top_ids_flat, 1, selected)
        return start + selected, num_candidates

    def _select_nodes(
        self, state: TreeBuildState, ctx: BeamScoreContext
    ) -> torch.Tensor:
        num_pool = state.num_nodes
        num_nodes = min(self.budget, num_pool)
        pool_scores = state.path_scores[:, 1 : num_pool + 1]
        order = torch.argsort(state.depths[:, :num_pool], dim=-1, stable=True)
        order = torch.gather(
            order,
            1,
            torch.argsort(
                torch.gather(pool_scores, 1, order),
                dim=-1,
                descending=True,
                stable=True,
            ),
        )
        return order[:, :num_nodes].sort(dim=-1).values + 1
