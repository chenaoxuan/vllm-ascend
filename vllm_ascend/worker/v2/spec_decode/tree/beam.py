import torch

from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout


def _markov_correct_logits(
    draft_model,
    depth_logits: torch.Tensor,
    frontier_tokens: torch.Tensor,
) -> torch.Tensor:
    """Beam-only one-depth logit correction (optional DSpark Markov bias).

    When ``draft_model`` is set, mirrors ``DSparkSpeculator._sample_sequential``:
    bias from the previous **target-vocab** token via ``markov_embed`` +
    ``markov_bias``. When ``draft_model`` is None, expand the shared-depth base
    logits across the frontier.
    """
    base = depth_logits.unsqueeze(1)
    if draft_model is None:
        return base.expand(-1, frontier_tokens.size(1), -1)
    markov_emb = draft_model.markov_embed(frontier_tokens)
    # NPU unquantized gemm is 2D-only; flatten then restore the frontier layout.
    bias = draft_model.markov_bias(markov_emb.reshape(-1, markov_emb.shape[-1]))
    return base + bias.view(*markov_emb.shape[:-1], -1)


class BeamTreeBuilder(TreeBuilder):
    """Level-wise beam (PCTree) expansion; optional DSpark Markov bias."""

    required_backend = "dspark"
    method = "beam"

    def __init__(self, budget: int, topk: int, draft_model=None):
        super().__init__(budget, topk)
        self.draft_model = draft_model

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
        draft_model = self.draft_model
        num_reqs, spec_num, vocab = draft_logits.shape
        k = min(topk, vocab)
        device = draft_logits.device

        frontier_tokens = root_token_ids.unsqueeze(1)
        frontier_scores = torch.zeros(num_reqs, 1, dtype=torch.float32, device=device)
        frontier_pools = torch.full((num_reqs, 1), -1, dtype=torch.long, device=device)

        pool_tokens = torch.empty(num_reqs, 0, dtype=torch.long, device=device)
        pool_scores = torch.empty(num_reqs, 0, dtype=torch.float32, device=device)
        pool_parents = torch.empty(num_reqs, 0, dtype=torch.long, device=device)
        pool_depth = torch.empty(num_reqs, 0, dtype=torch.long, device=device)

        for depth in range(spec_num):
            batch = frontier_tokens.size(1)
            step_logits = _markov_correct_logits(
                draft_model, draft_logits[:, depth], frontier_tokens
            )
            log_probs = torch.log_softmax(step_logits.float(), dim=-1)
            top_vals, top_ids = log_probs.topk(k, dim=-1)
            if draft_model is not None:
                top_ids = draft_model.map_draft_to_target(top_ids)
            candidate_scores = frontier_scores.unsqueeze(-1) + top_vals
            num_candidates = batch * k

            pool_tokens = torch.cat(
                [pool_tokens, top_ids.reshape(num_reqs, -1)], dim=-1
            )
            pool_scores = torch.cat(
                [pool_scores, candidate_scores.reshape(num_reqs, -1)], dim=-1
            )
            pool_parents = torch.cat(
                [pool_parents, frontier_pools.repeat_interleave(k, dim=-1)], dim=-1
            )
            pool_depth = torch.cat(
                [
                    pool_depth,
                    torch.full(
                        (num_reqs, num_candidates),
                        depth,
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=-1,
            )

            keep = min(k, num_candidates)
            cand_flat = candidate_scores.reshape(num_reqs, -1)
            top_ids_flat = top_ids.reshape(num_reqs, -1)
            selected = cand_flat.topk(keep, dim=-1).indices
            frontier_tokens = torch.gather(top_ids_flat, 1, selected)
            frontier_scores = torch.gather(cand_flat, 1, selected)
            frontier_pools = (pool_tokens.size(-1) - num_candidates) + selected

        num_pool = pool_tokens.size(-1)
        num_nodes = min(int(budget), num_pool)
        order = torch.argsort(pool_depth, dim=-1, stable=True)
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
        packed = order[:, :num_nodes].sort(dim=-1).values

        remap = torch.zeros(num_reqs, num_pool, dtype=torch.long, device=device)
        node_ids = torch.arange(
            1, num_nodes + 1, dtype=torch.long, device=device
        ).unsqueeze(0)
        remap.scatter_(1, packed, node_ids.expand(num_reqs, num_nodes))
        pool_parent_packed = torch.gather(pool_parents, 1, packed)
        non_root = pool_parent_packed >= 0
        raw_parent = torch.gather(remap, 1, pool_parent_packed.clamp(min=0))
        parent_ids = torch.where(
            non_root, raw_parent, torch.zeros_like(raw_parent)
        )

        tokens = torch.gather(pool_tokens, 1, packed)
        depths = (torch.gather(pool_depth, 1, packed) + 1).to(torch.int32)
        return finalize_tree_layout(out, tokens, depths, parent_ids, num_nodes)
