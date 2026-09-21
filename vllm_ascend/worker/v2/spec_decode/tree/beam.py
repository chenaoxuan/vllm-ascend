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


def _select_with_spine(
    cand_flat: torch.Tensor, n_add: int, k: int
) -> torch.Tensor:
    """Keep the best child of frontier[:, 0] as selected[:, 0].

    Later depths take a global top-``n_add``; that can drop the greedy chain
    so packed-causal DSA never sees a depth>1 ancestor path.
    """
    spine = cand_flat[:, :k].argmax(dim=-1)
    if n_add <= 1:
        return spine.unsqueeze(1)
    drop = torch.arange(cand_flat.shape[1], device=cand_flat.device).unsqueeze(
        0
    ) == spine.unsqueeze(1)
    rest = cand_flat.masked_fill(drop, float("-inf")).topk(n_add - 1, dim=-1).indices
    return torch.cat([spine.unsqueeze(1), rest], dim=1)


def _spine_first_pool_indices(
    pool_parents: torch.Tensor,
    pool_scores: torch.Tensor,
    pool_depth: torch.Tensor,
    num_nodes: int,
    spec_num: int,
) -> torch.Tensor:
    """Permute pool slots so the best path is packed as a causal prefix.

    Packed-causal DSA still benefits from a spine-first layout. Tree verify
    additionally passes ancestor slots via ori_sparse_indices.
    """
    num_reqs, num_pool = pool_parents.shape
    device = pool_parents.device
    req_idx = torch.arange(num_reqs, device=device)
    used = torch.zeros(num_reqs, num_pool, dtype=torch.bool, device=device)
    inf = num_pool + spec_num + 1
    spine_rank = torch.full(
        (num_reqs, num_pool), inf, dtype=torch.long, device=device
    )
    parent = torch.full((num_reqs,), -1, dtype=pool_parents.dtype, device=device)
    for depth in range(spec_num):
        cand = (pool_depth == depth) & (pool_parents == parent.unsqueeze(1)) & ~used
        scores = pool_scores.masked_fill(~cand, float("-inf"))
        pick = scores.argmax(dim=-1)
        valid = cand.any(dim=-1)
        safe = pick.clamp(min=0)
        cur = spine_rank[req_idx, safe]
        spine_rank[req_idx, safe] = torch.where(
            valid, torch.full_like(cur, depth), cur
        )
        used[req_idx, safe] = used[req_idx, safe] | valid
        parent = torch.where(valid, pick, parent)
    rest_rank = inf + torch.arange(num_pool, device=device, dtype=torch.long)
    rank = torch.where(used, spine_rank, rest_rank.unsqueeze(0))
    return rank.topk(num_nodes, largest=False, dim=-1).indices


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
        proposal_logits: torch.Tensor | None = None,
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

        # Temp proposal indexed by root=0 and pool slot i -> i+1. Remapped later.
        # FP32 to match tree_proposal_logits; NPU IndexPut rejects BF16.
        max_pool = spec_num * k
        prop_temp = None
        if proposal_logits is not None:
            prop_temp = torch.full(
                (num_reqs, max_pool + 1, vocab),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )

        # Commit only the kept beam at each depth. Dumping every k-ary
        # expansion into the pool and then taking a global top-``budget``
        # by cumulative log-score keeps only depth 1–2 (deeper scores are
        # more negative), so greedy can never accept past 2 drafts.
        #
        # Keep frontier[:, 0] as the best-path spine and always expand it:
        # otherwise the global top-n_add at later depths can drop the greedy
        # chain, and DSA packed-causal verify has no working tree mask.
        remaining = int(budget)
        for depth in range(spec_num):
            if remaining <= 0:
                break
            batch = frontier_tokens.size(1)
            step_logits = _markov_correct_logits(
                draft_model, draft_logits[:, depth], frontier_tokens
            )
            if prop_temp is not None:
                req_idx = torch.arange(num_reqs, device=device)
                step_f = step_logits.to(dtype=prop_temp.dtype)
                for j in range(batch):
                    parent_pool = frontier_pools[:, j]
                    temp_id = torch.where(
                        parent_pool < 0,
                        torch.zeros((), dtype=torch.long, device=device),
                        parent_pool + 1,
                    ).clamp(max=prop_temp.shape[1] - 1)
                    prop_temp[req_idx, temp_id] = step_f[:, j]
            log_probs = torch.log_softmax(step_logits.float(), dim=-1)
            top_vals, top_ids = log_probs.topk(k, dim=-1)
            if draft_model is not None:
                top_ids = draft_model.map_draft_to_target(top_ids)
            candidate_scores = frontier_scores.unsqueeze(-1) + top_vals
            num_candidates = batch * k
            depths_left = spec_num - depth
            if depth == 0:
                n_add = min(k, remaining)
            elif depths_left == 1:
                n_add = min(k, remaining)
            else:
                n_add = min(k, max(1, remaining // depths_left))
            n_add = min(n_add, num_candidates, remaining)
            cand_flat = candidate_scores.reshape(num_reqs, -1)
            top_ids_flat = top_ids.reshape(num_reqs, -1)
            parent_flat = frontier_pools.repeat_interleave(k, dim=-1)
            selected = _select_with_spine(cand_flat, n_add, k)
            sel_tokens = torch.gather(top_ids_flat, 1, selected)
            sel_scores = torch.gather(cand_flat, 1, selected)
            sel_parents = torch.gather(parent_flat, 1, selected)

            pool_tokens = torch.cat([pool_tokens, sel_tokens], dim=-1)
            pool_scores = torch.cat([pool_scores, sel_scores], dim=-1)
            pool_parents = torch.cat([pool_parents, sel_parents], dim=-1)
            pool_depth = torch.cat(
                [
                    pool_depth,
                    torch.full(
                        (num_reqs, n_add),
                        depth,
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=-1,
            )
            frontier_tokens = sel_tokens
            frontier_scores = sel_scores
            base = pool_tokens.size(-1) - n_add
            frontier_pools = base + torch.arange(
                n_add, device=device, dtype=torch.long
            ).unsqueeze(0).expand(num_reqs, -1)
            remaining -= n_add

        num_pool = pool_tokens.size(-1)
        num_nodes = min(int(budget), num_pool)
        packed = _spine_first_pool_indices(
            pool_parents, pool_scores, pool_depth, num_nodes, spec_num
        )

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
        finalize_tree_layout(out, tokens, depths, parent_ids, num_nodes)

        if proposal_logits is not None and prop_temp is not None:
            proposal_logits.fill_(float("-inf"))
            proposal_logits[:, 0] = prop_temp[:, 0]
            req_idx = torch.arange(num_reqs, device=device)
            for pool_idx in range(num_pool):
                final_id = remap[:, pool_idx]
                has = final_id > 0
                proposal_logits[req_idx, final_id.clamp(min=0)] = torch.where(
                    has.unsqueeze(-1),
                    prop_temp[:, pool_idx + 1],
                    proposal_logits[req_idx, final_id.clamp(min=0)],
                )
        return out
