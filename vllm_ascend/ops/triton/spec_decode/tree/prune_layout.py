"""Budget pruning for the shared speculative-decoding tree builder."""

from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_reqs", "num_candidates", "num_selected"])
def prune_tree_layout_kernel(
    priority_ptr,
    tokens_ptr,
    depths_ptr,
    parents_ptr,
    out_tokens_ptr,
    out_depths_ptr,
    out_parents_ptr,
    num_reqs,
    num_candidates,
    num_selected,
    stride_priority_r,
    stride_tokens_r,
    stride_depths_r,
    stride_parents_r,
    stride_out_tokens_r,
    stride_out_depths_r,
    stride_out_parents_r,
    N_CAND_PAD: tl.constexpr,
    BUDGET: tl.constexpr,
    BUDGET_PAD: tl.constexpr,
):
    """Select, order and remap a budget-sized ancestor-closed node pool.

    The reference implementation first stable-sorts scores, sorts the kept
    pool IDs, and then builds an old-ID to new-ID map for the parent column.
    This kernel keeps the same ordering rules while doing all three steps in
    one request program.  The pool is small (the speculative tree width times
    its depth), so a register-resident selection sort avoids materializing the
    intermediate index tensors used by Torch.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, BUDGET_PAD)
    candidate_lanes = tl.arange(0, N_CAND_PAD)
    candidate_ids = candidate_lanes.to(tl.int64)
    big = tl.full((), N_CAND_PAD, tl.int64)
    neg_inf = -float("inf")

    req = pid
    while req < num_reqs:
        valid_candidates = candidate_lanes < num_candidates
        scores = tl.load(
            priority_ptr + req * stride_priority_r + candidate_lanes,
            mask=valid_candidates,
            other=neg_inf,
        ).to(tl.float32)

        # Stable descending top-k: NaNs sort before finite values and the
        # smallest pool ID wins every tie, matching Torch stable argsort.
        taken = ~valid_candidates
        selected = tl.full((BUDGET_PAD,), -1, tl.int64)
        for rank in range(BUDGET_PAD):
            available = valid_candidates & ~taken
            is_nan = available & (scores != scores)
            has_nan = tl.sum(is_nan.to(tl.int32), axis=0) > 0
            values = tl.where(available & ~is_nan, scores, neg_inf)
            best_value = tl.max(values, axis=0)
            is_best = tl.where(has_nan, is_nan, (values == best_value) & available)
            best = tl.min(tl.where(is_best, candidate_ids, big), axis=0)
            selected = tl.where(
                lanes == rank,
                tl.where(rank < num_selected, best, -1),
                selected,
            )
            taken = taken | (candidate_ids == best)

        # The output layout is ordered by old node ID.  Repeated min over the
        # selected IDs is stable and makes every selected ancestor's new ID
        # available for the parent remap below.
        remaining = selected
        ordered = tl.full((BUDGET_PAD,), -1, tl.int64)
        for slot in range(BUDGET_PAD):
            minimum = tl.min(tl.where(remaining >= 0, remaining, big), axis=0)
            ordered = tl.where(
                lanes == slot,
                tl.where(slot < num_selected, minimum, -1),
                ordered,
            )
            remaining = tl.where(remaining == minimum, -1, remaining)

        valid_output = lanes < num_selected
        safe_slot = tl.where(valid_output, ordered, 0)
        old_node_id = safe_slot + 1
        token = tl.load(tokens_ptr + req * stride_tokens_r + safe_slot, valid_output, other=-1)
        depth = tl.load(depths_ptr + req * stride_depths_r + safe_slot, valid_output, other=0)
        old_parent = tl.load(parents_ptr + req * stride_parents_r + old_node_id, valid_output, other=0)

        # ``ordered`` stores zero-based pool slots while parents store
        # one-based node IDs.  The max returns the new one-based slot ID.
        parent_match = ((ordered[None, :] + 1) == old_parent[:, None]) & valid_output[None, :]
        new_parent = tl.max(tl.where(parent_match, lanes[None, :] + 1, 0), axis=1)
        parent = tl.where(old_parent > 0, new_parent, 0)

        tl.store(
            out_tokens_ptr + req * stride_out_tokens_r + lanes,
            tl.where(valid_output, token, -1),
            lanes < BUDGET,
        )
        tl.store(
            out_depths_ptr + req * stride_out_depths_r + lanes,
            tl.where(valid_output, depth, 0),
            lanes < BUDGET,
        )
        tl.store(
            out_parents_ptr + req * stride_out_parents_r + lanes,
            tl.where(valid_output, parent, -1),
            lanes < BUDGET,
        )
        req += nprog


def prune_tree_layout_triton(
    priority,
    tokens,
    depths,
    parents,
    out_tokens,
    out_depths,
    out_parents,
    num_nodes: int,
) -> None:
    """Run budget pruning without allocating Torch index intermediates."""
    num_reqs, num_candidates = priority.shape
    vec = get_vectorcore_num()
    grid = min(max(num_reqs, 1), max(vec, 1))
    n_candidates_pad = triton.next_power_of_2(max(num_candidates, 1))
    budget = out_tokens.shape[1]
    prune_tree_layout_kernel[(grid,)](
        priority,
        tokens,
        depths,
        parents,
        out_tokens,
        out_depths,
        out_parents,
        num_reqs,
        num_candidates,
        num_nodes,
        priority.stride(0),
        tokens.stride(0),
        depths.stride(0),
        parents.stride(0),
        out_tokens.stride(0),
        out_depths.stride(0),
        out_parents.stride(0),
        N_CAND_PAD=n_candidates_pad,
        BUDGET=budget,
        BUDGET_PAD=triton.next_power_of_2(budget),
    )
