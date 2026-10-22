from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs", "budget", "spec_len"])
def greedy_tree_reject_kernel(
    tokens_ptr,
    first_child_ptr,
    next_sibling_ptr,
    target_ids_ptr,
    sampled_ptr,
    path_ptr,
    num_reqs,
    budget,
    spec_len,
    stride_tokens_r,
    stride_fc_r,
    stride_ns_r,
    stride_tgt_r,
    stride_sampled_r,
    stride_path_r,
):
    # One request per program. Runtime tl.range — constexpr BUDGET×SPEC_LEN
    # unroll segfaults on Ascend for typical tree budgets.
    req = tl.program_id(0)
    if req >= num_reqs:
        return

    current = tl.full((), 0, tl.int64)
    alive = tl.full((), 1, tl.int32)
    for n_out in tl.range(0, spec_len):
        # current is a node id in [0, budget].
        t = tl.load(target_ids_ptr + req * stride_tgt_r + current)
        sampled_slot = sampled_ptr + req * stride_sampled_r + n_out
        old = tl.load(sampled_slot)
        tl.store(sampled_slot, tl.where(alive == 1, t, old))
        child = tl.load(first_child_ptr + req * stride_fc_r + current).to(tl.int64)
        found_child = tl.full((), -1, tl.int64)
        for _ in tl.range(0, budget):
            valid = (alive == 1) & (found_child < 0) & (child >= 0)
            slot = tl.where(child > 0, child - 1, 0)
            tok = tl.load(tokens_ptr + req * stride_tokens_r + slot)
            match = valid & (tok == t)
            found_child = tl.where(match, child, found_child)
            nxt = tl.load(
                next_sibling_ptr
                + req * stride_ns_r
                + tl.where(child > 0, child, 0)
            ).to(tl.int64)
            child = tl.where(valid & (found_child < 0), nxt, child)
        found = (alive == 1) & (found_child >= 0)
        path_slot = path_ptr + req * stride_path_r + n_out
        old_p = tl.load(path_slot)
        tl.store(path_slot, tl.where(found, found_child, old_p))
        current = tl.where(found, found_child, current)
        alive = tl.where(found, 1, 0)

    t = tl.load(target_ids_ptr + req * stride_tgt_r + current)
    bonus_slot = sampled_ptr + req * stride_sampled_r + spec_len
    old = tl.load(bonus_slot)
    tl.store(bonus_slot, tl.where(alive == 1, t, old))


def greedy_tree_reject_triton(
    tokens,
    first_child,
    next_sibling,
    target_token_ids,
    sampled_token_ids,
    path_out,
    spec_len: int,
) -> None:
    tokens = tokens.contiguous()
    first_child = first_child.contiguous()
    next_sibling = next_sibling.contiguous()
    target_token_ids = target_token_ids.contiguous()
    num_reqs, budget = tokens.shape
    # One program per request (no grid-stride while).
    grid = max(num_reqs, 1)
    greedy_tree_reject_kernel[(grid,)](
        tokens,
        first_child,
        next_sibling,
        target_token_ids,
        sampled_token_ids,
        path_out,
        num_reqs,
        budget,
        spec_len,
        tokens.stride(0),
        first_child.stride(0),
        next_sibling.stride(0),
        target_token_ids.stride(0),
        sampled_token_ids.stride(0),
        path_out.stride(0),
    )
