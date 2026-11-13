from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs", "budget", "spec_len"])
def greedy_tree_reject_kernel(
    tokens_ptr,
    first_child_ptr,
    next_sibling_ptr,
    target_ids_ptr,
    logits_cu_ptr,
    verified_nodes_ptr,
    sampled_ptr,
    path_ptr,
    num_sampled_ptr,
    num_accepted_ptr,
    endpoint_ptr,
    active_ptr,
    num_reqs,
    budget,
    spec_len,
    stride_tokens_r,
    stride_fc_r,
    stride_ns_r,
    stride_sampled_r,
    stride_path_r,
    BLOCK_OUT: tl.constexpr,
):
    # One request per program. Dynamic loops terminate as soon as a branch
    # rejects or its sibling list ends; they do not unroll budget * depth.
    req = tl.program_id(0)
    if req >= num_reqs:
        return
    columns = tl.arange(0, BLOCK_OUT)
    tl.store(sampled_ptr + req * stride_sampled_r + columns, -1, columns <= spec_len)
    tl.store(path_ptr + req * stride_path_r + columns, -1, columns < spec_len)
    start = tl.load(logits_cu_ptr + req).to(tl.int64)
    verified = tl.load(verified_nodes_ptr + req)
    alive = tl.load(active_ptr + req) != 0
    current = tl.full((), 0, tl.int64)
    accepted = tl.full((), 0, tl.int32)
    sampled = tl.full((), 0, tl.int32)
    while alive & (accepted < spec_len):
        target = tl.load(target_ids_ptr + start + current)
        tl.store(sampled_ptr + req * stride_sampled_r + accepted, target)
        sampled += 1
        child = tl.load(first_child_ptr + req * stride_fc_r + current).to(tl.int64)
        found = tl.full((), -1, tl.int64)
        visited = tl.full((), 0, tl.int32)
        while (child > 0) & (child <= budget) & (found < 0) & (visited < budget):
            eligible = child <= verified
            token = tl.load(tokens_ptr + req * stride_tokens_r + child - 1, mask=eligible, other=-1)
            found = tl.where(eligible & (token == target), child, found)
            child = tl.load(next_sibling_ptr + req * stride_ns_r + child).to(tl.int64)
            visited += 1
        alive = found > 0
        if alive:
            tl.store(path_ptr + req * stride_path_r + accepted, found)
            current = found
            accepted += 1
    if alive:
        target = tl.load(target_ids_ptr + start + current)
        tl.store(sampled_ptr + req * stride_sampled_r + spec_len, target)
        sampled += 1
    tl.store(num_sampled_ptr + req, sampled)
    tl.store(num_accepted_ptr + req, accepted)
    tl.store(endpoint_ptr + req, current)


def greedy_tree_reject_triton(
    tokens,
    first_child,
    next_sibling,
    target_token_ids,
    cu_num_logits,
    verified_nodes,
    sampled_token_ids,
    path_out,
    num_sampled,
    num_accepted,
    endpoint_node,
    active,
) -> None:
    num_reqs, budget = tokens.shape
    spec_len = path_out.shape[1]
    greedy_tree_reject_kernel[(max(num_reqs, 1),)](
        tokens,
        first_child,
        next_sibling,
        target_token_ids,
        cu_num_logits,
        verified_nodes,
        sampled_token_ids,
        path_out,
        num_sampled,
        num_accepted,
        endpoint_node,
        active,
        num_reqs,
        budget,
        spec_len,
        tokens.stride(0),
        first_child.stride(0),
        next_sibling.stride(0),
        sampled_token_ids.stride(0),
        path_out.stride(0),
        triton.next_power_of_2(spec_len + 1),
    )
