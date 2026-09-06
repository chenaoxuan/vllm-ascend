from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_reqs"])
def finalize_tree_layout_kernel(
    parent_ids_ptr,
    first_child_ptr,
    next_sibling_ptr,
    visibility_ptr,
    num_reqs,
    stride_par_r,
    stride_fc_r,
    stride_ns_r,
    stride_vis_r,
    stride_vis_i,
    NUM_NODES: tl.constexpr,
    BUDGET: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    req = pid
    while req < num_reqs:
        for n in range(NUM_NODES + 1):
            tl.store(first_child_ptr + req * stride_fc_r + n, -1)
            tl.store(next_sibling_ptr + req * stride_ns_r + n, -1)
        for slot in range(NUM_NODES):
            node_id = slot + 1
            parent_id = tl.load(parent_ids_ptr + req * stride_par_r + slot)
            cur_first = tl.load(first_child_ptr + req * stride_fc_r + parent_id)
            tl.store(next_sibling_ptr + req * stride_ns_r + node_id, cur_first)
            tl.store(first_child_ptr + req * stride_fc_r + parent_id, node_id)
        for slot in range(NUM_NODES):
            for j in range(BUDGET):
                tl.store(
                    visibility_ptr + req * stride_vis_r + slot * stride_vis_i + j,
                    0,
                )
            parent_id = tl.load(parent_ids_ptr + req * stride_par_r + slot)
            src = tl.where(parent_id > 0, parent_id - 1, 0)
            for j in range(slot):
                pv = tl.load(
                    visibility_ptr + req * stride_vis_r + src * stride_vis_i + j
                )
                val = tl.where((slot > 0) & (parent_id > 0), pv, 0)
                tl.store(
                    visibility_ptr + req * stride_vis_r + slot * stride_vis_i + j,
                    val,
                )
            tl.store(
                visibility_ptr + req * stride_vis_r + slot * stride_vis_i + slot,
                1,
            )
        req += nprog


def finalize_tree_layout_triton(
    parent_ids,
    first_child,
    next_sibling,
    visibility,
    num_nodes: int,
) -> None:
    import torch

    num_reqs = parent_ids.shape[0]
    budget = visibility.shape[1]
    vis_i8 = visibility.view(torch.int8) if visibility.dtype == torch.bool else visibility
    vec = get_vectorcore_num()
    grid = min(max(num_reqs, 1), max(vec, 1))
    finalize_tree_layout_kernel[(grid,)](
        parent_ids.contiguous().to(torch.long),
        first_child.contiguous(),
        next_sibling.contiguous(),
        vis_i8.contiguous(),
        num_reqs,
        parent_ids.stride(0),
        first_child.stride(0),
        next_sibling.stride(0),
        vis_i8.stride(0),
        vis_i8.stride(1),
        NUM_NODES=num_nodes,
        BUDGET=budget,
    )
