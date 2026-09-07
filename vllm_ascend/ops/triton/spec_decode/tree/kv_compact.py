from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs", "spec_len", "block_size"])
def compact_tree_kv_slots_kernel(
    block_table_ptr,  # [max_reqs, max_blocks]
    num_computed_ptr,  # [max_reqs]
    idx_mapping_ptr,  # [num_reqs]
    path_ptr,  # [num_reqs, spec_len]
    src_slots_ptr,  # [num_reqs, spec_len] out
    dst_slots_ptr,  # [num_reqs, spec_len] out
    num_reqs,
    spec_len,
    block_size,
    stride_bt_r,
    stride_path_r,
    stride_out_r,
):
    # 2D grid: one program per (req, depth). Fully parallel; no host-side
    # vectorcore tuning and no scalar loop over spec_len.
    req = tl.program_id(0)
    d = tl.program_id(1)
    if req >= num_reqs or d >= spec_len:
        return
    req_state = tl.load(idx_mapping_ptr + req)
    safe = tl.where(req_state >= 0, req_state, 0)
    prefix = tl.load(num_computed_ptr + safe)
    node = tl.load(path_ptr + req * stride_path_r + d)
    valid = (node >= 0) & (req_state >= 0)
    depth = d + 1
    dst_pos = prefix + depth
    src_pos = tl.where(valid, prefix + node, dst_pos)
    src_block = tl.load(
        block_table_ptr + safe * stride_bt_r + (src_pos // block_size)
    )
    dst_block = tl.load(
        block_table_ptr + safe * stride_bt_r + (dst_pos // block_size)
    )
    src_slot = src_block * block_size + (src_pos % block_size)
    dst_slot = dst_block * block_size + (dst_pos % block_size)
    tl.store(src_slots_ptr + req * stride_out_r + d, src_slot)
    tl.store(dst_slots_ptr + req * stride_out_r + d, dst_slot)


def compact_tree_kv_slots_triton(
    block_table,
    num_computed,
    idx_mapping,
    path_node_ids,
    src_slots,
    dst_slots,
    block_size: int,
) -> None:
    path = path_node_ids.contiguous()
    num_reqs, spec_len = path.shape
    compact_tree_kv_slots_kernel[(num_reqs, spec_len)](
        block_table,
        num_computed,
        idx_mapping,
        path,
        src_slots,
        dst_slots,
        num_reqs,
        spec_len,
        block_size,
        block_table.stride(0),
        path.stride(0),
        src_slots.stride(0),
    )
