from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


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
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    req = pid
    while req < num_reqs:
        req_state = tl.load(idx_mapping_ptr + req)
        safe = tl.where(req_state >= 0, req_state, 0)
        prefix = tl.load(num_computed_ptr + safe)
        for d in range(spec_len):
            node = tl.load(path_ptr + req * stride_path_r + d)
            valid = (node >= 0) & (req_state >= 0)
            depth = d + 1
            dst_pos = prefix + depth
            src_pos = tl.where(valid, prefix + node, dst_pos)
            src_block = tl.load(
                block_table_ptr
                + safe * stride_bt_r
                + (src_pos // block_size)
            )
            dst_block = tl.load(
                block_table_ptr
                + safe * stride_bt_r
                + (dst_pos // block_size)
            )
            src_slot = src_block * block_size + (src_pos % block_size)
            dst_slot = dst_block * block_size + (dst_pos % block_size)
            tl.store(src_slots_ptr + req * stride_out_r + d, src_slot)
            tl.store(dst_slots_ptr + req * stride_out_r + d, dst_slot)
        req += nprog


def compact_tree_kv_slots_triton(
    block_table,
    num_computed,
    idx_mapping,
    path_node_ids,
    src_slots,
    dst_slots,
    block_size: int,
) -> None:
    num_reqs, spec_len = path_node_ids.shape
    vec = get_vectorcore_num()
    grid = min(max(num_reqs, 1), max(vec, 1))
    compact_tree_kv_slots_kernel[(grid,)](
        block_table,
        num_computed,
        idx_mapping,
        path_node_ids,
        src_slots,
        dst_slots,
        num_reqs,
        spec_len,
        block_size,
        block_table.stride(0),
        path_node_ids.stride(0),
        src_slots.stride(0),
    )
