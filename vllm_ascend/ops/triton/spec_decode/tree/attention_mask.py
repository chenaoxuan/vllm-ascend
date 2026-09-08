from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(
    do_not_specialize=["num_mask", "query_len", "kv_len", "max_nodes"]
)
def tree_attention_mask_kernel(
    mask_ptr,  # [R, 1, Q, Kv] int8
    visibility_ptr,  # [R, B, B] int8
    prev_kv_ptr,  # [R] int32
    num_mask,
    query_len,
    kv_len,
    max_nodes,
    stride_mask_r,
    stride_mask_q,
    stride_mask_k,
    stride_vis_r,
    stride_vis_i,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: dim0 = request (grid-stride over num_mask), dim1 = query row.
    # Each program fills one (req, q) mask row, vectorized over KV in BLOCK_K
    # tiles. This replaces the old scalar `for q: for k:` double loop, so the
    # long KV dimension is consumed by SIMD instead of by loop iterations --
    # the dominant cost when batch (num_mask) grows.
    pid_r = tl.program_id(0)
    nprog_r = tl.num_programs(0)
    q = tl.program_id(1)

    req = pid_r
    while req < num_mask:
        prev = tl.load(prev_kv_ptr + req)
        row_base = mask_ptr + req * stride_mask_r + q * stride_mask_q
        k_off = tl.arange(0, BLOCK_K)

        if q == 0:
            # First query row only attends to the committed prefix; the whole
            # draft region stays masked. Skip the visibility gather entirely.
            for k_blk in tl.range(0, kv_len, BLOCK_K):
                k = k_blk + k_off
                in_range = k < kv_len
                masked = tl.full((BLOCK_K,), 1, tl.int8)
                masked = tl.where(in_range & (k <= prev), 0, masked)
                tl.store(row_base + k * stride_mask_k, masked, mask=in_range)
        else:
            vis_row = (
                visibility_ptr + req * stride_vis_r + (q - 1) * stride_vis_i
            )
            for k_blk in tl.range(0, kv_len, BLOCK_K):
                k = k_blk + k_off
                in_range = k < kv_len
                # default: masked out (1); committed prefix is visible (0).
                masked = tl.full((BLOCK_K,), 1, tl.int8)
                masked = tl.where(in_range & (k <= prev), 0, masked)
                # draft region: visible iff visibility[req, q-1, draft_col] != 0.
                draft_col = k - (prev + 1)
                in_draft = in_range & (k > prev) & (draft_col < max_nodes)
                vis = tl.load(vis_row + draft_col, mask=in_draft, other=0)
                masked = tl.where(in_draft & (vis != 0), 0, masked)
                masked = tl.where(in_draft & (vis == 0), 1, masked)
                tl.store(row_base + k * stride_mask_k, masked, mask=in_range)

        req += nprog_r


def fill_tree_attention_mask_triton(
    attn_mask,
    tree_visibility,
    prev_kv_lens,
) -> None:
    import torch

    num_mask = prev_kv_lens.shape[0]
    max_nodes = tree_visibility.shape[-1]
    query_len = attn_mask.shape[2]
    kv_len = attn_mask.shape[3]
    if num_mask == 0 or query_len == 0:
        return
    # Bool view must share storage with attn_mask (in-place FIA mask).
    mask_i8 = attn_mask.view(torch.int8)
    vis_i8 = tree_visibility.to(torch.int8).contiguous()
    prev = prev_kv_lens.to(torch.int32).contiguous()
    vec = get_vectorcore_num()
    # dim0: grid-stride over requests (capped at vectorcore count);
    # dim1: one program per query row for extra parallelism, which keeps the
    # vector cores saturated as batch grows (32 -> 64) instead of serializing
    # extra requests through the same programs.
    grid_r = min(num_mask, vec)
    grid_q = query_len
    raw = kv_len
    bk = 1 << (raw.bit_length() - 1) if raw > 0 else 256
    block_k = min(max(bk, 128), 1024)

    tree_attention_mask_kernel[(grid_r, grid_q)](
        mask_i8,
        vis_i8,
        prev,
        num_mask,
        query_len,
        kv_len,
        max_nodes,
        mask_i8.stride(0),
        mask_i8.stride(2),
        mask_i8.stride(3),
        vis_i8.stride(0),
        vis_i8.stride(1),
        BLOCK_K=block_k,
    )
