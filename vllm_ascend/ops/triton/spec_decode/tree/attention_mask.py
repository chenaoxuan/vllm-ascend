from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_mask"])
def tree_attention_mask_kernel(
    mask_ptr,  # [R, 1, Q, Kv] int8
    visibility_ptr,  # [R, B, B] int8
    prev_kv_ptr,  # [R] int32
    num_mask,
    stride_mask_r,
    stride_mask_q,
    stride_mask_k,
    stride_vis_r,
    stride_vis_i,
    MAX_NODES: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    KV_LEN: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    req = pid
    while req < num_mask:
        prev = tl.load(prev_kv_ptr + req)
        for q in range(QUERY_LEN):
            for k in range(KV_LEN):
                base = (
                    mask_ptr
                    + req * stride_mask_r
                    + q * stride_mask_q
                    + k * stride_mask_k
                )
                masked = tl.full((), 1, tl.int8)
                masked = tl.where(k <= prev, 0, masked)
                draft_col = k - (prev + 1)
                in_draft = (
                    (q >= 1)
                    & (k > prev)
                    & (draft_col >= 0)
                    & (draft_col < MAX_NODES)
                )
                vis = tl.load(
                    visibility_ptr
                    + req * stride_vis_r
                    + (q - 1) * stride_vis_i
                    + draft_col,
                    mask=in_draft,
                    other=0,
                )
                masked = tl.where(in_draft & (vis != 0), 0, masked)
                masked = tl.where(in_draft & (vis == 0), 1, masked)
                tl.store(base, masked)
        req += nprog


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
    mask_i8 = attn_mask.view(torch.int8)
    vis_i8 = tree_visibility.to(torch.int8).contiguous()
    vec = get_vectorcore_num()
    grid = min(max(num_mask, 1), max(vec, 1))
    tree_attention_mask_kernel[(grid,)](
        mask_i8,
        vis_i8,
        prev_kv_lens.to(torch.int32).contiguous(),
        num_mask,
        mask_i8.stride(0),
        mask_i8.stride(2),
        mask_i8.stride(3),
        vis_i8.stride(0),
        vis_i8.stride(1),
        MAX_NODES=max_nodes,
        QUERY_LEN=query_len,
        KV_LEN=kv_len,
    )
