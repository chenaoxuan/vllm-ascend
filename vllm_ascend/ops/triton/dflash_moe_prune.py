"""Fused DFlash stage-2 kernels: scatter union_mask, then write -inf on suffix rows."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

if HAS_TRITON:

    @triton.jit(do_not_specialize=["num_reqs", "prefix_k", "escape_rank", "max_rows"])
    def _fill_union_mask_kernel(
        topk_ids_ptr,
        topk_stride0,
        topk_stride1,
        bounds_ptr,
        mask_ptr,
        num_reqs,
        prefix_k,
        escape_rank,
        top_k,
        max_rows,
    ):
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        for req in tl.range(pid, num_reqs, num_programs):
            start = tl.load(bounds_ptr + req)
            end = tl.load(bounds_ptr + req + 1)
            n_rows = end - start
            protected = tl.minimum(prefix_k, n_rows)

            for t in tl.range(0, max_rows):
                row_valid = t < n_rows
                is_prefix = t < protected
                k_limit = tl.where(is_prefix, top_k, escape_rank)
                row = start + t
                row_ptr = topk_ids_ptr + row * topk_stride0
                for k in tl.range(0, top_k):
                    valid = row_valid & (k < k_limit)
                    eid = tl.load(row_ptr + k * topk_stride1, mask=valid, other=0)
                    tl.store(mask_ptr + eid, True, mask=valid)

    @triton.jit(do_not_specialize=["num_exposed", "num_experts", "n_e_blocks"])
    def _apply_union_mask_kernel(
        logits_ptr,
        logits_stride0,
        logits_stride1,
        exposed_rows_ptr,
        mask_ptr,
        num_exposed,
        num_experts,
        n_e_blocks,
        neg_inf,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        for exp_i in tl.range(pid, num_exposed, num_programs):
            row = tl.load(exposed_rows_ptr + exp_i)
            row_ptr = logits_ptr + row * logits_stride0
            for eb in tl.range(0, n_e_blocks):
                offs = eb * BLOCK + tl.arange(0, BLOCK)
                emask = offs < num_experts
                allowed = tl.load(mask_ptr + offs, mask=emask, other=False)
                logits = tl.load(row_ptr + offs * logits_stride1, mask=emask)
                neg = tl.full(logits.shape, neg_inf, dtype=logits.dtype)
                out = tl.where(allowed, logits, neg)
                tl.store(row_ptr + offs * logits_stride1, out, mask=emask)


def _grid_size(work_items: int) -> int:
    try:
        num_cores = get_vectorcore_num()
    except AssertionError:
        num_cores = 8
    return max(1, min(work_items, num_cores))


def apply_dflash_union_mask_triton(
    router_logits: torch.Tensor,
    all_topk_ids: torch.Tensor,
    req_boundaries: torch.Tensor,
    exposed_global_rows: torch.Tensor,
    union_mask: torch.Tensor,
    prefix_k: int,
    escape_rank: int,
    max_rows: int,
) -> None:
    """Fused stage-2: scatter prefix/escape ids into union_mask, then mask suffix logits."""
    num_reqs = req_boundaries.numel() - 1
    top_k = all_topk_ids.shape[-1]
    num_experts = union_mask.numel()
    num_exposed = exposed_global_rows.numel()
    if num_reqs <= 0 or num_exposed <= 0 or max_rows <= 0:
        return

    mask = union_mask.view(-1)
    block = 128
    n_e_blocks = (num_experts + block - 1) // block

    _fill_union_mask_kernel[(_grid_size(num_reqs),)](
        all_topk_ids,
        all_topk_ids.stride(0),
        all_topk_ids.stride(1),
        req_boundaries,
        mask,
        num_reqs,
        prefix_k,
        escape_rank,
        top_k,
        max_rows,
        multibuffer=False,
    )
    _apply_union_mask_kernel[(_grid_size(num_exposed),)](
        router_logits,
        router_logits.stride(0),
        router_logits.stride(1),
        exposed_global_rows,
        mask,
        num_exposed,
        num_experts,
        n_e_blocks,
        float(torch.finfo(router_logits.dtype).min),
        BLOCK=block,
        multibuffer=False,
    )
