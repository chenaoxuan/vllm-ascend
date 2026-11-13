# SPDX-License-Identifier: Apache-2.0
"""Masked writes of dense selected tree-convolution pages to a strided cache."""

import math

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.mamba.state_index import (
    STATE_IO_BLOCK_SIZE,
    _validate_scatter_inputs,
)


@triton.jit
def _scatter_tree_conv_pages_kernel(
    state_ptr,
    page_indices_ptr,
    valid_ptr,
    source_ptr,
    stride_state_page: tl.int64,
    stride_indices,
    stride_valid,
    stride_source_page: tl.int64,
    PAGE_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0)
    row = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    live = tl.load(valid_ptr + row * stride_valid).to(tl.int1)
    page = tl.load(page_indices_ptr + row * stride_indices, mask=live, other=0).to(tl.int64)
    values = tl.load(
        source_ptr + row * stride_source_page + offsets,
        mask=live & (offsets < PAGE_ELEMENTS),
        other=0,
    )
    tl.store(
        state_ptr + page * stride_state_page + offsets,
        values,
        mask=live & (offsets < PAGE_ELEMENTS),
    )


def scatter_tree_conv_pages_(
    state: torch.Tensor,
    page_indices: torch.Tensor,
    valid: torch.Tensor,
    source: torch.Tensor,
) -> None:
    """Store selected logical pages without touching physical-page gaps.

    Pages have contiguous inner dimensions and may have a padded page stride
    or a nonzero storage offset. Only live requests write; a padded request's
    harmless page ID may alias an active request's page. Active page IDs must
    be unique, as required by mutable per-request convolution state.
    """
    _validate_scatter_inputs(state, page_indices, source)
    if valid.ndim != 1 or valid.shape != page_indices.shape:
        raise ValueError("valid must be 1D and match page_indices")
    if valid.dtype != torch.bool:
        raise TypeError("valid must have bool dtype")
    if valid.device != state.device:
        raise ValueError("valid and state must be on the same device")
    if page_indices.numel() == 0:
        return
    page_elements = math.prod(state.shape[1:])
    grid = (triton.cdiv(page_elements, STATE_IO_BLOCK_SIZE), page_indices.numel())
    _scatter_tree_conv_pages_kernel[grid](
        state,
        page_indices,
        valid,
        source,
        state.stride(0),
        page_indices.stride(0),
        valid.stride(0),
        source.stride(0),
        PAGE_ELEMENTS=page_elements,
        BLOCK_SIZE=STATE_IO_BLOCK_SIZE,
        num_warps=8,
    )
