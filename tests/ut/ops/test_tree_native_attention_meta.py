# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the standalone binding first; these checks never initialize an NPU."""

import pytest
import torch


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("return_lse", [False, True])
def test_tree_pool_meta(batch_size, ratio, return_lse):
    if not hasattr(torch.ops._C_ascend, "npu_sparse_attn_sharedkv_tree"):
        pytest.skip("experimental tree binding not loaded")
    n, heads, dim = batch_size * 17, 8, 512
    q = torch.empty(n, heads, dim, dtype=torch.bfloat16, device="meta")
    output, lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(
        q,
        cmp_tree_kv=torch.empty(n, dim, dtype=q.dtype, device="meta"),
        tree_ancestors=torch.empty(n, 5, dtype=torch.int64, device="meta"),
        tree_prefix=torch.empty(n, dtype=torch.int32, device="meta"),
        cmp_ratio=ratio,
        layout_q="TND",
        return_softmax_lse=return_lse,
    )
    assert output.shape == q.shape and output.dtype == q.dtype
    assert lse.shape == ((n, heads, 1) if return_lse else (0,))
    assert lse.dtype == torch.float32
