# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.node_pool import build_ancestors


@pytest.mark.parametrize("max_depth", [1, 7, 16])
def test_ancestor_visibility(max_depth):
    """Sibling isolation, separate requests, roots, and padded node slots."""
    torch.npu.set_device(0)
    parents = [[0, 0, 1, 2, 3, 3, 4], [0, 1, 1, 0, 0, 0, 0]]
    depths = [[1, 1, 2, 2, 3, 3, 3], [1, 2, 2, 0, 0, 0, 0]]
    counts = [7, 3]
    device = "npu"
    args = [torch.tensor(x, dtype=torch.int32, device=device) for x in (parents, depths, counts)]
    req = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long, device=device)
    qsl = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    ancestors, visible, actual_depth, hidden, parent = build_ancestors(*args, req, qsl, max_depth)
    expected, mask, nd, hi, pi = [], [], [], [], []
    for r in range(2):
        for local in range(8):
            live = 0 < local <= counts[r]
            d = min(depths[r][local - 1], max_depth, local) if live else 0
            row = [r * 8] * (max_depth + 1)
            cursor = local if live else 0
            for offset in range(max_depth):
                col = max(d - offset, 0)
                if col:
                    row[col] = r * 8 + cursor
                cursor = parents[r][max(cursor - 1, 0)] if col else 0
            expected.append(row)
            mask.append([k <= d for k in range(max_depth + 1)])
            nd.append(d)
            hi.append(r * 8 + (local if live else 0))
            pi.append(r * 8 + (parents[r][local - 1] if live else 0))
    for actual, ref in ((ancestors, expected), (visible, mask), (actual_depth, nd), (hidden, hi), (parent, pi)):
        torch.testing.assert_close(actual.cpu(), torch.tensor(ref, dtype=actual.dtype), rtol=0, atol=0)
