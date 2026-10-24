# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.node_pool import build_ancestors
from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp
from vllm_ascend.worker.v2.spec_decode.tree.node_pool import commit_nodes, prepare_nodes, record_nodes


@pytest.mark.parametrize(
    "winner,enabled,ratio", [(0, True, 1), (3, True, 1), (4, True, 1), (4, False, 1), (3, True, 4), (4, True, 4)]
)
@pytest.mark.parametrize("alias", [False, True])
def test_node_commit_preserves_baseline_and_siblings(winner, enabled, ratio, alias):
    device = "npu"
    torch.npu.set_device(0)
    parents = torch.tensor([[0, 0, 1, 2]], dtype=torch.int32, device=device)
    depths = torch.tensor([[1, 1, 2, 2]], dtype=torch.int32, device=device)
    counts = torch.tensor([4], dtype=torch.int32, device=device)
    req = torch.zeros(5, dtype=torch.long, device=device)
    qsl = torch.tensor([0, 5], dtype=torch.int32, device=device)
    ancestors, visible, depth, hidden, parent = build_ancestors(parents, depths, counts, req, qsl, 3)
    plan = dict(
        n=5,
        max_q=4,
        req_index=[0] * 5,
        prefixes=[2] * 5,
        req_cu=[0, 5],
        req_ids=req,
        ancestors=ancestors,
        depths=depth,
        start=torch.full((5,), 2, dtype=torch.int32, device=device),
        host_uploads=[],
    )
    page_map = [1, 1, 1] if alias else [1, 2, 1]
    table = torch.tensor([page_map], dtype=torch.int32, device=device)
    original = torch.arange(3 * 2 * 16, dtype=torch.int32, device=device).reshape(3, 2, 1, 16)
    snapshot = torch.arange(5 * 16, dtype=torch.int32, device=device).reshape(5, 1, 16) + 1000
    before = original.clone()
    metadata = SimpleNamespace(cache_group_key="state", block_table=table)
    saved = cp._LAYOUT
    cp._LAYOUT = {"host_tables": {"state": ([page_map], [3])}}
    try:
        record_nodes(plan, original, snapshot, metadata, ratio)
        prepared = prepare_nodes(plan, torch.tensor([winner], device=device), torch.tensor([enabled], device=device))
        expected = before.cpu()
        path = [[0], [0, 1], [0, 2], [0, 1, 3], [0, 2, 4]][winner]
        touched = set()
        if enabled:
            for d, node in enumerate(path):
                pos = 2 + d
                if (pos + 1) % ratio:
                    continue
                p = (pos + 1) // ratio - 1
                page = page_map[p // 2]
                expected[page, p % 2] = snapshot[node].cpu()
                touched.add(page)
        # Simulate relocation after preparation. Active pages must restore
        # the original untouched rows; inactive pages must not be committed.
        original.add_(7000)
        for page in range(3):
            if page not in touched:
                expected[page].add_(7000)
        commit_nodes(prepared)
        torch.testing.assert_close(original.cpu(), expected, rtol=0, atol=0)
    finally:
        cp._LAYOUT = saved
