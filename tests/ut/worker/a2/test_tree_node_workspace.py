# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp
from vllm_ascend.worker.v2.spec_decode.tree.node_pool import node_workspace, record_nodes


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_workspace_round_reuse_and_live_node_isolation(batch):
    torch.npu.set_device(0)
    module = SimpleNamespace()
    original = torch.zeros((8, 4, 1, 32), dtype=torch.float32, device="npu")
    # Two packed views must retain their byte offsets and shared storage.
    caches = [original[..., :16], original[..., 16:]]
    previous = None
    for n in (batch * 5, batch * 3, batch * 9, batch * 5):
        plan = {}
        workspace = node_workspace(module, plan)
        nodes = workspace.buffers.acquire("nodes", (n, 32), torch.float32, "npu")
        views = workspace.cache_views(caches, 1 + (n + 3) // 4)
        if previous is not None and n <= previous[0]:
            assert nodes.data_ptr() == previous[1]
            assert views[0].data_ptr() == previous[2]
        assert views[0].untyped_storage().data_ptr() == views[1].untyped_storage().data_ptr()
        assert views[1].data_ptr() - views[0].data_ptr() == 16 * 4
        torch.testing.assert_close(views[0], torch.zeros_like(views[0]), atol=0, rtol=0)
        torch.testing.assert_close(views[1], torch.zeros_like(views[1]), atol=0, rtol=0)
        nodes.fill_(7)
        views[0].fill_(11)
        # A repeat invocation cannot overwrite state pending commitment.
        repeated = node_workspace(module, plan)
        other = repeated.buffers.acquire("nodes", (n, 32), torch.float32, "npu")
        other.fill_(99)
        other_views = repeated.cache_views(caches, 1 + (n + 3) // 4)
        other_views[0].fill_(88)
        torch.testing.assert_close(nodes, torch.full_like(nodes, 7), atol=0, rtol=0)
        torch.testing.assert_close(views[0], torch.full_like(views[0], 11), atol=0, rtol=0)
        previous = (n, nodes.data_ptr(), views[0].data_ptr())


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_page_metadata_matches_node_enumeration_across_rounds(batch, ratio):
    torch.npu.set_device(0)
    saved = cp._LAYOUT
    try:
        for offset in (0, 3):
            counts = [3 + req % 3 for req in range(batch)]
            cu = [0]
            for count in counts:
                cu.append(cu[-1] + count)
            prefixes = [ratio - 2 + offset + req for req in range(batch)]
            prefixes = [max(0, value) for value in prefixes]
            rows = [[1 + (col + req + offset) % 7 for col in range(8)] for req in range(batch)]
            cp._LAYOUT = {"host_tables": {"a": (rows, [8] * batch), "b": (rows, [8] * batch)}}
            plan = dict(
                max_q=4,
                req_cu=cu,
                req_index=[r for r, n in enumerate(counts) for _ in range(n)],
                prefixes=[p for p, n in zip(prefixes, counts) for _ in range(n)],
            )
            original = torch.empty((8, 2, 1, 16), dtype=torch.float32, device="npu")
            snapshot = torch.empty((cu[-1], 1, 16), dtype=torch.float32, device="npu")
            table = torch.tensor(rows, dtype=torch.int32, device="npu")
            expected = set()
            for node, req in enumerate(plan["req_index"]):
                start = plan["prefixes"][node]
                length = min(4, node - cu[req] + 1)
                begin = start if ratio == 1 else start // ratio
                end = start + length if ratio == 1 else (start + length) // ratio
                for pos in range(begin, end):
                    if pos // 2 < len(rows[req]):
                        expected.add(rows[req][pos // 2])
            for key in ("a", "b"):
                record_nodes(plan, original, snapshot, SimpleNamespace(cache_group_key=key, block_table=table), ratio)
            assert tuple(sorted(expected)) == plan["node_records"][0][3]
            assert plan["node_records"][0][3] is plan["node_records"][1][3]
            assert len(plan["node_page_sets"]) == 1
    finally:
        cp._LAYOUT = saved
