# SPDX-License-Identifier: Apache-2.0
"""BS1 NPU coverage for fixed-shape tree snapshot metadata writes."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.state_commit import commit_tree_snapshot_metadata


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("request_slot", [2, -1])
@pytest.mark.parametrize("tree_row", [True, False])
def test_snapshot_commit_bs1_graph_replays_live_inputs_and_clears(request_slot, tree_row):
    backing = torch.arange(4 * 18, dtype=torch.int32, device="npu").reshape(4, 18)
    original = backing.cpu().clone()
    previous_path = backing[:, 1:17]
    previous_sampled = backing[:, 0]
    mapping = torch.tensor([request_slot], dtype=torch.int32, device="npu")
    path = torch.tensor([[1, 3, 7, 16, -1, -1, -1]], dtype=torch.int32, device="npu")
    sampled = torch.tensor([5], dtype=torch.int32, device="npu")
    tree_rows = torch.tensor([tree_row], device="npu")

    def commit():
        commit_tree_snapshot_metadata(previous_path, previous_sampled, mapping, path, sampled, tree_rows)

    commit()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        commit()
    for nodes, count in (([1, 3, 7, 16, -1, -1, -1], 5), ([8, 16, -1, -1, -1, -1, -1], 3)):
        backing.copy_(original.to("npu"))
        path.copy_(torch.tensor([nodes], dtype=torch.int32, device="npu"))
        sampled.fill_(count)
        graph.replay()
        expected = original.clone()
        if request_slot >= 0:
            expected[request_slot, 1:8] = torch.tensor(nodes, dtype=torch.int32)
            expected[request_slot, 0] = count if tree_row else 0
        torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)

    # A valid request_slot's later linear step invalidates only its old path count.
    before_clear = backing.cpu().clone()
    commit_tree_snapshot_metadata(previous_path, previous_sampled, mapping)
    expected = before_clear.clone()
    if request_slot >= 0:
        expected[request_slot, 0] = 0
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
