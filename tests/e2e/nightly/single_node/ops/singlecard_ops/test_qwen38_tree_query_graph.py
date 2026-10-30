# SPDX-License-Identifier: Apache-2.0
"""BS1 real-NPU query graph replay, source aliasing and metadata lifetime."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401 - registers the NPU backend

from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path
from vllm_ascend.worker.v2.spec_decode.tree.qwen_query_graph import QwenTreeQueryGraphs

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Requires Ascend NPU")


def _views(storage):
    # Five auxiliary views partly overlap; last hidden aliases the first aux.
    aux = [storage[:, left : left + 2] for left in (0, 1, 3, 5, 7)]
    return [aux[0], *aux]


def _acceptance(nodes, *, active=True):
    path = torch.full((1, 7), -1, dtype=torch.long, device="npu")
    if nodes:
        path[0, : len(nodes)] = torch.tensor(nodes, device="npu")
    return TreeAcceptance.from_path(
        path, torch.tensor([len(nodes) + 1 if active else 0], dtype=torch.int32, device="npu")
    )


def test_bs1_two_round_nonprefix_alias_and_dynamic_metadata_query_graph():
    torch.npu.set_device(0)
    cache = QwenTreeQueryGraphs(
        SimpleNamespace(speculator=SimpleNamespace(tree_shape=SimpleNamespace(target_query_len=17)))
    )
    storage = torch.empty(24, 10, dtype=torch.float32, device="npu")
    positions = torch.empty(24, dtype=torch.int64, device="npu")
    # Each call has fresh qsl and acceptance objects. Their source pointers
    # must not create new graphs, and changed values must reach graph inputs.
    for round_id, (start, nodes, active) in enumerate(
        [(0, (), False), (0, (2, 3, 16), True), (2, (8, 12), True), (0, (2,), False)]
    ):
        storage.copy_(torch.arange(storage.numel(), device="npu").reshape_as(storage) + round_id * 1000)
        positions.copy_(torch.arange(24, device="npu") % 3 + round_id * 100)
        expected = storage.clone()
        expected_positions = positions.clone()
        untouched = storage.clone() if not active else None
        untouched_positions = positions.clone() if not active else None
        qsl = torch.tensor([start, start + 17], dtype=torch.int32, device="npu")
        acceptance = _acceptance(nodes, active=active)
        compact_tree_query_along_path(_views(expected), qsl, acceptance, expected_positions)
        cache.compact_queries(_views(storage), qsl, acceptance, positions)
        torch.npu.synchronize()
        torch.testing.assert_close(storage, expected, atol=0, rtol=0)
        torch.testing.assert_close(positions, expected_positions, atol=0, rtol=0)
        if not active:
            torch.testing.assert_close(storage, untouched, atol=0, rtol=0)
            torch.testing.assert_close(positions, untouched_positions, atol=0, rtol=0)
        assert len(cache.graphs) == 1
    entry = next(iter(cache.graphs.values()))
    assert entry.graph.replays == 4
    assert entry.tensors[0].data_ptr() == entry.tensors[1].data_ptr()
    assert entry.positions is positions
    # Changed real output storage and then changed positions storage both need
    # recapture; neither graph may continue writing the old target outputs.
    newer = storage.clone()
    newer_expected = newer.clone()
    newer_positions = positions.clone()
    newer_expected_positions = newer_positions.clone()
    qsl, acceptance = torch.tensor([0, 17], dtype=torch.int32, device="npu"), _acceptance((2, 3, 16))
    compact_tree_query_along_path(_views(newer_expected), qsl, acceptance, newer_expected_positions)
    cache.compact_queries(_views(newer), qsl, acceptance, newer_positions)
    torch.npu.synchronize()
    assert len(cache.graphs) == 2
    torch.testing.assert_close(newer, newer_expected, atol=0, rtol=0)
    torch.testing.assert_close(newer_positions, newer_expected_positions, atol=0, rtol=0)
    replacement_positions = newer_positions.clone()
    expected = newer.clone()
    expected_positions = replacement_positions.clone()
    compact_tree_query_along_path(_views(expected), qsl, acceptance, expected_positions)
    cache.compact_queries(_views(newer), qsl, acceptance, replacement_positions)
    torch.npu.synchronize()
    assert len(cache.graphs) == 3
    torch.testing.assert_close(newer, expected, atol=0, rtol=0)
    torch.testing.assert_close(replacement_positions, expected_positions, atol=0, rtol=0)
