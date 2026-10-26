# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Byte-exact NPU coverage for shared tree KV transport and graph replay."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.worker.v2.spec_decode.tree import chain_pack
from vllm_ascend.worker.v2.spec_decode.tree.kv_binding import bind_original_kv, record_original_kv
from vllm_ascend.worker.v2.spec_decode.tree.kv_project import TreeKvCompact
from vllm_ascend.worker.v2.spec_decode.tree.transport import TreeRowMove, relocate_tree_rows


@pytest.fixture(autouse=True)
def npu_device():
    torch.npu.set_device(0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float8_e4m3fn])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("graph", [False, True])
def test_transport_preserves_bytes_padding_and_overlapping_sources(dtype, padding, graph):
    width, pages, block = 16, 6, 4
    element_size = torch.empty((), dtype=dtype).element_size()
    storage = torch.randint(-128, 128, (pages, block + padding, width * element_size), device="npu", dtype=torch.int8)
    cache = storage.view(dtype)[:, :block]
    original = storage.cpu().clone()
    source = torch.tensor([[1, 3], [2, 1], [1, 0]], device="npu")
    destination = torch.tensor([[2, 1], [2, 2], [-1, -1]], device="npu")
    # The second binding conflicts at [2,1]; the first binding owns that row.
    # The offset view reads the original [2,1] before the first write.
    moves = [
        TreeRowMove(cache, source, destination, torch.empty((3, width), device="npu", dtype=dtype)),
        TreeRowMove(cache, torch.tensor([[3, 0], [4, 1]], device="npu"), torch.tensor([[2, 1], [4, 2]], device="npu")),
        TreeRowMove(cache[2:], torch.tensor([[0, 1]], device="npu"), torch.tensor([[0, 3]], device="npu")),
    ]
    if graph:
        relocate_tree_rows(moves, deduplicate=True)
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            relocate_tree_rows(moves, deduplicate=True)
    for first_source in (3, 2):
        storage.copy_(original.to("npu"))
        source.copy_(torch.tensor([[1, first_source], [2, 1], [1, 0]], device="npu"))
        if graph:
            captured.replay()
        else:
            relocate_tree_rows(moves, deduplicate=True)
        expected = original.clone()
        expected[2, 1] = original[1, first_source]
        expected[2, 2] = original[2, 1]
        expected[4, 2] = original[4, 1]
        expected[2, 3] = original[2, 1]
        # Includes untouched pages, disabled destinations, padding and FP8 bits.
        assert torch.equal(storage.cpu(), expected)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("graph", [False, True])
def test_compactor_replays_live_tables_paths_and_request_mapping(batch, padding, graph):
    device, block, pages, width = "npu", 4, 12, 16
    counts = torch.tensor([3, 0, 4, 7], device=device, dtype=torch.int32)
    runner = SimpleNamespace(
        device=torch.device(device),
        num_speculative_steps=3,
        max_num_reqs=4,
        max_model_len=16,
        req_states=SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=counts)),
    )
    compactor = TreeKvCompact(runner)
    table = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0], [5, 6, 7, 8], [9, 10, 11, 0]], device=device, dtype=torch.int32)
    storages = [
        torch.randint(-128, 128, (pages, block + padding, width), device=device, dtype=torch.int8) for _ in range(2)
    ]
    originals = [storage.cpu().clone() for storage in storages]
    caches = [storage[:, :block] for storage in storages]
    compactor._groups = [
        (caches, table, block, [torch.empty((12, width), device=device, dtype=torch.int8) for _ in caches])
    ]
    if graph:
        compactor._compact(batch)
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            compactor._compact(batch)
    for prefix in (4, 7):
        counts.copy_(torch.tensor([3, 0, prefix, 7], device=device, dtype=torch.int32))
        idx = [2, -1, 0][:batch]
        paths = [[3, 1, -1], [2, 1, 0], [1, 15, -1]][:batch]
        active = [True, False, True][:batch]
        compactor._idx[:batch].copy_(torch.tensor(idx, device=device, dtype=torch.int32))
        compactor._path[:batch].copy_(torch.tensor(paths, device=device))
        compactor._active[:batch].copy_(torch.tensor(active, device=device))
        # Change a live block mapping between graph replays.
        table[2].copy_(torch.tensor([5, 8, 7, 6], device=device, dtype=torch.int32))
        for storage, original in zip(storages, originals):
            storage.copy_(original.to(device))
        if graph:
            captured.replay()
        else:
            compactor._compact(batch)
        host_table, host_counts = table.cpu(), counts.cpu()
        for storage, original in zip(storages, originals):
            expected = original.clone()
            for req, state in enumerate(idx):
                if state < 0 or not active[req]:
                    continue
                for depth, node in enumerate(paths[req], 1):
                    src, dst = int(host_counts[state]) + node, int(host_counts[state]) + depth
                    if node < 0 or not (0 <= src < 16 and 0 <= dst < 16):
                        continue
                    src_page, dst_page = int(host_table[state, src // block]), int(host_table[state, dst // block])
                    if 0 < src_page < pages and 0 < dst_page < pages:
                        expected[dst_page, dst % block] = original[src_page, src % block]
            assert torch.equal(storage.cpu(), expected)


def test_original_kv_layers_share_round_snapshot_without_reusing_other_groups(monkeypatch):
    monkeypatch.setattr(chain_pack, "_LAYOUT", {"num_nodes": torch.tensor([3], device="npu")})
    table = torch.tensor([[1, 2, 3, 4]], device="npu", dtype=torch.int32)
    first_table = table.clone()
    caches = [torch.zeros((8, 4, 16), device="npu") for _ in range(3)]
    meta = SimpleNamespace(block_table=table, storage_block_size=4, cache_group_key="swa")
    record_original_kv("layer0", caches[0], meta)
    table.fill_(7)
    record_original_kv("layer1", caches[1], meta)
    meta.cache_group_key = "other"
    record_original_kv("layer2", caches[2], meta)
    compactor = SimpleNamespace(
        runner=SimpleNamespace(max_num_reqs=1),
        _idx=torch.zeros(1, device="npu", dtype=torch.int32),
        _binding_scratch={},
        max_num_reqs=1,
        spec_len=3,
    )
    groups = bind_original_kv(compactor, 1)
    assert [len(group[0]) for group in groups] == [2, 1]
    assert torch.equal(groups[0][1].cpu(), first_table.cpu())
    assert torch.equal(groups[1][1].cpu(), table.cpu())
