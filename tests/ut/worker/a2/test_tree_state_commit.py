# SPDX-License-Identifier: Apache-2.0
"""NPU byte-exact regression tests for accepted tree state commitment."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp


def reference_prepare(plan, selected):
    """Original first-enabled-dirty rule, including frozen baseline bytes."""
    copies = []
    for cache, page, candidates in plan["records"].values():
        value = cache[page].contiguous().view(torch.uint8).clone()
        chosen = torch.zeros((), dtype=torch.bool, device=cache.device)
        for owner, snapshot, begin, length, offsets, dirty in candidates:
            take = selected[owner] & dirty[begin : begin + length].any() & ~chosen
            if offsets is None:
                value = torch.where(take, snapshot[begin].contiguous().view(torch.uint8), value)
            else:
                rows = offsets[begin : begin + length]
                update = snapshot[begin : begin + length].contiguous().view(torch.uint8)
                value.index_copy_(0, rows, torch.where(take, update, value.index_select(0, rows)))
            chosen.logical_or_(take)
        copies.append((cache, page, value, chosen))
    return copies


@pytest.fixture(autouse=True)
def clean_plan():
    torch.npu.set_device(0)
    cp._PLAN = cp._LAYOUT = None
    yield
    torch.npu.synchronize()
    cp._PLAN = cp._LAYOUT = None


@pytest.mark.parametrize("dtype,width", [(torch.float32, 16), (torch.bfloat16, 32), (torch.uint8, 13)])
@pytest.mark.parametrize("mode", ["branch", "root", "inactive", "implicit_count"])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("nrows", [8, 128])
def test_frozen_pages_first_dirty_and_alias_order(dtype, width, mode, alias, nrows):
    torch.manual_seed(42)
    # Padded, strided physical rows exercise packed cache views. Two views of
    # the same storage deliberately overlap to require ordered commitment.
    storage = torch.randint(0, 50, (6, nrows, width + 8), dtype=dtype, device="npu")
    if dtype != torch.uint8:
        storage[1, 2, 0] = float("nan")
        storage[1, 2, 1] = -0.0
    caches = [storage[..., :width]]
    if alias:
        caches.append(storage[..., 4 : 4 + width])
    qsl = torch.tensor([0, 4, 7], dtype=torch.int32, device="npu")
    cp._LAYOUT = {"qsl": qsl}
    plan = {
        "n_req": 2,
        "req_ids": torch.tensor([0, 0, 0, 0, 1, 1, 1], device="npu"),
        "token_index": torch.arange(7, device="npu"),
        "committed": False,
        "pending": [],
        "records": {},
        "host_uploads": [],
        "workspace_layouts": {},
        "cache_views": {},
        "kv_tables": {},
        "ori_views": {},
        "buffers": SimpleNamespace(reset=lambda: None),
    }
    cp._PLAN = plan
    # All branches are snapshotted. Two invocations of the same cache test
    # first-dirty ordering, including an unchanged first selected candidate.
    for invocation in range(2):
        for cache_index, cache in enumerate(caches):
            entries = [(owner, 1 if owner < 4 else 3, owner + 1) for owner in range(7)]
            workspace = torch.empty((8, nrows, width), dtype=dtype, device="npu")
            before_rows, offsets, source_ids, spans = [], [], [], []
            for owner, dst, src in entries:
                workspace[src].copy_(cache[dst])
                rows = [0, 3, 7] if owner % 2 else [1, 6]
                spans.append((len(offsets), len(rows)))
                for row in rows:
                    before_rows.append(dst)
                    offsets.append(row)
                    source_ids.append(src)
                # For winner 3, first invocation is clean. Other winners and
                # second invocation change, with distinguishable payloads.
                if owner != 3 or invocation:
                    workspace[src, rows] = 70 + owner + invocation * 10 + cache_index
            layout = {
                "entries": entries,
                "state_indices": torch.tensor(
                    list(zip(before_rows, offsets, source_ids)), dtype=torch.long, device="npu"
                ).unbind(1),
                "state_spans": spans,
            }
            plan["pending"].append((cache, workspace, "state", layout))
        cp.finish_tree_layer()
    # Also test whole-page compressed KV and disabled writes.
    kv = torch.randint(0, 50, (4, 5, width), dtype=dtype, device="npu")
    work = torch.randint(80, 100, (3, 5, width), dtype=dtype, device="npu")
    layout = {
        "entries": [(3, 1, 1), (6, 2, 2)],
        "snapshot_indices": (None, torch.tensor([1, 2], device="npu")),
        "written": torch.tensor([True, False], device="npu"),
    }
    plan["pending"].append((kv, work, "kv", layout))
    cp.finish_tree_layer()
    path = torch.tensor([[1, 3, -1], [2, -1, -1]], dtype=torch.int32, device="npu")
    sampled = torch.tensor([3, 2], dtype=torch.int32, device="npu")
    if mode == "root":
        path.fill_(-1)
        sampled.fill_(1)
    if mode == "inactive":
        sampled.zero_()
    if mode == "implicit_count":
        sampled = None
    endpoint = torch.where(path != -1, path, 0).amax(1)
    active = (sampled > 0) if sampled is not None else torch.ones(2, dtype=torch.bool, device="npu")
    req = plan["req_ids"]
    selected = active[req] & (plan["token_index"] == (qsl[:-1] + endpoint)[req])
    reference = reference_prepare(plan, selected)
    prepared = cp.prepare_winner(path, sampled)
    # Simulate intervening KV compaction modifying *all* baseline bytes. The
    # active page must restore frozen untouched rows, the inactive page must
    # leave this compaction result untouched.
    storage.fill_(123)
    kv.fill_(124)
    expected_storage, expected_kv = storage.clone(), kv.clone()
    for cache, page, value, enabled in reference:
        if cache is kv:
            target = expected_kv[page].view(torch.uint8)
        else:
            start = (cache.data_ptr() - storage.data_ptr()) // storage.element_size()
            target = expected_storage[..., start : start + width][page].view(torch.uint8)
        target.masked_scatter_(enabled.expand_as(target), value.reshape(-1))
    cp.commit_winner(path, sampled, prepared=prepared)
    torch.npu.synchronize()
    assert torch.equal(storage.view(torch.uint8), expected_storage.view(torch.uint8))
    assert torch.equal(kv.view(torch.uint8), expected_kv.view(torch.uint8))
    assert plan["committed"] and not plan["records"]


def test_empty_records():
    cp._LAYOUT = {"qsl": torch.tensor([0, 1], dtype=torch.int32, device="npu")}
    cp._PLAN = {
        "n_req": 1,
        "req_ids": torch.tensor([0], device="npu"),
        "token_index": torch.tensor([0], device="npu"),
        "pending": [],
        "records": {},
        "host_uploads": [],
        "committed": False,
        "workspace_layouts": {},
    }
    cp.commit_winner(torch.tensor([[-1]], dtype=torch.int32, device="npu"))
    assert cp._PLAN["committed"]
