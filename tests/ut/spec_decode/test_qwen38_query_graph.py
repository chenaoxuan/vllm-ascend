# SPDX-License-Identifier: Apache-2.0
"""CPU routing, cache identity and lifetime checks for Qwen query graphs."""

import gc
import weakref
from dataclasses import dataclass, fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState
from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.commit import TreeCommitCoordinator
from vllm_ascend.worker.v2.spec_decode.tree.qwen_query_graph import QwenTreeQueryGraphs


@dataclass(frozen=True)
class _Desc:
    cg_mode: object = CUDAGraphMode.FULL
    num_reqs: int = 1
    num_tokens: int = 17
    uniform_token_count: int = 17


def _runner(*, tree=True, gdn=True, mode=CUDAGraphMode.FULL, dsv4=False):
    state = AscendMambaHybridModelState if gdn else AscendModelState
    return SimpleNamespace(
        tree_kv_compact=SimpleNamespace(),
        model_state=state.__new__(state),
        vllm_config=SimpleNamespace(additional_config={"tree_spec_config": {"enabled": tree, "budget": 16, "topk": 8}}),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        speculator=SimpleNamespace(
            packed_dsv4_verify=lambda: dsv4,
            tree_shape=SimpleNamespace(target_query_len=17),
        ),
        cudagraph_manager=SimpleNamespace(_graphs_captured=True, graphs={_Desc(): object()}),
    )


def _batch(**changes):
    values = dict(
        num_reqs=1,
        has_prefill=False,
        is_prefilling_np=np.array([False]),
        num_tokens=17,
        num_scheduled_tokens=np.array([17]),
        query_start_loc_np=np.array([0, 17]),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _acceptance(nodes=(2, 3, 16), sampled=None):
    path = torch.full((1, 7), -1, dtype=torch.long)
    path[0, : len(nodes)] = torch.tensor(nodes, dtype=torch.long)
    return TreeAcceptance.from_path(
        path, torch.tensor([len(nodes) + 1 if sampled is None else sampled], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(num_reqs=0),
        dict(has_prefill=True),
        dict(is_prefilling_np=np.array([True])),
        dict(num_tokens=8),
        dict(num_scheduled_tokens=np.array([8])),
        dict(query_start_loc_np=np.array([0, 8])),
    ],
)
def test_only_complete_bs1_decode_rows_are_query_graph_eligible(changes):
    cache = QwenTreeQueryGraphs(_runner())
    assert cache.eligible(_batch())
    assert not cache.eligible(_batch(**changes))
    assert not cache.eligible(None)


def test_eligibility_reads_scheduler_cpu_metadata_without_device_values():
    class DeviceValue:
        def __getattr__(self, name):
            raise AssertionError("device metadata read")

        def __bool__(self):
            raise AssertionError("device metadata read")

    batch = _batch()
    batch.query_start_loc = DeviceValue()
    batch.is_prefilling = DeviceValue()
    assert QwenTreeQueryGraphs(_runner()).eligible(batch)


@pytest.mark.parametrize("captured,desc", [(False, _Desc()), (True, _Desc(num_tokens=8)), (True, None)])
def test_configured_full_without_matching_captured_target_does_not_cache(captured, desc):
    runner = _runner()
    runner.cudagraph_manager._graphs_captured = captured
    runner.cudagraph_manager.graphs = {} if desc is None else {desc: object()}
    assert not QwenTreeQueryGraphs(runner).eligible(_batch())


@pytest.mark.parametrize(
    "options,enabled",
    [
        (dict(), True),
        (dict(tree=False), False),
        (dict(gdn=False), False),
        (dict(mode=CUDAGraphMode.NONE), False),
        (dict(dsv4=True), False),
    ],
)
def test_coordinator_enables_query_graphs_only_for_gdn_tree_full(options, enabled):
    assert (TreeCommitCoordinator(_runner(**options)).qwen_query_graphs is not None) == enabled


class _RecordingGraph:
    created = []

    def __init__(self, tensors, qsl, acceptance, positions):
        self.replays = []
        self.created.append(self)

    def replay(self, qsl, acceptance):
        self.replays.append(
            (qsl.clone(), {field.name: getattr(acceptance, field.name).clone() for field in fields(TreeAcceptance)})
        )


def test_new_metadata_objects_reuse_graph_and_copy_current_values(monkeypatch):
    _RecordingGraph.created = []
    monkeypatch.setattr("vllm_ascend.worker.v2.spec_decode.tree.qwen_query_graph.TreeQueryGraph", _RecordingGraph)
    cache = QwenTreeQueryGraphs(_runner())
    tensors, positions = [torch.empty(24, 4)], torch.arange(24)
    cache.compact_queries(tensors, torch.tensor([0, 17]), _acceptance(), positions)
    newer = _acceptance((8, 12))
    cache.compact_queries(tensors, torch.tensor([2, 19]), newer, positions)
    assert len(cache.graphs) == len(_RecordingGraph.created) == 1
    graph = _RecordingGraph.created[0]
    assert graph.replays[-1][0].tolist() == [2, 19]
    for field in fields(TreeAcceptance):
        torch.testing.assert_close(graph.replays[-1][1][field.name], getattr(newer, field.name))


def test_output_address_stride_and_positions_define_capture_identity_and_lifetime(monkeypatch):
    _RecordingGraph.created = []
    monkeypatch.setattr("vllm_ascend.worker.v2.spec_decode.tree.qwen_query_graph.TreeQueryGraph", _RecordingGraph)
    cache = QwenTreeQueryGraphs(_runner())
    backing = torch.empty(48, 4)
    first = backing[:24]
    reference = weakref.ref(first)
    positions = torch.arange(24)
    qsl, acceptance = torch.tensor([0, 17]), _acceptance()
    cache.compact_queries([first], qsl, acceptance, positions)
    cache.compact_queries([torch.empty_like(first)], qsl, acceptance, positions)
    cache.compact_queries([backing[::2]], qsl, acceptance, positions)
    cache.compact_queries([first], qsl, acceptance, positions.clone())
    assert len(cache.graphs) == 4
    del first
    gc.collect()
    assert reference() is not None


def test_capture_failure_propagates_and_does_not_insert_cache_entry(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("capture unsupported")

    monkeypatch.setattr("vllm_ascend.worker.v2.spec_decode.tree.qwen_query_graph.TreeQueryGraph", fail)
    cache = QwenTreeQueryGraphs(_runner())
    with pytest.raises(RuntimeError, match="capture unsupported"):
        cache.compact_queries([torch.empty(17, 4)], torch.tensor([0, 17]), _acceptance(), torch.arange(17))
    assert cache.graphs == {}


def test_coordinator_routes_only_full_decode_rows_to_query_graph(monkeypatch):
    coordinator = TreeCommitCoordinator(_runner())
    calls = []
    monkeypatch.setattr(coordinator.qwen_query_graphs, "compact_queries", lambda *args: calls.append("graph"))
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.tree.commit.compact_tree_query_along_path",
        lambda *args: calls.append("eager"),
    )
    args = [torch.empty(17, 4)], torch.tensor([0, 17]), _acceptance(), torch.arange(17)
    coordinator.compact_queries(*args, input_batch=_batch())
    coordinator.compact_queries(*args, input_batch=_batch(has_prefill=True))
    coordinator.compact_queries(*args, input_batch=_batch(num_tokens=8))
    assert calls == ["graph", "eager", "eager"]
