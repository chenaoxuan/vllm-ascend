# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
from dataclasses import dataclass

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.node_pool import build_ancestors
from vllm_ascend.worker.v2.spec_decode.tree.full_graph import GraphMetadata, GraphNodeCommit


def _report_graph_latency(stage, batch, eager, replay, **extra):
    timings = {}
    for name, run in (("eager", eager), ("full", replay)):
        for _ in range(3):
            run()
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            run()
        torch.npu.synchronize()
        timings[name + "_ms"] = (time.perf_counter() - start) * 50
    print("TREE_GRAPH_UT_BENCH", json.dumps(dict(stage=stage, batch=batch, **extra, **timings)), flush=True)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_full_graph_commit_uses_live_prefix_tables_and_winners(batch, ratio):
    torch.npu.set_device(0)
    device = "npu"
    n = batch * 5
    parents = torch.tensor([[0, 0, 1, 2]] * batch, dtype=torch.int32, device=device)
    depths = torch.tensor([[1, 1, 2, 2]] * batch, dtype=torch.int32, device=device)
    counts = torch.full((batch,), 4, dtype=torch.int32, device=device)
    req = torch.arange(batch, device=device).repeat_interleave(5).long()
    qsl = torch.arange(0, n + 1, 5, dtype=torch.int32, device=device)
    ancestors, _, depth, _, _ = build_ancestors(parents, depths, counts, req, qsl, 3)
    prefixes = torch.zeros(batch, dtype=torch.int32, device=device)
    table = torch.ones((batch, 8), dtype=torch.int32, device=device)
    original = torch.arange((batch * 8 + 1) * 2 * 16, device=device, dtype=torch.int32).reshape(-1, 2, 1, 16)
    baseline = original.clone()
    nodes = torch.arange(n * 16, device=device, dtype=torch.int32).reshape(n, 1, 16) + 100000
    plan = dict(
        n=n,
        n_req=batch,
        max_q=4,
        req_ids=req,
        ancestors=ancestors,
        depths=depth,
        start=torch.empty(n, dtype=torch.int32, device=device),
        host_uploads=[],
        node_records=[(original, nodes, table, (), ratio)],
    )
    commit = GraphNodeCommit(plan, prefixes)
    winner = torch.arange(batch, device=device) * 5 + 4
    active = torch.ones(batch, dtype=torch.bool, device=device)

    def execute():
        plan["start"].copy_(prefixes.index_select(0, req))
        commit.freeze(winner, active)
        original.add_(7000)  # Relocation must not destroy preserved page rows.
        commit.commit()

    with torch.inference_mode():
        execute()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            execute()
        for step in range(3):
            starts = [max(0, (step + 2) * ratio - 2 + r % 2) for r in range(batch)]
            maps = [[1 + (r * 8 + col + step) % (batch * 8) for col in range(8)] for r in range(batch)]
            # Include shared physical pages and inactive requests.
            if step == 2:
                maps = [[1 + (col + step) % 8 for col in range(8)] for _ in range(batch)]
            local_winners = [3 if (r + step) % 2 else 4 for r in range(batch)]
            enabled = [(r + step) % 3 != 1 for r in range(batch)]
            prefixes.copy_(torch.tensor(starts, dtype=torch.int32, device=device))
            table.copy_(torch.tensor(maps, dtype=torch.int32, device=device))
            winner.copy_(torch.tensor([r * 5 + w for r, w in enumerate(local_winners)], device=device))
            active.copy_(torch.tensor(enabled, device=device))
            original.copy_(baseline)
            graph.replay()
            expected = baseline.cpu()
            values = nodes.cpu()
            touched = set()
            for r in reversed(range(batch)):
                if not enabled[r]:
                    continue
                chain = [0, 1, 3] if local_winners[r] == 3 else [0, 2, 4]
                for d, node in enumerate(chain):
                    pos = starts[r] + d
                    if (pos + 1) % ratio:
                        continue
                    logical = (pos + 1) // ratio - 1
                    page = maps[r][logical // 2]
                    expected[page, logical % 2] = values[r * 5 + node]
                    touched.add(page)
            for page in range(expected.shape[0]):
                if page not in touched:
                    expected[page].add_(7000)
            torch.testing.assert_close(original.cpu(), expected, atol=0, rtol=0)


def test_graph_metadata_aliases_and_immutable_rope_tables():
    @dataclass
    class Metadata:
        block_table: torch.Tensor
        alias: torch.Tensor
        full_compress_cos: torch.Tensor

    tensor = torch.ones((2, 4), dtype=torch.int32, device="npu")
    rope = torch.ones((32, 64), device="npu")
    metadata = Metadata(tensor, tensor, rope)
    frozen = GraphMetadata({"layer": metadata})
    assert frozen.value["layer"].block_table is frozen.value["layer"].alias
    assert frozen.value["layer"].full_compress_cos is rope
    assert len(frozen.copies) == 1
    tensor.fill_(7)
    frozen.update({"layer": metadata})
    torch.testing.assert_close(frozen.value["layer"].block_table, tensor, atol=0, rtol=0)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_full_graph_plan_replays_topology_and_prefixes(batch):
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

    torch.npu.set_device(0)
    saved = cp._LAYOUT, cp._PLAN
    try:
        device = "npu"
        prefixes = torch.full((batch,), 5, dtype=torch.int32, device=device)
        layout = dict(
            parents=torch.tensor([[0, 0, 1, 2]] * batch, dtype=torch.int32, device=device),
            depths=torch.tensor([[1, 1, 2, 2]] * batch, dtype=torch.int32, device=device),
            num_nodes=torch.full((batch,), 4, dtype=torch.int32, device=device),
            qsl=torch.arange(0, 5 * batch + 1, 5, dtype=torch.int32, device=device),
            qsl_cpu=list(range(0, 5 * batch + 1, 5)),
            start_cpu=[5] * batch,
            max_depth=4,
            graph_prefixes=prefixes,
            graph_capacity=128,
        )
        cp._LAYOUT, cp._PLAN = layout, None
        with torch.inference_mode():
            cp._build_plan(5 * batch, device)
            cp._PLAN = None
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                plan = cp._build_plan(5 * batch, device)
            for step in range(3):
                prefixes.fill_(11 + step)
                parents = [0, 0, 1, 2] if step == 0 else [0, 1, 2, 3]
                depths = [1, 1, 2, 2] if step == 0 else [1, 2, 3, 4]
                layout["parents"].copy_(torch.tensor([parents] * batch, dtype=torch.int32, device=device))
                layout["depths"].copy_(torch.tensor([depths] * batch, dtype=torch.int32, device=device))
                graph.replay()
                torch.testing.assert_close(plan["start"], prefixes.repeat_interleave(5), atol=0, rtol=0)
                expected = torch.tensor(([0] + depths) * batch, dtype=plan["depths"].dtype, device=device)
                torch.testing.assert_close(plan["depths"], expected, atol=0, rtol=0)
                torch.testing.assert_close(plan["seqused"], plan["start"] + expected.int() + 1, atol=0, rtol=0)
                assert plan["max_kv"] == 128
    finally:
        cp._LAYOUT, cp._PLAN = saved


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_full_graph_workspace_gear_does_not_alias_eager_or_other_gear(batch):
    from types import SimpleNamespace

    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp
    from vllm_ascend.worker.v2.spec_decode.tree.node_pool import node_workspace

    torch.npu.set_device(0)
    saved = cp._LAYOUT
    try:
        module = SimpleNamespace()
        gears, buffers = [{"graph_workspaces": {}} for _ in range(2)], []
        for gear in [*gears, {}]:
            cp._LAYOUT = gear
            workspace = node_workspace(module, {})
            buffer = workspace.buffers.acquire("nodes", (batch * 5, 16), torch.float32, "npu")
            buffer.fill_(len(buffers) + 1)
            buffers.append(buffer)
        assert len({buffer.data_ptr() for buffer in buffers}) == 3
        for index, buffer in enumerate(buffers):
            torch.testing.assert_close(buffer, torch.full_like(buffer, index + 1), atol=0, rtol=0)
        cp._LAYOUT = gears[0]
        reused = node_workspace(module, {}).buffers.acquire("nodes", (batch * 5, 16), torch.float32, "npu")
        assert reused.data_ptr() == buffers[0].data_ptr()
    finally:
        cp._LAYOUT = saved


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_full_graph_beam_tree_matches_eager_after_logits_change(batch):
    from dataclasses import fields

    from vllm_ascend.worker.v2.spec_decode.tree.beam import BeamTreeBuilder
    from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, ensure_finalize_scratch

    torch.npu.set_device(0)
    device = torch.device("npu")
    budget, depth, vocab = 16, 7, 128

    def empty_layout():
        return TreeLayout(
            *[torch.empty((batch, budget), dtype=torch.long, device=device) for _ in range(3)],
            torch.empty(batch, dtype=torch.int32, device=device),
            torch.empty((batch, budget, budget), dtype=torch.bool, device=device),
            *[torch.empty((batch, budget + 1), dtype=torch.int32, device=device) for _ in range(2)],
        )

    torch.manual_seed(123)
    logits = torch.randn((batch, depth, vocab), device=device)
    roots = torch.zeros(batch, dtype=torch.long, device=device)
    builder, reference = BeamTreeBuilder(budget, 8), BeamTreeBuilder(budget, 8)
    output, expected = empty_layout(), empty_layout()
    builder.prepare_for_capture(batch, vocab, depth, device)
    ensure_finalize_scratch(batch, device)
    with torch.inference_mode():
        builder.build(logits, output, root_token_ids=roots, graph_safe=True)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            builder.build(logits, output, root_token_ids=roots, graph_safe=True)
        for step in range(3):
            logits.copy_(torch.randn_like(logits) + step)
            graph.replay()
            reference.build(logits, expected, root_token_ids=roots, graph_safe=False)
            for field in fields(output):
                torch.testing.assert_close(getattr(output, field.name), getattr(expected, field.name), atol=0, rtol=0)

        _report_graph_latency(
            "beam_build",
            batch,
            lambda: builder.build(logits, output, root_token_ids=roots, graph_safe=True),
            graph.replay,
        )


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_full_graph_query_compaction_preserves_capture_inputs_and_replays_paths(batch):
    from vllm_ascend.worker.v2.spec_decode.tree.full_graph import TreeQueryGraph
    from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path

    torch.npu.set_device(0)
    device = "npu"
    qsl = torch.arange(0, batch * 5 + 1, 5, dtype=torch.int32, device=device)
    path = torch.tensor([[1, 3, -1]] * batch, dtype=torch.long, device=device)
    sampled = torch.full((batch,), 3, dtype=torch.int32, device=device)
    baseline = torch.arange(batch * 5 * 16, device=device, dtype=torch.float32).reshape(batch * 5, 16)
    hidden = baseline.clone()
    positions = torch.arange(batch * 5, device=device)
    position_baseline = positions.clone()
    with torch.inference_mode():
        graph = TreeQueryGraph([hidden], qsl, path, positions, sampled)
        # Constructing the graph must not compact a real path during warmup.
        torch.testing.assert_close(hidden, baseline, atol=0, rtol=0)
        torch.testing.assert_close(positions, position_baseline, atol=0, rtol=0)
        for step in range(3):
            hidden.copy_(baseline + step)
            positions.copy_(position_baseline + step * 10)
            path.copy_(torch.tensor([[1, 3, -1] if step == 0 else [2, 4, -1]] * batch, device=device))
            if step == 2:
                sampled[::2].zero_()
            expected, expected_pos = hidden.clone(), positions.clone()
            compact_tree_query_along_path([expected], qsl, path, expected_pos, num_sampled=sampled)
            graph.replay(qsl, path, sampled)
            torch.testing.assert_close(hidden, expected, atol=0, rtol=0)
            torch.testing.assert_close(positions, expected_pos, atol=0, rtol=0)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_full_graph_tree_indices_follow_visibility_prefix_and_page_changes(batch):
    from vllm_ascend.attention.tree_spec import build_tree_ori_sparse_indices

    torch.npu.set_device(0)
    device = "npu"
    table = torch.arange(1, batch * 8 + 1, device=device, dtype=torch.int32).reshape(batch, 8)
    seq = torch.full((batch,), 15, device=device, dtype=torch.int32)
    qsl = torch.arange(0, batch * 5 + 1, 5, device=device, dtype=torch.int32)
    visibility = torch.eye(4, device=device, dtype=torch.bool).expand(batch, -1, -1).clone()
    output = torch.empty((batch * 5, 1, 128), device=device, dtype=torch.int32)
    expected = torch.empty_like(output)

    def run(buffer):
        return build_tree_ori_sparse_indices(table, seq, qsl, visibility, batch, batch * 5, 8, 4, buffer)

    with torch.inference_mode():
        run(output)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run(output)
        for step in range(3):
            seq.add_(1)
            table.copy_(table.roll(1, dims=1))
            visibility.copy_(torch.tril(torch.ones_like(visibility), diagonal=-step + 2))
            graph.replay()
            run(expected)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
        _report_graph_latency("tree_indices", batch, lambda: run(output), graph.replay)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("shared_pool", [False, True])
def test_full_target_coordinator_replays_live_metadata_and_topology(batch, shared_pool, monkeypatch):
    from contextlib import contextmanager, nullcontext
    from types import SimpleNamespace

    import vllm.config
    import vllm.distributed.parallel_state
    import vllm.forward_context

    from vllm_ascend.worker.v2 import aclgraph_utils, utils
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp
    from vllm_ascend.worker.v2.spec_decode.tree.full_graph import TreeTargetGraph

    torch.npu.set_device(0)
    device = torch.device("npu")
    saved = cp._LAYOUT, cp._PLAN
    metadata_context = {}

    @contextmanager
    def context(metadata, *args, **kwargs):
        metadata_context["value"] = metadata
        yield

    monkeypatch.setattr(vllm.config, "set_current_vllm_config", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(vllm.forward_context, "set_forward_context", context)
    monkeypatch.setattr(vllm.distributed.parallel_state, "graph_capture", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(aclgraph_utils, "ModelWithContext", lambda model: model)
    monkeypatch.setattr(utils, "communicator_switch", nullcontext)

    def model(input_ids, positions):
        plan = cp._build_plan(input_ids.shape[0], input_ids.device)
        return input_ids.float() + positions + plan["start"] + plan["depths"] + metadata_context["value"]["scale"]

    n = batch * 5
    inputs = SimpleNamespace(
        input_ids=torch.arange(n, device=device),
        positions=torch.arange(n, device=device),
        is_padding=torch.zeros(n, device=device, dtype=torch.bool),
    )
    state = SimpleNamespace(
        attn_metadata={"scale": torch.ones(n, device=device)}, prepare_dummy_inputs=lambda *args: {}
    )
    runner = SimpleNamespace(device=device, model=model, model_state=state, input_buffers=inputs, vllm_config=None)
    layout = dict(
        parents=torch.tensor([[0, 0, 1, 2]] * batch, dtype=torch.int32, device=device),
        depths=torch.tensor([[1, 1, 2, 2]] * batch, dtype=torch.int32, device=device),
        num_nodes=torch.full((batch,), 4, dtype=torch.int32, device=device),
        qsl=torch.arange(0, n + 1, 5, dtype=torch.int32, device=device),
        qsl_cpu=list(range(0, n + 1, 5)),
        start_cpu=[5] * batch,
        max_depth=4,
    )
    try:
        with torch.inference_mode():
            pool = torch.npu.graph_pool_handle() if shared_pool else None
            stream = torch.npu.Stream(device=device) if shared_pool else None
            graph = TreeTargetGraph(runner, SimpleNamespace(num_tokens=n), 128, layout, pool=pool, stream=stream)
            other = None
            if shared_pool:
                other = TreeTargetGraph(runner, SimpleNamespace(num_tokens=n), 256, layout, pool=pool, stream=stream)
                assert graph.graph.pool() == other.graph.pool() == pool
            for step in range(3):
                layout["start_cpu"] = [11 + step + r for r in range(batch)]
                inputs.input_ids.add_(1)
                state.attn_metadata = {"scale": torch.full((n,), float(step + 3), device=device)}
                if step:
                    layout["parents"].copy_(torch.tensor([[0, 1, 2, 3]] * batch, dtype=torch.int32, device=device))
                    layout["depths"].copy_(torch.tensor([[1, 2, 3, 4]] * batch, dtype=torch.int32, device=device))
                output = graph.replay(layout)
                depths = [0, 1, 1, 2, 2] if step == 0 else [0, 1, 2, 3, 4]
                expected = (
                    inputs.input_ids.float()
                    + inputs.positions
                    + state.attn_metadata["scale"]
                    + torch.tensor(layout["start_cpu"], device=device).repeat_interleave(5)
                    + torch.tensor(depths * batch, device=device)
                )
                torch.testing.assert_close(output, expected, atol=0, rtol=0)
                assert cp._PLAN is graph.plan and cp._LAYOUT is graph.layout
                assert graph.plan["committed"] is False
                if other is not None:
                    # A different gear runs between requests. Reusing its
                    # scratch must preserve the first gear's live outputs.
                    other_layout = {**layout, "start_cpu": [x + 32 for x in layout["start_cpu"]]}
                    other_output = other.replay(other_layout)
                    torch.testing.assert_close(other_output, expected + 32, atol=0, rtol=0)
                    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    finally:
        cp._LAYOUT, cp._PLAN = saved


@pytest.mark.parametrize("tokens", [5, 17])
def test_tree_full_dispatch_uses_exact_shape_and_preserves_regular_dispatch(tokens, monkeypatch):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

    regular = object()
    monkeypatch.setattr(ModelCudaGraphManager, "dispatch", lambda *args, **kwargs: regular)
    manager = object.__new__(ModelAclGraphManager)
    manager._graphs_captured = True
    manager.tree_full_graph_manager = object()
    manager.tree_batch_shape = (1, tokens)
    desc = manager.dispatch(1, tokens, None, 0, max_query_len=tokens)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_tokens == tokens and desc.num_reqs == 1
    assert manager.dispatch(1, tokens, None, 1, max_query_len=tokens) is regular
    manager.tree_batch_shape = None
    assert manager.dispatch(1, tokens, None, 0, max_query_len=tokens) is regular
    manager.tree_batch_shape = (1, tokens)
    manager.tree_full_graph_manager = None
    assert manager.dispatch(1, tokens, None, 0, max_query_len=tokens) is regular


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("rotate_prefill", [False, True])
def test_full_tree_sampling_stays_eager_and_reads_current_request_state(batch, rotate_prefill, monkeypatch):
    from contextlib import nullcontext
    from dataclasses import make_dataclass
    from types import SimpleNamespace

    import numpy as np
    import vllm.distributed.parallel_state

    from vllm_ascend.worker.v2 import model_runner as runner_module
    from vllm_ascend.worker.v2 import utils
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp
    from vllm_ascend.worker.v2.spec_decode.tree import rejection_sampler as reject
    from vllm_ascend.worker.v2.spec_decode.tree import triton_dispatch

    torch.npu.set_device(0)
    device = torch.device("npu")
    monkeypatch.setattr(vllm.distributed.parallel_state, "graph_capture", lambda **kw: nullcontext())
    monkeypatch.setattr(utils, "communicator_switch", nullcontext)
    monkeypatch.setattr(reject, "get_ascend_config", lambda: SimpleNamespace(tree_spec_config=SimpleNamespace(topk=8)))
    monkeypatch.setattr(triton_dispatch, "use_tree_triton", lambda: True)
    sampler = object.__new__(reject.TreeRejectionSampler)
    sampler.device, sampler.num_speculative_steps = device, 3
    sampler._path_node_ids_buf = sampler._sampled_buf = sampler._packed_logits_buf = None
    sampler._buf_cap_reqs = sampler._buf_cap_vocab = sampler._buf_cap_nodes = 0
    prefill = torch.zeros(batch, device=device, dtype=torch.int32)
    sampler.sampler = SimpleNamespace(
        compute_nans=False, req_states=SimpleNamespace(prefill_len=SimpleNamespace(gpu=prefill))
    )
    sampler.path_node_ids = None
    data = dict(
        num_reqs=batch,
        tree_tokens=torch.tensor([[1, 2, 3, 4]] * batch, device=device),
        tree_depths=torch.tensor([[1, 1, 2, 2]] * batch, device=device),
        tree_parents=torch.tensor([[0, 0, 1, 2]] * batch, device=device),
        tree_num_nodes=torch.full((batch,), 4, device=device, dtype=torch.int32),
        tree_visibility=torch.zeros((batch, 4, 4), device=device, dtype=torch.bool),
        tree_first_child=torch.full((batch, 5), -1, device=device, dtype=torch.long),
        tree_next_sibling=torch.full((batch, 5), -1, device=device, dtype=torch.long),
        cu_num_logits_np=np.arange(0, batch * 5 + 1, 5, dtype=np.int32),
        cu_num_logits=torch.arange(0, batch * 5 + 1, 5, device=device, dtype=torch.int32),
        query_start_loc=torch.arange(0, batch * 5 + 1, 5, device=device, dtype=torch.int32),
        seq_lens=torch.full((batch,), 12, device=device, dtype=torch.int32),
        idx_mapping=torch.arange(batch, device=device),
        path_node_ids=None,
    )
    batch_type = make_dataclass("SampleBatch", [(name, object) for name in data])
    input_batch = batch_type(**data)

    def reject_graph_sampling(*args):
        raise AssertionError("Tree rejection sampling must run outside the graph")

    runner = SimpleNamespace(
        device=device,
        rejection_sampler=sampler,
        tree_verification_plan=object(),
        speculator=SimpleNamespace(draft_logits=None),
        model=SimpleNamespace(compute_logits=lambda hidden: hidden),
        _lmhead_tp_max_num_logits=lambda: batch * 5,
    )
    monkeypatch.setattr(runner_module, "lmhead_tp_enable", lambda: True)
    input_batch.logits_indices = torch.arange(batch * 5, device=device)
    input_batch.num_draft_tokens = batch * 4
    logits = torch.zeros((batch * 5, 16), device=device)

    def sample(hidden, current, grammar):
        output = sampler(hidden, current)
        return output, output.num_sampled, output.num_rejected

    saved = cp._LAYOUT
    cp._LAYOUT = {"full_graph": SimpleNamespace(sample=reject_graph_sampling)}
    try:
        with torch.inference_mode():
            for step in range(3):
                logits.zero_()
                target_ids = [1, 3, 4, 9, 9] if step == 0 else [2, 3, 4, 9, 9]
                logits.scatter_(1, torch.tensor(target_ids * batch, device=device).unsqueeze(1), 10)
                input_batch.tree_tokens.add_(step == 2)
                if rotate_prefill and step == 1:
                    # Admission rotates UvaBackedTensor.gpu. Its former backing
                    # can hold a longer request and incorrectly suppress every
                    # token if replay retains that pointer.
                    prefill.fill_(100)
                    prefill = torch.zeros_like(prefill)
                    sampler.sampler.req_states.prefill_len.gpu = prefill
                if step == 2:
                    prefill[::2] = 20
                eager, counts, rejected = sample(logits, input_batch, None)
                expected = [x.clone() for x in (eager.sampled_token_ids, counts, rejected, sampler.path_node_ids)]
                result, counts, rejected = runner_module.NPUModelRunner.sample(runner, logits, input_batch, None)
                for actual, reference in zip(
                    (result.sampled_token_ids, counts, rejected, sampler.path_node_ids), expected
                ):
                    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    finally:
        cp._LAYOUT = saved
