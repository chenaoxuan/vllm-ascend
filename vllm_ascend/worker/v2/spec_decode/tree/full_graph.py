# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FULL target and accepted-state graphs for the DSV4 shared-node tree path.

Host code selects shape/capacity gears and copies scheduler metadata to stable
inputs. Tree topology, node state and commitment remain device computations.
Target sampling and rejection stay outside these graphs, as in chain decoding.
"""

import copy
from dataclasses import fields, is_dataclass
from math import prod

import torch
from vllm.logger import init_logger
from vllm.triton_utils import triton

from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp
from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.kv_binding import bind_original_kv
from vllm_ascend.worker.v2.spec_decode.tree.kv_project import TreeKvCompact
from vllm_ascend.worker.v2.spec_decode.tree.metrics import TreeGraphStats

logger = init_logger(__name__)


class PrefixStaging:
    """Reuse pinned prefix inputs only after their asynchronous H2D completes.

    Event queries never wait. If every retained slot is still in flight, grow
    the small staging pool instead of overwriting a live source or adding a
    synchronization. Steady sequential decoding reuses its existing slots.
    """

    def __init__(self, size, device):
        self.size = size
        self.is_npu = torch.device(device).type == "npu"
        self.slots = []

    def upload(self, destination, values):
        if len(values) != self.size:
            raise RuntimeError("Tree FULL prefix batch changed within a graph gear")
        for host, event in self.slots:
            if event is None or event.query():
                break
        else:
            host = torch.empty(self.size, dtype=torch.int32, device="cpu", pin_memory=self.is_npu)
            event = torch.npu.Event() if self.is_npu else None
            self.slots.append((host, event))
        host.numpy()[...] = values  # CPU-only scheduler metadata.
        destination.copy_(host, non_blocking=True)  # H2D at the scheduler boundary.
        if event is not None:
            event.record()


class GraphMetadata:
    """Clone mutable metadata tensors once, retaining aliases and source paths."""

    def __init__(self, metadata):
        self.copies = []
        memo = {}

        def clone(value, path=(), name=""):
            if name in ("full_compress_cos", "full_compress_sin", "hadamard"):
                return value
            if isinstance(value, torch.Tensor):
                key = (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
                if key not in memo:
                    memo[key] = value.clone()
                    self.copies.append((path, memo[key]))
                return memo[key]
            if is_dataclass(value) and not isinstance(value, type):
                result = copy.copy(value)
                for field in fields(value):
                    object.__setattr__(
                        result, field.name, clone(getattr(value, field.name), path + (field.name,), field.name)
                    )
                return result
            if isinstance(value, dict):
                return {key: clone(item, path + (key,), str(key)) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                result = [clone(item, path + (i,)) for i, item in enumerate(value)]
                return type(value)(*result) if hasattr(value, "_fields") else type(value)(result)
            return value

        self.value = clone(metadata)

    def update(self, metadata):
        for path, destination in self.copies:
            value = metadata
            for key in path:
                value = getattr(value, key) if is_dataclass(value) else value[key]
            if value.shape != destination.shape or value.dtype != destination.dtype:
                raise RuntimeError(f"Tree FULL metadata shape changed at {path}")
            destination.copy_(value)


class GraphNodeCommit:
    """Stable descriptors cover all pages a depth-limited accepted chain can touch."""

    def __init__(self, plan, prefixes):
        from vllm_ascend.ops.triton.spec_decode.tree.graph_commit import update_commit_descriptors

        self.plan, self.prefixes = plan, prefixes
        records, total, max_rows = [], 0, 1
        for original, nodes, table, _, ratio in plan["node_records"]:
            size = original.element_size()
            row_bytes = prod(original.shape[2:]) * size
            if nodes.stride(0) * nodes.element_size() != row_bytes:
                raise RuntimeError("Tree FULL commitment requires contiguous node rows")
            rows = original.shape[1]
            for req in range(plan["n_req"]):
                for page_offset in range(triton.cdiv(plan["max_q"] - 1, rows * ratio) + 1):
                    records.append(
                        (
                            original.data_ptr(),
                            original.stride(1) * size,
                            row_bytes,
                            rows,
                            nodes.data_ptr(),
                            table.data_ptr(),
                            table.shape[1],
                            ratio,
                            original.stride(0) * size,
                            req,
                            page_offset,
                            total,
                        )
                    )
                    total += rows * row_bytes
            max_rows = max(max_rows, rows)
        (self.base,) = cp._upload_metadata(
            prefixes.device, (records, torch.int64, (len(records), 12)), host_buffers=plan["host_uploads"]
        )
        self.descriptors = torch.empty((len(records), 10), dtype=torch.int64, device=prefixes.device)
        self.enabled = torch.empty(len(records), dtype=torch.bool, device=prefixes.device)
        self.staging = torch.empty(total, dtype=torch.uint8, device=prefixes.device)
        self.max_rows = max_rows
        self.update_descriptors = update_commit_descriptors

    def freeze(self, winner, active):
        from vllm_ascend.ops.triton.spec_decode.tree.node_commit import freeze_node_pages

        plan = self.plan
        jobs = self.descriptors.shape[0]
        self.update_descriptors[(jobs,)](self.base, self.prefixes, self.descriptors)
        freeze_node_pages[(jobs,)](
            self.descriptors,
            plan["ancestors"],
            plan["depths"],
            plan["start"],
            plan["req_ids"],
            winner,
            active,
            self.staging,
            self.enabled,
            plan["n"],
            plan["max_q"],
            triton.next_power_of_2(plan["n"]),
        )

    def commit(self):
        from vllm_ascend.ops.triton.spec_decode.tree.node_commit import commit_node_pages

        commit_node_pages[(self.descriptors.shape[0],)](
            self.descriptors,
            self.staging,
            self.enabled,
        )


class TreeTargetGraph:
    """One exact tree shape and prefix-capacity gear, with private storage.

    Capture is lazy because the real cache-group bindings only exist after
    request admission. Warmup writes original KV to the same tree slots and
    leaves compressed/state history untouched; accepted state is a later graph.
    """

    def __init__(self, runner, desc, capacity, layout, *, pool=None, stream=None, stats=None):
        from vllm.config import set_current_vllm_config
        from vllm.config.compilation import CUDAGraphMode
        from vllm.distributed.parallel_state import GraphCaptureContext, get_dp_group, get_pp_group, get_tp_group
        from vllm.forward_context import set_forward_context

        from vllm_ascend.attention.tree_spec import build_tree_ori_sparse_indices
        from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext
        from vllm_ascend.worker.v2.utils import communicator_switch

        self.runner, self.desc = runner, desc
        self.pool, self.stream = pool, stream
        self.stats = stats if stats is not None else TreeGraphStats()
        self.metadata = GraphMetadata(
            {"attention": runner.model_state.attn_metadata, "tree_indices": layout.get("graph_tree_indices", [])}
        )
        # Index outputs share the attention metadata views and are produced
        # inside the target graph; copying their old contents is unnecessary.
        output_pointers = {job["buffer"].data_ptr() for job in self.metadata.value["tree_indices"]}
        self.metadata.copies = [
            (path, tensor) for path, tensor in self.metadata.copies if tensor.data_ptr() not in output_pointers
        ]
        self.topology = GraphMetadata({key: layout[key] for key in ("parents", "depths", "num_nodes", "qsl")})
        self.layout = {**layout, **self.topology.value, "graph_capacity": capacity, "graph_workspaces": {}}
        self.prefixes = torch.empty(len(layout["start_cpu"]), dtype=torch.int32, device=runner.device)
        self.prefix_staging = PrefixStaging(len(layout["start_cpu"]), runner.device)
        self.prefix_staging.upload(self.prefixes, layout["start_cpu"])
        self.layout["graph_prefixes"] = self.prefixes
        self.layout["full_graph"] = self
        self.post_graph = None
        self.query_graphs = {}
        self.replays = 0
        n = desc.num_tokens
        model = ModelWithContext(runner.model)
        inputs = dict(
            input_ids=runner.input_buffers.input_ids[:n],
            positions=runner.input_buffers.positions[:n],
            **runner.model_state.prepare_dummy_inputs(len(layout["start_cpu"]), n),
        )

        def forward():
            cp._LAYOUT, cp._PLAN = self.layout, None
            self.layout.pop("_ori_commit_bindings", None)
            for arguments in self.metadata.value["tree_indices"]:
                build_tree_ori_sparse_indices(**arguments)
            with (
                set_current_vllm_config(runner.vllm_config),
                set_forward_context(
                    self.metadata.value["attention"],
                    runner.vllm_config,
                    num_tokens=n,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=None,
                    is_padding=runner.input_buffers.is_padding[:n],
                ),
            ):
                self.output = model(**inputs)
            self.plan = cp._PLAN

        # The upstream wrapper now always creates a new stream. Enter its
        # group contexts directly so all tree graphs retain the shared stream
        # required by their shared memory pool.
        capture_context = GraphCaptureContext(stream if stream is not None else torch.npu.Stream(device=runner.device))
        with (
            communicator_switch(),
            get_tp_group().graph_capture(capture_context),
            get_pp_group().graph_capture(capture_context),
            get_dp_group().graph_capture(capture_context),
            torch.inference_mode(),
        ):
            forward()
            # Build a fresh Python plan while recording all topology operations.
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, pool=pool, stream=stream):
                forward()
        if self.plan is None or self.plan["node_kv"]:
            raise RuntimeError("Tree FULL target requires finalized shared-node state")
        self.stats.record_capture("target")
        retained = [
            self.prefixes,
            *(tensor for _, tensor in self.metadata.copies),
            *(tensor for _, tensor in self.topology.copies),
        ]
        for workspace in self.layout["graph_workspaces"].values():
            retained.extend(tensor for buffers in workspace.buffers._buffers.values() for tensor in buffers)
            retained.extend(tensor for views in workspace.views.values() for tensor in views)
        self.stats.set_workspace_bytes("target", id(self), self.stats.tensor_bytes(retained))
        logger.info("Captured DSV4 tree FULL target tokens=%s capacity=%s", n, capacity)

    def replay(self, layout):
        self.topology.update({key: layout[key] for key in self.topology.value})
        self.metadata.update(
            {"attention": self.runner.model_state.attn_metadata, "tree_indices": layout.get("graph_tree_indices", [])}
        )
        self.prefix_staging.upload(self.prefixes, layout["start_cpu"])
        cp._LAYOUT, cp._PLAN = self.layout, self.plan
        self.plan["committed"] = False
        self.graph.replay()
        self.replays += 1
        self.stats.record_replay("target")
        return self.output

    def commit(self, idx_mapping, acceptance: TreeAcceptance):
        if self.plan["committed"]:
            raise RuntimeError("Tree FULL state committed twice")
        self.stats.record_lookup("post", self.post_graph is not None)
        if self.post_graph is None:
            self.post_graph = TreePostGraph(self)
        self.post_graph.replay(idx_mapping, acceptance)
        self.plan["committed"] = True

    def compact_queries(self, tensors, qsl, acceptance, positions):
        key = (
            tuple(cp._key(tensor) for tensor in [*tensors, positions]),
            tuple(acceptance.path_node_ids.shape),
            tuple(qsl.shape),
        )
        graph = self.query_graphs.get(key)
        self.stats.record_lookup("query", graph is not None)
        if graph is None:
            graph = TreeQueryGraph(
                tensors, qsl, acceptance, positions, pool=self.pool, stream=self.stream, stats=self.stats
            )
            self.query_graphs[key] = graph
        graph.replay(qsl, acceptance)


class TreeQueryGraph:
    """Accepted hidden/aux rows and logical positions for the next draft."""

    def __init__(self, tensors, qsl, acceptance, positions, *, pool=None, stream=None, stats=None):
        from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path

        self.qsl = qsl.clone()
        self.acceptance = TreeAcceptance.from_path(
            torch.full_like(acceptance.path_node_ids, -1), torch.zeros_like(acceptance.num_sampled)
        )
        self.stats = stats if stats is not None else TreeGraphStats()
        self.replays = 0

        def run():
            compact_tree_query_along_path(tensors, self.qsl, self.acceptance, positions)

        # Disabled request rows make warmup/capture non-mutating.
        with torch.inference_mode():
            run()
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, pool=pool, stream=stream):
                run()
        self.stats.record_capture("query")
        self.stats.set_workspace_bytes(
            "query",
            id(self),
            self.stats.tensor_bytes(
                [
                    self.qsl,
                    self.acceptance.path_node_ids,
                    self.acceptance.num_sampled,
                    self.acceptance.num_accepted,
                    self.acceptance.endpoint_node,
                    self.acceptance.active,
                ]
            ),
        )

    def replay(self, qsl, acceptance):
        self.qsl.copy_(qsl)
        for field in fields(TreeAcceptance):
            getattr(self.acceptance, field.name).copy_(getattr(acceptance, field.name))
        self.graph.replay()
        self.replays += 1
        self.stats.record_replay("query")


class TreePostGraph:
    """Freeze node state, relocate original KV, and commit the accepted chain."""

    def __init__(self, target):
        self.target = target
        runner, plan = target.runner, target.plan
        # Each target gear retains its own gather scratch and input addresses.
        # Other gears and intervening eager batches cannot replace these views.
        self.compactor = TreeKvCompact(runner, bind_groups=bind_original_kv)
        self.num_reqs = plan["n_req"]
        self.endpoint = torch.zeros(self.num_reqs, dtype=torch.long, device=runner.device)
        self.nodes = GraphNodeCommit(plan, target.prefixes)
        self.replays = 0
        # Warmup/capture must not relocate real KV twice. Empty paths are a
        # device no-op, while recording precisely the same operations.
        self.compactor._idx[: self.num_reqs].zero_()
        self.compactor._path[: self.num_reqs].fill_(-1)
        with torch.inference_mode():
            self._run()
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, pool=target.pool, stream=target.stream):
                self._run()
        target.stats.record_capture("post")
        retained = [
            self.endpoint,
            self.compactor._idx,
            self.compactor._path,
            self.compactor._active,
            self.nodes.base,
            self.nodes.descriptors,
            self.nodes.enabled,
            self.nodes.staging,
        ]
        retained.extend(self.compactor._binding_scratch.values())
        target.stats.set_workspace_bytes("post", id(self), target.stats.tensor_bytes(retained))
        logger.info("Captured DSV4 tree FULL accepted-state requests=%s", self.num_reqs)

    def _run(self):
        n, compactor = self.num_reqs, self.compactor
        winner = self.target.layout["qsl"][:n].long() + self.endpoint
        self.nodes.freeze(winner, compactor._active[:n])
        compactor._groups = compactor.bind_groups(compactor, n)
        compactor._compact(n)
        self.nodes.commit()

    def replay(self, idx_mapping, acceptance):
        n = self.num_reqs
        self.compactor._idx[:n].copy_(idx_mapping[:n])
        self.compactor._path[:n].copy_(acceptance.path_node_ids)
        self.compactor._active[:n].copy_(acceptance.active)
        self.endpoint.copy_(acceptance.endpoint_node)
        self.graph.replay()
        self.replays += 1
        self.target.stats.record_replay("post")


class TreeFullGraphManager:
    """Tree-only lazy FULL gears; regular model graph dispatch is unchanged."""

    def __init__(self, runner):
        self.runner = runner
        self.graphs = {}
        self.stats = TreeGraphStats()
        # Serialized tree phases/gears reuse temporary allocations. Persistent
        # metadata, outputs and plans remain owned by each graph. Separate pools
        # otherwise strand scratch memory needed by the next eager prefill.
        self.pool = torch.npu.graph_pool_handle()
        self.stream = torch.npu.Stream(device=runner.device)

    def run(self, desc):
        layout = cp._LAYOUT
        capacity = triton.next_power_of_2(max(layout["start_cpu"]) + layout["max_depth"] + 1)
        # Exact boundaries cover short final trees without reusing a graph
        # whose Python loop bounds or request layout differ.
        key = (desc, tuple(layout["qsl_cpu"]), capacity)
        target = self.graphs.get(key)
        self.stats.record_lookup("target", target is not None)
        if target is None:
            target = TreeTargetGraph(
                self.runner, desc, capacity, layout, pool=self.pool, stream=self.stream, stats=self.stats
            )
            self.graphs[key] = target
        return target.replay(layout)
