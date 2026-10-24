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

logger = init_logger(__name__)


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
        freeze_node_pages[(jobs, self.max_rows)](
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
            256,
        )

    def commit(self):
        from vllm_ascend.ops.triton.spec_decode.tree.node_commit import commit_node_pages

        commit_node_pages[(self.descriptors.shape[0], self.max_rows)](
            self.descriptors,
            self.staging,
            self.enabled,
            256,
        )


class TreeTargetGraph:
    """One exact tree shape and prefix-capacity gear, with private storage.

    Capture is lazy because the real cache-group bindings only exist after
    request admission. Warmup writes original KV to the same tree slots and
    leaves compressed/state history untouched; accepted state is a later graph.
    """

    def __init__(self, runner, desc, capacity, layout, *, pool=None, stream=None):
        from vllm.config import set_current_vllm_config
        from vllm.config.compilation import CUDAGraphMode
        from vllm.distributed.parallel_state import GraphCaptureContext, graph_capture
        from vllm.forward_context import set_forward_context

        from vllm_ascend.attention.tree_spec import build_tree_ori_sparse_indices
        from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext
        from vllm_ascend.worker.v2.utils import communicator_switch

        self.runner, self.desc = runner, desc
        self.pool, self.stream = pool, stream
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
        self.host_uploads = []
        (self.prefixes,) = cp._upload_metadata(
            runner.device,
            (layout["start_cpu"], torch.int32, (len(layout["start_cpu"]),)),
            host_buffers=self.host_uploads,
        )
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

        capture_context = GraphCaptureContext(stream) if stream is not None else None
        with (
            communicator_switch(),
            graph_capture(device=runner.device, graph_capture_context=capture_context),
            torch.inference_mode(),
        ):
            forward()
            # Build a fresh Python plan while recording all topology operations.
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, pool=pool, stream=stream):
                forward()
        if self.plan is None or self.plan["records"] or self.plan["pending"]:
            raise RuntimeError("Tree FULL target requires finalized shared-node state")
        logger.info("Captured DSV4 tree FULL target tokens=%s capacity=%s", n, capacity)

    def replay(self, layout):
        self.topology.update({key: layout[key] for key in self.topology.value})
        self.metadata.update(
            {"attention": self.runner.model_state.attn_metadata, "tree_indices": layout.get("graph_tree_indices", [])}
        )
        # H2D at the scheduler boundary; prefixes were already host metadata.
        host = torch.tensor(layout["start_cpu"], dtype=torch.int32, device="cpu", pin_memory=True)
        self.prefixes.copy_(host, non_blocking=True)
        self.prefix_host = host
        cp._LAYOUT, cp._PLAN = self.layout, self.plan
        self.plan["committed"] = False
        self.graph.replay()
        self.replays += 1
        return self.output

    def commit(self, idx_mapping, path, num_sampled):
        if self.plan["committed"]:
            raise RuntimeError("Tree FULL state committed twice")
        if self.post_graph is None:
            self.post_graph = TreePostGraph(self)
        self.post_graph.replay(idx_mapping, path, num_sampled)
        self.plan["committed"] = True

    def compact_queries(self, tensors, qsl, path, positions, sampled):
        key = tuple(cp._key(tensor) for tensor in [*tensors, positions])
        graph = self.query_graphs.get(key)
        if graph is None:
            graph = TreeQueryGraph(tensors, qsl, path, positions, sampled, pool=self.pool, stream=self.stream)
            self.query_graphs[key] = graph
        graph.replay(qsl, path, sampled)


class TreeQueryGraph:
    """Accepted hidden/aux rows and logical positions for the next draft."""

    def __init__(self, tensors, qsl, path, positions, sampled, *, pool=None, stream=None):
        from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path

        self.qsl, self.path = qsl.clone(), path.clone()
        self.sampled = torch.zeros_like(sampled)
        self.replays = 0

        def run():
            compact_tree_query_along_path(tensors, self.qsl, self.path, positions, num_sampled=self.sampled)

        # Disabled request rows make warmup/capture non-mutating.
        with torch.inference_mode():
            run()
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, pool=pool, stream=stream):
                run()

    def replay(self, qsl, path, sampled):
        self.qsl.copy_(qsl)
        self.path.copy_(path)
        self.sampled.copy_(sampled)
        self.graph.replay()
        self.replays += 1


class TreePostGraph:
    """Freeze node state, relocate original KV, and commit the accepted chain."""

    def __init__(self, target):
        self.target = target
        runner, plan = target.runner, target.plan
        self.compactor = runner.tree_kv_compact
        self.num_reqs = plan["n_req"]
        self.sampled = torch.zeros(self.num_reqs, dtype=torch.int32, device=runner.device)
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
        logger.info("Captured DSV4 tree FULL accepted-state requests=%s", self.num_reqs)

    def _run(self):
        n, compactor = self.num_reqs, self.compactor
        path = compactor._path[:n]
        endpoint = torch.where(path >= 0, path, 0).amax(dim=1)
        winner = self.target.layout["qsl"][:n].long() + endpoint
        self.nodes.freeze(winner, self.sampled > 0)
        compactor._bind_groups(n)
        compactor._compact(n)
        self.nodes.commit()

    def replay(self, idx_mapping, path, num_sampled):
        n = self.num_reqs
        self.compactor._idx[:n].copy_(idx_mapping[:n])
        self.compactor._path[:n].copy_(path[:n])
        self.sampled.copy_(num_sampled[:n])
        self.graph.replay()
        self.replays += 1


class TreeFullGraphManager:
    """Tree-only lazy FULL gears; regular model graph dispatch is unchanged."""

    def __init__(self, runner):
        self.runner = runner
        self.graphs = {}
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
        if target is None:
            target = TreeTargetGraph(self.runner, desc, capacity, layout, pool=self.pool, stream=self.stream)
            self.graphs[key] = target
        return target.replay(layout)
