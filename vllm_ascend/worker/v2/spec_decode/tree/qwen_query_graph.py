# SPDX-License-Identifier: Apache-2.0
"""Accepted query transport graphs for the Qwen GDN tree target."""

from dataclasses import dataclass, fields

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.full_graph import TreeQueryGraph


def _tensor_shape(tensor):
    return tuple(tensor.shape), tensor.dtype, tensor.device


def _tensor_view(tensor):
    return tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device


@dataclass
class _QueryGraphEntry:
    graph: TreeQueryGraph
    # The recorded graph refers to these actual target outputs and positions.
    # Retain their storage even after the caller replaces its Python views.
    tensors: tuple[torch.Tensor, ...]
    positions: torch.Tensor


class QwenTreeQueryGraphs:
    """Reuse one private query graph for each stable target output layout."""

    def __init__(self, runner):
        self.runner = runner
        self.query_len = runner.speculator.tree_shape.target_query_len
        self.graphs = {}

    def eligible(self, input_batch):
        # These fields are scheduler-owned CPU metadata, never device values.
        if input_batch is None or input_batch.num_reqs != 1 or input_batch.has_prefill:
            return False
        if input_batch.is_prefilling_np[:1].any():
            return False
        if input_batch.num_tokens != self.query_len:
            return False
        if not (input_batch.num_scheduled_tokens[:1] == self.query_len).all():
            return False
        qsl = input_batch.query_start_loc_np
        if qsl.shape != (2,) or qsl[1] - qsl[0] != self.query_len:
            return False
        # Configuring FULL is insufficient: its target output addresses must
        # belong to a captured FULL gear, rather than fresh eager allocations.
        manager = self.runner.cudagraph_manager
        return (
            manager is not None
            and manager._graphs_captured
            and any(
                desc.cg_mode == CUDAGraphMode.FULL
                and desc.num_reqs == 1
                and desc.num_tokens == self.query_len
                and desc.uniform_token_count == self.query_len
                for desc in manager.graphs
            )
        )

    def compact_queries(self, tensors, qsl, acceptance, positions):
        key = (
            tuple(_tensor_view(tensor) for tensor in tensors),
            _tensor_view(positions),
            _tensor_shape(qsl),
            tuple(_tensor_shape(getattr(acceptance, field.name)) for field in fields(TreeAcceptance)),
        )
        entry = self.graphs.get(key)
        if entry is None:
            # Capture failures propagate. The private acceptance starts inactive,
            # so warmup and capture cannot compact the live target outputs twice.
            graph = TreeQueryGraph(tensors, qsl, acceptance, positions)
            entry = _QueryGraphEntry(graph, tuple(tensors), positions)
            self.graphs[key] = entry
        entry.graph.replay(qsl, acceptance)
