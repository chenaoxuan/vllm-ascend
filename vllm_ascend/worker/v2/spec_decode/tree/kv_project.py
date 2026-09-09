import logging

import torch

from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import (
    iter_unique_kv_cache_tensors,
)

logger = logging.getLogger(__name__)


class TreeKvCompact:
    """Move accepted-path target KV from tree slots onto the linear prefix.

    Verify already wrote KV at ``prefix+node``. Packed RoPE at depth ``d``
    matches the linear dest ``prefix+d``, so a slot move is enough.

    Single-stream: ``run`` after target+reject and before
    ``num_computed`` increments. One ACLGraph per ``num_reqs`` so the
    per-layer gather/scatter is one replay instead of many launches.
    ``idx_mapping`` / ``path_node_ids`` are copied into persistent
    buffers; block tables and ``num_computed`` are live tensors captured
    by address. Graph gather uses preallocated scratch. Missing gears or
    capture failure fall back to the same eager ops.
    """

    def __init__(self, runner):
        self.runner = runner
        self.device = runner.device
        self.spec_len = runner.num_speculative_steps
        self.max_num_reqs = runner.max_num_reqs
        self._graphs: dict[int, object] = {}
        self._groups = None
        self._idx = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        self._path = torch.full(
            (self.max_num_reqs, self.spec_len),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._depth = (
            torch.arange(self.spec_len, device=self.device, dtype=torch.int32) + 1
        )

    def capture(self, sizes: list[int]) -> None:
        if not sizes or self.runner.model_config.enforce_eager:
            return
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            return
        from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
        from vllm.platforms import current_platform

        self._bind_groups()
        self._idx.fill_(0)
        self._path.fill_(-1)
        pool = current_platform.get_global_graph_pool()
        for num_reqs in sizes:
            try:
                validate_cudagraph_capturing_enabled()
                with torch.inference_mode():
                    self._compact(num_reqs)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, pool=pool):
                        self._compact(num_reqs)
                self._graphs[num_reqs] = graph
                logger.info("Captured tree KV compact ACLGraph num_reqs=%s", num_reqs)
            except Exception as exc:
                logger.warning(
                    "Tree KV compact ACLGraph capture failed for num_reqs=%s; "
                    "eager fallback for this size. %s",
                    num_reqs,
                    exc,
                )

    def run(self, idx_mapping: torch.Tensor, path_node_ids: torch.Tensor) -> None:
        num_reqs = path_node_ids.shape[0]
        self._idx[:num_reqs].copy_(idx_mapping[:num_reqs])
        self._path[:num_reqs].copy_(path_node_ids[:num_reqs])
        graph = self._graphs.get(num_reqs)
        if graph is not None:
            graph.replay()
            return
        self._bind_groups()
        self._compact(num_reqs)

    def _bind_groups(self) -> None:
        if self._groups is not None:
            return
        ctx = self.runner.compilation_config.static_forward_context
        groups = []
        seen: set[int] = set()
        for group_id, group in enumerate(self.runner.kv_cache_config.kv_cache_groups):
            caches: list[torch.Tensor] = []
            gathers: list[torch.Tensor] = []
            for layer_name in group.layer_names:
                layer = ctx.get(layer_name)
                if layer is None:
                    continue
                for tensor in iter_unique_kv_cache_tensors(
                    getattr(layer, "kv_cache", None)
                ):
                    ptr = tensor.untyped_storage().data_ptr()
                    if ptr in seen:
                        continue
                    seen.add(ptr)
                    caches.append(tensor)
                    tail = tensor.shape[2:]
                    gathers.append(
                        torch.empty(
                            (self.max_num_reqs * self.spec_len, *tail),
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                    )
            if not caches:
                continue
            groups.append(
                (
                    caches,
                    self.runner.block_tables.block_tables[group_id].gpu,
                    self.runner.block_tables.kernel_block_sizes[group_id],
                    gathers,
                )
            )
        self._groups = groups

    def _compact(self, num_reqs: int) -> None:
        idx = self._idx[:num_reqs]
        path = self._path[:num_reqs]
        num_computed = self.runner.req_states.num_computed_tokens.gpu
        nslot = num_reqs * self.spec_len
        safe_idx = idx.clamp(min=0)
        prefix = num_computed[safe_idx]
        dst_pos = prefix.unsqueeze(1) + self._depth
        src_pos = prefix.unsqueeze(1) + path.clamp(min=0).to(dtype=prefix.dtype)
        valid = (path >= 0) & (idx >= 0).unsqueeze(1)
        src_pos = torch.where(valid, src_pos, dst_pos)
        req_f = safe_idx.unsqueeze(1).expand(num_reqs, self.spec_len)
        for caches, block_table, block_size, gathers in self._groups:
            src_block = block_table[
                req_f, torch.div(src_pos, block_size, rounding_mode="floor")
            ]
            dst_block = block_table[
                req_f, torch.div(dst_pos, block_size, rounding_mode="floor")
            ]
            src_flat = (src_block * block_size + src_pos % block_size).to(
                dtype=torch.long
            ).reshape(-1)
            dst_flat = (dst_block * block_size + dst_pos % block_size).to(
                dtype=torch.long
            ).reshape(-1)
            for cache, gather in zip(caches, gathers):
                tail = cache.shape[2:]
                flat = cache.reshape(cache.shape[0] * cache.shape[1], *tail)
                scratch = gather[:nslot]
                torch.index_select(flat, 0, src_flat, out=scratch)
                flat.index_copy_(0, dst_flat, scratch)
