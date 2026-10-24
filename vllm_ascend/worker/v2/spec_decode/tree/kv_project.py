import torch
import torch_npu
from vllm.logger import init_logger

from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import (
    iter_unique_kv_cache_tensors,
)

logger = init_logger("vllm." + __name__)


try:
    from vllm.v1.kv_cache_interface import CircularBufferSpec as _CircularBufferSpec
except ImportError:
    _CircularBufferSpec = None


def _spec_is_circular_buffer(spec) -> bool:
    if _CircularBufferSpec is None or spec is None:
        return False
    if isinstance(spec, _CircularBufferSpec):
        return True
    first = getattr(spec, "first_spec", None)
    return first is not None and isinstance(first, _CircularBufferSpec)


def _group_is_linear_token_cache(group) -> bool:
    """True when compact can treat the group as 1 token per slot."""
    names = getattr(group, "layer_names", ()) or ()
    if any("state_cache" in name for name in names):
        return False
    spec = group.kv_cache_spec
    if _spec_is_circular_buffer(spec) or getattr(spec, "cache_role", None):
        return False
    inner = getattr(spec, "kv_cache_specs", None)
    specs = list(inner.values()) if inner else [spec]
    for item in specs:
        if _spec_is_circular_buffer(item) or getattr(item, "cache_role", None):
            return False
        ratio = getattr(item, "tokens_per_state", None)
        if ratio is None:
            ratio = getattr(item, "compress_ratio", 1)
        if ratio is not None and int(ratio) > 1:
            return False
    return True


def needs_causal_kv_repair(runner) -> bool:
    """True when a KV group cannot be compacted by paged slot after tree verify."""
    groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None)
    if not groups:
        return False
    return any(not _group_is_linear_token_cache(group) for group in groups)


class TreeKvCompact:
    """Move accepted-path target KV from tree slots onto the linear prefix.

    Verify wrote ori KV at packed ``prefix+node``. Accepted drafts
    start at ``prefix+1``; the root remains at P. C4/indexer/state are
    compacted separately by the winning chain.

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
        self._binding_scratch = {}
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

    def run(self, idx_mapping, path_node_ids, *, dst_from_zero=False) -> None:
        if dst_from_zero:
            raise ValueError("Tree KV compaction must preserve the root at prefix P")
        num_reqs = path_node_ids.shape[0]
        if num_reqs > self.max_num_reqs or path_node_ids.shape[1] != self.spec_len:
            raise RuntimeError("Accepted path exceeds KV compaction capacity")
        self._idx[:num_reqs].copy_(idx_mapping[:num_reqs])
        self._path[:num_reqs].copy_(path_node_ids[:num_reqs])
        if num_reqs == 0:
            return
        self._bind_groups(num_reqs)
        graph = self._graphs.get(num_reqs)
        if graph is not None:
            graph.replay()
        else:
            self._compact(num_reqs)

    def _bind_groups(self, num_reqs=None) -> None:
        # dspark-ori-binding-fix:v1
        if getattr(self.runner.speculator, "packed_dsv4_verify", lambda: False)():
            from vllm_ascend.worker.v2.spec_decode.tree.kv_binding import bind_original_kv
            self._groups = bind_original_kv(self, num_reqs)
            return
        if self._groups is not None:
            return
        ctx = self.runner.compilation_config.static_forward_context
        speculator = self.runner.speculator
        draft_names = set(getattr(speculator, "draft_attn_layer_names", ()) or ())
        groups, seen = [], set()
        for group_id, group in enumerate(self.runner.kv_cache_config.kv_cache_groups):
            if not _group_is_linear_token_cache(group):
                continue
            caches, gathers = [], []
            block_size = self.runner.block_tables.kernel_block_sizes[group_id]
            for layer_name in group.layer_names:
                if layer_name in draft_names:
                    continue
                layer = ctx.get(layer_name)
                if layer is None:
                    continue
                for tensor in iter_unique_kv_cache_tensors(getattr(layer, "kv_cache", None)):
                    key = (group_id, tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
                    if key in seen:
                        continue
                    seen.add(key)
                    if int(tensor.shape[1]) != int(block_size):
                        raise RuntimeError("Target KV page size differs from block-table metadata")
                    caches.append(tensor)
                    gathers.append(torch.empty(
                        (self.max_num_reqs * self.spec_len, *tensor.shape[2:]),
                        dtype=tensor.dtype, device=tensor.device,
                    ))
            if caches:
                groups.append((
                    caches,
                    self.runner.block_tables.block_tables[group_id].gpu,
                    block_size,
                    gathers,
                ))
        self._groups = groups

    def _compact(self, num_reqs: int) -> None:
        if not self._groups:
            return
        idx = self._idx[:num_reqs]
        path = self._path[:num_reqs]
        num_computed = self.runner.req_states.num_computed_tokens.gpu
        nslot = num_reqs * self.spec_len
        live_req = (idx >= 0) & (idx < num_computed.shape[0])
        safe_idx = idx.clamp(0, num_computed.shape[0] - 1).long()
        prefix = num_computed[safe_idx]
        dst_pos = prefix.unsqueeze(1) + self._depth
        src_pos = prefix.unsqueeze(1) + path.clamp(min=0).to(dtype=prefix.dtype)
        present = path != -1
        valid = present & live_req.unsqueeze(1)
        max_pos = self.runner.max_model_len - 1
        in_range = (src_pos >= 0) & (src_pos <= max_pos) & (dst_pos >= 0) & (dst_pos <= max_pos)
        valid &= in_range
        dst_pos = dst_pos.clamp(0, max_pos)
        src_pos = src_pos.clamp(0, max_pos)
        req_f = safe_idx.unsqueeze(1).expand(num_reqs, self.spec_len)
        transfers = {}
        for caches, block_table, block_size, gathers in self._groups:
            max_block = block_table.shape[1] - 1
            src_bi = torch.div(src_pos, block_size, rounding_mode="floor")
            dst_bi = torch.div(dst_pos, block_size, rounding_mode="floor")
            columns_valid = (src_bi <= max_block) & (dst_bi <= max_block)
            src_block = block_table[req_f, src_bi.clamp(max=max_block)]
            dst_block = block_table[req_f, dst_bi.clamp(max=max_block)]
            src_flat = (src_block * block_size + src_pos % block_size).to(
                dtype=torch.long
            ).reshape(-1)
            dst_flat = (dst_block * block_size + dst_pos % block_size).to(
                dtype=torch.long
            ).reshape(-1)
            for cache, gather in zip(caches, gathers):
                pages_valid = (
                    (src_block > 0) & (dst_block > 0)
                    & (src_block < cache.shape[0]) & (dst_block < cache.shape[0])
                )
                keep = (valid & columns_valid & pages_valid).reshape(-1)
                # Gather every cache/view before any target write, including
                # different live mappings of the same physical storage.
                scratch = gather[:nslot].view(torch.int8)
                safe_src = src_flat.clamp(0, cache.shape[0] * block_size - 1)
                scratch.copy_(cache.view(torch.int8)[safe_src // block_size, safe_src % block_size])
                key = (cache.data_ptr(), tuple(cache.shape), tuple(cache.stride()), cache.dtype)
                transfer = transfers.setdefault(key, (cache, [], [], []))
                transfer[1].append(dst_flat)
                transfer[2].append(scratch)
                transfer[3].append(keep)

        writes = []
        for cache, destinations, values, masks in transfers.values():
            dest = torch.cat(destinations)
            keep = torch.cat(masks)
            # Invalid rows sort after real slots. Keep one writer per live slot.
            dest = torch.where(keep, dest, cache.shape[0] * cache.shape[1])
            dest, order = dest.sort()
            keep = keep[order]
            data = torch.cat(values).contiguous().view(torch.int8).index_select(0, order)
            duplicate = torch.zeros_like(keep)
            duplicate[1:] = keep[1:] & keep[:-1] & (dest[1:] == dest[:-1])
            writes.append((cache, dest, data, keep & ~duplicate))

        for cache, dest, data, keep in writes:
            indices = torch.stack((dest // cache.shape[1], dest % cache.shape[1]), dim=1)
            indices = torch.where(keep.unsqueeze(1), indices, -1)
            # Same fixed [-1, -1] padding contract used by DSA cache scatter.
            # Byte transport retains FP8 payloads and signed zero exactly.
            torch_npu.npu_scatter_nd_update_(cache.view(torch.int8), indices, data)
