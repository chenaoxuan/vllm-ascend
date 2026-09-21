import torch
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

    Verify wrote ori KV at packed ``prefix+node``. DSV4 sampled tokens
    start at ``prefix+0`` (``dst_from_zero``). C4/indexer/state are
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
        self._held = None
        self._held_keep = None
        self._hold = False
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
        self._depth0 = torch.arange(self.spec_len, device=self.device, dtype=torch.int32)
        self._dst_off = self._depth

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
        self._hold = False
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

    def run(
        self,
        idx_mapping: torch.Tensor,
        path_node_ids: torch.Tensor,
        *,
        dst_from_zero: bool = False,
    ) -> None:
        num_reqs = path_node_ids.shape[0]
        self._idx[:num_reqs].copy_(idx_mapping[:num_reqs])
        self._path[:num_reqs].copy_(path_node_ids[:num_reqs])
        self._dst_off = self._depth0 if dst_from_zero else self._depth
        self._hold = dst_from_zero
        graph = self._graphs.get(num_reqs)
        if graph is not None and not dst_from_zero:
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
            if not _group_is_linear_token_cache(group):
                continue
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
        if not self._groups:
            self._held = None
            self._held_keep = None
            return
        idx = self._idx[:num_reqs]
        path = self._path[:num_reqs]
        num_computed = self.runner.req_states.num_computed_tokens.gpu
        nslot = num_reqs * self.spec_len
        safe_idx = idx.clamp(min=0)
        prefix = num_computed[safe_idx]
        if num_reqs > 0:
            try:
                from debug_trace import cmp_log

                path0 = path[0].detach().to("cpu").tolist()
                accepted = [int(n) for n in path0 if int(n) >= 0]
                cmp_log(
                    {
                        "ori_move": 1,
                        "dst_from_zero": bool(self._hold),
                        "prefix": int(prefix[0].item()),
                        "src": accepted[:8],
                        "dst_off": [int(x) for x in self._dst_off[: len(accepted)].tolist()],
                    }
                )
            except Exception:
                pass
        dst_pos = prefix.unsqueeze(1) + self._dst_off
        src_pos = prefix.unsqueeze(1) + path.clamp(min=0).to(dtype=prefix.dtype)
        valid = (path >= 0) & (idx >= 0).unsqueeze(1)
        src_pos = torch.where(valid, src_pos, dst_pos)
        max_pos = self.runner.max_model_len - 1
        dst_pos = dst_pos.clamp(max=max_pos)
        src_pos = src_pos.clamp(max=max_pos)
        req_f = safe_idx.unsqueeze(1).expand(num_reqs, self.spec_len)
        hold = self._hold
        held = []
        for caches, block_table, block_size, gathers in self._groups:
            max_block = block_table.shape[1] - 1
            src_bi = torch.div(src_pos, block_size, rounding_mode="floor").clamp(
                min=0, max=max_block
            )
            dst_bi = torch.div(dst_pos, block_size, rounding_mode="floor").clamp(
                min=0, max=max_block
            )
            src_block = block_table[req_f, src_bi]
            dst_block = block_table[req_f, dst_bi]
            src_flat = (src_block * block_size + src_pos % block_size).to(
                dtype=torch.long
            ).reshape(-1)
            dst_flat = (dst_block * block_size + dst_pos % block_size).to(
                dtype=torch.long
            ).reshape(-1)
            dst_keep = dst_flat.clone() if hold else None
            for cache, gather in zip(caches, gathers):
                tail = cache.shape[2:]
                flat = cache.reshape(cache.shape[0] * cache.shape[1], *tail)
                scratch = gather[:nslot]
                torch.index_select(flat, 0, src_flat, out=scratch)
                flat.index_copy_(0, dst_flat, scratch)
                if hold:
                    held.append((cache, scratch, dst_keep))
        self._held = held if hold else None
        self._held_keep = valid.reshape(-1) if hold else None


def _cmp_case_log(prefix, path_nodes, parents, depths, num_nodes, moves) -> None:
    """Classify CSA/HCA windows for request 0. Host-side, last rank only."""
    try:
        from debug_trace import cmp_log
    except Exception:
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "debug_trace", "/home/specdec/spec260922/debug_trace.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cmp_log = mod.cmp_log
    p0 = int(prefix[0].item()) if hasattr(prefix, "numel") else int(prefix)
    nodes = [int(x) for x in path_nodes[:16]]
    n = int(num_nodes)
    dep = [int(x) for x in depths[:n]]
    par = [int(x) for x in parents[:n]]
    by_ratio = {}
    for ratio in (4, 128):
        residual = p0 % ratio
        groups: dict[int, list[int]] = {}
        rope_groups: dict[int, list[int]] = {}
        for i in range(n):
            nid = i + 1
            slot_c = (p0 + nid) // ratio
            rope_c = (p0 + dep[i]) // ratio
            groups.setdefault(slot_c, []).append(nid)
            rope_groups.setdefault(rope_c, []).append(nid)
        slot_stomp = []
        chain_mix = 0
        for ids in groups.values():
            if len(ids) < 2:
                continue
            slot_stomp.append(ids[:8])
            # A real window is one parent chain. Mixed ids are siblings or
            # unrelated nodes written into the same compressed slot.
            idset = set(ids)
            linked = 0
            for nid in ids:
                parent = par[nid - 1] if nid - 1 < len(par) else -1
                if parent in idset or parent == 0:
                    linked += 1
            if linked < len(ids):
                chain_mix += 1
            if len(slot_stomp) >= 6:
                break
        # Siblings share a RoPE depth, so they fall in one compressed window.
        branch = []
        for ids in rope_groups.values():
            if len(ids) < 2:
                continue
            d0 = dep[ids[0] - 1]
            if any(dep[j - 1] == d0 for j in ids[1:]):
                branch.append(ids[:8])
            if len(branch) >= 4:
                break
        by_ratio["r" + str(ratio)] = {
            "residual": residual,
            "partial": residual != 0,
            "n_slot": len(groups),
            "slot_stomp": slot_stomp,
            "rope_branch": branch,
        }
    cmp_log(
        {
            "prefix": p0,
            "path": nodes,
            "moves": moves,
            "cmp": by_ratio,
        }
    )


def move_accepted_compress_slots(
    runner, idx_mapping, path_node_ids, num_sampled, tree=None
) -> dict:
    """Move accepted-path compressed windows onto the linear prefix.

    Verify writes each tree query at ``prefix + node_id``. After accept, draft
    ``i`` belongs at ``prefix + i``. When that position closes a ratio-4 or
    ratio-128 window, copy the compressed slot from the tree position.
    """
    groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None)
    if (
        not groups
        or path_node_ids is None
        or num_sampled is None
        or int(num_sampled.shape[0]) == 0
    ):
        return {"groups": 0, "copies": 0}
    from vllm_ascend.worker.v2.spec_decode.tree.chain_pack import commit_winner

    committed = commit_winner(path_node_ids)
    if committed is not None:
        return committed
    num_reqs = int(num_sampled.shape[0])
    idx = idx_mapping[:num_reqs].clamp(min=0)
    prefix = runner.req_states.num_computed_tokens.gpu[idx]
    spec = int(path_node_ids.shape[1])
    node = path_node_ids[:num_reqs].to(dtype=torch.long)
    valid = node >= 0
    offs = torch.arange(spec, device=prefix.device)
    ctx = runner.compilation_config.static_forward_context
    n_groups = 0
    n_copies = 0
    move_rows = []
    for gid, group in enumerate(groups):
        if _group_is_linear_token_cache(group):
            continue
        spec = group.kv_cache_spec
        inner = getattr(spec, "kv_cache_specs", None)
        items = list(inner.values()) if inner else [spec]
        ratio = 0
        for item in items:
            ratio = int(
                getattr(item, "compress_ratio", 0)
                or getattr(item, "tokens_per_state", 0)
                or 0
            )
            if ratio > 1:
                break
        if ratio not in (4, 128):
            continue
        block_size = int(runner.block_tables.kernel_block_sizes[gid])
        if block_size % ratio != 0:
            continue
        page = block_size // ratio
        block_table = runner.block_tables.block_tables[gid].gpu
        tensors: list[torch.Tensor] = []
        seen: set[int] = set()
        for name in group.layer_names:
            layer = ctx.get(name)
            if layer is None:
                continue
            for tensor in iter_unique_kv_cache_tensors(getattr(layer, "kv_cache", None)):
                ptr = tensor.untyped_storage().data_ptr()
                if ptr in seen:
                    continue
                seen.add(ptr)
                tensors.append(tensor)
        if not tensors:
            continue
        pos = prefix.unsqueeze(1) + offs
        src_pos = prefix.unsqueeze(1) + node.clamp(min=0)
        boundary = valid & torch.remainder(pos + 1, ratio).eq(0)
        dst_cpos = torch.div(pos, ratio, rounding_mode="floor")
        src_cpos = torch.div(src_pos, ratio, rounding_mode="floor")
        take = boundary & src_cpos.ne(dst_cpos)
        if ratio in (4, 128) and len(move_rows) < 4:
            taken = take[0].tolist()
            src_l = src_cpos[0].tolist()
            dst_l = dst_cpos[0].tolist()
            pairs = [
                [int(s), int(d)]
                for s, d, ok in zip(src_l, dst_l, taken)
                if ok
            ]
            dst_count: dict[int, int] = {}
            for _, d in pairs:
                dst_count[d] = dst_count.get(d, 0) + 1
            move_rows.append(
                {
                    "ratio": ratio,
                    "closed": len(pairs),
                    "dst_stomp": [d for d, c in dst_count.items() if c > 1][:4],
                    "pairs": pairs[:8],
                }
            )
        if not bool(take.any().item()):
            continue
        req = idx.unsqueeze(1).expand_as(pos)
        max_block = block_table.shape[1] - 1

        def _flat(cpos: torch.Tensor) -> torch.Tensor:
            bi = torch.div(cpos, page, rounding_mode="floor").clamp(0, max_block)
            bo = torch.remainder(cpos, page)
            blocks = block_table[req, bi]
            return (blocks * page + bo).to(dtype=torch.long)

        src_flat = _flat(src_cpos.clamp(min=0))[take]
        dst_flat = _flat(dst_cpos)[take]
        ranked = [t for t in tensors if t.ndim >= 3]
        if not ranked:
            continue
        n_groups += 1
        limit = min(int(t.shape[0] * t.shape[1]) for t in ranked)
        in_range = (src_flat >= 0) & (dst_flat >= 0) & (src_flat < limit) & (dst_flat < limit)
        src_flat = src_flat[in_range]
        dst_flat = dst_flat[in_range]
        if src_flat.numel() == 0:
            continue
        n_copies += int(src_flat.numel())
        for tensor in ranked:
            tail = tensor.shape[2:]
            flat = tensor.reshape(tensor.shape[0] * tensor.shape[1], *tail)
            flat.index_copy_(0, dst_flat, flat.index_select(0, src_flat))
    try:
        if tree is not None:
            n_nodes = int(tree.num_nodes[0].item())
            _cmp_case_log(
                prefix,
                node[0].tolist(),
                tree.parents[0, :n_nodes].tolist(),
                tree.depths[0, :n_nodes].tolist(),
                n_nodes,
                move_rows,
            )
    except Exception:
        pass
    return {"groups": n_groups, "copies": n_copies}

