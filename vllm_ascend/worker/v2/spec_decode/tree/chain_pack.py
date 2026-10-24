"""Eager tree verification with private cache tensors per layer.

Persistent caches are read during verification. Only accepted branch snapshots
are copied back. Block 0 remains the null block in every temporary cache.
"""
from __future__ import annotations

from contextlib import contextmanager
from math import prod

import torch
import torch_npu
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.spec_decode.tree.node_pool import build_ancestors
from vllm_ascend.ops.triton.spec_decode.tree.state_commit import (
    copy_tree_state_rows_kernel,
    select_tree_state_kernel,
)

# CANN CompressorMetadata slot_mapping_format for (block, offset) outputs.
_SLOT_MAPPING_BLOCK_OFFSET = 2
_LAYOUT: dict | None = None
_PLAN: dict | None = None
_KV_STACK: list[list[torch.Tensor]] = []


class TreeHostMetadata:
    """CPU-only block IDs supplied by the scheduler, owned by the tree runner.

    This observes request updates without changing framework block allocation,
    staged device writes or synchronization. New requests replace old slot data.
    """

    def __init__(self):
        self.blocks: dict[str, list[list[int]]] = {}

    def add_requests(self, requests):
        for request in requests:
            self.blocks[request.req_id] = [list(ids) for ids in request.block_ids]

    def update_requests(self, requests, live_req_ids):
        self.blocks = {key: value for key, value in self.blocks.items() if key in live_req_ids}
        for req_id, groups in zip(requests.req_ids, requests.new_block_ids):
            if groups is not None:
                for saved, new in zip(self.blocks[req_id], groups):
                    saved.extend(new)

    def snapshot(self, req_ids, block_tables, cache_config):
        result = {}
        for group_id, group in enumerate(cache_config.kv_cache_groups):
            width = block_tables.block_tables[group_id].gpu.shape[1]
            expansion = block_tables.blocks_per_kv_block[group_id]
            rows, lengths = [], []
            for req_id in req_ids:
                ids = [block * expansion + offset
                       for block in self.blocks[req_id][group_id] for offset in range(expansion)]
                if len(ids) > width:
                    raise RuntimeError("Tree CPU block-table mirror exceeds device row capacity")
                lengths.append(len(ids))
                rows.append(ids + [0] * (width - len(ids)))
            for layer_name in group.layer_names:
                result[layer_name] = rows, lengths
        return result


class _LayerBuffers:
    """Scratch storage owned by one verification plan, reused between layers.

    Every acquire in a layer has distinct storage. Reset only after attention
    and snapshot gathers have been enqueued on the same stream. Plan inputs,
    cached TopK and winner snapshots must never use this pool.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple, list[torch.Tensor]] = {}
        self._used: dict[tuple, int] = {}

    def acquire(
        self,
        kind: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        key = (kind, torch.device(device), dtype, tuple(shape[1:]))
        buffers = self._buffers.setdefault(key, [])
        index = self._used.get(key, 0)
        self._used[key] = index + 1
        if index == len(buffers):
            buffers.append(torch.empty(shape, dtype=dtype, device=device))
        elif buffers[index].shape[0] < shape[0]:
            buffers[index] = torch.empty(shape, dtype=dtype, device=device)
        return buffers[index][:shape[0]]

    def reset(self) -> None:
        self._used.clear()


def _upload_metadata(
    device: torch.device,
    *fields: tuple[list, torch.dtype, tuple[int, ...]],
    host_buffers: list[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Upload immutable plan metadata once, independently of layer scratch.

    Typed views retain each field's dtype and alignment. The NumPy views
    reference CPU storage only; no device values are read on host.
    """
    offsets, nbytes = [], 0
    for _, dtype, shape in fields:
        size = dtype.itemsize
        begin = (nbytes + size - 1) // size * size
        nbytes = begin + prod(shape) * size
        offsets.append((begin, nbytes))
    host = torch.empty(nbytes, dtype=torch.uint8, device="cpu",
                       pin_memory=torch.device(device).type == "npu")
    target = torch.empty(nbytes, dtype=torch.uint8, device=device)
    result = []
    for (values, dtype, shape), (begin, end) in zip(fields, offsets):
        if end > begin:
            host[begin:end].view(dtype).view(shape).numpy()[...] = values
        result.append(target[begin:end].view(dtype).view(shape))
    # H2D: the runner retains this immutable pinned source with the round until
    # the next metadata boundary has completed. Layer scratch never owns it.
    host_buffers.append(host)
    target.copy_(host, non_blocking=True)
    return tuple(result)


def _key(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def set_tree_chain_layout(parents, num_nodes, depths, query_start_loc, *,
                          query_start_loc_cpu, start_pos_cpu, max_depth, host_tables) -> dict:
    global _LAYOUT, _PLAN
    if _PLAN is not None and not _PLAN.get("committed", False):
        raise RuntimeError("Previous tree verification was not committed")
    # The runner owns this round. Snapshot device inputs because proposal and
    # metadata buffers can reuse their addresses before the next verification.
    _LAYOUT = dict(parents=parents.clone(), num_nodes=num_nodes.clone(), depths=depths.clone(),
                   qsl=query_start_loc.clone(), plan=None,
                   qsl_cpu=[int(value) for value in query_start_loc_cpu],
                   start_cpu=[int(value) for value in start_pos_cpu],
                   max_depth=int(max_depth), host_tables=host_tables)
    _PLAN = None
    return _LAYOUT


def clear_tree_chain_layout() -> None:
    global _LAYOUT, _PLAN
    if _PLAN is not None and not _PLAN.get("committed", False):
        raise RuntimeError("Cannot discard uncommitted tree state")
    _LAYOUT = None
    _PLAN = None


def chain_layout_active() -> bool:
    return _LAYOUT is not None


@contextmanager
def chain_kv_scope(caches):
    _KV_STACK.append([cache for cache in caches if cache is not None])
    try:
        yield
    finally:
        _KV_STACK.pop()


def _build_plan(num_tokens: int, device) -> dict | None:
    """Build a fixed-capacity device plan; host inputs were already CPU metadata."""
    global _PLAN
    if _PLAN is not None:
        return _PLAN
    if _LAYOUT is None:
        return None
    layout = _LAYOUT
    qsl, prefixes = layout["qsl_cpu"], layout["start_cpu"]
    n_req, depth_limit = len(prefixes), layout["max_depth"]
    if (n_req == 0 or depth_limit <= 0 or len(qsl) != n_req + 1
            or qsl[0] != 0 or qsl[-1] > num_tokens
            or any(end <= begin for begin, end in zip(qsl, qsl[1:]))
            or min(prefixes) < 0):
        raise RuntimeError("Invalid host tree query boundaries/capacity")
    req_index = [req for req in range(n_req) for _ in range(qsl[req], qsl[req + 1])]
    prefix_of = [prefixes[req] for req in req_index]
    n = len(req_index)
    host_buffers = []
    req_ids, host_qsl, start = _upload_metadata(
        device,
        (req_index, torch.long, (n,)),
        (qsl, torch.int32, (n_req + 1,)),
        (prefixes, torch.int32, (n_req,)),
        host_buffers=host_buffers,
    )
    tokens = torch.arange(n, dtype=torch.long, device=device)
    ancestor_ids, valid, depths, hidden_ids, parent_ids = build_ancestors(
        layout["parents"], layout["depths"], layout["num_nodes"], req_ids,
        host_qsl, depth_limit,
    )
    lens = (depths + 1).to(torch.int32)
    cu = torch.cat((lens.new_zeros(1), lens.cumsum(0, dtype=torch.int32)))
    steps = torch.arange(depth_limit + 1, device=device).unsqueeze(0)
    capacity = n * (depth_limit + 1)
    # Live chain positions are unique. Give every inactive entry its own tail
    # slot as well: ordinary scatter needs no reduction or duplicate writes.
    padding = capacity + torch.arange(capacity, device=device).view(n, depth_limit + 1)
    destinations = torch.where(valid, cu[:-1].unsqueeze(1) + steps, padding).long()
    rows = torch.zeros(2 * capacity, dtype=torch.long, device=device)
    rows.scatter_(0, destinations.reshape(-1), torch.where(valid, ancestor_ids, 0).reshape(-1))
    start = start.index_select(0, req_ids)
    _PLAN = {
        "n": n, "n_req": n_req, "req_cu": qsl, "max_depth": depth_limit,
        "req_ids": req_ids, "req_index": req_index, "prefixes": prefix_of,
        "depths": depths, "lens": lens, "ancestors": ancestor_ids,
        "parent_ids": parent_ids, "visibility": valid,
        "hidden_ids": hidden_ids,
        "max_q": depth_limit + 1, "max_kv": max(prefixes) + depth_limit + 1,
        "token_count": capacity, "rows": rows[:capacity], "cu": cu,
        "query_cu": torch.arange(n + 1, dtype=torch.int32, device=device),
        "end_rows": (cu[1:] - 1).long(), "start": start, "seqused": start + lens,
        "token_index": tokens, "host_uploads": host_buffers,
        "cache_views": {}, "kv_tables": {}, "ori_views": {}, "index_topk": {},
        "length_cus": {}, "indexer_lengths": None, "query_metadata": {},
        "compressor_metadata": {}, "compressor_batches": {},
        "buffers": _LayerBuffers(), "workspace_layouts": {},
        "pending": [], "records": {}, "committed": False,
    }
    layout["plan"] = _PLAN
    return _PLAN


def _new_cache_views(buffers: _LayerBuffers, caches, num_pages):
    """Preserve aliases such as packed FP8 key/scale/full-cache views."""
    groups, result = [], [None] * len(caches)
    order = sorted(range(len(caches)), key=lambda i: caches[i].data_ptr())
    for i in order:
        cache = caches[i]
        if cache.ndim < 2 or any(stride < 0 for stride in cache.stride()):
            raise RuntimeError("Unsupported cache tensor strides")
        size = cache.element_size()
        step = int(cache.stride(0)) * size
        span = size + sum((int(n) - 1) * int(s) * size
                          for n, s in zip(cache.shape[1:], cache.stride()[1:]))
        if step <= 0 or span > step:
            raise RuntimeError("Overlapping/invalid physical cache pages")
        storage = cache.untyped_storage().data_ptr()
        group = next((g for g in groups if g["storage"] == storage
                      and g["step"] == step
                      and 0 <= cache.data_ptr() - g["base"]
                      and cache.data_ptr() - g["base"] + span <= step), None)
        if group is None:
            group = {"storage": storage, "step": step, "base": cache.data_ptr(),
                     "bytes": buffers.acquire("cache", (num_pages * step,), torch.uint8,
                                               cache.device)}
            # Clear the full active workspace, including null page and padding.
            group["bytes"].zero_()
            groups.append(group)
        offset = cache.data_ptr() - group["base"]
        if offset % size:
            raise RuntimeError("Misaligned packed-cache view")
        result[i] = torch.empty(0, dtype=cache.dtype, device=cache.device).set_(
            group["bytes"].untyped_storage(), offset // size,
            (num_pages, *cache.shape[1:]), tuple(cache.stride()),
        )
    return result


def _workspace_layout(plan, block_table, metadata, kind, block_size, *, ratio=1,
                      window_size=0):
    """Plan bounded private pages from CPU block IDs."""
    group = metadata.cache_group_key
    key = (group, kind, block_size, ratio, window_size)
    if key in plan["workspace_layouts"]:
        return plan["workspace_layouts"][key]
    if group not in _LAYOUT["host_tables"]:
        raise RuntimeError("Tree metadata group has no scheduler block-table snapshot")
    real, allocated_lengths = _LAYOUT["host_tables"][group]
    width = int(block_table.shape[1])
    if len(real) != block_table.shape[0] or any(len(row) != width for row in real):
        raise RuntimeError("Tree CPU/device block-table shapes differ")
    rows = [[0] * width for _ in range(plan["n"])]
    sources, allocated = [0], {}
    private = [[] for _ in rows]
    capacities = []
    for endpoint, req in enumerate(plan["req_index"]):
        original = real[req]
        prefix = plan["prefixes"][endpoint]
        length = min(plan["max_q"], endpoint - plan["req_cu"][req] + 1)
        capacities.append(length)
        if kind == "kv":
            begin, end = prefix // ratio, (prefix + length) // ratio
            write_cols = {position // block_size for position in range(begin, end)}
            columns = range(min(width, (end + block_size - 1) // block_size))
        elif kind == "state":
            write_cols = set(range(prefix // block_size, (prefix + length - 1) // block_size + 1))
            columns = range(width)
        else:
            write_cols = {(prefix + depth) // block_size for depth in range(1, length)}
            lo, hi = max(0, prefix - window_size + 1), prefix + length - 1
            columns = range(lo // block_size, min(width, hi // block_size + 1))
        if max(write_cols, default=-1) >= width:
            raise RuntimeError("Tree scheduled capacity exceeds block-table row width")
        # Capacity may extend beyond actual scheduled depth or allocated pages.
        # Only a live device write requires a non-null page.
        writable = {original[col] for col in write_cols if 0 <= col < width and original[col] > 0}
        for col in sorted(set(columns) | write_cols):
            if not 0 <= col < width:
                continue
            src = original[col]
            if src <= 0:
                continue
            owner = endpoint if src in writable else -1
            tag = owner, src
            if tag not in allocated:
                allocated[tag] = len(sources)
                sources.append(src)
                if owner >= 0:
                    private[endpoint].append((src, allocated[tag]))
            rows[endpoint][col] = allocated[tag]

    entries = [] if kind == "ori" else [
        (endpoint, src, dst) for endpoint, pages in enumerate(private) for src, dst in pages
    ]
    state_rows, state_spans, minimum_lengths = [], [], []
    for endpoint, src, dst in entries:
        prefix, length = plan["prefixes"][endpoint], capacities[endpoint]
        original = real[plan["req_index"][endpoint]]
        if kind == "state":
            offsets = sorted({position % block_size for position in range(prefix, prefix + length)
                              if position // block_size < width and original[position // block_size] == src})
            state_spans.append((len(state_rows), len(offsets)))
            state_rows.extend((src, offset, dst) for offset in offsets)
        else:
            minimum_lengths.append(min(
                (position + 1) * ratio - prefix
                for position in range(prefix // ratio, (prefix + length) // ratio)
                if position // block_size < width and original[position // block_size] == src
            ))
    ori_rows = []
    if kind == "ori":
        for endpoint, length in enumerate(capacities):
            prefix = plan["prefixes"][endpoint]
            for depth in range(1, length):
                col, offset = divmod(prefix + depth, block_size)
                ori_rows.append((endpoint, depth, rows[endpoint][col] if col < width else 0, offset))
    original_rows = real if kind == "ori" else []
    original_lengths = allocated_lengths if kind == "ori" else []
    (table, source_ids, snapshot_indices, state_indices, owners, minimum,
     original_table, valid_lengths, ori_ids) = _upload_metadata(
        block_table.device,
        (rows, torch.int32, (plan["n"], width)),
        (sources[1:], torch.long, (len(sources) - 1,)),
        ([(src, dst) for _, src, dst in entries], torch.long, (len(entries), 2)),
        (state_rows, torch.long, (len(state_rows), 3)),
        ([endpoint for endpoint, _, _ in entries], torch.long, (len(entries),)),
        (minimum_lengths, torch.int32, (len(minimum_lengths),)),
        (original_rows, torch.int32, (len(original_rows), width)),
        (original_lengths, torch.long, (len(original_lengths),)),
        (ori_rows, torch.long, (len(ori_rows), 4)),
        host_buffers=plan["host_uploads"],
    )
    ori_indices = None
    if kind == "ori" and ori_rows:
        endpoints, depths, destinations, offsets = ori_ids.unbind(1)
        requests = plan["req_ids"].index_select(0, endpoints)
        roots = _LAYOUT["qsl"].long().index_select(0, requests)
        nodes = plan["ancestors"][endpoints, depths] - roots
        positions = plan["start"].index_select(0, endpoints) + nodes
        columns = positions // block_size
        safe_columns = columns.clamp(0, width - 1)
        pages = original_table[requests, safe_columns].long()
        active = depths <= plan["depths"].index_select(0, endpoints)
        valid = ((columns < width) & (pages > 0) & (destinations > 0)
                 & (safe_columns < valid_lengths.index_select(0, requests)))
        active = active & valid
        # Sort by destination, then input order, so the last live candidate
        # wins each aliased slot. This avoids NPU scatter-reduction fallback
        # and only allocates per candidate, not per physical workspace row.
        capacity = len(sources) * block_size
        row = torch.where(active, destinations * block_size + offsets, capacity)
        order = torch.arange(endpoints.numel(), device=block_table.device)
        order = (row * endpoints.numel() + order).sort().indices
        row = row.index_select(0, order)
        last = torch.cat((row[:-1] != row[1:], active.new_ones(1)))
        keep = active.index_select(0, order) & last
        indices = torch.stack((destinations, offsets), 1).index_select(0, order)
        indices = torch.where(keep.unsqueeze(1), indices, -1)
        ori_indices = (indices, pages.clamp(min=0).index_select(0, order),
                       (positions % block_size).index_select(0, order))
    layout = {
        "metadata": metadata,
        "table": table, "table_rows": rows, "source_ids": source_ids,
        "num_pages": len(sources),
        "entries": entries, "owners": owners,
        "written": None if kind != "kv" else plan["lens"].index_select(0, owners) >= minimum,
        "snapshot_indices": snapshot_indices.unbind(1),
        "ori_indices": ori_indices,
        "state_indices": state_indices.unbind(1), "state_spans": state_spans,
        "capacities": capacities,
    }
    plan["workspace_layouts"][key] = layout
    return layout


def _workspace(plan, caches, layout, kind):
    """Refresh private cache values on device using the frozen page mapping.

    No temporary block ID refers to persistent storage. Writable aliases
    within one chain are preserved and never shared with another chain.
    """
    views = _new_cache_views(plan["buffers"], caches, layout["num_pages"])
    table = layout["table"]
    for cache, view in zip(caches, views):
        if layout["num_pages"] > 1:
            view[1:].copy_(cache.index_select(0, layout["source_ids"]))
        for binding in (cache, view):
            plan["cache_views"][_key(binding)] = view
            plan["kv_tables"][_key(binding)] = table
        if kind != "ori" and layout["entries"]:
            plan["pending"].append((cache, view, kind, layout))
    return views, table


def tree_cache_view(cache):
    """Resolve the cache matching temporary compressor slot IDs/block tables."""
    if cache is None or _LAYOUT is None:
        return cache
    if _PLAN is None or _key(cache) not in _PLAN["cache_views"]:
        raise RuntimeError("Tree cache writer/reader has no private workspace")
    return _PLAN["cache_views"][_key(cache)]


def lookup_kv_table(cache):
    if _LAYOUT is None:
        return None
    if _PLAN is None or _key(cache) not in _PLAN["kv_tables"]:
        raise RuntimeError("Tree attention/indexer has no private block table")
    return _PLAN["kv_tables"][_key(cache)]


def _compressor_depth_batches(plan, state_layout, block_size, ratio, cos, sin):
    """Use a fixed root + speculative-depth loop with device-only membership.

    Inactive rows have one zero token at position zero and a null state table.
    C4/C128 therefore emit no compressed token and SaveState skips their writes.
    Keeping their input length at one also avoids all-empty compressor batches.
    """
    device, n = cos.device, plan["n"]
    steps = torch.arange(plan["max_q"], device=device).unsqueeze(0)
    positions = plan["start"].long().unsqueeze(1) + steps
    live = steps <= plan["depths"].unsqueeze(1)
    columns = positions // block_size
    table = state_layout["table"]
    pages = table.gather(1, columns.clamp(0, table.shape[1] - 1))
    valid_state = (columns < table.shape[1]) & (pages > 0)

    # Splitting a chain is unsafe if a later input aliases history that the
    # fused full-chain call still reads. Use fixed-stride TH input and device
    # seqused in that case; neither the shape nor the fallback decision needs
    # actual depths on the host. BSH is not compiled by the local CANN kernels.
    tables = state_layout["table_rows"]
    alias = False
    for endpoint, length in enumerate(state_layout["capacities"]):
        prefix = plan["prefixes"][endpoint]
        history = max(0, prefix // ratio * ratio - (ratio if ratio == 4 else 0))
        seen = set()
        for position in range(history, prefix + length):
            column, offset = divmod(position, block_size)
            page = tables[endpoint][column] if column < table.shape[1] else 0
            if page > 0:
                slot = page, offset
                alias = alias or slot in seen
                seen.add(slot)
    full_count = (plan["start"] + plan["lens"]) // ratio - plan["start"] // ratio
    if alias:
        return {
            "batches": None,
            "tokens": plan["ancestors"].reshape(-1),
            "cu": plan["query_cu"] * plan["max_q"],
            "seqused": plan["lens"],
            "output_valid": torch.arange(cos.shape[0], device=device) < full_count.sum(),
        }

    # Copy only the parent's input-position rows, preserving both state halves.
    # The list is bounded by host scheduling; the source parent and active mask
    # are determined on device. Duplicate physical destinations are eliminated.
    fields = []
    for depth in range(plan["max_q"]):
        copies = []
        for endpoint, length in enumerate(state_layout["capacities"]):
            if length <= depth:
                continue
            seen = set()
            prefix = plan["prefixes"][endpoint]
            for position in range(prefix, prefix + depth):
                column, offset = divmod(position, block_size)
                destination = tables[endpoint][column] if column < table.shape[1] else 0
                if (destination, offset) in seen:
                    continue
                seen.add((destination, offset))
                copies.append((endpoint, column, destination, offset))
        fields.append((copies, torch.long, (len(copies), 4)))
    uploaded = _upload_metadata(device, *fields, host_buffers=plan["host_uploads"])

    output_begin = full_count.cumsum(0) - full_count
    padded_cos = torch.cat((cos, cos.new_zeros((1, *cos.shape[1:]))))
    padded_sin = torch.cat((sin, sin.new_zeros((1, *sin.shape[1:]))))
    batches, emitted_rows = [], []
    for depth in range(plan["max_q"]):
        active = (plan["depths"] == depth) & valid_state[:, depth]
        start = torch.where(active, plan["start"] + depth, 0)
        emit = active & ((start + 1) % ratio == 0)
        rank = emit.long().cumsum(0) - 1
        emitted_rows.append(depth * n + rank)
        # TH emits only complete compression windows, densely in request order.
        # Emission ranks are unique; inactive rows use distinct scratch slots.
        rope_ids = torch.full((2 * n,), cos.shape[0], dtype=torch.long, device=device)
        rope_ids.scatter_(
            0, torch.where(emit, rank, n + plan["token_index"]),
            torch.where(emit, output_begin + full_count - 1, cos.shape[0]).long(),
        )
        rope_ids = rope_ids[:n]
        owners, columns, destinations, offsets = uploaded[depth].unbind(1)
        parents = plan["parent_ids"].index_select(0, owners)
        source = table[parents, columns.clamp(0, table.shape[1] - 1)].long()
        copying = active.index_select(0, owners)
        valid_copy = (columns < table.shape[1]) & (destinations > 0) & (source > 0)
        copy_indices = torch.stack((destinations, offsets), dim=1)
        copy_indices = torch.where((copying & valid_copy).unsqueeze(1), copy_indices, -1)
        batches.append({
            "active": active, "tokens": plan["hidden_ids"],
            "copy_indices": copy_indices, "copy_source": (source, offsets),
            "cu": plan["query_cu"], "start": start,
            "state_bt": torch.where(active.unsqueeze(1), table, 0),
            "cos": padded_cos.index_select(0, rope_ids),
            "sin": padded_sin.index_select(0, rope_ids),
        })

    # Expand each ancestor's single projection back to full-chain emission order.
    # Padding points at an explicit zero row, never at uninitialized op output.
    emitted = torch.stack(emitted_rows, dim=1)
    raw_ids = emitted.gather(0, plan["ancestors"])
    emitting = live & ((positions + 1) % ratio == 0)
    destinations = emitting.reshape(-1).long().cumsum(0) - 1
    raw_capacity = n * plan["max_q"]
    padding = cos.shape[0] + torch.arange(raw_capacity, device=device)
    output_ids = torch.full((cos.shape[0] + raw_capacity,), -1, dtype=torch.long, device=device)
    output_ids.scatter_(
        0, torch.where(emitting.reshape(-1), destinations, padding),
        torch.where(emitting, raw_ids, -1).reshape(-1),
    )
    output_ids = output_ids[:cos.shape[0]]
    output_ids = torch.where(output_ids >= 0, output_ids, raw_capacity)
    return {"batches": batches, "output_ids": output_ids}


def chain_compressor_batch(hidden, state_cache, kv_block_table, state_block_table,
                           storage_block_size, ratio, req_metadata, slot_format, *, state_metadata):
    if _LAYOUT is None:
        return None
    if not _KV_STACK or not _KV_STACK[-1]:
        raise RuntimeError("Tree compressor called without its output cache bindings")
    plan = _build_plan(int(hidden.shape[0]), hidden.device)
    ratio, bs = int(ratio), int(storage_block_size)
    if ratio not in (4, 128) or bs <= 0:
        raise RuntimeError("Unsupported tree compression layout")
    caches = _KV_STACK[-1]
    if int(caches[0].shape[1]) != bs:
        raise RuntimeError("Compressed cache page size differs from metadata")
    kv_layout = _workspace_layout(
        plan, kv_block_table, req_metadata, "kv", bs, ratio=ratio
    )
    _, kv_rows = _workspace(plan, caches, kv_layout, "kv")

    # State caches can reuse physical pages through circular/aliased tables.
    # A round-owned mapping snapshot covers every alias of the write interval.
    state_layout = _workspace_layout(
        plan, state_block_table, state_metadata, "state", int(state_cache.shape[1])
    )
    state_views, state_rows = _workspace(plan, [state_cache], state_layout, "state")
    full_cos, full_sin = req_metadata.full_compress_cos, req_metadata.full_compress_sin
    cos = full_cos.view(full_cos.shape[0], full_cos.shape[-1])
    sin = full_sin.view(full_sin.shape[0], full_sin.shape[-1])
    token_size = plan["token_count"]
    n_rows = max(1, min(token_size, token_size // ratio + plan["n"]))
    rope_shape = (n_rows, 1, 1, cos.shape[1])
    slot_shape = (n_rows, 2) if int(slot_format) == _SLOT_MAPPING_BLOCK_OFFSET else (n_rows,)
    group = req_metadata.cache_group_key
    metadata_key = (
        group, bs, ratio, int(slot_format), rope_shape,
        cos.shape, cos.dtype, cos.device, sin.shape, sin.dtype,
    )
    packed_metadata = plan["compressor_metadata"].get(metadata_key)
    if packed_metadata is None:
        # These outputs contain physical workspace slots. They are round-owned,
        # tied to the group's mapping, and never enter layer scratch.
        compress_cos = torch.empty(rope_shape, dtype=cos.dtype, device=cos.device)
        compress_sin = torch.empty(rope_shape, dtype=sin.dtype, device=sin.device)
        slots = torch.empty(slot_shape, dtype=torch.int32, device=kv_rows.device)
        torch.ops._C_ascend.compressor_metadata_out(
            cos, sin, plan["cu"], plan["start"], kv_rows, bs, int(slot_format),
            ratio, plan["n"], compress_cos, compress_sin, slots,
        )
        packed_metadata = compress_cos, compress_sin, slots
        plan["compressor_metadata"][metadata_key] = packed_metadata
    compress_cos, compress_sin, slots = packed_metadata
    batch_key = (metadata_key, state_metadata.cache_group_key, int(state_cache.shape[1]))
    if batch_key not in plan["compressor_batches"]:
        depth_batches = _compressor_depth_batches(
            plan, state_layout, int(state_cache.shape[1]), ratio, compress_cos, compress_sin,
        )
        plan["compressor_batches"][batch_key] = depth_batches
    else:
        depth_batches = plan["compressor_batches"][batch_key]
    return {
        "hidden": hidden,
        "cu": plan["cu"], "start": plan["start"], "kv_bt": kv_rows,
        "state_bt": state_rows, "state": state_views[0],
        "cos": compress_cos, "sin": compress_sin, "slot": slots, "depth_batches": depth_batches,
    }


def pack_chain_tokens(tensor):
    if tensor is None or _PLAN is None:
        return tensor
    return tensor.index_select(0, _PLAN["rows"])


def unpack_chain_tokens(packed, n_tokens):
    if _PLAN is None:
        return packed
    ends = packed.index_select(0, _PLAN["end_rows"])
    out = packed.new_zeros((n_tokens, *ends.shape[1:]))
    out.index_copy_(0, _PLAN["token_index"], ends)
    return out


def select_tree_queries(tensor):
    """Select one attention/indexer query per endpoint."""
    if tensor is None or _PLAN is None:
        return tensor
    return tensor.index_select(0, _PLAN["token_index"])


def restore_tree_queries(values, n_tokens):
    if _PLAN is None:
        return values
    out = values.new_zeros((n_tokens, *values.shape[1:]))
    out.index_copy_(0, _PLAN["token_index"], values)
    return out


def tree_query_metadata(key, build):
    """Cache configuration-specific metadata on immutable round-owned lengths."""
    if _PLAN is None:
        raise RuntimeError("Query metadata requires a tree verification plan")
    cache = _PLAN["query_metadata"]
    if key not in cache:
        cache[key] = build()
    return cache[key]


def chain_length_cus(seqused, cmp_ratio):
    # Only plan-owned lengths are immutable across layers. Other callers keep
    # the standalone calculation, with no pointer-based content cache.
    cached = _PLAN is not None and seqused is _PLAN["seqused"]
    if cached and cmp_ratio in _PLAN["length_cus"]:
        return _PLAN["length_cus"][cmp_ratio]
    ori = seqused.new_zeros(seqused.numel() + 1)
    cmp = seqused.new_zeros(seqused.numel() + 1)
    ori[1:] = torch.cumsum(seqused, dim=0)
    cmp[1:] = torch.cumsum(torch.div(seqused, cmp_ratio, rounding_mode="floor"), dim=0)
    if cached:
        _PLAN["length_cus"][cmp_ratio] = (ori, cmp)
    return ori, cmp


def chain_attention_view(ori_cache, ori_block_table, block_size, window_size, *,
                         req_metadata=None):
    if _LAYOUT is None:
        return None
    if _PLAN is None:
        raise RuntimeError("Tree attention ran before branch compression")
    plan = _PLAN
    key = _key(ori_cache)
    if key in plan["ori_views"]:
        return plan["ori_views"][key]
    bs = int(block_size)
    if int(ori_cache.shape[1]) != bs or int(window_size) <= 0:
        raise RuntimeError("Invalid original-KV page/window size")
    layout = _workspace_layout(
        plan, ori_block_table, req_metadata, "ori", bs, window_size=int(window_size)
    )
    views, table = _workspace(plan, [ori_cache], layout, "ori")
    work = views[0]
    if layout["ori_indices"] is not None:
        indices, source_pages, source_offsets = layout["ori_indices"]
        # Byte transport preserves packed FP8 data and scales without casting.
        values = ori_cache.view(torch.int8)[source_pages, source_offsets]
        torch_npu.npu_scatter_nd_update_(work.view(torch.int8), indices, values)
    view = {"cu": plan["query_cu"], "seqused": plan["seqused"],
            "max_q": 1, "max_kv": plan["max_kv"],
            "ori_bt": table, "ori_cache": work}
    plan["ori_views"][key] = view
    return view


def node_kv_active():
    return _PLAN is not None and bool(_PLAN.get("node_kv"))


def indexer_batch(key_cache):
    if _LAYOUT is None:
        return None
    if _PLAN is not None and _key(key_cache) in _PLAN.get("node_kv", {}):
        return {"node_cache": _PLAN["node_kv"][_key(key_cache)], "plan": _PLAN}
    table = lookup_kv_table(key_cache)
    plan = _PLAN
    seq = plan["seqused"]
    if plan["indexer_lengths"] is None:
        plan["indexer_lengths"] = (
            torch.div(seq, 4, rounding_mode="floor"), torch.remainder(seq, 4)
        )
    seqused_k, residual = plan["indexer_lengths"]
    return {"cu": plan["query_cu"], "seqused_k": seqused_k,
            "residual": residual, "block_table": table,
            "max_q": 1, "max_k": plan["max_kv"] // 4,
            "batch": plan["n"]}


def cache_tree_topk(buffer, indices, offset=0):
    if _PLAN is None or buffer is None or offset != 0:
        raise RuntimeError("Invalid packed TopK cache write")
    if int(indices.shape[0]) != _PLAN["n"]:
        raise RuntimeError("TopK output does not use unique endpoint query rows")
    # The indexer owns a fresh output tensor; subsequent layers only read it.
    _PLAN["index_topk"][_key(buffer)] = indices


def get_tree_topk(buffer, offset=0):
    if _PLAN is None or buffer is None or offset != 0:
        raise RuntimeError("Invalid packed TopK cache read")
    indices = _PLAN["index_topk"].get(_key(buffer))
    if indices is None:
        raise RuntimeError("No TopK producer for this tree verification step")
    return indices


def finish_tree_layer() -> None:
    """Retain candidate deltas and dirty masks without a host round trip.

    SaveState only writes input-position rows, so state snapshots retain that
    physical write set. Read-only history stays in the persistent baseline.
    Commit reconstructs the selected page from the baseline and its row deltas.
    """
    if _PLAN is None:
        return
    if _KV_STACK:
        raise RuntimeError("Tree layer finished inside a compressor scope")
    plan = _PLAN
    for original, workspace, kind, layout in plan["pending"]:
        entries = layout["entries"]
        if kind == "state":
            destination_ids, offsets, source_ids = layout["state_indices"]
            snapshot = workspace[source_ids, offsets]
            before = original[destination_ids, offsets].contiguous().view(torch.uint8)
            after = snapshot.contiguous().view(torch.uint8)
            dirty = (before != after).reshape(snapshot.shape[0], -1).any(dim=1)
            del before, after
        else:
            _, source_ids = layout["snapshot_indices"]
            # Compressed KV (including packed FP8/scale views) retains its page
            # representation and invocation order; only state uses row deltas.
            snapshot = workspace.index_select(0, source_ids)
            dirty = layout["written"]
        # Keep whole tensors alive; descriptors use host offsets instead of
        # constructing a device tensor view and reduction for every candidate.
        for row, (chain, dst, _) in enumerate(entries):
            key = (_key(original), dst)
            record = plan["records"].setdefault(key, (original, dst, []))
            if kind == "state":
                begin, length = layout["state_spans"][row]
                record[2].append((chain, snapshot, begin, length, offsets, dirty))
            else:
                record[2].append((chain, snapshot, row, 1, None, dirty))

    plan["pending"].clear()
    plan["cache_views"].clear()
    plan["kv_tables"].clear()
    plan["ori_views"].clear()
    # The eager tree path queues all consumers and snapshots on one stream.
    # Later layers can now reuse scratch storage without overwriting winners.
    plan["buffers"].reset()


def prepare_winner(path_node_ids, num_sampled=None):
    """Select compressed/state writes using device-side winner and dirty masks."""
    if _PLAN is None or path_node_ids is None:
        raise RuntimeError("Missing tree verification plan/accepted path")
    plan = _PLAN
    if plan["committed"] or plan["pending"]:
        raise RuntimeError("Tree state committed twice or a layer was not finalized")
    n_req = plan["n_req"]
    if path_node_ids.shape[0] != n_req:
        raise RuntimeError("Accepted-path batch differs from verification batch")
    path = path_node_ids.long()
    present = path != -1
    sampled = present.sum(dim=1) + 1 if num_sampled is None else num_sampled[:n_req]
    active = sampled > 0
    qsl = _LAYOUT["qsl"][:n_req + 1]
    # The root is endpoint zero when a bonus is sampled without any draft.
    endpoint = torch.where(present, path, 0).amax(dim=1)
    winner = qsl[:-1] + endpoint
    selected = active[plan["req_ids"]] & (plan["token_index"] == winner[plan["req_ids"]])
    if plan.get("node_records"):
        from vllm_ascend.worker.v2.spec_decode.tree.node_pool import prepare_nodes

        plan["prepared_nodes"] = prepare_nodes(plan, winner, active)
    return _prepare_state_pages(plan, selected)


def _prepare_state_pages(plan, selected):
    """Batch descriptors on host; select/freeze pages entirely on device.

    Snapshots remain alive in records until commitment. Overlapping packed
    cache views are assigned ordered commit waves, never concurrent writers.
    The source-row map includes untouched baseline rows, since KV compaction
    may change those bytes between preparation and commitment.
    """
    pages, candidates, waves = [], [], []
    intervals = {}
    total_rows = total_bytes = 0
    max_rows = max_delta = 1
    word_copy = True
    for cache, page, entries in plan["records"].values():
        size = cache.element_size()
        row_bytes = prod(cache.shape[2:]) * size
        expected = 1
        for dim, stride in zip(reversed(cache.shape[2:]), reversed(cache.stride()[2:])):
            if dim > 1 and stride != expected:
                raise RuntimeError("Tree state requires contiguous bytes within each row")
            expected *= dim
        row_stride = cache.stride(1) * size
        if row_stride < row_bytes:
            raise RuntimeError("Tree state rows overlap")
        base = cache.data_ptr() + page * cache.stride(0) * size
        nrows = cache.shape[1]
        start = len(candidates)
        for owner, snapshot, begin, length, offsets, dirty in entries:
            source = snapshot.data_ptr() + begin * snapshot.stride(0) * size
            is_delta = offsets is not None
            row_ptr = offsets.data_ptr() + begin * offsets.stride(0) * 8 if is_delta else 0
            candidates.append((owner, source, row_ptr,
                               offsets.stride(0) if is_delta else 0, length,
                               dirty.data_ptr() + begin * dirty.stride(0),
                               dirty.stride(0), int(is_delta)))
            max_delta = max(max_delta, length)
            word_copy &= source % 8 == 0
        page_id = len(pages)
        pages.append((base, row_stride, row_bytes, nrows, start, len(candidates), total_rows))
        max_rows = max(max_rows, nrows)
        word_copy &= (base % 8 == 0 and row_bytes % 8 == 0 and row_stride % 8 == 0)
        end = base + (nrows - 1) * row_stride + row_bytes
        storage = cache.untyped_storage().data_ptr()
        previous = intervals.setdefault(storage, [])
        wave = max((w + 1 for lo, hi, w in previous if base < hi and lo < end), default=0)
        previous.append((base, end, wave))
        while len(waves) <= wave:
            waves.append([])
        waves[wave].extend((page_id, total_rows + row, total_bytes + row * row_bytes,
                            row_bytes, base + row * row_stride) for row in range(nrows))
        total_rows += nrows
        total_bytes += nrows * row_bytes
    jobs = [job for wave in waves for job in wave]
    device = selected.device
    page_desc, candidate_desc, job_desc = _upload_metadata(
        device,
        (pages, torch.int64, (len(pages), 7)),
        (candidates, torch.int64, (len(candidates), 8)),
        (jobs, torch.int64, (len(jobs), 5)),
        host_buffers=plan["host_uploads"],
    )
    source_rows = torch.empty(total_rows, dtype=torch.int64, device=device)
    active = torch.empty(len(pages), dtype=torch.bool, device=device)
    staging = torch.empty(total_bytes, dtype=torch.uint8, device=device)
    if pages:
        select_tree_state_kernel[(len(pages),)](
            page_desc, candidate_desc, selected, source_rows, active,
            PAGE_ROWS=triton.next_power_of_2(max_rows),
            DELTA_ROWS=triton.next_power_of_2(max_delta),
        )
        copy_tree_state_rows_kernel[(len(jobs),)](
            job_desc, source_rows, active, staging, 0,
            PREPARE=True, WORD_COPY=word_copy, BLOCK=256,
        )
    return (job_desc, source_rows, active, staging, [len(w) for w in waves], word_copy)


def commit_winner(path_node_ids, num_sampled=None, *, prepared=None):
    if _PLAN is None or _PLAN["committed"] or _PLAN["pending"]:
        raise RuntimeError("Missing/unfinalized tree plan or duplicate commitment")
    prepared = prepare_winner(path_node_ids, num_sampled) if prepared is None else prepared
    jobs, source_rows, active, staging, wave_sizes, word_copy = prepared
    start = 0
    for count in wave_sizes:
        copy_tree_state_rows_kernel[(count,)](
            jobs, source_rows, active, staging, start,
            PREPARE=False, WORD_COPY=word_copy, BLOCK=256,
        )
        start += count
    if "prepared_nodes" in _PLAN:
        from vllm_ascend.worker.v2.spec_decode.tree.node_pool import commit_nodes

        commit_nodes(_PLAN.pop("prepared_nodes"))
        _PLAN["node_records"].clear()
    _PLAN["committed"] = True
    _PLAN["records"].clear()
    _PLAN["workspace_layouts"].clear()
