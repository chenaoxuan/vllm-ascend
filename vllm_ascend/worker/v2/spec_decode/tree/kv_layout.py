import torch

# Cached arange / column buffers keyed by device (grow on demand).
_depth_arange: dict[torch.device, torch.Tensor] = {}
_dst_off_arange: dict[torch.device, torch.Tensor] = {}
_root_col: dict[torch.device, torch.Tensor] = {}
_valid_root_col: dict[torch.device, torch.Tensor] = {}
_token_arange: dict[torch.device, torch.Tensor] = {}
# Growable src/dst slot buffers for the triton path, keyed by device.
_kv_src_slots: dict[torch.device, torch.Tensor] = {}
_kv_dst_slots: dict[torch.device, torch.Tensor] = {}


def _cached_arange(cache: dict, n: int, device: torch.device, dtype=torch.long) -> torch.Tensor:
    buf = cache.get(device)
    if buf is None or buf.numel() < n or buf.dtype != dtype:
        buf = torch.arange(n, device=device, dtype=dtype)
        cache[device] = buf
    return buf[:n]


def _cached_col(
    cache: dict,
    num_reqs: int,
    device: torch.device,
    dtype,
    fill_value,
) -> torch.Tensor:
    buf = cache.get(device)
    if buf is None or buf.shape[0] < num_reqs or buf.dtype != dtype:
        buf = torch.empty((num_reqs, 1), dtype=dtype, device=device)
        cache[device] = buf
    out = buf[:num_reqs]
    out.fill_(fill_value)
    return out


def compact_tree_kv_along_path(
    caches: list[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    idx_mapping: torch.Tensor,
    num_computed: torch.Tensor,
    path_node_ids: torch.Tensor,
) -> None:
    """Move accepted-path KV from tree slots onto the linear prefix.

    Each cache is slot-major ``[num_blocks, block_size, ...]``. ``block_table``
    is ``[max_reqs, max_blocks]`` indexed by req_state. ``path_node_ids`` is
    ``[num_reqs, spec_len]`` with ``-1`` unused. ``num_computed`` is
    ``[max_reqs]`` at the start of the verify step. All tensors are device-side
    except the Python ``block_size``.

    Reads are gathered into a temporary before scatter so overlapping src/dst
    slots (including swaps) stay correct.
    """
    from vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch import use_tree_triton

    if use_tree_triton():
        _compact_tree_kv_along_path_triton(
            caches, block_table, block_size, idx_mapping, num_computed, path_node_ids
        )
        return
    compact_tree_kv_along_path_torch(
        caches, block_table, block_size, idx_mapping, num_computed, path_node_ids
    )


def _compact_tree_kv_along_path_triton(
    caches: list[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    idx_mapping: torch.Tensor,
    num_computed: torch.Tensor,
    path_node_ids: torch.Tensor,
) -> None:
    from vllm_ascend.ops.triton.spec_decode.tree.kv_compact import (
        compact_tree_kv_slots_triton,
    )

    # Skip the copy when already int64+contiguous; only convert otherwise.
    if path_node_ids.dtype == torch.long and path_node_ids.is_contiguous():
        node = path_node_ids
    else:
        node = path_node_ids.to(dtype=torch.long).contiguous()
    num_reqs, spec_len = node.shape
    device = node.device
    # Reuse growable slot buffers (keyed by device) instead of malloc per call.
    src_full = _kv_src_slots.get(device)
    dst_full = _kv_dst_slots.get(device)
    if (
        src_full is None
        or src_full.shape[0] < num_reqs
        or src_full.shape[1] < spec_len
    ):
        r = max(num_reqs, src_full.shape[0] if src_full is not None else 0)
        s = max(spec_len, src_full.shape[1] if src_full is not None else 0)
        src_full = torch.empty((r, s), dtype=torch.long, device=device)
        dst_full = torch.empty_like(src_full)
        _kv_src_slots[device] = src_full
        _kv_dst_slots[device] = dst_full
    src_slots = src_full[:num_reqs, :spec_len]
    dst_slots = dst_full[:num_reqs, :spec_len]
    compact_tree_kv_slots_triton(
        block_table,
        num_computed,
        idx_mapping,
        node,
        src_slots,
        dst_slots,
        block_size,
    )
    src_flat = src_slots.reshape(-1)
    dst_flat = dst_slots.reshape(-1)
    for cache in caches:
        tail = cache.shape[2:]
        flat = cache.reshape(cache.shape[0] * cache.shape[1], *tail)
        # index_select returns a fresh copy (gather), so no .clone() is needed
        # before the in-place scatter. Overlapping src/dst slots stay correct
        # because `gathered` is an independent snapshot of the pre-scatter state.
        gathered = torch.index_select(flat, 0, src_flat).view(
            num_reqs, spec_len, *tail
        )
        flat.index_copy_(0, dst_flat, gathered.reshape(-1, *tail))


def compact_tree_kv_along_path_torch(
    caches: list[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    idx_mapping: torch.Tensor,
    num_computed: torch.Tensor,
    path_node_ids: torch.Tensor,
) -> None:
    node = path_node_ids.to(dtype=torch.long)
    req_idx = idx_mapping[: node.shape[0]]
    safe_idx = req_idx.clamp(min=0)
    prefix = num_computed[safe_idx]
    spec_len = node.shape[1]
    depth = _cached_arange(_depth_arange, spec_len, node.device, dtype=prefix.dtype) + 1
    dst_pos = prefix.unsqueeze(1) + depth
    src_pos = prefix.unsqueeze(1) + node.clamp(min=0)
    valid = (node >= 0) & (req_idx >= 0).unsqueeze(1)
    src_pos = torch.where(valid, src_pos, dst_pos)
    req_f = safe_idx.unsqueeze(1).expand_as(node)
    src_block = block_table[req_f, torch.div(src_pos, block_size, rounding_mode="floor")]
    dst_block = block_table[req_f, torch.div(dst_pos, block_size, rounding_mode="floor")]
    src_slots = (src_block * block_size + src_pos % block_size).to(dtype=torch.long)
    dst_slots = (dst_block * block_size + dst_pos % block_size).to(dtype=torch.long)
    for cache in caches:
        tail = cache.shape[2:]
        flat = cache.reshape(cache.shape[0] * cache.shape[1], *tail)
        gathered = flat[src_slots].clone()
        flat[dst_slots] = gathered


def compact_tree_query_along_path(
    tensors: list[torch.Tensor],
    query_start_loc: torch.Tensor,
    path_node_ids: torch.Tensor,
    linearize_positions: torch.Tensor,
) -> None:
    """Move accepted-path query rows onto the linear prefix of each request.

    ``path_node_ids`` is ``[num_reqs, spec_len]`` with ``-1`` unused (device).
    ``query_start_loc`` is ``[num_reqs + 1]`` (device). Each tensor in
    ``tensors`` is token-major ``[num_tokens, ...]`` (hidden / aux). Root is
    query offset 0; draft node ``node_id`` is offset ``node_id``. Invalid path
    slots are no-ops (src = dst). Gather clones before scatter so overlapping
    src/dst rows stay correct.

    ``linearize_positions`` is ``[num_tokens]`` (device). Destination rows
    become ``root_pos + 0..k`` rather than packed RoPE (siblings may share a
    packed position). Leftover rejected rows keep their packed RoPE; DFlash
    context-slot writes must PAD that suffix
    (``mask_rejected_dflash_context_slots`` / kernel ``is_valid_ctx``).
    """
    num_reqs, spec_len = path_node_ids.shape
    node = path_node_ids.to(dtype=torch.long)
    qsl = query_start_loc[:num_reqs].to(dtype=torch.long)
    root = _cached_col(_root_col, num_reqs, node.device, torch.long, 0)
    src_off = torch.cat([root, node.clamp(min=0)], dim=1)
    dst_off = (
        _cached_arange(_dst_off_arange, spec_len + 1, node.device, dtype=torch.long)
        .unsqueeze(0)
        .expand(num_reqs, -1)
    )
    valid_root = _cached_col(
        _valid_root_col, num_reqs, node.device, torch.bool, True
    )
    valid = torch.cat([valid_root, node >= 0], dim=1)
    src_off = torch.where(valid, src_off, dst_off)
    src_idx = qsl.unsqueeze(1) + src_off
    dst_idx = qsl.unsqueeze(1) + dst_off
    for tensor in tensors:
        gathered = tensor[src_idx].clone()
        tensor[dst_idx] = gathered
    base = linearize_positions[qsl].unsqueeze(1)
    new_pos = base + dst_off.to(dtype=linearize_positions.dtype)
    cur = linearize_positions[dst_idx]
    linearize_positions[dst_idx] = torch.where(valid, new_pos, cur)


def mask_rejected_dflash_context_slots(
    context_slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_rejected: torch.Tensor,
    pad_slot_id: int,
) -> None:
    """PAD draft-context slots on each request's rejected query suffix.

    After ``compact_tree_query_along_path``, accepted tokens are a linear
    prefix of length ``query_len - num_rejected``. Leftover siblings can still
    share RoPE with that prefix; those rows must not write draft KV.
    All tensors are device-side. ``query_start_loc`` is ``[num_reqs + 1]``.
    """
    num_reqs = num_rejected.shape[0]
    starts = query_start_loc[:num_reqs].to(dtype=torch.long)
    ends = query_start_loc[1 : num_reqs + 1].to(dtype=torch.long)
    valid_ends = ends - num_rejected.to(dtype=torch.long)
    n = context_slot_mapping.shape[0]
    idx = _cached_arange(
        _token_arange, n, context_slot_mapping.device, dtype=torch.long
    )
    req = torch.searchsorted(ends, idx, right=True).clamp(max=num_reqs - 1)
    in_req = (idx >= starts[req]) & (idx < ends[req])
    rejected = in_req & (idx >= valid_ends[req])
    context_slot_mapping.masked_fill_(rejected, pad_slot_id)


def iter_unique_kv_cache_tensors(kv_cache) -> list[torch.Tensor]:
    """Yield slot-major K/V tensors from a layer ``kv_cache`` binding."""
    if kv_cache is None:
        return []
    if isinstance(kv_cache, (tuple, list)):
        return [t for t in kv_cache if isinstance(t, torch.Tensor) and t.ndim >= 2]
    if isinstance(kv_cache, torch.Tensor) and kv_cache.ndim >= 2:
        if kv_cache.shape[0] == 2 and kv_cache.ndim >= 3:
            return [kv_cache[0], kv_cache[1]]
        return [kv_cache]
    return []
