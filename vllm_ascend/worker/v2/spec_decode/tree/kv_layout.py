import torch


# Cached arange / column buffers keyed by device (grow on demand).
_dst_off_arange: dict[torch.device, torch.Tensor] = {}
_root_col: dict[torch.device, torch.Tensor] = {}
_valid_root_col: dict[torch.device, torch.Tensor] = {}
_token_arange: dict[torch.device, torch.Tensor] = {}


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


def iter_unique_kv_cache_tensors(kv_cache) -> list[torch.Tensor]:
    """Yield slot-major K/V tensors from a layer ``kv_cache`` binding."""
    if kv_cache is None:
        return []
    if isinstance(kv_cache, (tuple, list)):
        out: list[torch.Tensor] = []
        for item in kv_cache:
            if isinstance(item, torch.Tensor) and item.ndim >= 2:
                out.append(item)
            elif isinstance(item, (tuple, list)):
                out.extend(iter_unique_kv_cache_tensors(item))
        return out
    if isinstance(kv_cache, torch.Tensor) and kv_cache.ndim >= 2:
        if kv_cache.shape[0] == 2 and kv_cache.ndim >= 3:
            return [kv_cache[0], kv_cache[1]]
        return [kv_cache]
    return []


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
    # Truncated near max_seq_len can be 16 tokens on a 17-token graph while
    # path_node_ids still hold full-tree ids (e.g. 19). Stay in-bounds.
    n_tok = tensors[0].shape[0]
    last = n_tok - 1
    in_bound = (src_idx >= 0) & (src_idx <= last) & (dst_idx >= 0) & (dst_idx <= last)
    valid = valid & in_bound
    safe_dst = dst_idx.clamp(min=0, max=last)
    src_idx = torch.where(valid, src_idx, safe_dst)
    dst_idx = safe_dst
    for tensor in tensors:
        gathered = tensor[src_idx].clone()
        tensor[dst_idx] = gathered
    base = linearize_positions[qsl.clamp(min=0, max=last)].unsqueeze(1)
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
