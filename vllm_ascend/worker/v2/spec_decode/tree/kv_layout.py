import torch


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
    if not caches:
        return
    node = path_node_ids.to(dtype=torch.long)
    req_idx = idx_mapping[: node.shape[0]]
    safe_idx = req_idx.clamp(min=0)
    prefix = num_computed[safe_idx]
    spec_len = node.shape[1]
    depth = torch.arange(spec_len, device=node.device, dtype=prefix.dtype) + 1
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
    linearize_positions: torch.Tensor | None = None,
) -> None:
    """Move accepted-path query rows onto the linear prefix of each request.

    ``path_node_ids`` is ``[num_reqs, spec_len]`` with ``-1`` unused (device).
    ``query_start_loc`` is ``[num_reqs + 1]`` (device). Each tensor in
    ``tensors`` is token-major ``[num_tokens, ...]`` (hidden / aux). Root is
    query offset 0; draft node ``node_id`` is offset ``node_id``. Invalid path
    slots are no-ops (src = dst). Gather clones before scatter so overlapping
    src/dst rows stay correct.

    When ``linearize_positions`` is set (``[num_tokens]`` device), the same
    destination rows are rewritten as ``root_pos + 0..k`` rather than packed
    RoPE (siblings may share a packed position). Leftover rejected rows keep
    their packed RoPE; DFlash context-slot writes must PAD that suffix
    (``mask_rejected_dflash_context_slots`` / kernel ``is_valid_ctx``).
    """
    num_reqs, spec_len = path_node_ids.shape
    if num_reqs == 0:
        return
    node = path_node_ids.to(dtype=torch.long)
    qsl = query_start_loc[:num_reqs].to(dtype=torch.long)
    root = torch.zeros((num_reqs, 1), dtype=torch.long, device=node.device)
    src_off = torch.cat([root, node.clamp(min=0)], dim=1)
    dst_off = torch.arange(
        spec_len + 1, device=node.device, dtype=torch.long
    ).expand(num_reqs, -1)
    valid = torch.cat(
        [
            torch.ones((num_reqs, 1), dtype=torch.bool, device=node.device),
            node >= 0,
        ],
        dim=1,
    )
    src_off = torch.where(valid, src_off, dst_off)
    src_idx = qsl.unsqueeze(1) + src_off
    dst_idx = qsl.unsqueeze(1) + dst_off
    for tensor in tensors:
        gathered = tensor[src_idx].clone()
        tensor[dst_idx] = gathered
    if linearize_positions is None:
        return
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
    if num_reqs == 0 or context_slot_mapping.numel() == 0:
        return
    starts = query_start_loc[:num_reqs].to(dtype=torch.long)
    ends = query_start_loc[1 : num_reqs + 1].to(dtype=torch.long)
    valid_ends = ends - num_rejected.to(dtype=torch.long)
    idx = torch.arange(
        context_slot_mapping.shape[0],
        device=context_slot_mapping.device,
        dtype=torch.long,
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
