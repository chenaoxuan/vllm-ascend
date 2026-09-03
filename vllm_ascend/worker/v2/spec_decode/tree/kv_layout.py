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
