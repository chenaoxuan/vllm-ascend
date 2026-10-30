"""Round-owned original-KV bindings for accepted-path compaction.

Host keys describe cache views and metadata groups, never live page contents.
Snapshot each binding once so later table-buffer reuse cannot change it.
"""

from __future__ import annotations

import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec

from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import iter_unique_kv_cache_tensors


def _layout():
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack

    return chain_pack._LAYOUT


def _view_key(tensor):
    return (str(tensor.device), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def _group_is_linear_token_cache(group) -> bool:
    """Only ordinary token caches participate in slot relocation."""
    if any("state_cache" in name for name in group.layer_names):
        return False
    spec = group.kv_cache_spec
    if isinstance(spec, (CircularBufferSpec, MambaSpec)) or getattr(spec, "cache_role", None):
        return False
    inner = getattr(spec, "kv_cache_specs", None)
    for item in inner.values() if inner else [spec]:
        if isinstance(item, (CircularBufferSpec, MambaSpec)) or getattr(item, "cache_role", None):
            return False
        ratio = getattr(item, "tokens_per_state", 1)
        if ratio is not None and ratio > 1:
            return False
    return True


def bind_linear_kv(compactor, num_reqs=None):
    """Bind target token caches once, excluding draft and recurrent states."""
    if compactor._groups is not None:
        return compactor._groups
    runner = compactor.runner
    ctx = runner.compilation_config.static_forward_context
    draft_names = set(runner.speculator.draft_attn_layer_names)
    groups, seen = [], set()
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if not _group_is_linear_token_cache(group):
            continue
        caches, gathers = [], []
        block_size = runner.block_tables.kernel_block_sizes[group_id]
        for layer_name in group.layer_names:
            if layer_name in draft_names:
                continue
            layer = ctx.get(layer_name)
            if layer is None:
                continue
            for tensor in iter_unique_kv_cache_tensors(layer.kv_cache):
                key = (group_id, _view_key(tensor))
                if key in seen:
                    continue
                seen.add(key)
                if tensor.shape[1] != block_size:
                    raise RuntimeError("Target KV page size differs from block-table metadata")
                caches.append(tensor)
                gathers.append(
                    torch.empty(
                        (compactor.max_num_reqs * compactor.spec_len, *tensor.shape[2:]),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                )
        if caches:
            groups.append((caches, runner.block_tables.block_tables[group_id].gpu, block_size, gathers))
    return groups


def record_original_kv(layer_name, cache, metadata):
    """Observe a target cache before expansion and retain its block table."""
    layout = _layout()
    if layout is None:
        return
    tree_indices = getattr(metadata, "tree_ori_indices", None)
    if getattr(metadata, "dspark_swa_indices", None) is not None and tree_indices is None:
        return
    context = layout.get("_ori_commit_bindings")
    if context is None:
        n = layout["num_nodes"].shape[0]
        context = dict(num_reqs=n, bindings={}, tables={})
        layout["_ori_commit_bindings"] = context
    n = context["num_reqs"]
    table, bs = metadata.block_table, int(metadata.storage_block_size)
    table = table[:n]
    cache_key = _view_key(cache)
    group = str(getattr(metadata, "cache_group_key", "") or "")
    owner = "group:" + group if group else "layer:" + str(layer_name)
    key = (owner, cache_key)
    if key in context["bindings"]:
        return
    # One scheduler group has one original block mapping for this round.
    # Retain it before builders reuse their metadata storage, and share the
    # snapshot across layers so compaction builds one movement plan per group.
    table_key = (owner, bs)
    if table_key not in context["tables"]:
        context["tables"][table_key] = table.clone()
    context["bindings"][key] = dict(
        cache=cache,
        table=context["tables"][table_key],
        block_size=bs,
    )


def bind_original_kv(compactor, num_reqs):
    """Bind round snapshots and persistent gather scratch to the common compactor."""
    runner = compactor.runner
    layout = _layout()
    context = layout["_ori_commit_bindings"]
    n = num_reqs
    idx = compactor._idx[:n].long()
    # Invert the request mapping with fixed capacity and mask absent slots.
    state_ids = torch.arange(runner.max_num_reqs, device=idx.device)
    matches = state_ids.unsqueeze(1) == idx.unsqueeze(0)
    rows = matches.to(torch.int32).argmax(dim=1)
    present = matches.any(dim=1)
    groups = {}
    used_scratch = {}
    for key, record in context["bindings"].items():
        cache, table, bs = record["cache"], record["table"], record["block_size"]
        table_key = (_view_key(table), bs)
        if table_key not in groups:
            state_table = torch.where(present.unsqueeze(1), table.index_select(0, rows), 0)
            groups[table_key] = ([], state_table, bs, [])
        scratch = compactor._binding_scratch.get(key)
        shape = (compactor.max_num_reqs * compactor.spec_len, *cache.shape[2:])
        if scratch is None or scratch.shape != shape:
            scratch = torch.empty(shape, dtype=cache.dtype, device=cache.device)
        used_scratch[key] = scratch
        groups[table_key][0].append(cache)
        groups[table_key][3].append(scratch)
    # Reused slots never retain old mappings; only allocation scratch survives.
    compactor._binding_scratch = used_scratch
    return list(groups.values())
