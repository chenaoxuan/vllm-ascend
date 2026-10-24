"""Round-owned original-KV bindings for accepted-path compaction.

Host keys describe cache views and metadata groups, never live page contents.
Snapshot each binding once so later table-buffer reuse cannot change it.
"""
from __future__ import annotations

import torch


def _layout():
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack
    return chain_pack._LAYOUT


def _view_key(tensor):
    return (str(tensor.device), tensor.data_ptr(), tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype)


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
        context = dict(num_reqs=n, bindings={})
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

    context["bindings"][key] = dict(
        cache=cache, table=table.clone(), block_size=bs,
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
    groups = []
    used_scratch = {}
    for key, record in context["bindings"].items():
        cache, table, bs = record["cache"], record["table"], record["block_size"]
        state_table = torch.where(present.unsqueeze(1), table.index_select(0, rows), 0)
        scratch = compactor._binding_scratch.get(key)
        shape = (compactor.max_num_reqs * compactor.spec_len, *cache.shape[2:])
        if scratch is None or scratch.shape != shape:
            scratch = torch.empty(shape, dtype=cache.dtype, device=cache.device)
        used_scratch[key] = scratch
        groups.append(([cache], state_table, bs, [scratch]))
    # Reused slots never retain old mappings; only allocation scratch survives.
    compactor._binding_scratch = used_scratch
    return groups
