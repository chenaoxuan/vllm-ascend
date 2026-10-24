# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in DSV4 node pool. Device data is indexed by immutable tree ancestry."""

from math import prod

import torch
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.spec_decode.tree.node_commit import commit_node_pages, freeze_node_pages
from vllm_ascend.ops.triton.spec_decode.tree.node_compressor import aggregate_nodes


def compress_nodes(module, hidden, state, metadata):
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

    plan = cp._build_plan(hidden.shape[0], hidden.device)
    cache_meta = metadata.cache.req_metadata
    state_meta = metadata.state.req_metadata
    ratio, head = module.compress_ratio, module.head_dim
    n, device = plan["n"], hidden.device
    key = ("projection", ratio)
    saved = plan["query_metadata"].get(key)
    if saved is None:
        # Projection uses one independent row per node. Its logical residue
        # preserves APE exactly; the disposable native aggregation sees only
        # null history. Triton performs the real branch aggregation afterwards.
        residue = ((plan["start"] + plan["depths"]) % ratio).to(torch.int32)
        table = torch.zeros((n, ratio), dtype=torch.int32, device=device)
        table.scatter_(1, residue.long().unsqueeze(1), (plan["token_index"] + 1).int().unsqueeze(1))
        saved = (
            residue,
            table,
            torch.ones((n, module.rope_head_dim), device=device),
            torch.zeros((n, module.rope_head_dim), device=device),
        )
        plan["query_metadata"][key] = saved
    residue, projection_table, dummy_cos, dummy_sin = saved
    nodes = torch.zeros((n + 1, 1, 1, 2 * module.coff * head), dtype=state.dtype, device=device)
    module._compress(
        hidden.index_select(0, plan["hidden_ids"]),
        nodes,
        projection_table,
        plan["query_cu"],
        residue,
        dummy_cos,
        dummy_sin,
    )
    full_cos, full_sin = cache_meta.full_compress_cos, cache_meta.full_compress_sin
    cos = full_cos.view(full_cos.shape[0], full_cos.shape[-1])
    sin = full_sin.view(full_sin.shape[0], full_sin.shape[-1])
    output = aggregate_nodes(
        state,
        nodes,
        state_meta.block_table,
        plan,
        module.norm.weight,
        cos,
        sin,
        ratio,
        head,
        module.rope_head_dim,
        module.norm_eps,
        hidden.dtype,
    )
    # Retain each node once until acceptance. History is never copied per branch.
    record_nodes(plan, state, nodes[1:], state_meta, 1)

    # Transitional adapter for the existing indexer/attention consumers. This
    # expansion is removed when those consumers can read the pool directly.
    caches = cp._KV_STACK[-1]
    bs = int(cache_meta.storage_block_size)
    layout = cp._workspace_layout(plan, cache_meta.block_table, cache_meta, "kv", bs, ratio=ratio)
    _, kv_table = cp._workspace(plan, caches, layout, "kv")
    count = max(1, min(plan["token_count"], plan["token_count"] // ratio + n))
    meta_key = ("node_compressed", cache_meta.cache_group_key, ratio, bs)
    packed = plan["query_metadata"].get(meta_key)
    if packed is None:
        rope_shape = (count, 1, 1, cos.shape[-1])
        rc = torch.empty(rope_shape, dtype=cos.dtype, device=device)
        rs = torch.empty_like(rc)
        slots = torch.empty((count, 2), dtype=torch.int32, device=device)
        torch.ops._C_ascend.compressor_metadata_out(
            cos, sin, plan["cu"], plan["start"], kv_table, bs, 2, ratio, n, rc, rs, slots
        )
        steps = torch.arange(plan["max_q"], device=device).unsqueeze(0)
        positions = plan["start"].unsqueeze(1) + steps
        emit = plan["visibility"] & ((positions + 1) % ratio == 0)
        rank = emit.reshape(-1).long().cumsum(0) - 1
        tail = count + torch.arange(emit.numel(), device=device)
        ids = torch.full((count + emit.numel(),), n, dtype=torch.long, device=device)
        ids.scatter_(0, torch.where(emit.reshape(-1), rank, tail), torch.where(emit, plan["ancestors"], n).reshape(-1))
        packed = slots, ids[:count]
        plan["query_metadata"][meta_key] = packed
    slots, ids = packed
    output = torch.cat((output, output.new_zeros((1, head)))).index_select(0, ids)
    return output, slots


def record_nodes(plan, original, snapshot, metadata, ratio):
    """Keep node rows and device block mappings alive; no data copy or readback."""
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

    real, _ = cp._LAYOUT["host_tables"][metadata.cache_group_key]
    pages = set()
    bs = original.shape[1]
    for node, req in enumerate(plan["req_index"]):
        start = plan["prefixes"][node]
        length = min(plan["max_q"], node - plan["req_cu"][req] + 1)
        begin = start if ratio == 1 else start // ratio
        end = start + length if ratio == 1 else (start + length) // ratio
        for pos in range(begin, end):
            col = pos // bs
            if col < len(real[req]) and real[req][col] > 0:
                pages.add(real[req][col])
    plan.setdefault("node_records", []).append((original, snapshot, metadata.block_table, sorted(pages), ratio))


def prepare_nodes(plan, winner, active):
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

    descriptors, total, max_rows = [], 0, 1
    for original, snapshot, table, pages, ratio in plan.get("node_records", []):
        size = original.element_size()
        row_bytes = prod(original.shape[2:]) * size
        if snapshot.stride(0) * snapshot.element_size() != row_bytes:
            raise RuntimeError("Node snapshots must have contiguous rows")
        for page in pages:
            descriptors.append(
                (
                    original.data_ptr() + page * original.stride(0) * size,
                    original.stride(1) * size,
                    row_bytes,
                    original.shape[1],
                    snapshot.data_ptr(),
                    table.data_ptr(),
                    table.shape[1],
                    page,
                    total,
                    ratio,
                )
            )
            total += original.shape[1] * row_bytes
            max_rows = max(max_rows, original.shape[1])
    (desc,) = cp._upload_metadata(
        winner.device, (descriptors, torch.int64, (len(descriptors), 10)), host_buffers=plan["host_uploads"]
    )
    staging = torch.empty(total, dtype=torch.uint8, device=winner.device)
    enabled = torch.empty(len(descriptors), dtype=torch.bool, device=winner.device)
    if descriptors:
        freeze_node_pages[(len(descriptors), max_rows)](
            desc,
            plan["ancestors"],
            plan["depths"],
            plan["start"],
            plan["req_ids"],
            winner,
            active,
            staging,
            enabled,
            plan["n"],
            plan["max_q"],
            triton.next_power_of_2(plan["n"]),
            256,
        )
    return desc, staging, enabled, max_rows


def commit_nodes(prepared):
    desc, staging, enabled, max_rows = prepared
    if desc.shape[0]:
        commit_node_pages[(desc.shape[0], max_rows)](desc, staging, enabled, 256)
