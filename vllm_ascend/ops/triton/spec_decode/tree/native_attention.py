# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native tree attention over shared KV; only small indices are materialized."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _original_slots(
    Table,
    Ancestors,
    Starts,
    Depths,
    Requests,
    Slots,
    DEPTH: tl.constexpr,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    TS0: tl.constexpr,
    TS1: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, WIDTH)
    start = tl.load(Starts + row)
    end = start + tl.load(Depths + row) + 1
    p = tl.maximum(0, end - WINDOW) + i
    valid = (i < WINDOW) & (p < end)
    tree = p >= start
    ancestor = tl.load(Ancestors + row * DEPTH + p - start, valid & tree, other=0)
    root = tl.load(Ancestors + row * DEPTH)
    address = tl.where(tree, start + ancestor - root, p)
    req = tl.load(Requests + row)
    page = tl.load(Table + req * TS0 + address // BLOCK * TS1, valid, other=0)
    slot = page * BLOCK + address % BLOCK
    tl.store(Slots + row * WIDTH + i, tl.where(valid & (page > 0), slot, -1))


def native_attention(
    query, original, compressed, nodes, original_table, compressed_table, plan, topk, sinks, ratio, window, scale
):
    """Consume the same node plan as Triton, retaining accepted-chain semantics.

    Metadata is shared across layers with matching page tables and geometry.
    Its producers execute inside FULL capture, so prefix/topology changes are
    read again on replay. No host readback or KV chain expansion is performed.
    """
    n, heads, dim = query.shape
    cache = plan["query_metadata"]
    slot_key = ("native_slots", original_table.data_ptr(), original_table.stride(), original.shape[1], window)
    slots = cache.get(slot_key)
    if slots is None:
        width = triton.next_power_of_2(max(128, window))
        slots = torch.empty((n, 1, width), dtype=torch.int32, device=query.device)
        _original_slots[(n,)](
            original_table,
            plan["ancestors"],
            plan["start"],
            plan["depths"],
            plan["req_ids"],
            slots,
            plan["ancestors"].shape[1],
            window,
            width,
            original.shape[1],
            *original_table.stride(),
        )
        cache[slot_key] = slots

    key = ("native_metadata", ratio, heads, dim, window, topk.shape[-1] if topk is not None else 0)
    entry = cache.get(key)
    if entry is None:
        seq = plan["seqused"].to(torch.int32)
        args = dict(
            cu_seqlens_q=plan["query_cu"],
            cu_seqlens_ori_kv=torch.cat((seq.new_zeros(1), seq.cumsum(0, dtype=torch.int32))),
            cu_seqlens_cmp_kv=torch.cat((seq.new_zeros(1), (seq // ratio).cumsum(0, dtype=torch.int32))),
            seqused_kv=seq,
            cmp_ratio=ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )
        metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=heads,
            num_heads_kv=1,
            head_dim=dim,
            seqused_q=seq.new_empty(0),
            batch_size=n,
            max_seqlen_q=1,
            max_seqlen_kv=plan["max_kv"],
            ori_topk=0,
            cmp_topk=topk.shape[-1] if ratio == 4 else 0,
            has_ori_kv=True,
            has_cmp_kv=True,
            device=str(query.device),
            **args,
        )
        entry = cache[key] = (args, metadata)
    args, metadata = entry

    def query_table(table):
        table_key = ("native_table", table.data_ptr(), table.shape, table.stride())
        if table_key not in cache:
            cache[table_key] = table.index_select(0, plan["req_ids"])
        return cache[table_key]

    return torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(
        query,
        ori_kv=original,
        cmp_kv=compressed,
        ori_sparse_indices=slots,
        cmp_sparse_indices=topk,
        ori_block_table=query_table(original_table),
        cmp_block_table=query_table(compressed_table),
        sinks=sinks,
        metadata=metadata,
        cmp_tree_kv=nodes.view(n, dim),
        tree_ancestors=plan["ancestors"],
        tree_prefix=plan["start"],
        softmax_scale=scale,
        **args,
    )[0]
