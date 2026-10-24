# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 indexer over shared prefix KV and sparse ancestor node rows."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _score_heads(
    Q,
    Weight,
    QScale,
    History,
    NodeKV,
    Table,
    Ancestors,
    Starts,
    Depths,
    Req,
    Partial,
    H: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BS: tl.constexpr,
    HP: tl.constexpr,
    HR: tl.constexpr,
    WIDTH: tl.constexpr,
    MAX_Q: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    BK: tl.constexpr,
):
    node, head, k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, D)
    prefix = tl.load(Starts + node)
    depth = tl.load(Depths + node)
    req = tl.load(Req + node)
    valid = k < (prefix + depth + 1) // 4
    tree = k >= prefix // 4
    step = (k + 1) * 4 - 1 - prefix
    ancestor = tl.load(Ancestors + node * MAX_Q + step, valid & tree, other=0)
    page = tl.load(Table + req * WIDTH + k // BS, valid & ~tree, other=0)
    old = tl.load(History + page * HP + (k % BS) * HR + d, valid & ~tree, other=0).to(tl.float32)
    new = tl.load(NodeKV + ancestor * D + d, valid & tree, other=0).to(tl.float32)
    query = tl.load(Q + node * QS0 + head * QS1 + d).to(tl.float32)
    dot = tl.sum(tl.where(tree, new, old) * query, 0)
    # INT8 dot products fit exactly in FP32 at D=128. Retain both native
    # INT8 QLI FP16 rounding boundaries before reducing across heads.
    relu = tl.maximum(dot / 1024.0, 0).to(tl.float16).to(tl.float32)
    weight = tl.load(Weight + node * WS0 + head * WS1).to(tl.float16).to(tl.float32)
    scale = tl.load(QScale + node * SS0 + head * SS1).to(tl.float32)
    factor = (weight * scale).to(tl.float16).to(tl.float32)
    tl.store(Partial + (node * H + head) * K + k, relu * factor)


@triton.jit
def _reduce_heads(
    Partial,
    HistoryScale,
    NodeScale,
    Table,
    Ancestors,
    Starts,
    Depths,
    Req,
    Scores,
    H: tl.constexpr,
    K: tl.constexpr,
    BS: tl.constexpr,
    HP: tl.constexpr,
    HR: tl.constexpr,
    WIDTH: tl.constexpr,
    MAX_Q: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    node, k = tl.program_id(0), tl.program_id(1)
    h = tl.arange(0, BH)
    prefix = tl.load(Starts + node)
    depth = tl.load(Depths + node)
    req = tl.load(Req + node)
    valid = k < (prefix + depth + 1) // 4
    tree = k >= prefix // 4
    ancestor = tl.load(Ancestors + node * MAX_Q + (k + 1) * 4 - 1 - prefix, valid & tree, other=0)
    page = tl.load(Table + req * WIDTH + k // BS, valid & ~tree, other=0)
    old = tl.load(HistoryScale + page * HP + (k % BS) * HR, valid & ~tree, other=0).to(tl.float32)
    new = tl.load(NodeScale + ancestor, valid & tree, other=0).to(tl.float32)
    values = tl.load(Partial + (node * H + h) * K + k, h < H, other=0)
    result = tl.sum(values, 0) * tl.where(tree, new, old)
    tl.store(Scores + node * K + k, tl.where(valid, result, -float("inf")))


def node_indexer(query, weights, query_scale, history, history_scale, nodes, node_scale, table, plan, topk):
    """Return logical compressed positions, masked before ranking.

    Inputs are device tensors. History is shared; node rows are addressed only
    through ancestors. Top-K retains the platform's stable device sort.
    """
    n, heads, dim = query.shape
    capacity = max(1, plan["max_kv"] // 4)
    lengths = (plan["start"] + plan["depths"] + 1) // 4
    if capacity <= topk:
        positions = torch.arange(topk, dtype=torch.int32, device=query.device).expand(n, -1)
        return torch.where(positions < lengths.unsqueeze(1), positions, -1).unsqueeze(1)
    partial = torch.empty((n, heads, capacity), dtype=torch.float32, device=query.device)
    scores = torch.empty((n, capacity), dtype=torch.float32, device=query.device)
    weights = weights.reshape(n, heads)
    query_scale = query_scale.reshape(n, heads)
    _score_heads[(n, heads, capacity)](
        query,
        weights,
        query_scale,
        history,
        nodes,
        table,
        plan["ancestors"],
        plan["start"],
        plan["depths"],
        plan["req_ids"],
        partial,
        heads,
        dim,
        capacity,
        history.shape[1],
        history.stride(0),
        history.stride(1),
        table.shape[1],
        plan["max_q"],
        *query.stride()[:2],
        *weights.stride(),
        *query_scale.stride(),
        16,
        enable_fp_fusion=False,
    )
    _reduce_heads[(n, capacity)](
        partial,
        history_scale,
        node_scale,
        table,
        plan["ancestors"],
        plan["start"],
        plan["depths"],
        plan["req_ids"],
        scores,
        heads,
        capacity,
        history_scale.shape[1],
        history_scale.stride(0),
        history_scale.stride(1),
        table.shape[1],
        plan["max_q"],
        triton.next_power_of_2(heads),
        32,
    )
    values, ids = scores.sort(dim=-1, descending=True, stable=True)
    return torch.where(values[:, :topk].isfinite(), ids[:, :topk].int(), -1).unsqueeze(1)
