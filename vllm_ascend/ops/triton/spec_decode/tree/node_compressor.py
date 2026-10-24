# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate branch windows directly from shared history and unique node state."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _aggregate(
    State,
    Nodes,
    Table,
    Ancestors,
    Starts,
    Depths,
    Req,
    Output,
    BS: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    MAX_Q: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BD: tl.constexpr,
):
    node = tl.program_id(0)
    start = tl.load(Starts + node)
    depth = tl.load(Depths + node)
    pos = start + depth
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    if (pos + 1) % R == 0:
        req = tl.load(Req + node)
        w = tl.arange(0, R * C)
        # C4's native reduction interleaves the previous/current window.
        half = w % C
        p = pos + 1 - R * C + w // C + half * R
        live = p >= 0
        tree = p >= start
        ancestor = tl.load(Ancestors + node * MAX_Q + p - start, live & tree & (p - start <= depth), other=0)
        page = tl.load(Table + req * WIDTH + p // BS, live & ~tree & (p // BS < WIDTH), other=0)
        hist_ptr = State + page[:, None] * PAGE_STRIDE + (p % BS)[:, None] * ROW_STRIDE
        nd_ptr = Nodes + (ancestor[:, None] + 1) * (2 * C * D)
        col = half[:, None] * D + d[None, :]
        kv_hist = tl.load(hist_ptr + col, live[:, None] & ~tree[:, None] & (d[None, :] < D), other=0)
        sc_hist = tl.load(hist_ptr + C * D + col, live[:, None] & ~tree[:, None] & (d[None, :] < D), other=0)
        kv_node = tl.load(nd_ptr + col, live[:, None] & tree[:, None] & (d[None, :] < D), other=0)
        sc_node = tl.load(nd_ptr + C * D + col, live[:, None] & tree[:, None] & (d[None, :] < D), other=0)
        kv = tl.where(tree[:, None], kv_node, kv_hist)
        score = tl.where(live[:, None], tl.where(tree[:, None], sc_node, sc_hist), -float("inf"))
        exp = tl.exp(score - tl.max(score, 0)[None, :])
        probability = exp / tl.sum(exp, 0)[None, :]
        result = tl.sum(probability * kv, 0)
    else:
        result = tl.full((BD,), 0, tl.float32)
    tl.store(Output + node * D + d, result, d < D)


@triton.jit
def _epilogue(
    Aggregated,
    Weight,
    Cos,
    Sin,
    Starts,
    Depths,
    Output,
    D: tl.constexpr,
    RD: tl.constexpr,
    R: tl.constexpr,
    EPS: tl.constexpr,
    COS_STRIDE: tl.constexpr,
    SIN_STRIDE: tl.constexpr,
):
    node = tl.program_id(0)
    d = tl.arange(0, D)
    x = tl.load(Aggregated + node * D + d)
    denominator = tl.sqrt(tl.sum(x * x, 0) / D + EPS)
    x = x / denominator * tl.load(Weight + d)
    paired = tl.load(Aggregated + node * D + (d ^ 1)) / denominator * tl.load(Weight + (d ^ 1))
    pos = tl.load(Starts + node) + tl.load(Depths + node)
    emitted = (pos + 1) % R == 0
    rope_pos = tl.maximum(pos + 1 - R, 0)
    rd = d - (D - RD)
    c = tl.load(Cos + rope_pos * COS_STRIDE + rd, (rd >= 0) & emitted, other=1)
    s = tl.load(Sin + rope_pos * SIN_STRIDE + rd, (rd >= 0) & emitted, other=0)
    y = x * c + tl.where(d % 2 == 0, -paired, paired) * s
    tl.store(Output + node * D + d, tl.where(emitted, y, 0))


def aggregate_nodes(state, nodes, table, plan, norm, cos, sin, ratio, head_dim, rope_dim, eps, dtype, workspace=None):
    """All tensors are on device. The cache table addresses read-only history."""
    n = plan["n"]
    if workspace is None:
        aggregate = torch.empty((n, head_dim), dtype=torch.float32, device=state.device)
        output = torch.empty((n, head_dim), dtype=dtype, device=state.device)
    else:
        aggregate = workspace.acquire("aggregate", (n, head_dim), torch.float32, state.device)
        output = workspace.acquire("compressed", (n, head_dim), dtype, state.device)
    coeff = 2 if ratio == 4 else 1
    _aggregate[(n, triton.cdiv(head_dim, 32))](
        state,
        nodes,
        table,
        plan["ancestors"],
        plan["start"],
        plan["depths"],
        plan["req_ids"],
        aggregate,
        state.shape[1],
        state.stride(0),
        state.stride(1),
        table.shape[1],
        plan["max_q"],
        head_dim,
        ratio,
        coeff,
        32,
        enable_fp_fusion=False,
    )
    _epilogue[(n,)](
        aggregate,
        norm,
        cos,
        sin,
        plan["start"],
        plan["depths"],
        output,
        head_dim,
        rope_dim,
        ratio,
        eps,
        cos.stride(0),
        sin.stride(0),
        enable_fp_fusion=False,
    )
    return output
