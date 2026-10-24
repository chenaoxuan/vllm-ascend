# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-K attention directly over shared history and visible tree ancestors."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _read_attention(
    Q,
    Original,
    Compressed,
    Nodes,
    OriginalTable,
    CompressedTable,
    Ancestors,
    Starts,
    Depths,
    Req,
    TopK,
    Partial,
    Scores,
    Probability,
    SCORE_ONLY: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    MAX_Q: tl.constexpr,
    WINDOW: tl.constexpr,
    CK: tl.constexpr,
    TK_STRIDE: tl.constexpr,
    R: tl.constexpr,
    OBS: tl.constexpr,
    OP: tl.constexpr,
    OR: tl.constexpr,
    OW: tl.constexpr,
    CBS: tl.constexpr,
    CP: tl.constexpr,
    CR: tl.constexpr,
    CW: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    BK: tl.constexpr,
):
    node, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = split * BK + tl.arange(0, BK)
    d = tl.arange(0, D)
    prefix = tl.load(Starts + node)
    depth = tl.load(Depths + node)
    pos = prefix + depth
    req = tl.load(Req + node)
    root = tl.load(Ancestors + node * MAX_Q)
    is_ori = t < WINDOW
    p = pos - WINDOW + 1 + t
    ori_valid = is_ori & (p >= 0)
    tree_ori = p >= prefix
    ancestor = tl.load(Ancestors + node * MAX_Q + p - prefix, ori_valid & tree_ori, other=0)
    address = tl.where(tree_ori, prefix + ancestor - root, p)
    page = tl.load(OriginalTable + req * OW + address // OBS, ori_valid, other=0)
    ori = tl.load(
        Original + page[:, None] * OP + (address % OBS)[:, None] * OR + d[None, :],
        ori_valid[:, None] & (page[:, None] > 0),
        other=0,
    ).to(tl.float32)
    ci = t - WINDOW
    if R == 4:
        k = tl.load(TopK + node * TK_STRIDE + ci, (ci >= 0) & (ci < CK), other=-1)
    else:
        k = ci
    cmp_valid = (ci >= 0) & (ci < CK) & (k >= 0) & (k < (pos + 1) // R)
    tree_cmp = k >= prefix // R
    ca = tl.load(Ancestors + node * MAX_Q + (k + 1) * R - 1 - prefix, cmp_valid & tree_cmp, other=0)
    cp = tl.load(CompressedTable + req * CW + k // CBS, cmp_valid & ~tree_cmp, other=0)
    old = tl.load(
        Compressed + cp[:, None] * CP + (k % CBS)[:, None] * CR + d[None, :],
        cmp_valid[:, None] & ~tree_cmp[:, None],
        other=0,
    ).to(tl.float32)
    new = tl.load(Nodes + ca[:, None] * D + d[None, :], cmp_valid[:, None] & tree_cmp[:, None], other=0).to(tl.float32)
    kv = tl.where(is_ori[:, None], ori, tl.where(tree_cmp[:, None], new, old))
    if SCORE_ONLY:
        query = tl.load(Q + node * QS0 + head * QS1 + d).to(tl.float32)
        valid = tl.where(is_ori, ori_valid & (page > 0), cmp_valid)
        score = tl.sum(kv * query[None, :], 1) * SCALE
        tl.store(Scores + (node * H + head) * (WINDOW + CK) + t, tl.where(valid, score, -float("inf")), t < WINDOW + CK)
    else:
        probability = tl.load(Probability + (node * H + head) * (WINDOW + CK) + t, t < WINDOW + CK, other=0).to(
            tl.float32
        )
        weighted = tl.sum(probability[:, None] * kv, 0)
        tl.store(Partial + ((node * H + head) * SPLITS + split) * D + d, weighted)


@triton.jit
def _attention_probabilities(
    Scores,
    Sinks,
    Probability,
    Stats,
    H: tl.constexpr,
    WINDOW: tl.constexpr,
    CK: tl.constexpr,
    GROUPS: tl.constexpr,
    B: tl.constexpr,
):
    node, head = tl.program_id(0), tl.program_id(1)
    t = tl.arange(0, B)
    total = WINDOW + CK
    score = tl.load(Scores + (node * H + head) * total + t, t < total, other=-float("inf"))
    group_id = tl.where(t < WINDOW, 0, 1 + (t - WINDOW) // 512)
    maximum = tl.load(Sinks + head).to(tl.float32)
    denominator = 1.0
    p = tl.full((B,), 0, tl.float32)
    # Match A2 native online softmax: original window first, then compressed
    # chunks of 512. P is rounded to BF16 before the value product; sums stay FP32.
    for group in range(GROUPS):
        chosen = (group_id == group) & (t < total)
        current = tl.maximum(maximum, tl.max(tl.where(chosen, score, -float("inf")), 0))
        values = tl.exp(score - current)
        denominator = denominator * tl.exp(maximum - current) + tl.sum(tl.where(chosen, values, 0), 0)
        p = tl.where(chosen, values, p)
        maximum = current
        tl.store(Stats + (node * H + head) * (GROUPS + 1) + group, maximum)
    tl.store(Stats + (node * H + head) * (GROUPS + 1) + GROUPS, denominator)
    tl.store(Probability + (node * H + head) * total + t, p, t < total)


@triton.jit
def _merge_attention(
    Partial,
    Stats,
    Output,
    H: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    GROUPS: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    node, head = tl.program_id(0), tl.program_id(1)
    split = tl.arange(0, BS)
    d = tl.program_id(2) * BD + tl.arange(0, BD)
    base = Partial + ((node * H + head) * SPLITS + split) * D
    values = tl.load(base[:, None] + d[None, :], split[:, None] < SPLITS, other=0)
    group_id = tl.where(split * 16 < WINDOW, 0, 1 + (split * 16 - WINDOW) // 512)
    numerator = tl.full((BD,), 0, tl.float32)
    previous = tl.load(Stats + (node * H + head) * (GROUPS + 1))
    for group in range(GROUPS):
        current = tl.load(Stats + (node * H + head) * (GROUPS + 1) + group)
        numerator = numerator * tl.exp(previous - current) + tl.sum(
            tl.where((group_id == group)[:, None], values, 0), 0
        )
        previous = current
    denominator = tl.load(Stats + (node * H + head) * (GROUPS + 1) + GROUPS)
    tl.store(Output + (node * H + head) * D + d, numerator / denominator)


def node_attention(
    query, original, compressed, nodes, original_table, compressed_table, plan, topk, sinks, ratio, window, scale
):
    """Device-only gather, visibility, attention and sink normalization."""
    n, heads, dim = query.shape
    if query.dtype != torch.bfloat16 or original.dtype != torch.bfloat16 or compressed.dtype != torch.bfloat16:
        raise ValueError("The node-pool prototype currently requires A2 BF16 attention KV")
    if not 0 < window <= 512 or window % 16:
        raise ValueError("The A2 prototype requires a window divisible by 16, at most 512")
    # Bucket the host capacity so growing prefixes do not recompile every step.
    capacity = triton.next_power_of_2(max(1, plan["max_kv"] // ratio))
    count = min(topk.shape[-1], capacity) if ratio == 4 else capacity
    splits = triton.cdiv(window + count, 16)
    groups = 1 + triton.cdiv(count, 512)
    partial = torch.empty((n, heads, splits, dim), dtype=torch.float32, device=query.device)
    scores = torch.empty((n, heads, window + count), dtype=torch.float32, device=query.device)
    probability = torch.empty_like(scores, dtype=query.dtype)
    stats = torch.empty((n, heads, groups + 1), dtype=torch.float32, device=query.device)
    output = torch.empty_like(query)
    args = (
        query,
        original,
        compressed,
        nodes,
        original_table,
        compressed_table,
        plan["ancestors"],
        plan["start"],
        plan["depths"],
        plan["req_ids"],
        topk if topk is not None else plan["depths"],
        partial,
        scores,
        probability,
    )
    shape = (
        heads,
        dim,
        plan["max_q"],
        window,
        count,
        topk.stride(0) if topk is not None else 0,
        ratio,
        original.shape[1],
        original.stride(0),
        original.stride(1),
        original_table.shape[1],
        compressed.shape[1],
        compressed.stride(0),
        compressed.stride(1),
        compressed_table.shape[1],
        *query.stride()[:2],
        scale,
        splits,
        16,
    )
    _read_attention[(n, heads, splits)](*args, True, *shape, enable_fp_fusion=False)
    _attention_probabilities[(n, heads)](
        scores,
        sinks,
        probability,
        stats,
        heads,
        window,
        count,
        groups,
        triton.next_power_of_2(window + count),
        enable_fp_fusion=False,
    )
    _read_attention[(n, heads, splits)](*args, False, *shape, enable_fp_fusion=False)
    _merge_attention[(n, heads, triton.cdiv(dim, 64))](
        partial,
        stats,
        output,
        heads,
        dim,
        window,
        groups,
        splits,
        triton.next_power_of_2(splits),
        64,
        enable_fp_fusion=False,
    )
    return output
