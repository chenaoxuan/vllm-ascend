# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze accepted node deltas over baseline pages before KV relocation."""

from vllm.triton_utils import tl, triton


@triton.jit
def freeze_node_pages(
    Desc,
    Ancestors,
    Depths,
    Starts,
    Req,
    Winners,
    Active,
    Staging,
    PageActive,
    N: tl.constexpr,
    MAX_Q: tl.constexpr,
    BN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    job = tl.program_id(0)
    row = tl.program_id(1)
    d = Desc + job * 10
    original = tl.load(d).to(tl.pointer_type(tl.uint8))
    row_stride = tl.load(d + 1)
    row_bytes = tl.load(d + 2)
    rows = tl.load(d + 3)
    node_data = tl.load(d + 4).to(tl.pointer_type(tl.uint8))
    table = tl.load(d + 5).to(tl.pointer_type(tl.int32))
    width = tl.load(d + 6)
    page_id = tl.load(d + 7)
    stage_offset = tl.load(d + 8)
    ratio = tl.load(d + 9)
    node = tl.arange(0, BN)
    req = tl.load(Req + node, node < N, other=0)
    winner = tl.load(Winners + req)
    depth = tl.load(Depths + node, node < N, other=0)
    win_depth = tl.load(Depths + winner)
    ancestor = tl.load(Ancestors + winner * MAX_Q + depth)
    selected = (node < N) & (depth <= win_depth) & (ancestor == node) & tl.load(Active + req)
    pos = tl.load(Starts + node, node < N, other=0) + depth
    # ratio=1 denotes state; compressed KV is emitted at the window boundary.
    selected = selected & ((pos + 1) % ratio == 0)
    position = (pos + 1) // ratio - 1
    col = position // rows
    page = tl.load(table + req * width + col, (node < N) & (col < width), other=0)
    on_page = selected & (page == page_id)
    enabled = tl.sum(on_page.to(tl.int32), 0) > 0
    if row == 0:
        tl.store(PageActive + job, enabled)
    if row < rows and enabled:
        matches = on_page & (position % rows == row)
        # Ascending request priority, latest accepted depth within a request.
        priority = tl.where(matches, (N - req) * (MAX_Q + 1) + depth, -1)
        best = tl.max(priority, 0)
        chosen = tl.max(tl.where(matches & (priority == best), node, -1), 0)
        lanes = tl.arange(0, BLOCK)
        for begin in range(0, row_bytes, BLOCK):
            offset = begin + lanes
            old = tl.load(original + row * row_stride + offset, offset < row_bytes, other=0)
            value = tl.load(
                node_data + tl.maximum(chosen, 0) * row_bytes + offset, (offset < row_bytes) & (chosen >= 0), other=0
            )
            tl.store(
                Staging + stage_offset + row * row_bytes + offset, tl.where(chosen >= 0, value, old), offset < row_bytes
            )


@triton.jit
def commit_node_pages(Desc, Staging, Active, BLOCK: tl.constexpr):
    job = tl.program_id(0)
    row = tl.program_id(1)
    d = Desc + job * 10
    if tl.load(Active + job) and row < tl.load(d + 3):
        target = tl.load(d).to(tl.pointer_type(tl.uint8))
        stride = tl.load(d + 1)
        size = tl.load(d + 2)
        source = Staging + tl.load(d + 8) + row * size
        lanes = tl.arange(0, BLOCK)
        for begin in range(0, size, BLOCK):
            value = tl.load(source + begin + lanes, begin + lanes < size, other=0)
            tl.store(target + row * stride + begin + lanes, value, begin + lanes < size)
