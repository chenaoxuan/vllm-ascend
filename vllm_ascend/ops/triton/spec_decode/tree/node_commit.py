# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the VLLM project
"""Freeze accepted node deltas over baseline pages before KV relocation.

Grid layout is one program per job: each program resolves the accepted node
set for its job once as BN-wide vectors, then walks the page rows in a loop
overlaying the winning node row (or the original row) into staging.

History and rationale: the original one-program-per-page-row grid multiplied
per-program descriptor loads and NPU program-launch overhead by ``max_rows``
(a few hundred page rows x hundreds of jobs per DSV4 round) while moving only
a few MB. An intermediate one-program-per-row-chunk grid removed most of that
but still paid, per program, a 2D ``(RBLK x BN)`` reduction and
``(RBLK x CBLOCK)`` copy tiles in the Ascend core unified buffer (~192 KiB
per core; 32x1024 tiles overflowed it with
"ub overflow, requires 2763264 bits while 1572864 bits available"). One
program per job keeps the unified buffer usage tiny (one row's copy tile plus
BN-wide vectors), avoids 2D reductions, and runs the per-node gathers once
per job instead of once per row chunk. The per-row selection and copy rules
are unchanged.
"""

from vllm.triton_utils import tl, triton

# Byte-vector width for the row copies (one row is copied in slices of this
# many bytes; small so the copy tiles stay out of the unified buffer hot
# path).
BYTE_BLOCK = 512


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
    CBLOCK: tl.constexpr = BYTE_BLOCK,
):
    job = tl.program_id(0)
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
    row_of = position % rows
    col_of = position // rows
    page = tl.load(table + req * width + col_of, (node < N) & (col_of < width), other=0)
    on_page = selected & (page == page_id)
    enabled = tl.sum(on_page.to(tl.int32), 0) > 0
    tl.store(PageActive + job, enabled)
    if enabled:
        # Ascending request priority, latest accepted depth within a request.
        priority = (N - req) * (MAX_Q + 1) + depth
        lanes = tl.arange(0, CBLOCK)
        for r in range(0, rows):
            match = on_page & (row_of == r)
            best = tl.max(tl.where(match, priority, -1))
            chosen = tl.max(tl.where(match & (priority == best), node, -1))
            if chosen >= 0:
                src = node_data + chosen * row_bytes
            else:
                src = original + r * row_stride
            for begin in range(0, row_bytes, CBLOCK):
                offset = begin + lanes
                value = tl.load(src + offset, offset < row_bytes, other=0)
                tl.store(Staging + stage_offset + r * row_bytes + offset, value, offset < row_bytes)


@triton.jit
def commit_node_pages(Desc, Staging, Active, CBLOCK: tl.constexpr = BYTE_BLOCK):
    job = tl.program_id(0)
    d = Desc + job * 10
    if tl.load(Active + job):
        rows = tl.load(d + 3)
        target = tl.load(d).to(tl.pointer_type(tl.uint8))
        stride = tl.load(d + 1)
        size = tl.load(d + 2)
        stage = tl.load(d + 8)
        lanes = tl.arange(0, CBLOCK)
        for r in range(0, rows):
            for begin in range(0, size, CBLOCK):
                offset = begin + lanes
                value = tl.load(Staging + stage + r * size + offset, offset < size, other=0)
                tl.store(target + r * stride + offset, value, offset < size)
