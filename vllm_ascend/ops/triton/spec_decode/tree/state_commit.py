# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-exact accepted-tree state preparation and ordered commitment.

All descriptors and predicates are device resident. Host code supplies only
addresses, shapes and immutable scheduler metadata; kernels never read a
device predicate back to the host. Preparation freezes complete winning pages
before KV compaction, including baseline rows outside the selected delta.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def select_tree_state_kernel(
    pages,
    candidates,
    selected,
    source_rows,
    active,
    PAGE_ROWS: tl.constexpr,
    DELTA_ROWS: tl.constexpr,
):
    page = tl.program_id(0)
    p = pages + page * 7
    base = tl.load(p)
    stride = tl.load(p + 1)
    row_bytes = tl.load(p + 2)
    nrows = tl.load(p + 3)
    begin = tl.load(p + 4)
    end = tl.load(p + 5)
    out = tl.load(p + 6)
    dirty_rows = tl.arange(0, DELTA_ROWS)
    chosen = tl.full((), -1, tl.int64)
    # Runtime iteration avoids specializing/unrolling the candidate count on
    # Ascend. Invocation order, including the first dirty candidate, is exact.
    for candidate in range(begin, end):
        c = candidates + candidate * 8
        owner = tl.load(c)
        enabled = tl.load(selected + owner)
        if enabled and chosen < 0:
            count = tl.load(c + 4)
            dirty = tl.load(c + 5).to(tl.pointer_type(tl.uint8))
            dirty_stride = tl.load(c + 6)
            changed = tl.load(dirty + dirty_rows * dirty_stride, dirty_rows < count, other=0)
            if tl.sum(changed.to(tl.int32), 0) > 0:
                chosen = candidate
    tl.store(active + page, chosen >= 0)
    if chosen >= 0:
        c = candidates + chosen * 8
        source = tl.load(c + 1)
        rows = tl.arange(0, PAGE_ROWS)
        addresses = base + rows.to(tl.int64) * stride
        delta = tl.load(c + 7)
        if delta:
            offsets = tl.load(c + 2).to(tl.pointer_type(tl.int64))
            offset_stride = tl.load(c + 3)
            count = tl.load(c + 4)
            for r in range(count):
                row = tl.load(offsets + r * offset_stride)
                addresses = tl.where(rows == row, source + r * row_bytes, addresses)
        else:
            addresses = source + rows.to(tl.int64) * row_bytes
        tl.store(source_rows + out + rows, addresses, rows < nrows)


@triton.jit
def copy_tree_state_rows_kernel(
    jobs,
    source_rows,
    active,
    staging,
    job_start,
    PREPARE: tl.constexpr,
    WORD_COPY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    job = (tl.program_id(0) + job_start) * 5
    page = tl.load(jobs + job)
    if tl.load(active + page):
        row = tl.load(jobs + job + 1)
        offset = tl.load(jobs + job + 2)
        size = tl.load(jobs + job + 3)
        if WORD_COPY:
            stage_ptr = staging.to(tl.pointer_type(tl.uint64)) + offset // 8
            if PREPARE:
                src_ptr = tl.load(source_rows + row).to(tl.pointer_type(tl.uint64))
                dst_ptr = stage_ptr
            else:
                src_ptr = stage_ptr
                dst_ptr = tl.load(jobs + job + 4).to(tl.pointer_type(tl.uint64))
            size = size // 8
        else:
            stage_ptr = staging + offset
            if PREPARE:
                src_ptr = tl.load(source_rows + row).to(tl.pointer_type(tl.uint8))
                dst_ptr = stage_ptr
            else:
                src_ptr = stage_ptr
                dst_ptr = tl.load(jobs + job + 4).to(tl.pointer_type(tl.uint8))
        # Cast once outside the loop: required by Ascend pointer analysis.
        lanes = tl.arange(0, BLOCK)
        for begin in range(0, size, BLOCK):
            data = tl.load(src_ptr + begin + lanes, begin + lanes < size, other=0)
            tl.store(dst_ptr + begin + lanes, data, begin + lanes < size)
