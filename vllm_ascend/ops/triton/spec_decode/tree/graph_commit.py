# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve graph-stable cache descriptors from live request prefixes/tables."""

from vllm.triton_utils import tl, triton


@triton.jit
def update_commit_descriptors(Base, Prefix, Output):
    job = tl.program_id(0)
    b = Base + job * 12
    original = tl.load(b)
    stride = tl.load(b + 1)
    size = tl.load(b + 2)
    rows = tl.load(b + 3)
    nodes = tl.load(b + 4)
    table_address = tl.load(b + 5)
    width = tl.load(b + 6)
    ratio = tl.load(b + 7)
    page_stride = tl.load(b + 8)
    req = tl.load(b + 9)
    offset = tl.load(b + 10)
    stage = tl.load(b + 11)
    column = tl.load(Prefix + req) // ratio // rows + offset
    table = table_address.to(tl.pointer_type(tl.int32))
    page = tl.load(table + req * width + column, (column >= 0) & (column < width), other=0)
    d = Output + job * 10
    tl.store(d, original + tl.maximum(page, 0).to(tl.int64) * page_stride)
    tl.store(d + 1, stride)
    tl.store(d + 2, size)
    tl.store(d + 3, rows)
    tl.store(d + 4, nodes)
    tl.store(d + 5, table_address)
    tl.store(d + 6, width)
    tl.store(d + 7, tl.where(page > 0, page, -1))
    tl.store(d + 8, stage)
    tl.store(d + 9, ratio)
