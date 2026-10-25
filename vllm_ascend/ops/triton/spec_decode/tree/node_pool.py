# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-only ancestor addressing for a flat tree node pool."""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _ancestors_kernel(Parents, Depths, Counts, Req, Qsl, Ancestors, Visible,
                      NodeDepth, Hidden, Parent, PWIDTH: tl.constexpr,
                      MAX_DEPTH: tl.constexpr, N: tl.constexpr,
                      BLOCK: tl.constexpr):
    node = tl.program_id(0)
    req = tl.load(Req + node)
    root = tl.load(Qsl + req)
    end = tl.load(Qsl + req + 1)
    local = node - root
    count = tl.minimum(tl.load(Counts + req), end - root - 1)
    live = (local > 0) & (local <= count)
    slot = tl.minimum(tl.maximum(local - 1, 0), PWIDTH - 1)
    depth = tl.load(Depths + req * PWIDTH + slot)
    depth = tl.where(live, tl.minimum(tl.minimum(tl.maximum(depth, 1), MAX_DEPTH), local), 0)
    parent = tl.load(Parents + req * PWIDTH + slot)
    parent = tl.where(live, tl.minimum(tl.maximum(parent, 0), tl.maximum(local - 1, 0)), 0)
    tl.store(NodeDepth + node, depth)
    tl.store(Hidden + node, tl.where(live, node, root))
    tl.store(Parent + node, root + parent)
    cols = tl.arange(0, BLOCK)
    ancestors = tl.full((BLOCK,), root, tl.int32)
    cursor = tl.where(live, local, 0)
    for offset in range(MAX_DEPTH):
        column = tl.maximum(depth - offset, 0)
        ancestors = tl.where((cols == column) & (column > 0), root + cursor, ancestors)
        p = tl.load(Parents + req * PWIDTH + tl.minimum(tl.maximum(cursor - 1, 0), PWIDTH - 1))
        cursor = tl.where(column > 0, tl.minimum(tl.maximum(p, 0), tl.maximum(cursor - 1, 0)), 0)
    tl.store(Ancestors + node * (MAX_DEPTH + 1) + cols, ancestors, cols <= MAX_DEPTH)
    tl.store(Visible + node * (MAX_DEPTH + 1) + cols, cols <= depth, cols <= MAX_DEPTH)


def build_ancestors(parents, depths, counts, req_ids, qsl, max_depth):
    """Return device ancestor IDs and validity; host supplies only capacities.

    Each row contains the root followed by its ancestors through the node.
    Padding uses a safe root address and a false visibility bit.
    """
    n = req_ids.numel()
    ancestors = torch.empty((n, max_depth + 1), dtype=torch.long, device=parents.device)
    visible = torch.empty_like(ancestors, dtype=torch.bool)
    node_depth = torch.empty((n,), dtype=torch.long, device=parents.device)
    hidden = torch.empty_like(node_depth)
    parent = torch.empty_like(node_depth)
    _ancestors_kernel[(n,)](
        parents, depths, counts, req_ids, qsl, ancestors, visible, node_depth,
        hidden, parent, parents.shape[1], max_depth, n,
        triton.next_power_of_2(max_depth + 1),
    )
    return ancestors, visible, node_depth, hidden, parent
