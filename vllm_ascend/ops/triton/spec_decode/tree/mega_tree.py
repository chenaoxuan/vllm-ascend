# SPDX-License-Identifier: Apache-2.0
"""Best-first tree selection from normalized DFlash2 transition rows."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _load_row(ptr, lanes, prefix_score, K: tl.constexpr):
    score = tl.load(ptr + lanes, mask=lanes < K, other=0.0).to(tl.float32)
    total = prefix_score + score
    # Use a distinct inactive sentinel so a real -inf-probability edge can
    # still fill the available budget, with its score ordered below finite rows.
    lowest = -3.4028234663852886e38
    total = tl.where(total == total, tl.maximum(total, lowest), lowest)
    return tl.where(lanes < K, total, -float("inf"))


@triton.jit
def _search_kernel(
    candidate_ids,
    child_slots,
    child_log_probs,
    out_tokens,
    out_depths,
    out_parents,
    out_first,
    out_next,
    out_visibility,
    out_count,
    stride_tokens_r,
    D: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    BUDGET: tl.constexpr,
    BUDGET_LANES: tl.constexpr,
    BUDGET_COLS: tl.constexpr,
    K_PAD: tl.constexpr,
    STORE_NUM_NODES: tl.constexpr,
):
    req = tl.program_id(0)
    id_base = req * D * C
    score_base = id_base * K
    lanes = tl.arange(0, BUDGET_LANES)
    children = tl.arange(0, K_PAD)
    neg_inf = -float("inf")
    root_scores = _load_row(child_log_probs + score_base, children, 0.0, K)
    root_best = tl.max(root_scores, 0)
    root_pick = tl.min(tl.where(root_scores == root_best, children, K_PAD), 0)
    queue = tl.where(lanes[:, None] == 0, root_scores[None, :], neg_inf)
    pending = tl.where(lanes == 0, root_best, neg_inf)
    next_child = tl.where(lanes == 0, root_pick, 0)
    source_slot = tl.full((BUDGET_LANES,), 0, tl.int32)
    depths = tl.full((BUDGET_LANES,), 0, tl.int32)
    parents = tl.full((BUDGET_LANES,), -1, tl.int32)
    tokens = tl.full((BUDGET_LANES,), -1, tl.int64)
    first_child = tl.full((BUDGET_LANES,), -1, tl.int32)
    next_sibling = tl.full((BUDGET_LANES,), -1, tl.int32)
    visible_lo = tl.full((BUDGET_LANES,), 0, tl.uint32)
    visible_hi = tl.full((BUDGET_LANES,), 0, tl.uint32)
    count = 0

    # Each active parent keeps its best remaining child. Promote the global
    # winner, advance that parent, and open a row for the new prefix.
    for new_node in tl.range(1, BUDGET + 1):
        best = tl.max(pending, 0)
        if best > neg_inf:
            parent = tl.min(tl.where(pending == best, lanes, BUDGET_LANES), 0)
            own_entry = lanes == parent
            depth = tl.sum(tl.where(own_entry, depths, 0), 0)
            child = tl.sum(tl.where(own_entry, next_child, 0), 0)
            old_first = tl.sum(tl.where(own_entry, first_child, 0), 0)

            qflat = tl.reshape(queue, (BUDGET_LANES * K_PAD,))
            sibling_row = tl.gather(qflat, parent * K_PAD + children, axis=0)
            sibling_row = tl.where(children == child, neg_inf, sibling_row)
            sibling_score = tl.max(sibling_row, 0)
            sibling_pick = tl.min(tl.where(sibling_row == sibling_score, children, K_PAD), 0)
            queue = tl.where(own_entry[:, None], sibling_row[None, :], queue)
            pending = tl.where(own_entry, sibling_score, pending)
            next_child = tl.where(own_entry, sibling_pick, next_child)

            row = score_base + (depth * C + tl.sum(tl.where(own_entry, source_slot, 0), 0)) * K
            child_slot = tl.load(child_slots + row + child).to(tl.int32)
            child_depth = depth + 1
            new_entry = lanes == new_node
            if child_depth < D:
                row_offset = score_base + (child_depth * C + child_slot) * K
                child_scores = _load_row(child_log_probs + row_offset, children, best, K)
                child_best = tl.max(child_scores, 0)
                child_pick = tl.min(tl.where(child_scores == child_best, children, K_PAD), 0)
                queue = tl.where(new_entry[:, None], child_scores[None, :], queue)
                pending = tl.where(new_entry, child_best, pending)
                next_child = tl.where(new_entry, child_pick, next_child)

            token = tl.load(candidate_ids + id_base + depth * C + child_slot)
            source_slot = tl.where(new_entry, child_slot, source_slot)
            depths = tl.where(new_entry, child_depth, depths)
            parents = tl.where(new_entry, parent, parents)
            tokens = tl.where(new_entry, token, tokens)
            first_child = tl.where(own_entry, new_node, first_child)
            next_sibling = tl.where(new_entry, old_first, next_sibling)
            parent_lo = tl.sum(tl.where(own_entry, visible_lo, 0), 0).to(tl.uint32)
            parent_hi = tl.sum(tl.where(own_entry, visible_hi, 0), 0).to(tl.uint32)
            bit = tl.full((), 1, tl.uint32) << ((new_node - 1) % 32)
            visible_lo = tl.where(new_entry, parent_lo | tl.where(new_node <= 32, bit, 0), visible_lo)
            visible_hi = tl.where(new_entry, parent_hi | tl.where(new_node > 32, bit, 0), visible_hi)
            count += 1

    nodes = (lanes > 0) & (lanes <= BUDGET)
    tl.store(out_tokens + req * stride_tokens_r + lanes - 1, tokens, nodes)
    tl.store(out_depths + req * BUDGET + lanes - 1, depths, nodes)
    tl.store(out_parents + req * BUDGET + lanes - 1, parents, nodes)
    tl.store(out_first + req * (BUDGET + 1) + lanes, first_child, lanes <= BUDGET)
    tl.store(out_next + req * (BUDGET + 1) + lanes, next_sibling, lanes <= BUDGET)
    columns = tl.arange(0, BUDGET_COLS)
    masks = tl.where(columns[None, :] < 32, visible_lo[:, None], visible_hi[:, None])
    visible = (masks >> (columns[None, :] % 32)) & 1
    tl.store(
        out_visibility + (req * BUDGET + lanes[:, None] - 1) * BUDGET + columns[None, :],
        visible,
        nodes[:, None] & (columns[None, :] < BUDGET),
    )
    if STORE_NUM_NODES:
        tl.store(out_count + req, count)


def best_first_tree(
    candidate_ids,
    child_slots,
    child_log_probs,
    out_tokens,
    out_depths,
    out_parents,
    out_first_child,
    out_next_sibling,
    out_visibility,
    *,
    out_num_nodes=None,
):
    """Select global top prefixes under the normalized transition probabilities.

    Candidate IDs have shape [R,D,C]. Child slots and nonpositive log
    probabilities have shape [R,D,C,K] with each row sorted by descending
    score. Root uses parent row zero. Each prefix is distinct even when its
    transition row (depth, last token slot) is shared with another prefix.

    B is inferred from [R,B] outputs. K=min(B,C) suffices for exact global
    top B (a lower-ranked sibling already has B at least as good siblings).
    Smaller K searches only the supplied edges. Equal scores use the existing
    selected parent ID (smaller first) and then the supplied sibling order.
    Pruned rows are padded with the usual TreeLayout sentinels. The selected
    nodes are in topological order, with potentially interleaved depths.
    """
    num_reqs, depth, width = candidate_ids.shape
    batch, budget = out_tokens.shape
    children = child_slots.shape[-1]
    if batch != num_reqs:
        raise ValueError("candidate and output batch sizes must match")
    if not (1 <= depth <= 32 and 1 <= children <= width <= 64 and 1 <= budget <= 64):
        raise ValueError("expected 1<=D<=32, 1<=K<=C<=64, 1<=B<=64")
    if child_slots.shape != (num_reqs, depth, width, children) or child_log_probs.shape != child_slots.shape:
        raise ValueError("transition tensors must have the same [R,D,C,K] shape")
    if out_depths.shape != out_tokens.shape or out_parents.shape != out_tokens.shape:
        raise ValueError("tokens, depths and parents must have shape [R,B]")
    if out_next_sibling.shape != (num_reqs, budget + 1) or out_first_child.shape != out_next_sibling.shape:
        raise ValueError("sibling and first-child outputs must have shape [R,B+1]")
    if out_visibility.shape != (num_reqs, budget, budget):
        raise ValueError("visibility must have shape [R,B,B]")
    buffers = (
        candidate_ids,
        child_slots,
        child_log_probs,
        out_depths,
        out_parents,
        out_first_child,
        out_next_sibling,
        out_visibility,
    )
    if out_num_nodes is not None:
        if out_num_nodes.shape != (num_reqs,):
            raise ValueError("node counts must have shape [R]")
        buffers += (out_num_nodes,)
    if any(not tensor.is_contiguous() for tensor in buffers) or out_tokens.stride(1) != 1:
        raise ValueError("best-first tree requires contiguous buffers and packed token rows")
    if num_reqs == 0:
        return
    grid = (num_reqs,)
    visibility = out_visibility.view(torch.int8) if out_visibility.dtype == torch.bool else out_visibility
    _search_kernel[grid](
        candidate_ids,
        child_slots,
        child_log_probs,
        out_tokens,
        out_depths,
        out_parents,
        out_first_child,
        out_next_sibling,
        visibility,
        out_tokens if out_num_nodes is None else out_num_nodes,
        out_tokens.stride(0),
        depth,
        width,
        children,
        budget,
        triton.next_power_of_2(budget + 1),
        triton.next_power_of_2(budget),
        triton.next_power_of_2(children),
        out_num_nodes is not None,
    )
