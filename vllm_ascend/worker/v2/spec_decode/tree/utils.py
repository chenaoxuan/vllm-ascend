import heapq
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TreeLayout:
    """Flattened draft tree for a batch of DFlash draft logits.

    A node is a tree vertex: id 0 is the already-accepted root, ids 1..N are draft nodes. 
    A non-root node is packed at tensor slot i = node i+1(root is not a slot).

    Slot-indexed (non-root): tokens, depths, parents, visibility, num_nodes.
    Node-id-indexed (includes root 0): values in parents, first_child,
    and next_sibling.

    Attributes:
        tokens: [R, budget] token id of non-root node i+1.
        depths: [R, budget] depth of non-root node i+1 (from 1).
        parents: [R, budget] parent node id of non-root node i+1.
        num_nodes: [R] number of valid non-root nodes.
        visibility: [R, budget, budget] whether non-root node i+1 can see
            non-root node j+1 (seeing root is implicit).
        first_child: [R, budget+1] first child node id of node p; -1 if none.
        next_sibling: [R, budget+1] next sibling node id of node c; -1 if none.
    """

    tokens: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor
    num_nodes: torch.Tensor
    visibility: torch.Tensor
    first_child: torch.Tensor
    next_sibling: torch.Tensor


def empty_tree_layout(
    num_reqs: int,
    budget: int,
    device: torch.device | str = "cpu",
) -> TreeLayout:
    """Allocate a padded TreeLayout for a builder to fill in place."""
    return TreeLayout(
        tokens=torch.full((num_reqs, budget), -1, dtype=torch.long, device=device),
        depths=torch.zeros((num_reqs, budget), dtype=torch.int32, device=device),
        parents=torch.full((num_reqs, budget), -1, dtype=torch.int32, device=device),
        num_nodes=torch.zeros((num_reqs,), dtype=torch.int32, device=device),
        visibility=torch.zeros((num_reqs, budget, budget), dtype=torch.bool, device=device),
        first_child=torch.full((num_reqs, budget + 1), -1, dtype=torch.int32, device=device),
        next_sibling=torch.full((num_reqs, budget + 1), -1, dtype=torch.int32, device=device),
    )


def build_trees(
    draft_logits: torch.Tensor,
    budget: int,
    topk: int,
    out: TreeLayout,
) -> TreeLayout:
    """Expand shared-depth DFlash logits into a best-first draft tree.

    Path scores are sums of log-softmax values along the unique token ranks
    chosen at each depth. Results are written into out and returned.

    Args:
        draft_logits: [num_reqs, spec_num, vocab] shared-depth draft logits.
        budget: maximum number of non-root nodes per request.
        topk: number of candidate tokens kept at each depth.
        out: TreeLayout to fill in place.
    """
    num_reqs, _, _ = draft_logits.shape
    top_logits, top_token_ids = torch.topk(draft_logits, k=topk, dim=-1)
    log_z = torch.logsumexp(draft_logits, dim=-1, keepdim=True)
    top_log_probs_np = (top_logits - log_z).detach().to(device="cpu", dtype=torch.float32).numpy()
    top_token_ids_np = top_token_ids.detach().to(device="cpu", dtype=torch.long).numpy()

    tokens_np = np.full((num_reqs, budget), -1, dtype=np.int64)
    depths_np = np.zeros((num_reqs, budget), dtype=np.int32)
    parents_np = np.full((num_reqs, budget), -1, dtype=np.int32)
    num_nodes_np = np.zeros((num_reqs,), dtype=np.int32)
    visibility_np = np.zeros((num_reqs, budget, budget), dtype=np.bool_)
    first_child_np = np.full((num_reqs, budget + 1), -1, dtype=np.int32)
    next_sibling_np = np.full((num_reqs, budget + 1), -1, dtype=np.int32)

    for req_idx in range(num_reqs):
        node_count = _expand_one_tree(
            top_log_probs_np[req_idx],
            top_token_ids_np[req_idx],
            budget,
            tokens_np[req_idx],
            depths_np[req_idx],
            parents_np[req_idx],
            visibility_np[req_idx],
            first_child_np[req_idx],
            next_sibling_np[req_idx],
        )
        num_nodes_np[req_idx] = node_count

    out.tokens.copy_(torch.from_numpy(tokens_np))
    out.depths.copy_(torch.from_numpy(depths_np))
    out.parents.copy_(torch.from_numpy(parents_np))
    out.num_nodes.copy_(torch.from_numpy(num_nodes_np))
    out.visibility.copy_(torch.from_numpy(visibility_np))
    out.first_child.copy_(torch.from_numpy(first_child_np))
    out.next_sibling.copy_(torch.from_numpy(next_sibling_np))
    return out


def _expand_one_tree(
    top_log_probs: np.ndarray,
    top_token_ids: np.ndarray,
    budget: int,
    tokens_out: np.ndarray,
    depths_out: np.ndarray,
    parents_out: np.ndarray,
    visibility_out: np.ndarray,
    first_child_out: np.ndarray,
    next_sibling_out: np.ndarray,
) -> int:
    depth_limit, topk = top_log_probs.shape

    first_logw = float(top_log_probs[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int]] = [
        (-first_logw, (0,), 0, 1, 0),
    ]
    node_count = 0

    while heap and node_count < budget:
        neg_logw, ranks, parent_index, depth, rank = heapq.heappop(heap)
        logw = -neg_logw
        token_id = int(top_token_ids[depth - 1, rank])
        node_index = node_count + 1
        tokens_out[node_count] = token_id
        depths_out[node_count] = depth
        parents_out[node_count] = parent_index
        next_sibling_out[node_index] = first_child_out[parent_index]
        first_child_out[parent_index] = node_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw - top_log_probs[depth - 1, rank] + top_log_probs[depth - 1, rank + 1]
            )
            heapq.heappush(
                heap,
                (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + top_log_probs[depth, 0]
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, node_index, depth + 1, 0),
            )

    for i in range(node_count):
        parent_node_idx = parents_out[i]
        if parent_node_idx > 0:
            visibility_out[i, :i] = visibility_out[parent_node_idx - 1, :i]
        visibility_out[i, i] = True
    return node_count


def _markov_correct_logits(
    draft_model,
    depth_logits: torch.Tensor,
    frontier_tokens: torch.Tensor,
) -> torch.Tensor:
    """Apply the DSpark markov logit-bias correction to one depth's base logits.

    Mirrors the existing Ascend DSpark drafting loop
    (``vllm_ascend/spec_decode/llm_base_proposer.py``): the bias is computed from
    the frontier (previous) token via ``markov_embed`` + ``markov_bias`` and added
    to the base logits.

    Args:
        draft_model: DSpark draft model exposing ``markov_embed`` / ``markov_bias``.
        depth_logits: [R, vocab] uncorrected base logits of this depth.
        frontier_tokens: [R, batch] token ids of the frontier nodes (the "previous"
            tokens), one per frontier node per request.

    Returns:
        [R, batch, vocab] corrected logits, one row per frontier node.
    """
    base = depth_logits.unsqueeze(1)  # [R, 1, vocab]
    markov_emb = draft_model.markov_embed(frontier_tokens)      # [R, batch, markov_rank]
    return base + draft_model.markov_bias(markov_emb)           # [R, batch, vocab]


def build_trees(
    draft_logits: torch.Tensor,
    budget: int,
    topk: int,
    out: TreeLayout,
    draft_model,
    root_token_ids: torch.Tensor,
):
    """
    Build trees from DSpark draft logits, applying the markov logit-bias
    correction at each depth.

    Port of DeepSpec's level-by-level (beam) construction, generalized to a batch
    of requests. All requests share the same tree *structure* (frontier size is
    1 at depth 0 then ``topk`` thereafter, so pool and node counts are identical
    across requests), so everything is kept with a leading [R, ...] batch dim and
    processed in one batched pass. At each depth the frontier's base logits are
    corrected with the draft model's markov bias, the top-``topk`` candidates are
    merged into a per-request pool, and the pool is re-ranked by (depth asc,
    score desc) and truncated to ``budget`` nodes.

    Args:
        draft_logits: [num_reqs, spec_num, vocab] shared-depth draft base logits.
        budget: maximum number of non-root nodes per request.
        topk: number of candidate tokens kept at each depth.
        out: TreeLayout to fill in place.
        draft_model: DSpark draft model exposing ``markov_embed`` / ``markov_bias``.
        root_token_ids: [num_reqs] already-accepted anchor token per request, used
            as the depth-0 "previous" token for the markov bias.
    """
    num_reqs, spec_num, vocab = draft_logits.shape
    k = max(1, min(topk, vocab))
    device = draft_logits.device

    # Frontier = the set of nodes at the current depth. Root has no pool slot.
    frontier_tokens = root_token_ids.unsqueeze(1)  # [R, 1]
    frontier_scores = torch.zeros(num_reqs, 1, dtype=torch.float32, device=device)
    frontier_pools = torch.full((num_reqs, 1), -1, dtype=torch.long, device=device)

    pool_tokens = torch.empty(num_reqs, 0, dtype=torch.long, device=device)
    pool_scores = torch.empty(num_reqs, 0, dtype=torch.float32, device=device)
    pool_parents = torch.empty(num_reqs, 0, dtype=torch.long, device=device)
    pool_depth = torch.empty(num_reqs, 0, dtype=torch.long, device=device)

    for depth in range(spec_num):
        batch = frontier_tokens.size(1)
        step_logits = _markov_correct_logits(draft_model, draft_logits[:, depth], frontier_tokens)  # [R, batch, vocab]
        log_probs = torch.log_softmax(step_logits.float(), dim=-1)
        top_vals, top_ids = log_probs.topk(k, dim=-1)  # [R, batch, k]
        candidate_scores = frontier_scores.unsqueeze(-1) + top_vals  # [R, batch, k]
        num_candidates = batch * k

        pool_tokens = torch.cat([pool_tokens, top_ids.reshape(num_reqs, -1)], dim=-1)
        pool_scores = torch.cat([pool_scores, candidate_scores.reshape(num_reqs, -1)], dim=-1)
        pool_parents = torch.cat([pool_parents, frontier_pools.repeat_interleave(k, dim=-1)], dim=-1)
        pool_depth = torch.cat(
            [pool_depth, torch.full((num_reqs, num_candidates), depth, dtype=torch.long, device=device)],
            dim=-1,
        )

        keep = min(k, num_candidates)
        cand_flat = candidate_scores.reshape(num_reqs, -1)  # [R, num_candidates]
        top_ids_flat = top_ids.reshape(num_reqs, -1)        # [R, num_candidates]
        selected = cand_flat.topk(keep, dim=-1).indices     # [R, keep]
        frontier_tokens = torch.gather(top_ids_flat, 1, selected)
        frontier_scores = torch.gather(cand_flat, 1, selected)
        frontier_pools = (pool_tokens.size(-1) - num_candidates) + selected

    # Re-rank the whole pool by (depth asc, score desc) and keep budget nodes.
    num_pool = pool_tokens.size(-1)
    num_selected = min(int(budget), num_pool)
    order = torch.argsort(pool_depth, dim=-1, stable=True)  # [R, num_pool]
    order = torch.gather(
        order, 1,
        torch.argsort(torch.gather(pool_scores, 1, order), dim=-1, descending=True, stable=True),
    )
    selected = order[:, :num_selected]
    packed = selected.sort(dim=-1).values                   # [R, num_selected]
    num_nodes = num_selected

    remap = torch.zeros(num_reqs, num_pool, dtype=torch.long, device=device)
    node_ids = torch.arange(1, num_nodes + 1, dtype=torch.long, device=device).unsqueeze(0)
    remap.scatter_(1, packed, node_ids.expand(num_reqs, num_nodes))
    pool_parent_packed = torch.gather(pool_parents, 1, packed)
    non_root = pool_parent_packed >= 0
    raw_parent = torch.gather(remap, 1, pool_parent_packed.clamp(min=0))
    parent_indices = torch.where(non_root, raw_parent, torch.zeros_like(raw_parent))  # [R, num_nodes]

    out.tokens[:, :num_nodes] = torch.gather(pool_tokens, 1, packed)
    out.depths[:, :num_nodes] = (torch.gather(pool_depth, 1, packed) + 1).to(torch.int32)
    out.parents[:, :num_nodes] = parent_indices.to(torch.int32)
    out.num_nodes[:] = num_nodes

    # first_child / next_sibling: linked-list, sequential over node slots (batched over R).
    for i in range(num_nodes):
        node_id = i + 1
        parent_id = parent_indices[:, i]  # [R]
        cur_first = torch.gather(out.first_child, 1, parent_id.unsqueeze(1)).squeeze(1)  # [R]
        out.next_sibling[:, node_id] = cur_first
        out.first_child.scatter_(
            1, parent_id.unsqueeze(1),
            torch.full((num_reqs, 1), node_id, dtype=torch.int32, device=device),
        )

    # visibility: a node sees its parent's visible set plus itself (batched over R).
    for i in range(num_nodes):
        parent_id = parent_indices[:, i]  # [R]
        if i > 0:
            src_idx = (parent_id - 1).clamp(min=0)  # [R]
            parent_vis = torch.gather(
                out.visibility[:, :num_nodes, :i],
                1,
                src_idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, i),
            ).squeeze(1)  # [R, i]
            out.visibility[:, i, :i] = torch.where(
                (parent_id > 0).unsqueeze(-1), parent_vis, out.visibility[:, i, :i]
            )
        out.visibility[:, i, i] = True

    return out
