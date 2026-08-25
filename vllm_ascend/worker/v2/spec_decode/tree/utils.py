from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TreeLayout:
    """Flattened draft tree for a batch of DFlash draft logits.

    A node is a tree vertex: id 0 is the already-accepted root, ids 1..N are draft nodes. 
    A non-root node is packed at tensor slot i = node i+1(root is not a slot).

    Slot-indexed (non-root): tokens, depths, parents, visibility, num_nodes.
    Node-id-indexed (includes root 0): values in parents and keys of child_maps.

    Attributes:
        tokens: [R, budget] token id of non-root node i+1.
        depths: [R, budget] depth of non-root node i+1 (from 1).
        parents: [R, budget] parent node id of non-root node i+1.
        num_nodes: [R] number of valid non-root nodes.
        visibility: [R, budget, budget] whether non-root node i+1 can see
            non-root node j+1 (seeing root is implicit).
        child_maps: per request, child_maps[p][token] = child, where p and
            child are node ids.
    """

    tokens: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor
    num_nodes: torch.Tensor
    visibility: torch.Tensor
    child_maps: list[list[dict[int, int]]] = field(default_factory=list)


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
        child_maps=[[{}] for _ in range(num_reqs)],
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
    # tree mask among non-root nodes; seeing root is implicit
    visibility_np = np.zeros((num_reqs, budget, budget), dtype=np.bool_)

    for req_idx in range(num_reqs):
        node_count, req_child_maps = _expand_one_tree(
            top_log_probs_np[req_idx],
            top_token_ids_np[req_idx],
            budget,
            tokens_np[req_idx],
            depths_np[req_idx],
            parents_np[req_idx],
            visibility_np[req_idx],
        )
        num_nodes_np[req_idx] = node_count
        out.child_maps[req_idx] = req_child_maps

    out.tokens.copy_(torch.from_numpy(tokens_np), non_blocking=False)
    out.depths.copy_(torch.from_numpy(depths_np), non_blocking=False)
    out.parents.copy_(torch.from_numpy(parents_np), non_blocking=False)
    out.num_nodes.copy_(torch.from_numpy(num_nodes_np), non_blocking=False)
    out.visibility.copy_(torch.from_numpy(visibility_np), non_blocking=False)
    return out


def _expand_one_tree(
    top_log_probs: np.ndarray,
    top_token_ids: np.ndarray,
    budget: int,
    tokens_out: np.ndarray,
    depths_out: np.ndarray,
    parents_out: np.ndarray,
    visibility_out: np.ndarray,
) -> tuple[int, list[dict[int, int]]]:
    depth_limit, topk = top_log_probs.shape
    child_maps: list[dict[int, int]] = [dict()]

    # start from the argmax token in pos 0
    first_logw = float(top_log_probs[0, 0])
    # heap: (-logw, ranks_along_path, parent_index, current_depth, current_rank). -logw is the min-heap key.
    # depth from 1 to sepc_num (depth=0 means the root), but rank from 0 to topk-1
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
        child_maps.append(dict())
        child_maps[parent_index][token_id] = node_index
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
    return node_count, child_maps
