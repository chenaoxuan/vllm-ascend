from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TreeLayout:
    """Flattened draft tree for a batch of DFlash draft logits.

    Index 0 of ``parents`` / ``visibility`` is the already-accepted root and is
    not stored in ``tokens``. ``tokens[r, i]`` is tree node ``i + 1``.
    """

    tokens: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor
    num_nodes: torch.Tensor
    visibility: torch.Tensor
    child_maps: list[list[dict[int, int]]] = field(default_factory=list)


def build_trees(
    draft_logits: torch.Tensor,
    budget: int,
) -> TreeLayout:
    """Expand shared-depth DFlash logits into a best-first draft tree.

    ``draft_logits`` has shape ``[num_reqs, depth, vocab]``. ``budget`` is the
    maximum number of non-root nodes per request. Path scores are sums of
    log-softmax values along the unique token ranks chosen at each depth.
    """
    if draft_logits.ndim != 3:
        raise ValueError(
            f"draft_logits must have shape [num_reqs, depth, vocab], got {tuple(draft_logits.shape)}"
        )
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")

    num_reqs, depth, vocab_size = draft_logits.shape
    device = draft_logits.device
    max_len = budget + 1

    tokens = torch.zeros((num_reqs, budget), dtype=torch.long, device=device)
    depths = torch.zeros((num_reqs, budget), dtype=torch.int32, device=device)
    parents = torch.full((num_reqs, max_len), -1, dtype=torch.int32, device=device)
    num_nodes = torch.ones((num_reqs,), dtype=torch.int32, device=device)
    visibility = torch.zeros((num_reqs, max_len, max_len), dtype=torch.bool, device=device)
    visibility[:, 0, 0] = True
    child_maps: list[list[dict[int, int]]] = [[{}] for _ in range(num_reqs)]

    if budget == 0 or depth == 0 or vocab_size == 0 or num_reqs == 0:
        return TreeLayout(tokens, depths, parents, num_nodes, visibility, child_maps)

    topk = min(budget, vocab_size)
    logits_f = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits_f, k=topk, dim=-1)
    log_z = torch.logsumexp(logits_f, dim=-1, keepdim=True)
    top_log_probs_np = (top_logits - log_z).detach().to(device="cpu", dtype=torch.float32).numpy()
    top_token_ids_np = top_token_ids.detach().to(device="cpu", dtype=torch.long).numpy()

    tokens_np = np.zeros((num_reqs, budget), dtype=np.int64)
    depths_np = np.zeros((num_reqs, budget), dtype=np.int32)
    parents_np = np.full((num_reqs, max_len), -1, dtype=np.int32)
    num_nodes_np = np.ones((num_reqs,), dtype=np.int32)
    visibility_np = np.zeros((num_reqs, max_len, max_len), dtype=np.bool_)
    visibility_np[:, 0, 0] = True

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
        num_nodes_np[req_idx] = 1 + node_count
        child_maps[req_idx] = req_child_maps

    tokens.copy_(torch.from_numpy(tokens_np), non_blocking=False)
    depths.copy_(torch.from_numpy(depths_np), non_blocking=False)
    parents.copy_(torch.from_numpy(parents_np), non_blocking=False)
    num_nodes.copy_(torch.from_numpy(num_nodes_np), non_blocking=False)
    visibility.copy_(torch.from_numpy(visibility_np), non_blocking=False)
    return TreeLayout(tokens, depths, parents, num_nodes, visibility, child_maps)


def _expand_one_tree(
    top_log_probs: np.ndarray,
    top_token_ids: np.ndarray,
    budget: int,
    tokens_out: np.ndarray,
    depths_out: np.ndarray,
    parents_out: np.ndarray,
    visibility_out: np.ndarray,
) -> tuple[int, list[dict[int, int]]]:
    depth_limit = int(top_log_probs.shape[0])
    topk = int(top_log_probs.shape[1])
    child_maps: list[dict[int, int]] = [dict()]
    if depth_limit == 0 or topk == 0:
        return 0, child_maps

    first_logw = float(top_log_probs[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [
        (-first_logw, (0,), 0, 1, 0, first_logw),
    ]
    parents_out[0] = -1
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)
        token_id = int(top_token_ids[depth - 1, rank])
        current_index = node_count + 1
        tokens_out[node_count] = token_id
        depths_out[node_count] = depth
        parents_out[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw - float(top_log_probs[depth - 1, rank]) + float(top_log_probs[depth - 1, rank + 1])
            )
            heapq.heappush(
                heap,
                (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs[depth, 0])
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw),
            )

    current_length = 1 + node_count
    visibility_out[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_out[index])
        visibility_out[index, :index] = visibility_out[parent_index, :index]
        visibility_out[index, index] = True
    return node_count, child_maps
