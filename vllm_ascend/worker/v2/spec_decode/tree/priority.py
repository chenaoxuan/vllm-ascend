import heapq

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout


class PriorityTreeBuilder(TreeBuilder):
    """Best-first (priority-queue) expansion of shared-depth draft logits."""

    required_backend = "dflash"
    method = "priority"

    def build(
        self,
        draft_logits: torch.Tensor,
        out: TreeLayout,
        *,
        root_token_ids: torch.Tensor | None = None,
        draft_hidden: torch.Tensor | None = None,
    ) -> TreeLayout:
        budget = self.budget
        topk = self.topk
        num_reqs, _, _ = draft_logits.shape
        top_logits, top_token_ids = torch.topk(draft_logits, k=topk, dim=-1)
        log_z = torch.logsumexp(draft_logits, dim=-1, keepdim=True)
        # D2H: priority expand is host-side (numpy + heapq).
        top_log_probs_np = (
            (top_logits - log_z).detach().to(device="cpu", dtype=torch.float32).numpy()
        )
        top_token_ids_np = top_token_ids.detach().to(device="cpu", dtype=torch.long).numpy()

        tokens_np = np.full((num_reqs, budget), -1, dtype=np.int64)
        depths_np = np.zeros((num_reqs, budget), dtype=np.int32)
        parents_np = np.full((num_reqs, budget), -1, dtype=np.int32)
        num_nodes_np = np.zeros((num_reqs,), dtype=np.int32)

        for req_idx in range(num_reqs):
            num_nodes_np[req_idx] = _expand_one_priority_tree(
                top_log_probs_np[req_idx],
                top_token_ids_np[req_idx],
                budget,
                tokens_np[req_idx],
                depths_np[req_idx],
                parents_np[req_idx],
            )

        device = out.tokens.device
        # H2D when ``out`` lives on NPU.
        tokens = torch.from_numpy(tokens_np).to(device=device, dtype=torch.long)
        depths = torch.from_numpy(depths_np).to(device=device)
        parent_ids = torch.from_numpy(parents_np).to(device=device, dtype=torch.long)
        num_nodes = int(num_nodes_np.max()) if num_reqs > 0 else 0

        if num_nodes == 0:
            out.tokens.fill_(-1)
            out.depths.zero_()
            out.parents.fill_(-1)
            out.visibility.zero_()
            out.first_child.fill_(-1)
            out.next_sibling.fill_(-1)
            out.num_nodes.zero_()
            return out

        if np.all(num_nodes_np == num_nodes):
            return finalize_tree_layout(
                out,
                tokens[:, :num_nodes],
                depths[:, :num_nodes],
                parent_ids[:, :num_nodes],
                num_nodes,
            )

        # Per-request sizes differ: finalize each row (pad width is not shared).
        out.tokens.fill_(-1)
        out.depths.zero_()
        out.parents.fill_(-1)
        out.visibility.zero_()
        out.first_child.fill_(-1)
        out.next_sibling.fill_(-1)
        out.num_nodes.zero_()
        for req_idx in range(num_reqs):
            n = int(num_nodes_np[req_idx])
            if n == 0:
                continue
            req_out = TreeLayout(
                tokens=out.tokens[req_idx : req_idx + 1],
                depths=out.depths[req_idx : req_idx + 1],
                parents=out.parents[req_idx : req_idx + 1],
                num_nodes=out.num_nodes[req_idx : req_idx + 1],
                visibility=out.visibility[req_idx : req_idx + 1],
                first_child=out.first_child[req_idx : req_idx + 1],
                next_sibling=out.next_sibling[req_idx : req_idx + 1],
            )
            finalize_tree_layout(
                req_out,
                tokens[req_idx : req_idx + 1, :n],
                depths[req_idx : req_idx + 1, :n],
                parent_ids[req_idx : req_idx + 1, :n],
                n,
            )
        return out


def _expand_one_priority_tree(
    top_log_probs: np.ndarray,
    top_token_ids: np.ndarray,
    budget: int,
    tokens: np.ndarray,
    depths: np.ndarray,
    parents: np.ndarray,
) -> int:
    """Expand one request into slot buffers; returns non-root node count."""
    depth_limit, topk = top_log_probs.shape

    first_logw = float(top_log_probs[0, 0])
    pq: list[tuple[float, tuple[int, ...], int, int, int]] = [
        (-first_logw, (0,), 0, 1, 0),
    ]
    num_nodes = 0

    while pq and num_nodes < budget:
        neg_logw, ranks, parent_id, depth, rank = heapq.heappop(pq)
        logw = -neg_logw
        token_id = int(top_token_ids[depth - 1, rank])
        node_id = num_nodes + 1
        tokens[num_nodes] = token_id
        depths[num_nodes] = depth
        parents[num_nodes] = parent_id
        num_nodes += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = (
                logw - top_log_probs[depth - 1, rank] + top_log_probs[depth - 1, rank + 1]
            )
            heapq.heappush(
                pq,
                (-sibling_logw, sibling_ranks, parent_id, depth, rank + 1),
            )

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + top_log_probs[depth, 0]
            heapq.heappush(
                pq,
                (-child_logw, child_ranks, node_id, depth + 1, 0),
            )

    return num_nodes
