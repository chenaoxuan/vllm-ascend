from dataclasses import dataclass

import torch


@dataclass
class TreeLayout:
    """Flattened draft tree for a batch of parallel-draft logits.

    Naming:
        node_id: vertex id including root (root=0, draft nodes 1..N).
        slot: non-root layout index ``0..budget-1`` for node ``slot+1``.
        parent_id: parent ``node_id`` of a non-root node.
        num_nodes: number of valid non-root nodes.
        spec_num: draft depth dimension of ``draft_logits``.

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
        visibility=torch.zeros(
            (num_reqs, budget, budget), dtype=torch.bool, device=device
        ),
        first_child=torch.full(
            (num_reqs, budget + 1), -1, dtype=torch.int32, device=device
        ),
        next_sibling=torch.full(
            (num_reqs, budget + 1), -1, dtype=torch.int32, device=device
        ),
    )


def finalize_tree_layout(
    out: TreeLayout,
    tokens: torch.Tensor,
    depths: torch.Tensor,
    parent_ids: torch.Tensor,
    num_nodes: int,
) -> TreeLayout:
    """Write selected nodes into ``out`` and fill sibling links + visibility.

    Args:
        out: destination layout (cleared then filled).
        tokens: [R, num_nodes] token ids of kept non-root nodes.
        depths: [R, num_nodes] depths (from 1).
        parent_ids: [R, num_nodes] parent node ids (0 = root).
        num_nodes: number of non-root nodes (same for every request in the batch).
    """
    num_reqs = tokens.shape[0]
    device = tokens.device

    out.tokens.fill_(-1)
    out.depths.zero_()
    out.parents.fill_(-1)
    out.visibility.zero_()
    out.first_child.fill_(-1)
    out.next_sibling.fill_(-1)

    out.tokens[:, :num_nodes] = tokens.to(out.tokens.dtype)
    out.depths[:, :num_nodes] = depths.to(out.depths.dtype)
    out.parents[:, :num_nodes] = parent_ids.to(out.parents.dtype)
    out.num_nodes[:] = num_nodes

    for slot in range(num_nodes):
        node_id = slot + 1
        parent_id = parent_ids[:, slot]
        cur_first = torch.gather(
            out.first_child[:, : num_nodes + 1], 1, parent_id.unsqueeze(1)
        ).squeeze(1)
        out.next_sibling[:, node_id] = cur_first
        out.first_child.scatter_(
            1,
            parent_id.unsqueeze(1),
            torch.full((num_reqs, 1), node_id, dtype=torch.int32, device=device),
        )

    for slot in range(num_nodes):
        parent_id = parent_ids[:, slot]
        if slot > 0:
            src_idx = (parent_id - 1).clamp(min=0)
            parent_vis = torch.gather(
                out.visibility[:, :num_nodes, :slot],
                1,
                src_idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, slot),
            ).squeeze(1)
            out.visibility[:, slot, :slot] = torch.where(
                (parent_id > 0).unsqueeze(-1),
                parent_vis,
                out.visibility[:, slot, :slot],
            )
        out.visibility[:, slot, slot] = True
    return out
