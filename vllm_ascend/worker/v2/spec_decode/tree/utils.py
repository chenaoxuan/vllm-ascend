import heapq
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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


def build_heap_trees(
    draft_logits: torch.Tensor,
    budget: int,
    topk: int,
    out: TreeLayout,
) -> TreeLayout:
    """Expand shared-depth DFlash logits into a heap (best-first) draft tree.

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
        node_count = _expand_one_heap_tree(
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


def _expand_one_heap_tree(
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


# DARTree Codes
def _select_topb_prefix_tree_tensor(
    path_scores: torch.Tensor,
    depths: torch.Tensor,
    budget: int,
    depth_bonus: float,
) -> torch.Tensor:
    """Select the global Top-B supertree nodes, batched over requests.

    Mirrors ``select_topb_prefix_tree_tensor`` in the DARTree reference. The
    path scores are prefix-monotone (they never increase along the tree) and
    ``depth_bonus <= 0`` keeps the bonus-adjusted scores monotone, so the
    global Top-B set is prefix-closed (every selected node's parent is also
    selected).

    Args:
        path_scores: [R, node_count] path score per non-root supertree node.
        depths: [R, node_count] depth per non-root supertree node.
        budget: number of non-root nodes to keep (per request).
        depth_bonus: per-depth score bonus favoring shallower nodes (<= 0).

    Returns:
        [R, budget] ascending node ids (1-based, column offsets into the
        ``path_scores`` / ``depths`` tensors) of the kept non-root nodes.
    """
    node_count = int(depths.shape[1])
    budget = max(0, min(int(budget), node_count))
    if budget <= 0:
        return torch.empty(
            (path_scores.shape[0], 0), dtype=torch.long, device=path_scores.device
        )
    candidate_scores = path_scores.float() + float(depth_bonus) * depths.float()
    selected = torch.topk(
        candidate_scores, k=budget, largest=True, sorted=False, dim=-1
    ).indices  # [R, budget] 0-based column offsets into non-root nodes
    selected_ids = selected + 1
    return torch.sort(selected_ids, dim=-1).values


class DominoCorrectionScorer:
    """Domino correction head scorer for DARTree (reference CPU path).

    Wraps the draft model's ``embed_proj`` correction MLP and ``prefix_gru``,
    together with the target model's token embedding, into fixed-weight
    projections used to bias the base logits during DARTree construction. See
    ``DominoCorrectionScorer`` in ``E:/repos/DARTree/utils/correction.py``.
    """

    def __init__(self, draft_model, target_model, hidden_dim=None):
        fc1, self.middle, fc2 = self._unwrap_correction_mlp(draft_model.embed_proj)
        gru = draft_model.prefix_gru
        if gru.num_layers != 1 or gru.bidirectional:
            raise ValueError(
                "Domino correction requires a single-layer unidirectional prefix_gru"
            )
        self.gru_hidden_dim = int(gru.hidden_size)
        if hidden_dim is None:
            hidden_dim = fc1.in_features - self.gru_hidden_dim
        self.hidden_dim = int(hidden_dim)
        if fc1.in_features != self.hidden_dim + self.gru_hidden_dim:
            raise ValueError(
                "Domino correction MLP input dim mismatch: got "
                f"{fc1.in_features}, expected {self.hidden_dim + self.gru_hidden_dim}"
            )
        # Split the first correction linear into the (z, s) branches.
        self.w_z = fc1.weight[:, :self.hidden_dim].detach().contiguous()  # [E, H]
        self.w_s = fc1.weight[:, self.hidden_dim:].detach().contiguous()  # [E, G]
        self.fc1_bias = fc1.bias.detach() if fc1.bias is not None else None
        self.fc2_weight = fc2.weight.detach().contiguous()  # [V, E]
        self.fc2_bias = fc2.bias.detach() if fc2.bias is not None else None
        # GRU weights.
        self.gru_w_ih = gru.weight_ih_l0.detach().contiguous()  # [3G, H]
        self.gru_w_hh = gru.weight_hh_l0.detach().contiguous()  # [3G, G]
        self.gru_b_ih = gru.bias_ih_l0.detach() if gru.bias else None
        self.gru_b_hh = gru.bias_hh_l0.detach() if gru.bias else None
        # Precompute token embedding @ W_ih.T into a GRU-input lookup table.
        embed_weight = target_model.model.embed_tokens.weight  # [V, H]
        self._gru_input_proj_table = F.linear(
            embed_weight, self.gru_w_ih, self.gru_b_ih
        ).contiguous()  # [V, 3G]

    @staticmethod
    def _unwrap_correction_mlp(embed_proj):
        """Split ``embed_proj = Sequential(Linear, ..., Linear)`` into (fc1, middle, fc2)."""
        if not isinstance(embed_proj, nn.Sequential):
            raise TypeError(
                "Domino correction requires draft_model.embed_proj to be nn.Sequential"
            )
        modules = list(embed_proj.children())
        if len(modules) < 2:
            raise ValueError("embed_proj must contain at least two layers")
        if not isinstance(modules[0], nn.Linear) or not isinstance(modules[-1], nn.Linear):
            raise ValueError("embed_proj first/last layers must be nn.Linear")
        fc1 = modules[0]
        fc2 = modules[-1]
        middle = (
            nn.Sequential(*modules[1:-1]) if len(modules) > 2 else nn.Identity()
        )
        return fc1, middle, fc2

    def project_z(self, parallel_hiddens: torch.Tensor) -> torch.Tensor:
        """Project draft backbone hidden states to the correction latent (z branch)."""
        return F.linear(parallel_hiddens, self.w_z, self.fc1_bias)

    def update_hidden(
        self, token_ids: torch.Tensor, h_state: torch.Tensor
    ) -> torch.Tensor:
        """Advance the GRU hidden state by one token."""
        token_ids = token_ids.reshape(-1)
        gi = self._gru_input_proj_table.index_select(0, token_ids)  # [B, 3G]
        gh = F.linear(h_state, self.gru_w_hh, self.gru_b_hh)  # [B, 3G]
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h_state


def _dar_corrected_candidates(
    scorer,
    z: torch.Tensor,
    parent_hidden: torch.Tensor,
    candidate_vals: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_weight: torch.Tensor,
    candidate_bias: torch.Tensor | None,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a parent batch's candidate tokens with the Domino correction head.

    Implements the correction of ``DominoCorrectionScorer``
    (``utils/correction.py`` in the DARTree reference): the base logits of the
    candidate tokens are adjusted by the ``embed_proj`` MLP bias
    ``SiLU(w_z@z + w_s@s) @ W2 + b2``, where ``z`` is the per-depth draft
    feature (shared by all parents of a depth) and ``s`` is the per-parent GRU
    hidden state. Scores are returned as softmax-normalized log-probs.

    Args:
        scorer: correction scorer exposing ``w_s``, ``middle``, ``fc2_weight``
            and ``fc2_bias`` (mirrors ``DominoCorrectionScorer``).
        z: [R, mid] per-request draft feature for this depth.
        parent_hidden: [R, W, gru_hidden] GRU hidden state per parent.
        candidate_vals: [R, C] base logits of the candidate tokens.
        candidate_ids: [R, C] candidate token ids.
        candidate_weight: [R, C, mid] ``fc2`` rows gathered by candidate id.
        candidate_bias: [R, C] or None, ``fc2`` bias rows for the candidates.
        k: number of top candidates to keep per parent.

    Returns:
        (top_scores, sel_ids): [R, W, k] softmax-normalized log-prob scores and
        [R, W, k] selected candidate token ids.
    """
    s_proj = F.linear(parent_hidden, scorer.w_s, None)  # [R, W, mid]
    mid = scorer.middle(z.unsqueeze(1) + s_proj)  # [R, W, mid]
    bias = torch.einsum("rwm,rcm->rwc", mid, candidate_weight)  # [R, W, C]
    if candidate_bias is not None:
        bias = bias + candidate_bias.unsqueeze(1)
    candidate_logits = candidate_vals.unsqueeze(1).to(bias.dtype) + bias  # [R, W, C]
    candidate_logits = candidate_logits.float()  # topk / logsumexp in fp32
    top_vals, top_ids = torch.topk(candidate_logits, k=k, dim=-1)  # [R, W, k]
    log_z = torch.logsumexp(candidate_logits, dim=-1, keepdim=True)  # [R, W, 1]
    top_scores = top_vals - log_z  # [R, W, k]
    width = parent_hidden.size(1)
    cand_ids = candidate_ids.unsqueeze(1).expand(-1, width, -1)  # [R, W, C]
    sel_ids = torch.gather(cand_ids, 2, top_ids)  # [R, W, k]
    return top_scores, sel_ids


def build_multi_order_trees(
    draft_logits: torch.Tensor,
    budget: int,
    topk: int,
    out: TreeLayout,
    *,
    depth_bonus: float = -0.2,
    supertree_width: int | None = None,
    pruned: bool = True,
    draft_model=None,
    target_model=None,
    draft_hidden: torch.Tensor | None = None,
    root_token_ids: torch.Tensor | None = None,
    prefix_len: int = 0,
    candidate_vocab_size: int | None = None,
    correction_scorer: DominoCorrectionScorer | None = None,
) -> TreeLayout:
    """Expand base logits into DARTrees, batched over requests.

    Implements the DARTree supertree construction and global Top-B pruning
    (see ``build_dartree_supertree`` in the DARTree reference). When
    ``draft_model`` and ``target_model`` are provided, candidate tokens are
    scored with the Domino correction head (``embed_proj`` MLP bias on top of
    the base logits, driven by the per-depth draft feature and a per-parent GRU
    hidden state); otherwise the tree is grown from the ``log_softmax`` of
    ``draft_logits`` directly. All requests are processed together with tensor
    ops; the supertree has the same shape for every request, so construction has
    no per-request loop. The only Python loops are the node-level
    linked-list / visibility fills, each batched over all requests.

    Path scores are sums of per-node log-probs, which are prefix-monotone (a
    node never outranks its parent); this keeps the Top-B prune prefix-closed.
    With ``pruned=True``, keep ``depth_bonus <= 0`` so deeper nodes are not
    favored by the prune.

    Args:
        draft_logits: [num_reqs, spec_num, vocab] base logits per draft depth.
        budget: maximum number of non-root nodes per request (post-prune).
        topk: candidate tokens kept per parent at each depth (``expansion_k``).
        out: TreeLayout to fill in place.
        depth_bonus: per-depth score bonus applied during the Top-B prune.
        supertree_width: per-depth supertree width for the ``pruned`` variant;
            defaults to ``topk`` when ``None``.
        pruned: whether to grow a wider supertree and prune it to ``budget``.
        draft_model: optional Domino draft model exposing ``embed_proj`` and
            ``prefix_gru`` (see ``DominoCorrectionScorer``).
        target_model: optional target model exposing ``model.embed_tokens``;
            required together with ``draft_model`` for Domino correction.
        draft_hidden: [num_reqs, spec_num, hidden] draft backbone output;
            required when ``draft_model`` is set (source of ``z``).
        root_token_ids: [num_reqs] already-accepted anchor token per request;
            required when ``draft_model`` is set.
        prefix_len: number of leading depths scored from base logits without
            correction (mirrors ``pure_draft_prefix_len`` in the reference).
        candidate_vocab_size: number of candidate tokens gathered per depth for
            the correction scorer; defaults to ``topk``.
        correction_scorer: optional prebuilt ``DominoCorrectionScorer`` so the
            GRU lookup table is not rebuilt every draft step.
    """
    num_reqs, block_size, vocab = draft_logits.shape
    device = draft_logits.device
    budget = max(0, int(budget))
    topk = max(1, int(topk))
    k = min(topk, vocab, budget)  # candidate tokens kept per parent

    if budget <= 0 or block_size <= 0 or k <= 0:
        out.num_nodes.zero_()
        return out

    if pruned:
        width = supertree_width if supertree_width is not None else k
        width = max(1, min(int(width), k))
    else:
        width = max(1, min(int(np.ceil(budget / block_size)), k, budget))
    supertree_budget = width * block_size

    with_correction = correction_scorer is not None or (
        draft_model is not None and target_model is not None
    )
    if with_correction:
        if correction_scorer is None:
            correction_scorer = DominoCorrectionScorer(
                draft_model, target_model, hidden_dim=draft_hidden.shape[-1]
            )
        candidate_count = k if candidate_vocab_size is None else min(vocab, int(candidate_vocab_size))
        base_float = draft_logits.float()
        candidate_vals, candidate_ids = torch.topk(
            base_float, k=candidate_count, dim=-1
        )  # [R, spec, C]
        flat_cids = candidate_ids.reshape(-1)
        candidate_weight = correction_scorer.fc2_weight.index_select(
            0, flat_cids
        ).view(num_reqs, block_size, candidate_count, -1)  # [R, spec, C, mid]
        candidate_bias = None
        if correction_scorer.fc2_bias is not None:
            candidate_bias = correction_scorer.fc2_bias.index_select(
                0, flat_cids
            ).view(num_reqs, block_size, candidate_count)  # [R, spec, C]
        z_parts = correction_scorer.project_z(
            draft_hidden[:, :block_size]
        )  # [R, spec, mid]
        gru_hidden_dim = correction_scorer.gru_hidden_dim
    else:
        log_probs = torch.log_softmax(draft_logits.float(), dim=-1)  # [R, spec, vocab]

    # Node buffers. Node 0 is the root (parent -1); non-root node i+1 has
    # token/depth in slot i and its parent id in parents[i+1].
    max_nodes = supertree_budget + 1
    tokens = torch.full(
        (num_reqs, supertree_budget), -1, dtype=torch.long, device=device
    )
    depths = torch.zeros((num_reqs, supertree_budget), dtype=torch.long, device=device)
    parents = torch.zeros((num_reqs, max_nodes), dtype=torch.long, device=device)
    path_scores = torch.zeros((num_reqs, max_nodes), dtype=torch.float32, device=device)
    hidden_states = None
    if with_correction:
        # GRU hidden state per node; root is computed from the anchor token.
        root_h0 = torch.zeros(
            (num_reqs, gru_hidden_dim), dtype=draft_logits.dtype, device=device
        )
        root_hidden = correction_scorer.update_hidden(
            root_token_ids.reshape(-1), root_h0
        )
        hidden_states = torch.zeros(
            (num_reqs, max_nodes, gru_hidden_dim),
            dtype=root_hidden.dtype,
            device=device,
        )
        hidden_states[:, 0] = root_hidden

    # Grow the supertree breadth-first. Each layer expands the current frontier
    # (uniform width across the batch) with the global top-``width`` children by
    # path score.
    frontier = torch.zeros((num_reqs, width), dtype=torch.long, device=device)
    frontier_len = 1  # layer 1 expands only the root
    node_count = 0
    for child_depth in range(1, block_size + 1):
        if node_count >= supertree_budget:
            break
        take = min(width, frontier_len * k, supertree_budget - node_count)
        if take <= 0:
            break
        slot = child_depth - 1
        valid_parent = torch.arange(width, device=device)[None, :] < frontier_len

        if with_correction:
            parent_hidden = torch.gather(
                hidden_states, 1, frontier.unsqueeze(-1).expand(-1, -1, gru_hidden_dim)
            )  # [R, width, G]
            if slot < prefix_len:
                # Leading depths use the base logits without correction.
                logits = draft_logits[:, slot].float()  # [R, vocab]
                top_vals, top_ids = torch.topk(logits, k=k, dim=-1)  # [R, k]
                log_z = torch.logsumexp(logits, dim=-1, keepdim=True)  # [R, 1]
                top_scores = (top_vals - log_z).unsqueeze(1).expand(-1, width, -1)
                cand_ids = top_ids.unsqueeze(1).expand(-1, width, -1)
            else:
                top_scores, cand_ids = _dar_corrected_candidates(
                    correction_scorer,
                    z_parts[:, slot],                                    # [R, mid]
                    parent_hidden,                                       # [R, width, G]
                    candidate_vals[:, slot],                             # [R, C]
                    candidate_ids[:, slot],                              # [R, C]
                    candidate_weight[:, slot],                           # [R, C, mid]
                    candidate_bias[:, slot] if candidate_bias is not None else None,
                    k,
                )  # [R, width, k] log-prob scores and candidate token ids
        else:
            top_vals, top_ids = torch.topk(log_probs[:, slot, :], k=k, dim=-1)  # [R, k]
            top_scores = top_vals.unsqueeze(1).expand(-1, width, -1)
            cand_ids = top_ids.unsqueeze(1).expand(-1, width, -1)

        # Only the first ``frontier_len`` parents are valid this layer.
        top_scores = torch.where(
            valid_parent[:, :, None],
            top_scores,
            torch.full_like(top_scores, float("-inf")),
        )
        parent_scores = torch.gather(path_scores, 1, frontier)  # [R, width]
        cand_scores = parent_scores.unsqueeze(-1) + top_scores  # [R, width, k]
        flat = cand_scores.reshape(num_reqs, -1)  # [R, width*k]
        vals, sel = torch.topk(flat, k=take, dim=-1)  # [R, take]
        parent_pos = sel // k
        rank = sel % k
        sel_tokens = torch.gather(cand_ids.reshape(num_reqs, -1), 1, sel)  # [R, take]
        sel_parents = torch.gather(frontier, 1, parent_pos)  # [R, take]

        start = node_count + 1
        tokens[:, node_count:node_count + take] = sel_tokens
        depths[:, node_count:node_count + take] = child_depth
        parents[:, start:start + take] = sel_parents
        path_scores[:, start:start + take] = vals

        if with_correction:
            # Update the GRU hidden state of the selected children.
            parent_hidden_sel = parent_hidden.gather(
                1, parent_pos.unsqueeze(-1).expand(-1, -1, gru_hidden_dim)
            )  # [R, take, G]
            child_hidden = correction_scorer.update_hidden(
                sel_tokens.reshape(-1), parent_hidden_sel.reshape(-1, gru_hidden_dim)
            ).reshape(num_reqs, take, gru_hidden_dim)
            hidden_states[:, start:start + take] = child_hidden

        new_ids = torch.arange(
            start, start + take, dtype=torch.long, device=device
        ).unsqueeze(0)
        if take < width:
            new_ids = torch.cat(
                [
                    new_ids,
                    torch.zeros(
                        (num_reqs, width - take), dtype=torch.long, device=device
                    ),
                ],
                dim=-1,
            )
        frontier = new_ids
        frontier_len = take
        node_count += take

    # Optional global Top-B prune back to the final budget.
    if node_count > budget:
        kept_ids = _select_topb_prefix_tree_tensor(
            path_scores[:, 1:node_count + 1],
            depths[:, :node_count],
            budget,
            depth_bonus,
        )  # [R, budget] node ids
        tokens = torch.gather(tokens, 1, kept_ids - 1)
        depths = torch.gather(depths, 1, kept_ids - 1)
        old_parents = torch.gather(parents, 1, kept_ids)  # old parent node ids
        old_to_new = torch.zeros(
            (num_reqs, node_count + 1), dtype=torch.long, device=device
        )
        new_ids = torch.arange(
            1, budget + 1, dtype=torch.long, device=device
        ).unsqueeze(0)
        old_to_new.scatter_(1, kept_ids, new_ids.expand(num_reqs, budget))
        parent_ids = torch.gather(old_to_new, 1, old_parents)  # new parent node ids
        node_count = budget
    else:
        tokens = tokens[:, :node_count]
        depths = depths[:, :node_count]
        parent_ids = parents[:, 1:node_count + 1]

    # Write the TreeLayout. Node-level linked-list / visibility fills are
    # sequential over node slots but batched over all requests.
    out.tokens.fill_(-1)
    out.depths.zero_()
    out.parents.fill_(-1)
    out.visibility.zero_()
    out.first_child.fill_(-1)
    out.next_sibling.fill_(-1)
    out.tokens[:, :node_count] = tokens.to(out.tokens.dtype)
    out.depths[:, :node_count] = depths.to(out.depths.dtype)
    out.parents[:, :node_count] = parent_ids.to(out.parents.dtype)
    out.num_nodes[:] = node_count
    for i in range(node_count):
        node_id = i + 1
        parent_id = parent_ids[:, i]  # [R]
        cur_first = torch.gather(
            out.first_child[:, :node_count + 1], 1, parent_id.unsqueeze(1)
        ).squeeze(1)  # [R]
        out.next_sibling[:, node_id] = cur_first
        out.first_child.scatter_(
            1,
            parent_id.unsqueeze(1),
            torch.full((num_reqs, 1), node_id, dtype=torch.int32, device=device),
        )
    for i in range(node_count):
        parent_id = parent_ids[:, i]  # [R]
        if i > 0:
            src_idx = (parent_id - 1).clamp(min=0)  # [R]
            parent_vis = torch.gather(
                out.visibility[:, :node_count, :i],
                1,
                src_idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, i),
            ).squeeze(1)  # [R, i]
            out.visibility[:, i, :i] = torch.where(
                (parent_id > 0).unsqueeze(-1),
                parent_vis,
                out.visibility[:, i, :i],
            )
        out.visibility[:, i, i] = True
    return out


# Beam Tree Codes
def _markov_correct_logits(
    draft_model,
    depth_logits: torch.Tensor,
    frontier_tokens: torch.Tensor,
) -> torch.Tensor:
    """Apply optional DSpark markov logit-bias to one depth's base logits.

    When ``draft_model`` is set, mirrors ``DSparkSpeculator._sample_sequential``:
    bias from the previous **target-vocab** token via ``markov_embed`` +
    ``markov_bias``. When ``draft_model`` is None, expand the shared-depth base
    logits across the frontier.

    Args:
        draft_model: DSpark draft exposing ``markov_embed`` / ``markov_bias``, or
            None to skip correction.
        depth_logits: [R, vocab] uncorrected base logits of this depth.
        frontier_tokens: [R, batch] target-vocab ids of the frontier nodes (the
            previous tokens), one per frontier node per request.

    Returns:
        [R, batch, vocab] logits, one row per frontier node.
    """
    base = depth_logits.unsqueeze(1)  # [R, 1, vocab]
    if draft_model is None:
        return base.expand(-1, frontier_tokens.size(1), -1)
    markov_emb = draft_model.markov_embed(frontier_tokens)  # [R, batch, markov_rank]
    # NPU unquantized gemm is 2D-only; flatten then restore the frontier layout.
    bias = draft_model.markov_bias(markov_emb.reshape(-1, markov_emb.shape[-1]))
    return base + bias.view(*markov_emb.shape[:-1], -1)


def build_beam_trees(
    draft_logits: torch.Tensor,
    budget: int,
    topk: int,
    out: TreeLayout,
    root_token_ids: torch.Tensor,
    draft_model=None,
) -> TreeLayout:
    """Expand shared-depth draft logits into a beam (PCTree) draft tree.

    Each depth optionally applies Markov bias from the parent token, keeps the
    top-``topk`` children of the current frontier, then truncates the pool to
    ``budget`` nodes. Requests are batched in one pass.

    Selected ids live in draft vocab. When ``draft_model`` is set they are
    remapped with ``map_draft_to_target`` before the next Markov step and
    before writing ``out.tokens``, matching ``_sample_sequential``.

    Args:
        draft_logits: [num_reqs, spec_num, vocab] shared-depth draft base logits.
        budget: maximum number of non-root nodes per request.
        topk: number of candidate tokens kept at each depth.
        out: TreeLayout to fill in place.
        root_token_ids: [num_reqs] already-accepted anchor token (target vocab).
        draft_model: optional DSpark draft exposing ``markov_embed`` /
            ``markov_bias`` / ``map_draft_to_target``; None uses base logits only.
    """
    num_reqs, spec_num, vocab = draft_logits.shape
    k = max(1, min(topk, vocab))
    device = draft_logits.device

    out.first_child.fill_(-1)
    out.next_sibling.fill_(-1)
    out.visibility.zero_()
    out.tokens.fill_(-1)
    out.depths.zero_()
    out.parents.fill_(-1)

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
        top_vals, top_ids = log_probs.topk(k, dim=-1)  # [R, batch, k] draft ids
        if draft_model is not None:
            top_ids = draft_model.map_draft_to_target(top_ids)
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
