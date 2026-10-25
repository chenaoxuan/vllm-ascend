"""One topology builder; model adapters supply only edge scores and state."""

from dataclasses import dataclass

import torch

from vllm_ascend import envs
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout
from vllm_ascend.worker.v2.spec_decode.tree.scoring import DominoScorer, IndependentScorer, MarkovScorer


@dataclass
class TreeBuildState:
    """Per-build device buffers; ``width`` and ``num_nodes`` are host ints.

    Nodes use one-based IDs, with zero reserved for the root. Token and depth
    buffers omit the root; parent and path-score buffers include it.
    """

    tokens: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor
    path_scores: torch.Tensor
    frontier: torch.Tensor
    width: int
    num_nodes: int = 0


class TreeBuilder:
    """Shared level selection, ancestor-closed pruning and workspace ownership.

    Device tensors contain all topology/scoring state. Depth, width and buffer
    capacities are host integers. One builder is serialized across rounds;
    prepare_for_capture freezes its backing storage before graph capture.
    Expansion uses cumulative log probability and stable pool order on ties.
    Each scorer retains its nonnegative depth penalty for global pruning.
    """

    def __init__(self, budget, topk, scorer=None):
        if budget < 1 or topk < 1:
            raise ValueError("Tree budget and topk must be positive")
        self.budget, self.topk = budget, topk
        self.scorer = scorer if scorer is not None else IndependentScorer()
        if self.scorer.depth_penalty < 0:
            raise ValueError("Tree depth penalty must preserve ancestor-first priority")
        self.keep_greedy = envs.VLLM_ASCEND_TREE_KEEP_GREEDY
        self.num_selected_nodes = 0
        self._scratch_reqs = self._scratch_vocab = self._scratch_spec = 0
        self._tokens_buf = self._depths_buf = self._parents_buf = None
        self._path_scores_buf = self._frontier_buf = self._old_to_new_buf = None
        self._node_arange = self._budget_ids = self._selection_buf = None
        self._scratch_frozen = False

    def prepare_for_capture(
        self,
        num_reqs: int,
        vocab: int,
        spec_num: int,
        device: torch.device,
    ) -> None:
        """Reserve topology and scoring buffers before capturing any graphs.

        The caller supplies the maximum batch size and the model's fixed
        vocabulary, depth and device. Subsequent builds must stay within these
        bounds. Freezing after preparation keeps captured addresses stable.
        This method prepares resources; it does not enable graph execution.
        """
        self._ensure_scratch(num_reqs, vocab, spec_num, device)
        self._scratch_frozen = True

    def _frontier_width(self, vocab: int) -> int:
        return min(self.topk, vocab, self.budget)

    def workspace_tensors(self):
        """Expose owned storage for graph accounting, without device reads."""
        buffers = (
            self._tokens_buf,
            self._depths_buf,
            self._parents_buf,
            self._path_scores_buf,
            self._frontier_buf,
            self._old_to_new_buf,
            self._node_arange,
            self._budget_ids,
            self._selection_buf,
        )
        return tuple(buffer for buffer in buffers if buffer is not None) + self.scorer.workspace_tensors()

    def _pool_capacity(self, width: int, spec_num: int) -> int:
        return width * spec_num

    def _ensure_scratch(
        self,
        num_reqs: int,
        vocab: int,
        spec_num: int,
        device: torch.device,
    ) -> None:
        if self._scratch_frozen:
            if (
                num_reqs > self._scratch_reqs
                or vocab > self._scratch_vocab
                or spec_num > self._scratch_spec
                or self._tokens_buf.device != device
            ):
                raise ValueError("Tree build exceeds its captured workspace capacity")
            return
        grow = (
            self._tokens_buf is None
            or num_reqs > self._scratch_reqs
            or vocab > self._scratch_vocab
            or spec_num > self._scratch_spec
            or self._tokens_buf.device != device
        )
        if grow:
            self._scratch_reqs = max(num_reqs, self._scratch_reqs)
            self._scratch_vocab = max(vocab, self._scratch_vocab)
            self._scratch_spec = max(spec_num, self._scratch_spec)
            r = self._scratch_reqs
            w = self._frontier_width(self._scratch_vocab)
            capacity = self._pool_capacity(w, self._scratch_spec)
            max_nodes = capacity + 1
            self._tokens_buf = torch.empty((r, capacity), dtype=torch.long, device=device)
            self._depths_buf = torch.empty((r, capacity), dtype=torch.long, device=device)
            self._parents_buf = torch.empty((r, max_nodes), dtype=torch.long, device=device)
            self._path_scores_buf = torch.empty((r, max_nodes), dtype=torch.float32, device=device)
            self._frontier_buf = torch.empty((r, w), dtype=torch.long, device=device)
            self._old_to_new_buf = torch.empty((r, max_nodes), dtype=torch.long, device=device)
            self._node_arange = torch.arange(max_nodes + 1, device=device, dtype=torch.long)
            self._budget_ids = torch.arange(1, self.budget + 1, device=device, dtype=torch.long)

            self._selection_buf = torch.empty((r, w), dtype=torch.long, device=device)
            self.scorer.reserve(r, max_nodes, device)

    def _expand_layer(self, state, layer, child_depth, graph_safe, greedy_candidate):
        r, width = state.tokens.shape[0], state.width
        scores = layer.scores.expand(-1, width, -1)
        tokens = layer.token_ids.expand(-1, width, -1)
        live = 1 if child_depth == 1 else width
        if scores.device.type == "npu" and not graph_safe and not self.keep_greedy:
            from vllm_ascend.ops.triton.spec_decode.tree.expand import expand_tree_layer_triton

            expand_tree_layer_triton(
                scores,
                tokens,
                state.path_scores,
                state.frontier,
                state.tokens,
                state.depths,
                state.parents,
                self._selection_buf[:r, :width],
                live,
                state.num_nodes,
                child_depth,
            )
            selected = self._selection_buf[:r, :width]
        else:
            parent_scores = state.path_scores.gather(1, state.frontier)
            valid = torch.arange(width, device=scores.device) < live
            flat = (
                (parent_scores.unsqueeze(-1) + scores).masked_fill(~valid[None, :, None], float("-inf")).reshape(r, -1)
            )
            selected = flat.argsort(dim=-1, descending=True, stable=True)[:, :width].clone()
            if self.keep_greedy:
                present = (selected == greedy_candidate[:, None]).any(-1)
                selected[:, -1] = torch.where(present, selected[:, -1], greedy_candidate)
            start = state.num_nodes + 1
            state.tokens[:, state.num_nodes : state.num_nodes + width] = tokens.reshape(r, -1).gather(1, selected)
            state.depths[:, state.num_nodes : state.num_nodes + width] = child_depth
            state.parents[:, start : start + width] = state.frontier.gather(1, selected // width)
            state.path_scores[:, start : start + width] = flat.gather(1, selected)
        start = state.num_nodes + 1
        frontier = self._node_arange[start : start + width].unsqueeze(0).expand(r, -1)
        return frontier, selected

    def build(self, draft_logits, out: TreeLayout, *, root_token_ids, draft_hidden=None, graph_safe=False):
        r, depth, vocab = draft_logits.shape
        width = self._frontier_width(vocab)
        capacity = self._pool_capacity(width, depth)
        if self.keep_greedy and self.budget < depth:
            raise ValueError("Keeping the complete greedy chain requires budget >= depth")
        self._ensure_scratch(r, vocab, depth, draft_logits.device)
        state = TreeBuildState(
            self._tokens_buf[:r, :capacity],
            self._depths_buf[:r, :capacity],
            self._parents_buf[:r, : capacity + 1],
            self._path_scores_buf[:r, : capacity + 1],
            self._frontier_buf[:r, :width],
            width,
        )
        state.tokens.fill_(-1)
        state.depths.zero_()
        state.parents.zero_()
        state.path_scores.zero_()
        state.frontier.zero_()
        ctx = self.scorer.prepare(state, draft_logits, root_token_ids, draft_hidden, keep_greedy=self.keep_greedy)
        greedy_pools = []
        greedy_frontier = torch.zeros(r, dtype=torch.long, device=draft_logits.device) if self.keep_greedy else None
        for slot in range(depth):
            layer = self.scorer.score(state, ctx, slot, self.keep_greedy)
            greedy_candidate = None
            if self.keep_greedy:
                child = layer.greedy_child.expand(-1, width).gather(1, greedy_frontier[:, None])[:, 0]
                greedy_candidate = greedy_frontier * width + child
            state.frontier, selected = self._expand_layer(state, layer, slot + 1, graph_safe, greedy_candidate)
            if self.keep_greedy:
                greedy_frontier = (selected == greedy_candidate[:, None]).long().argmax(-1)
                greedy_pools.append(state.frontier.gather(1, greedy_frontier[:, None])[:, 0])
            self.scorer.update(state, ctx, width)
            state.num_nodes += width
        num_nodes = min(self.budget, state.num_nodes)
        priority = state.path_scores[:, 1 : state.num_nodes + 1]
        if self.scorer.depth_penalty:
            priority = priority - self.scorer.depth_penalty * state.depths[:, : state.num_nodes].float()
        if self.keep_greedy:
            priority = priority.clone()
            priority.scatter_(1, torch.stack(greedy_pools, dim=1) - 1, float("inf"))
        # Nonpositive log-probabilities, nonnegative depth penalties and stable
        # ancestor-first ties ensure closure. Preserve the existing pool IDs.
        kept = priority.argsort(dim=-1, descending=True, stable=True)[:, :num_nodes]
        kept_ids = (kept + 1).sort(dim=-1).values
        old_to_new = self._old_to_new_buf[:r, : state.num_nodes + 1]
        old_to_new.zero_()
        old_to_new.scatter_(1, kept_ids, self._budget_ids[:num_nodes].unsqueeze(0).expand(r, -1))
        parent_ids = old_to_new.gather(1, state.parents.gather(1, kept_ids))
        self.num_selected_nodes = num_nodes
        return finalize_tree_layout(
            out,
            state.tokens.gather(1, kept_ids - 1),
            state.depths.gather(1, kept_ids - 1),
            parent_ids,
            num_nodes,
            force_torch=graph_safe,
        )


def create_tree_builder(
    budget, topk, draft_backend, *, draft_model=None, correction_scorer=None, prefix_len=0, candidate_size=None
):
    """Resolve the scorer once at model load; tree construction is shared."""
    if draft_backend == "dspark":
        if candidate_size is not None:
            raise ValueError("candidate_size is for Domino correction; DSpark retains full-vocabulary scoring")
        scorer = MarkovScorer(draft_model)
    elif draft_backend == "dflash":
        scorer = (
            IndependentScorer()
            if correction_scorer is None
            else DominoScorer(correction_scorer, prefix_len, candidate_size)
        )
        if correction_scorer is None and candidate_size is not None:
            raise ValueError("candidate_size requires a Domino correction head")
    else:
        raise ValueError(f"Unsupported tree draft backend: {draft_backend!r}")
    return TreeBuilder(budget, topk, scorer)
