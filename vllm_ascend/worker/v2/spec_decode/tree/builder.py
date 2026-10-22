from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import torch

from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout

SUPPORTED_TREE_METHODS: tuple[str, ...] = ("beam", "prefix")

# tree_spec_config.method -> required speculative draft backend
METHOD_REQUIRED_BACKEND: dict[str, str] = {
    "beam": "dspark",
    "prefix": "dflash",
}

ScoreContextT = TypeVar("ScoreContextT")


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


class TreeBuilder(ABC, Generic[ScoreContextT]):
    """Shared level-wise tree construction and layout packing.

    Backend policies supply scores, retained layer nodes, optional recurrent
    state updates and the final budget selection. Buffer ownership, the depth
    loop and parent remapping are common to both policies. Model-specific
    scoring buffers belong to the concrete builder. Calls on one builder
    must be serialized because its workspace is reused.
    """

    required_backend: ClassVar[str]
    method: ClassVar[str]

    def __init__(self, budget: int, topk: int):
        self.budget = budget
        self.topk = topk
        self._scratch_reqs = 0
        self._scratch_vocab = 0
        self._scratch_spec = 0
        self._tokens_buf: torch.Tensor | None = None
        self._depths_buf: torch.Tensor | None = None
        self._parents_buf: torch.Tensor | None = None
        self._path_scores_buf: torch.Tensor | None = None
        self._frontier_buf: torch.Tensor | None = None
        self._old_to_new_buf: torch.Tensor | None = None
        self._node_arange: torch.Tensor | None = None
        self._budget_ids: torch.Tensor | None = None
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

    @abstractmethod
    def _beam_width(self, vocab: int) -> int:
        """Return the backend's frontier width."""

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
            w = self._beam_width(self._scratch_vocab)
            capacity = self._pool_capacity(w, self._scratch_spec)
            max_nodes = capacity + 1
            self._tokens_buf = torch.empty((r, capacity), dtype=torch.long, device=device)
            self._depths_buf = torch.empty((r, capacity), dtype=torch.long, device=device)
            self._parents_buf = torch.empty((r, max_nodes), dtype=torch.long, device=device)
            self._path_scores_buf = torch.empty(
                (r, max_nodes), dtype=torch.float32, device=device
            )
            self._frontier_buf = torch.empty((r, w), dtype=torch.long, device=device)
            self._old_to_new_buf = torch.empty((r, max_nodes), dtype=torch.long, device=device)
            self._node_arange = torch.arange(max_nodes + 1, device=device, dtype=torch.long)
            self._budget_ids = torch.arange(
                1, self.budget + 1, device=device, dtype=torch.long
            )

    @abstractmethod
    def _prepare_scores(
        self,
        state: TreeBuildState,
        draft_logits: torch.Tensor,
        root_token_ids: torch.Tensor,
        draft_hidden: torch.Tensor | None,
    ) -> ScoreContextT:
        """Prepare backend scoring state for this build."""

    @abstractmethod
    def _score_layer(
        self, state: TreeBuildState, ctx: ScoreContextT, depth_slot: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [requests, parents, children] edge scores and target token IDs.

        The scoring policy owns normalization (full vocabulary or shortlist);
        the shared loop must not renormalize these log-probabilities.
        """

    @abstractmethod
    def _expand_layer(
        self,
        state: TreeBuildState,
        ctx: ScoreContextT,
        top_scores: torch.Tensor,
        cand_ids: torch.Tensor,
        child_depth: int,
        graph_safe: bool,
    ) -> tuple[torch.Tensor, int]:
        """Append retained nodes and return the next frontier and added count.

        Write from the current ``state.num_nodes`` slot without changing that
        count. Returned frontier IDs refer to the root-inclusive node space;
        a policy may retain more pool nodes than it puts on the frontier.
        """

    def _update_state(
        self,
        state: TreeBuildState,
        ctx: ScoreContextT,
        num_added: int,
        *,
        graph_safe: bool,
    ) -> None:
        """Advance model state for the nodes just written, if needed.

        ``frontier`` already identifies the next layer; ``num_nodes`` still
        marks the start of the appended slice. The shared loop increments it
        after this hook. Graph-safe updates must not introduce timer syncs.
        """

    @abstractmethod
    def _select_nodes(self, state: TreeBuildState, ctx: ScoreContextT) -> torch.Tensor | None:
        """Return one-based IDs in topological order, or None for the whole pool.

        A selection must retain every selected node's ancestors and fit the
        output budget. The shared loop only remaps these IDs; it does not
        change the policy's score ordering, tie-breaking or pruning rule.
        """

    def build(
        self,
        draft_logits: torch.Tensor,
        out: TreeLayout,
        *,
        root_token_ids: torch.Tensor,
        draft_hidden: torch.Tensor | None = None,
        graph_safe: bool = False,
    ) -> TreeLayout:
        """Expand logits below the supplied roots into ``out``.

        ``root_token_ids`` contains one target-vocabulary token per request.
        Graph-safe builds skip timers and Triton; Domino also needs the draft
        hidden states used to produce these logits.
        """
        from vllm_ascend.worker.v2.spec_decode.tree.timer import (
            tree_time,
            tree_time_accum,
            tree_time_accum_flush,
        )

        num_reqs, spec_num, vocab = draft_logits.shape
        width = self._beam_width(vocab)
        capacity = self._pool_capacity(width, spec_num)
        with nullcontext() if graph_safe else tree_time("build_precompute"):
            self._ensure_scratch(
                num_reqs,
                vocab,
                spec_num,
                draft_logits.device,
            )
            state = TreeBuildState(
                tokens=self._tokens_buf[:num_reqs, :capacity],
                depths=self._depths_buf[:num_reqs, :capacity],
                parents=self._parents_buf[:num_reqs, : capacity + 1],
                path_scores=self._path_scores_buf[:num_reqs, : capacity + 1],
                frontier=self._frontier_buf[:num_reqs, :width],
                width=width,
            )
            state.tokens.fill_(-1)
            state.depths.zero_()
            state.parents.zero_()
            state.path_scores.zero_()
            state.frontier.zero_()
            ctx = self._prepare_scores(state, draft_logits, root_token_ids, draft_hidden)

        for depth_slot in range(spec_num):
            with nullcontext() if graph_safe else tree_time_accum("build_expand_score"):
                top_scores, cand_ids = self._score_layer(state, ctx, depth_slot)
            state.frontier, num_added = self._expand_layer(
                state, ctx, top_scores, cand_ids, depth_slot + 1, graph_safe
            )
            self._update_state(state, ctx, num_added, graph_safe=graph_safe)
            state.num_nodes += num_added

        if not graph_safe:
            tree_time_accum_flush()
        with nullcontext() if graph_safe else tree_time("build_prune"):
            kept_ids = self._select_nodes(state, ctx)
            num_nodes = state.num_nodes
            if kept_ids is None:
                tokens = state.tokens[:, :num_nodes]
                depths = state.depths[:, :num_nodes]
                parent_ids = state.parents[:, 1 : num_nodes + 1]
            else:
                tokens = torch.gather(state.tokens, 1, kept_ids - 1)
                depths = torch.gather(state.depths, 1, kept_ids - 1)
                old_parents = torch.gather(state.parents, 1, kept_ids)
                old_to_new = self._old_to_new_buf[:num_reqs, : num_nodes + 1]
                old_to_new.zero_()
                num_nodes = kept_ids.shape[1]
                new_ids = self._budget_ids[:num_nodes].unsqueeze(0)
                old_to_new.scatter_(1, kept_ids, new_ids.expand(num_reqs, -1))
                parent_ids = torch.gather(old_to_new, 1, old_parents)
        with nullcontext() if graph_safe else tree_time("build_finalize_layout"):
            return finalize_tree_layout(
                out, tokens, depths, parent_ids, num_nodes, force_torch=graph_safe
            )


def validate_tree_method_backend(method: str, draft_backend: str) -> None:
    """Raise if ``tree_spec_config.method`` does not match draft backend."""
    required = METHOD_REQUIRED_BACKEND.get(method)
    if required is None:
        raise ValueError(
            f"tree_spec_config.method must be one of {SUPPORTED_TREE_METHODS}, "
            f"got {method!r}"
        )
    if draft_backend != required:
        raise ValueError(
            f"tree_spec_config.method={method!r} requires "
            f"speculative_config method {required!r}, got {draft_backend!r}"
        )


def create_tree_builder(
    method: str,
    budget: int,
    topk: int,
    draft_backend: str,
    *,
    draft_model=None,
    correction_scorer=None,
    prefix_len: int = 0,
    params: dict | None = None,
) -> TreeBuilder:
    """Construct the builder for ``method`` after backend pairing check.

    ``params`` is forwarded to prefix (``candidate_size`` only).
    Beam ignores it.
    """
    validate_tree_method_backend(method, draft_backend)
    if method == "beam":
        from vllm_ascend.worker.v2.spec_decode.tree.beam import BeamTreeBuilder

        return BeamTreeBuilder(budget, topk, draft_model=draft_model)
    if method == "prefix":
        from vllm_ascend.worker.v2.spec_decode.tree.prefix import PrefixTreeBuilder

        return PrefixTreeBuilder(
            budget,
            topk,
            correction_scorer=correction_scorer,
            prefix_len=prefix_len,
            params=params,
        )
    raise ValueError(
        f"tree_spec_config.method must be one of {SUPPORTED_TREE_METHODS}, "
        f"got {method!r}"
    )
