from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout

SUPPORTED_TREE_METHODS: tuple[str, ...] = ("priority", "beam", "prefix")

# tree_spec_config.method -> required speculative draft backend
METHOD_REQUIRED_BACKEND: dict[str, str] = {
    "priority": "dflash",
    "beam": "dspark",
    "prefix": "dflash",
}


class TreeBuilder(ABC):
    """Base for draft-tree topology builders.

    Axis A (draft backend) is ``required_backend`` / ``speculative_config``.
    Axis B (topology) is the concrete subclass selected by
    ``tree_spec_config.method``.

    ``build()`` writes into ``out`` and returns it. Subclasses ignore kwargs
    they do not need (no probing fallbacks).
    """

    required_backend: ClassVar[str]
    method: ClassVar[str]

    def __init__(self, budget: int, topk: int):
        self.budget = budget
        self.topk = topk

    @abstractmethod
    def build(
        self,
        draft_logits: torch.Tensor,
        out: TreeLayout,
        *,
        root_token_ids: torch.Tensor | None = None,
        draft_hidden: torch.Tensor | None = None,
    ) -> TreeLayout:
        """Expand ``draft_logits`` [R, spec_num, vocab] into ``out``."""


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
    depth_bonus: float = -0.2,
    supertree_width: int | None = None,
    pruned: bool = True,
) -> TreeBuilder:
    """Construct the builder for ``method`` after backend pairing check."""
    validate_tree_method_backend(method, draft_backend)
    if method == "priority":
        from vllm_ascend.worker.v2.spec_decode.tree.priority import (
            PriorityTreeBuilder,
        )

        return PriorityTreeBuilder(budget, topk)
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
            depth_bonus=depth_bonus,
            supertree_width=supertree_width,
            pruned=pruned,
        )
    raise ValueError(
        f"tree_spec_config.method must be one of {SUPPORTED_TREE_METHODS}, "
        f"got {method!r}"
    )
