# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tree dimensions shared by configuration, capture and publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class TreeShape:
    """Separate draft depth K, root-excluded storage B and published nodes S.

    Node 0 is the accepted root; token/parent/depth storage contains nodes
    1..B. S is a host shape supplied by the builder, never a device readback.
    The drafter's query width is model-owned: anchor/shift-label handling must
    not be inferred from the target verification capacity.
    """

    depth: int
    budget: int
    topk: int
    dsv4_dspark: bool = False

    def __post_init__(self) -> None:
        if self.depth < 1 or self.budget < self.depth or self.topk < 1:
            raise ValueError("Tree dimensions require depth >= 1, budget >= depth and topk >= 1")

    @property
    def target_query_len(self) -> int:
        """Maximum target rows per request, including its accepted root."""
        return 1 + (self.depth if self.dsv4_dspark and self.topk == 1 else self.budget)

    def validate_selected_nodes(self, count: int) -> int:
        """Validate the builder's host publication width against target capacity."""
        if not 0 <= count < self.target_query_len:
            raise ValueError(f"Selected tree nodes {count} exceed capacity {self.target_query_len - 1}")
        return count

    @classmethod
    def from_config(cls, vllm_config: VllmConfig) -> TreeShape | None:
        """Resolve early platform defaults without requiring Ascend's singleton."""
        tree_config = (vllm_config.additional_config or {}).get("tree_spec_config") or {}
        if not tree_config.get("enabled", False):
            return None
        spec = vllm_config.speculative_config
        if spec is None:
            raise ValueError("Tree speculation requires speculative_config")
        from vllm_ascend.worker.v2.spec_decode import dsv4_dspark_draft

        return cls(
            depth=int(spec.num_speculative_tokens),
            budget=int(tree_config["budget"]),
            topk=int(tree_config["topk"]),
            dsv4_dspark=dsv4_dspark_draft(vllm_config, spec),
        )
