from __future__ import annotations

import logging
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.tree.utils import TreeLayout, build_trees

logger = logging.getLogger(__name__)


class AscendTreeSpeculator(AscendDFlashSpeculator):
    """DFlash speculator that expands mask-token logits into a draft tree.

    Draft model forward is unchanged (one parallel pass). Tree construction
    replaces per-position single-token sampling.

    ``propose()`` still returns flattened non-root tokens so the existing
    runner call stays valid. The tree itself is ``TreeLayout``; read it with
    ``get_tree()`` after ``propose()`` / ``_generate_draft()``.
    """

    _speculator_name = "DFlashTree"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if self.use_local_argmax_reduction:
            raise ValueError(
                "DFlash tree speculator needs full draft logits; "
                "disable use_local_argmax_reduction."
            )

        tree_cfg = get_ascend_config().tree_spec_config
        if tree_cfg.max_nodes is None:
            self.tree_budget = self.num_speculative_steps
        else:
            self.tree_budget = tree_cfg.max_nodes
        if self.tree_budget > self.draft_tokens.shape[1]:
            self.draft_tokens = torch.zeros(
                self.max_num_reqs,
                self.tree_budget,
                dtype=self.draft_tokens.dtype,
                device=device,
            )

        max_len = self.tree_budget + 1
        self.tree_parents = torch.full(
            (self.max_num_reqs, max_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.tree_depths = torch.zeros(
            (self.max_num_reqs, max_len),
            dtype=torch.int32,
            device=device,
        )
        self.tree_num_nodes = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.tree_visibility = torch.zeros(
            (self.max_num_reqs, max_len, max_len),
            dtype=torch.bool,
            device=device,
        )
        self.tree_child_maps: list[list[dict[int, int]]] = [[{}] for _ in range(self.max_num_reqs)]
        self.tree: TreeLayout | None = None
        logger.info(
            "DFlash tree speculator enabled: budget=%s depth=%s "
            "(target tree verify is not wired yet)",
            self.tree_budget,
            self.num_speculative_steps,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if cudagraph_mode != CUDAGraphMode.NONE:
            logger.warning(
                "DFlash tree speculator builds the draft tree on CPU; "
                "disabling full ACL graphs for the draft query."
            )
        super().init_cudagraph_manager(CUDAGraphMode.NONE)

    def get_tree(self) -> TreeLayout:
        """Return the last draft tree built by ``propose`` / ``_generate_draft``."""
        if self.tree is None:
            raise RuntimeError("Draft tree has not been built yet.")
        return self.tree

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> TreeLayout:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        logits = self.model.compute_logits(sample_hidden_states)
        logits = logits.view(num_reqs, self.num_speculative_steps, -1)
        layout = build_trees(logits, self.tree_budget)
        self._store_tree_layout(layout, num_reqs)
        self.tree = self._layout_from_buffers(num_reqs)
        return self.tree

    def _reset_tree_buffers(self, num_reqs: int) -> None:
        self.draft_tokens[:num_reqs].fill_(-1)
        self.tree_parents[:num_reqs].fill_(-1)
        self.tree_depths[:num_reqs].zero_()
        self.tree_num_nodes[:num_reqs].fill_(1)
        self.tree_visibility[:num_reqs].zero_()
        self.tree_visibility[:num_reqs, 0, 0] = True
        for req_idx in range(num_reqs):
            self.tree_child_maps[req_idx] = [{}]

    def _layout_from_buffers(self, num_reqs: int) -> TreeLayout:
        budget = self.tree_budget
        return TreeLayout(
            tokens=self.draft_tokens[:num_reqs, :budget],
            depths=self.tree_depths[:num_reqs, 1 : budget + 1],
            parents=self.tree_parents[:num_reqs, : budget + 1],
            num_nodes=self.tree_num_nodes[:num_reqs],
            visibility=self.tree_visibility[:num_reqs, : budget + 1, : budget + 1],
            child_maps=self.tree_child_maps[:num_reqs],
        )

    def _store_tree_layout(self, layout: TreeLayout, num_reqs: int) -> None:
        budget = layout.tokens.shape[1]
        self._reset_tree_buffers(num_reqs)
        if budget == 0:
            return
        tokens = layout.tokens[:num_reqs].to(device=self.draft_tokens.device, dtype=self.draft_tokens.dtype)
        self.draft_tokens[:num_reqs, :budget].copy_(tokens)
        non_root_nodes = layout.num_nodes[:num_reqs].to(device=self.draft_tokens.device) - 1
        unused = torch.arange(budget, device=self.draft_tokens.device).unsqueeze(0) >= non_root_nodes.unsqueeze(1)
        self.draft_tokens[:num_reqs, :budget].masked_fill_(unused, -1)
        self.tree_parents[:num_reqs, : budget + 1].copy_(
            layout.parents[:num_reqs].to(device=self.tree_parents.device, dtype=self.tree_parents.dtype)
        )
        self.tree_depths[:num_reqs, 1 : budget + 1].copy_(
            layout.depths[:num_reqs].to(device=self.tree_depths.device, dtype=self.tree_depths.dtype)
        )
        self.tree_num_nodes[:num_reqs].copy_(
            layout.num_nodes[:num_reqs].to(device=self.tree_num_nodes.device, dtype=self.tree_num_nodes.dtype)
        )
        vis_len = layout.visibility.shape[-1]
        self.tree_visibility[:num_reqs, :vis_len, :vis_len].copy_(
            layout.visibility[:num_reqs].to(device=self.tree_visibility.device, dtype=self.tree_visibility.dtype)
        )
        for req_idx in range(num_reqs):
            self.tree_child_maps[req_idx] = layout.child_maps[req_idx]
