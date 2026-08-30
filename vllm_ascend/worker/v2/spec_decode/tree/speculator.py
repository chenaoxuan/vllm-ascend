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
        if tree_cfg.budget is None:
            self.budget = self.num_speculative_steps
        else:
            self.budget = tree_cfg.budget
        if self.budget < self.num_speculative_steps:
            raise ValueError(
                "tree_spec_config.budget must be >= num_speculative_tokens "
                f"({self.num_speculative_steps}), got {self.budget}"
            )
        self.topk = tree_cfg.topk
        if self.budget > self.draft_tokens.shape[1]:
            self.draft_tokens = torch.zeros(
                self.max_num_reqs,
                self.budget,
                dtype=self.draft_tokens.dtype,
                device=device,
            )

        self.tree_parents = torch.full(
            (self.max_num_reqs, self.budget),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.tree_depths = torch.zeros(
            (self.max_num_reqs, self.budget),
            dtype=torch.int32,
            device=device,
        )
        self.tree_num_nodes = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.tree_visibility = torch.zeros(
            (self.max_num_reqs, self.budget, self.budget),
            dtype=torch.bool,
            device=device,
        )
        self.tree_first_child = torch.full(
            (self.max_num_reqs, self.budget + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.tree_next_sibling = torch.full(
            (self.max_num_reqs, self.budget + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.tree: TreeLayout | None = None
        logger.info(
            "DFlash tree speculator enabled: budget=%s topk=%s depth=%s "
            "(target tree verify is not wired yet)",
            self.budget,
            self.topk,
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
        # [all_tokens, dim] all_tokens include the anchor token
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        # skip the anchor and only use the masked pos(true draft pos)
        sample_hidden_states = last_hidden_states[self.sample_indices[:num_sample]]
        logits = self.model.compute_logits(sample_hidden_states)
        # [bs, spec_num, vocab_size]
        logits = logits.view(num_reqs, self.num_speculative_steps, -1)
        vocab_size = logits.shape[-1]
        topk = vocab_size if self.topk is None else self.topk
        self._reset_tree_buffers(num_reqs)
        layout = self._load_layout_from_buffers(num_reqs)
        build_trees(logits, self.budget, topk, layout)
        self.tree = layout
        return self.tree

    def _reset_tree_buffers(self, num_reqs: int) -> None:
        self.draft_tokens[:num_reqs].fill_(-1)
        self.tree_parents[:num_reqs].fill_(-1)
        self.tree_depths[:num_reqs].zero_()
        self.tree_num_nodes[:num_reqs].zero_()
        self.tree_visibility[:num_reqs].zero_()
        self.tree_first_child[:num_reqs].fill_(-1)
        self.tree_next_sibling[:num_reqs].fill_(-1)

    def _load_layout_from_buffers(self, num_reqs: int) -> TreeLayout:
        """Views into persistent buffers."""
        budget = self.budget
        return TreeLayout(
            tokens=self.draft_tokens[:num_reqs, :budget],
            depths=self.tree_depths[:num_reqs, :budget],
            parents=self.tree_parents[:num_reqs, :budget],
            num_nodes=self.tree_num_nodes[:num_reqs],
            visibility=self.tree_visibility[:num_reqs, :budget, :budget],
            first_child=self.tree_first_child[:num_reqs, : budget + 1],
            next_sibling=self.tree_next_sibling[:num_reqs, : budget + 1],
        )
