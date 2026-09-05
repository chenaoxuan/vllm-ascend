import logging
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path
from vllm_ascend.worker.v2.spec_decode.tree.utils import (
    TreeLayout,
    build_beam_trees,
    build_heap_trees,
    build_multi_order_trees,
)

logger = logging.getLogger(__name__)


class AscendTreeSpeculator(AscendDFlashSpeculator):
    """DFlash speculator that expands mask-token logits into a draft tree.

    Draft model forward is unchanged (one parallel pass). Tree construction
    replaces per-position single-token sampling.

    ``propose()`` still returns flattened non-root tokens so the existing
    runner call stays valid. The tree itself is ``self.tree`` after
    ``propose()`` / ``_generate_draft()``.
    """

    _speculator_name = "DFlashTree"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Tree query layout keeps the anchor as the bonus token. DSpark drafts
        # may set sample_from_anchor; force the DFlash 1+N layout before super().
        draft_hf = vllm_config.speculative_config.draft_model_config.hf_config
        dflash_cfg = getattr(draft_hf, "dflash_config", None)
        if isinstance(dflash_cfg, dict) and dflash_cfg.get("sample_from_anchor"):
            draft_hf.dflash_config = {**dflash_cfg, "sample_from_anchor": False}
        elif dflash_cfg is not None and getattr(dflash_cfg, "sample_from_anchor", False):
            dflash_cfg.sample_from_anchor = False
        super().__init__(vllm_config, device)
        if self.use_local_argmax_reduction:
            raise ValueError(
                "DFlash tree speculator needs full draft logits; "
                "disable use_local_argmax_reduction."
            )

        tree_cfg = get_ascend_config().tree_spec_config
        self.method = tree_cfg.method
        self.budget = tree_cfg.budget
        self.topk = tree_cfg.topk
        self._beam_draft_model = None
        self._domino_scorer = None
        self._domino_prefix_len = 0
        if self.budget < self.num_speculative_steps:
            raise ValueError(
                "tree_spec_config.budget must be >= num_speculative_tokens "
                f"({self.num_speculative_steps}), got {self.budget}"
            )
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
        self.tree = self._load_layout_from_buffers(self.max_num_reqs)
        logger.info(
            "DFlash tree speculator enabled: method=%s budget=%s topk=%s depth=%s",
            self.method,
            self.budget,
            self.topk,
            self.num_speculative_steps,
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        if self.speculative_config.use_dspark():
            from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

            return load_dspark_model(target_model, self.vllm_config)
        return super().load_draft_model(target_model, target_attn_layer_names)

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self._bind_correction_heads(target_model)

    def _bind_correction_heads(self, target_model: nn.Module) -> None:
        """Resolve markov / Domino heads once after the draft is loaded."""
        from vllm_ascend.worker.v2.spec_decode.tree.utils import (
            DominoCorrectionScorer,
        )

        model = self.model
        if self.method == "beam" and self.speculative_config.use_dspark():
            self._beam_draft_model = model
        if (
            self.method == "multi_order"
            and self.speculative_config.use_dflash()
            and model.projector_type == "domino"
        ):
            self._domino_prefix_len = int(model.pure_draft_prefix_len)
            embed = target_model.get_language_model().model.embed_tokens
            self._domino_scorer = DominoCorrectionScorer(
                model,
                SimpleNamespace(model=SimpleNamespace(embed_tokens=embed)),
            )
        logger.info(
            "DFlash tree correction heads: markov=%s domino=%s prefix_len=%s",
            self._beam_draft_model is not None,
            self._domino_scorer is not None,
            self._domino_prefix_len,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if cudagraph_mode != CUDAGraphMode.NONE:
            logger.warning(
                "DFlash tree speculator builds the draft tree on CPU; "
                "disabling full ACL graphs for the draft query."
            )
        super().init_cudagraph_manager(CUDAGraphMode.NONE)

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        path_node_ids = getattr(input_batch, "path_node_ids", None)
        if path_node_ids is not None and not dummy_run:
            tensors = [last_hidden_states]
            if aux_hidden_states:
                tensors.extend(aux_hidden_states)
            compact_tree_query_along_path(
                tensors,
                input_batch.query_start_loc,
                path_node_ids,
                linearize_positions=input_batch.positions,
            )
        return super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            num_tokens_across_dp,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile=is_profile,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
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
        if self._beam_draft_model is not None:
            logits = self.model.compute_draft_logits(sample_hidden_states)
        else:
            logits = self.model.compute_logits(sample_hidden_states)
        # [bs, spec_num, vocab_size]
        logits = logits.view(num_reqs, self.num_speculative_steps, -1)
        layout = self._load_layout_from_buffers(num_reqs)
        if self.method == "heap":
            build_heap_trees(logits, self.budget, self.topk, layout)
        elif self.method == "beam":
            nqp = self.num_query_per_req
            root_token_ids = self.input_buffers.input_ids[: num_reqs * nqp].view(
                num_reqs, nqp
            )[:, 0]
            build_beam_trees(
                logits,
                self.budget,
                self.topk,
                layout,
                root_token_ids,
                self._beam_draft_model,
            )
            if self._beam_draft_model is not None:
                mapped = self.model.map_draft_to_target(layout.tokens.clamp(min=0))
                layout.tokens.copy_(
                    torch.where(layout.tokens >= 0, mapped, layout.tokens)
                )
        elif self.method == "multi_order":
            nqp = self.num_query_per_req
            root_token_ids = self.input_buffers.input_ids[: num_reqs * nqp].view(
                num_reqs, nqp
            )[:, 0]
            build_multi_order_trees(
                logits,
                self.budget,
                self.topk,
                layout,
                draft_hidden=sample_hidden_states.view(
                    num_reqs, self.num_speculative_steps, -1
                ),
                root_token_ids=root_token_ids,
                prefix_len=self._domino_prefix_len,
                correction_scorer=self._domino_scorer,
            )
        self.tree = layout

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
