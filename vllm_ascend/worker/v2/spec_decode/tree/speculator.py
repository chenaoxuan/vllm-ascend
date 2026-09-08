import logging
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
from vllm_ascend.worker.v2.spec_decode.tree.builder import (
    create_tree_builder,
)
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout

logger = logging.getLogger(__name__)


def _hf_dflash_config(hf_config) -> dict:
    raw = getattr(hf_config, "dflash_config", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    return vars(raw)


class AscendTreeSpeculator(AscendDFlashSpeculator):
    """Parallel-draft tree host (DFlash or DSpark draft) + topology builder.

    Draft model forward is one parallel pass. Tree construction replaces
    per-position single-token sampling via ``self.tree_builder``.

    Domino checkpoints with ``shift_label=true`` use an N-query layout
    (bonus + N-1 masks) and sample the bonus hidden as draft slot 0.
    Vanilla DFlash / priority / beam keep the 1+N mask-only layout (beam uses
    DSpark's own query layout).

    ``propose()`` still returns flattened non-root tokens so the existing
    runner call stays valid. The tree itself is ``self.tree`` after
    ``propose()`` / ``_generate_draft()``.
    """

    _speculator_name = "DFlashTree"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # DFlashSpeculator.__init__ rejects dflash_config.sample_from_anchor.
        # Clear that so super() can run. DSpark and Domino+shift_label then
        # set the instance flags (N queries, sample the bonus as slot 0).
        draft_hf = vllm_config.speculative_config.draft_model_config.hf_config
        dflash_cfg = getattr(draft_hf, "dflash_config", None)
        if isinstance(dflash_cfg, dict) and dflash_cfg.get("sample_from_anchor"):
            draft_hf.dflash_config = {**dflash_cfg, "sample_from_anchor": False}
        elif dflash_cfg is not None and getattr(dflash_cfg, "sample_from_anchor", False):
            dflash_cfg.sample_from_anchor = False
        super().__init__(vllm_config, device)
        if self.speculative_config.use_dspark():
            self.sample_from_anchor = getattr(draft_hf, "sample_from_anchor", True)
            if self.sample_from_anchor:
                self.num_query_per_req = self.num_speculative_steps
            else:
                self.num_query_per_req = 1 + self.num_speculative_steps
        if self.use_local_argmax_reduction:
            raise ValueError(
                "DFlash tree speculator needs full draft logits; "
                "disable use_local_argmax_reduction."
            )

        tree_cfg = get_ascend_config().tree_spec_config
        self.method = tree_cfg.method
        self.budget = tree_cfg.budget
        self.topk = tree_cfg.topk
        self.params = tree_cfg.params
        self.draft_backend = (
            "dspark" if self.speculative_config.use_dspark() else "dflash"
        )

        self.tree_builder = None
        self._domino_scorer = None
        self._domino_prefix_len = 0
        self._tree_finalized = True
        dflash_cfg = _hf_dflash_config(draft_hf)
        self._domino_shift_label = (
            self.method == "prefix"
            and self.draft_backend == "dflash"
            and dflash_cfg.get("projector_type") == "domino"
            and bool(dflash_cfg.get("shift_label", False))
        )
        if self._domino_shift_label:
            self.sample_from_anchor = True
            self.num_query_per_req = self.num_speculative_steps
        # Persistent so FULL replay can update hidden without re-entering Python.
        self._draft_hidden_buf = torch.empty(
            self.max_num_reqs * self.num_query_per_req,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )
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
        self.tree_proposal_logits: torch.Tensor | None = None
        if tree_cfg.rejection_sampler == "magicmtp":
            # Node-indexed proposal for MagicMTP: column j = M_s at node j.
            # Must carry Domino / Markov corrections from the builders.
            self.tree_proposal_logits = torch.full(
                (
                    self.max_num_reqs,
                    self.budget + 1,
                    self.vocab_size,
                ),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )
        logger.info(
            "Tree speculator enabled: method=%s budget=%s topk=%s "
            "depth=%s sample_from_anchor=%s num_query_per_req=%s "
            "draft_backend=%s domino_shift_label=%s magicmtp=%s",
            self.method,
            self.budget,
            self.topk,
            self.num_speculative_steps,
            self.sample_from_anchor,
            self.num_query_per_req,
            self.draft_backend,
            self._domino_shift_label,
            self.tree_proposal_logits is not None,
        )
        from vllm_ascend.worker.v2.spec_decode.tree.timer import configure_tree_timer
        from vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch import (
            use_tree_triton,
        )

        self._prefix_graphs: dict[int, Any] = {}
        timer_backend = "triton" if use_tree_triton() else "torch"
        configure_tree_timer(
            enabled=bool(tree_cfg.enable_timer),
            backend=timer_backend,
            meta={
                "method": self.method,
                "budget": self.budget,
                "topk": self.topk,
                "depth": self.num_speculative_steps,
                "rejection_sampler": tree_cfg.rejection_sampler,
                "enable_triton": bool(tree_cfg.enable_triton),
            },
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        if self.draft_backend == "dspark":
            from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

            return load_dspark_model(target_model, self.vllm_config)
        return super().load_draft_model(target_model, target_attn_layer_names)

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self._bind_correction_heads(target_model)

    def _bind_correction_heads(self, target_model: nn.Module) -> None:
        """Resolve Domino heads (prefix) then construct the tree builder."""
        from vllm_ascend.worker.v2.spec_decode.tree.prefix import (
            DominoCorrectionScorer,
        )

        model = self.model
        draft_model = None
        if self.method == "beam":
            draft_model = model
        if (
            self.method == "prefix"
            and getattr(model, "projector_type", None) == "domino"
        ):
            self._domino_prefix_len = int(model.pure_draft_prefix_len)
            language_model = (
                target_model.get_language_model()
                if hasattr(target_model, "get_language_model")
                else target_model
            )
            self._domino_scorer = DominoCorrectionScorer(model, language_model)

        self.tree_builder = create_tree_builder(
            method=self.method,
            budget=self.budget,
            topk=self.topk,
            draft_backend=self.draft_backend,
            draft_model=draft_model,
            correction_scorer=self._domino_scorer,
            prefix_len=self._domino_prefix_len,
            params=self.params,
        )
        logger.info(
            "Tree correction heads: markov=%s domino=%s prefix_len=%s "
            "shift_label=%s",
            draft_model is not None,
            self._domino_scorer is not None,
            self._domino_prefix_len,
            self._domino_shift_label,
        )

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
            from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time

            tensors = [last_hidden_states]
            if aux_hidden_states:
                tensors.extend(aux_hidden_states)
            with tree_time("compact_query_path"):
                compact_tree_query_along_path(
                    tensors,
                    input_batch.query_start_loc,
                    path_node_ids,
                    linearize_positions=input_batch.positions,
                )
        self._tree_finalized = False
        tokens = super().propose(
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
        # FULL replay only runs draft forward; prefix may replay a second graph.
        if not dummy_run:
            self._finalize_tree(input_batch.num_reqs)
        return tokens

    def capture(self) -> None:
        logger.info("Capturing model for %s speculator...", self._speculator_name)
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        self.query_cudagraph_manager.capture(
            self._run_draft_forward,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=self._group_causal,
            progress_bar_desc=f"Capturing {self._speculator_name.lower()} CUDA graphs",
        )
        self._capture_prefix_graphs()

    def _draft_capture_num_reqs(self) -> list[int]:
        manager = self.query_cudagraph_manager
        if manager is None:
            return []
        seen: set[int] = set()
        qlen = max(int(self.num_query_per_req), 1)
        for descs in manager._capture_descs.values():
            for desc in descs:
                n = desc.num_reqs
                if not n:
                    n = desc.num_tokens // qlen
                if n:
                    seen.add(int(n))
        return sorted(seen)

    def _prepare_prefix_graph_scratch(self) -> None:
        from vllm_ascend.worker.v2.spec_decode.tree.layout import (
            ensure_finalize_scratch,
        )

        gru_hidden_dim = 0
        if self._domino_scorer is not None:
            gru_hidden_dim = self._domino_scorer.gru_hidden_dim
        self.tree_builder._ensure_scratch(
            self.max_num_reqs,
            self.vocab_size,
            self.num_speculative_steps,
            self.device,
            need_proposal=self.tree_proposal_logits is not None,
            gru_hidden_dim=gru_hidden_dim,
            dtype=self.dtype,
        )
        self.tree_builder.freeze_scratch()
        ensure_finalize_scratch(self.max_num_reqs, self.device)

    def _run_prefix_finalize(self, num_reqs: int, *, graph_safe: bool) -> None:
        """compute_logits + prefix build; captured as one NPUGraph per num_reqs."""
        hidden = self._draft_hidden_buf
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = hidden[self.sample_indices[:num_sample]]
        logits = self.model.compute_logits(sample_hidden_states)
        logits = logits.view(num_reqs, self.num_speculative_steps, -1)
        layout = self._load_layout_from_buffers(num_reqs)
        nqp = self.num_query_per_req
        root_token_ids = self.input_buffers.input_ids[: num_reqs * nqp].view(
            num_reqs, nqp
        )[:, 0]
        proposal = None
        if self.tree_proposal_logits is not None:
            proposal = self.tree_proposal_logits[:num_reqs, : self.budget + 1]
        self.tree = self.tree_builder.build(
            logits,
            layout,
            root_token_ids=root_token_ids,
            draft_hidden=sample_hidden_states.view(
                num_reqs, self.num_speculative_steps, -1
            ),
            proposal_logits=proposal,
            graph_safe=graph_safe,
        )

    def _capture_prefix_graphs(self) -> None:
        if self.method != "prefix" or self.tree_builder is None:
            return
        manager = self.query_cudagraph_manager
        if manager is None or not manager.needs_capture():
            return
        sizes = self._draft_capture_num_reqs()
        if not sizes:
            return
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            return
        from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
        from vllm.platforms import current_platform

        self._prepare_prefix_graph_scratch()
        pool = current_platform.get_global_graph_pool()
        for num_reqs in sizes:
            try:
                validate_cudagraph_capturing_enabled()
                with torch.inference_mode():
                    self._run_prefix_finalize(num_reqs, graph_safe=True)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, pool=pool):
                        self._run_prefix_finalize(num_reqs, graph_safe=True)
                self._prefix_graphs[num_reqs] = graph
                logger.info("Captured prefix tree ACLGraph num_reqs=%s", num_reqs)
            except Exception as exc:
                logger.warning(
                    "Prefix tree ACLGraph capture failed for num_reqs=%s; "
                    "eager fallback for this size. %s",
                    num_reqs,
                    exc,
                )

    def _run_draft_forward(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Draft model forward only; prefix tree uses a separate ACLGraph."""
        from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time

        with tree_time("draft_model_forward"):
            hidden = self._run_model(
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        self._draft_hidden_buf[:num_tokens_padded].copy_(hidden)
        self._tree_finalized = False

    def _finalize_tree(self, num_reqs: int) -> None:
        if self.tree_builder is None or self._tree_finalized:
            return
        from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time

        if self.method == "prefix":
            graph = self._prefix_graphs.get(num_reqs)
            if graph is not None:
                graph.replay()
                self.tree = self._load_layout_from_buffers(num_reqs)
                self._tree_finalized = True
                return
            self._run_prefix_finalize(num_reqs, graph_safe=False)
            self._tree_finalized = True
            return

        hidden = self._draft_hidden_buf
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = hidden[self.sample_indices[:num_sample]]
        if self.method == "beam":
            logits = self.model.compute_draft_logits(sample_hidden_states)
        else:
            logits = self.model.compute_logits(sample_hidden_states)
        logits = logits.view(num_reqs, self.num_speculative_steps, -1)
        layout = self._load_layout_from_buffers(num_reqs)
        nqp = self.num_query_per_req
        root_token_ids = self.input_buffers.input_ids[: num_reqs * nqp].view(
            num_reqs, nqp
        )[:, 0]
        proposal = None
        if self.tree_proposal_logits is not None:
            proposal = self.tree_proposal_logits[:num_reqs, : self.budget + 1]
        build_kwargs = dict(
            root_token_ids=root_token_ids,
            draft_hidden=sample_hidden_states.view(
                num_reqs, self.num_speculative_steps, -1
            ),
            proposal_logits=proposal,
        )
        with tree_time("build_draft_tree"):
            self.tree = self.tree_builder.build(logits, layout, **build_kwargs)
        self._tree_finalized = True

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._run_draft_forward(
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._finalize_tree(num_reqs)

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
