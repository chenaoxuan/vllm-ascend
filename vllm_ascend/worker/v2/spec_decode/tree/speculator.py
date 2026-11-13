import copy
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.tree.builder import (
    create_tree_builder,
)
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout
from vllm_ascend.worker.v2.spec_decode.tree.metrics import TreeGraphStats
from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape

logger = init_logger("vllm." + __name__)


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
    Vanilla DFlash keeps the 1+N mask-only layout; beam uses DSpark's own
    query layout.

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
            raise ValueError("DFlash tree speculator needs full draft logits; disable use_local_argmax_reduction.")

        tree_cfg = get_ascend_config().tree_spec_config
        self.budget = tree_cfg.budget
        self.topk = tree_cfg.topk
        self.candidate_size = tree_cfg.candidate_size
        self.depth_penalty = tree_cfg.depth_penalty
        self.keep_origin_chain = tree_cfg.keep_origin_chain
        self.draft_backend = "dspark" if self.speculative_config.use_dspark() else "dflash"

        self.tree_shape = TreeShape(self.num_speculative_steps, self.budget, self.topk)
        self.tree_graph_stats = TreeGraphStats()

        self.tree_builder = None
        self._domino_scorer = None
        self._domino_prefix_len = 0
        self._tree_finalized = True
        self.tree_kv_compact = None
        self.tree_commit = None
        dflash_cfg = _hf_dflash_config(draft_hf)
        self._domino_shift_label = (
            self.draft_backend == "dflash"
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
        logger.info(
            "Tree speculator enabled: draft_backend=%s budget=%s topk=%s "
            "depth=%s sample_from_anchor=%s num_query_per_req=%s "
            "domino_shift_label=%s",
            self.draft_backend,
            self.budget,
            self.topk,
            self.num_speculative_steps,
            self.sample_from_anchor,
            self.num_query_per_req,
            self._domino_shift_label,
        )
        self._tree_graphs: dict[int, Any] = {}

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        if self.draft_backend == "dspark":
            from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

            from vllm_ascend.models.qwen3_dspark import process_weight
            from vllm_ascend.utils import get_rotation_matrix, get_rotation_path

            model = load_dspark_model(target_model, self.vllm_config)
            rotation_path = get_rotation_path(self.vllm_config)
            if rotation_path is not None and hasattr(model.model, "fc"):
                rotation_weight = get_rotation_matrix(rotation_path)
                fc = model.model.fc
                with torch.no_grad():
                    fc.weight.data.copy_(process_weight(fc.weight.data.cpu(), rotation_weight))
            return model
        if "DFlash2DraftModel" in self.draft_model_config.architectures and not self.speculative_config.enforce_eager:
            from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model

            from vllm_ascend.worker.v2.spec_decode.pcp_utils import disable_profiling_chunk_for_draft

            # Capture the fixed query forward directly with ACLGraph. A private
            # compiler policy avoids retracing DFlash2 RoPE through NPU meta
            # kernels, while retaining the shared attention-layer registrations.
            config = copy.copy(self.vllm_config)
            config.compilation_config = copy.copy(config.compilation_config)
            config.compilation_config.mode = CompilationMode.NONE
            with disable_profiling_chunk_for_draft(config):
                return load_dflash_model(target_model, config)
        return super().load_draft_model(target_model, target_attn_layer_names)

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self._bind_correction_heads(target_model)

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self.supports_decode_metadata_update = False

    def _bind_correction_heads(self, target_model: nn.Module) -> None:
        """Bind model scoring heads once, then create the common tree builder."""
        from vllm_ascend.worker.v2.spec_decode.tree.scoring import (
            DominoCorrectionScorer,
        )

        model = self.model
        draft_model = None
        if self.draft_backend == "dspark":
            draft_model = model
        elif self.draft_backend == "dflash" and hasattr(getattr(model, "model", model), "candidate_selector"):
            # DFlash2 exposes a candidate-selector correction head. Pass the
            # draft model to the tree builder so it can score edges with that
            # head instead of treating each depth as independent logits.
            draft_model = model
        if self.draft_backend == "dflash" and getattr(model, "projector_type", None) == "domino":
            self._domino_prefix_len = int(model.pure_draft_prefix_len)
            language_model = (
                target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
            )
            self._domino_scorer = DominoCorrectionScorer(model, language_model)

        self.tree_builder = create_tree_builder(
            budget=self.budget,
            topk=self.topk,
            draft_backend=self.draft_backend,
            draft_model=draft_model,
            correction_scorer=self._domino_scorer,
            prefix_len=self._domino_prefix_len,
            candidate_size=self.candidate_size,
            depth_penalty=self.depth_penalty,
            keep_origin_chain=self.keep_origin_chain,
        )
        logger.info(
            "Tree correction heads: markov=%s domino=%s prefix_len=%s shift_label=%s",
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
        dp_sync_state: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        if num_speculative_tokens not in (None, 0, self.num_speculative_steps):
            raise ValueError(
                "Tree speculation requires a fixed draft depth; dynamic speculative token counts are not supported."
            )
        self._draft_graph_used = False
        acceptance = None if dummy_run else input_batch.tree_acceptance
        if acceptance is not None and not dummy_run:
            tensors = [last_hidden_states]
            if aux_hidden_states:
                tensors.extend(aux_hidden_states)
            self.tree_commit.compact_queries(
                tensors,
                input_batch.query_start_loc,
                acceptance,
                input_batch.positions,
                input_batch=input_batch,
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
            dp_sync_state,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )
        if num_speculative_tokens == 0:
            # Upstream skips proposals when the remaining output budget needs
            # no drafts. Its context-only pass still updates the drafter KV.
            return tokens
        # FULL replay only runs draft forward; tree construction has its own graph.
        if not dummy_run:
            self.tree_graph_stats.record_lookup("draft", self._draft_graph_used)
            if self._draft_graph_used:
                self.tree_graph_stats.record_replay("draft")
            self._finalize_tree(input_batch.num_reqs)
            if self.tree_builder is not None:
                # The target verifier must receive every node selected by the
                # tree builder. ``super().propose`` returns the chain-shaped
                # draft view, while the finalized layout contains budget nodes.
                out = self.tree.tokens[: input_batch.num_reqs, : self.budget]
                return out
        return tokens

    def _prepare_graph_replay(
        self,
        input_batch,
        num_graph_reqs,
        last_hidden_states,
        aux_hidden_states,
        num_rejected,
        dummy_run,
    ) -> None:
        self._draft_graph_used = True
        self._store_hidden_states(input_batch.num_tokens, last_hidden_states, aux_hidden_states)
        self.prepare_context_anchor(input_batch.num_reqs, input_batch.query_start_loc, num_rejected)
        self._precompute_context_kv(0, input_batch.num_tokens, dummy_run)
        # In-place-updatable draft backends (FA3) refresh their device inputs
        # here: with supports_decode_metadata_update the core speculator
        # skips the eager draft metadata rebuild in FULL graph mode, and the
        # runner-owned length buffers the graph reads are already refreshed
        # in place, so only the FA3 tiling needs an explicit per-replay
        # recompute.
        for groups in self.attn_groups:
            for group in groups:
                if group.supports_draft_decode_metadata_update:
                    group.get_metadata_builder().update_draft_decode_metadata()

    def _prepare_dflash2_graph_attention(self) -> None:
        if self.tree_builder is None or getattr(self.tree_builder.scorer, "name", None) != "dflash2":
            return
        from vllm_ascend.attention.attention_mask import AttentionMaskBuilder

        context = self.vllm_config.compilation_config.static_forward_context
        impls = [context[name].impl for name in self.draft_attn_layer_names]
        windowed = [impl for impl in impls if impl.sliding_window]
        if windowed:
            mask = AttentionMaskBuilder(self.device).get_splitfuse_attn_mask()
            for impl in windowed:
                impl.configure_tree_draft_window(mask)

    def capture(self) -> None:
        logger.info("Capturing model for %s speculator...", self._speculator_name)
        self._prepare_dflash2_graph_attention()
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        existing_draft_graphs = set(self.query_cudagraph_manager.graphs)
        self.query_cudagraph_manager.capture(
            self._run_draft_forward,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=self._group_causal,
            # Tree stages context eagerly in _prepare_graph_replay; the draft
            # graph must keep capturing only the parallel query forward.
            prepare_forward=lambda num_reqs, attn_metadata: None,
            progress_bar_desc=f"Capturing {self._speculator_name.lower()} CUDA graphs",
        )
        for _ in self.query_cudagraph_manager.graphs.keys() - existing_draft_graphs:
            self.tree_graph_stats.record_capture("draft")
        self.tree_graph_stats.set_workspace_bytes(
            "draft", "hidden", self.tree_graph_stats.tensor_bytes((self._draft_hidden_buf,))
        )
        self._capture_tree_graphs()
        if self.tree_kv_compact is not None:
            sizes = self._draft_capture_num_reqs()
            self.tree_kv_compact.capture(sizes)
            if self.tree_builder is not None and self.tree_builder.scorer.name == "dflash2":
                missing = set(sizes) - self.tree_kv_compact._graphs.keys()
                if missing:
                    raise RuntimeError(f"FULL DFlash2 tree KV compaction capture failed for batches {sorted(missing)}")

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

    def _run_tree_finalize(self, num_reqs: int) -> None:
        """Compute logits and build a tree from the current persistent inputs.

        Eager execution and graph capture use the same operations and output
        buffers. DSpark keeps logits in draft-vocabulary space until its
        builder applies the Markov correction and maps the selected tokens.
        """
        hidden = self._draft_hidden_buf
        num_sample = num_reqs * self.num_speculative_steps
        sample_hidden_states = hidden[self.sample_indices[:num_sample]]
        draft_hidden = sample_hidden_states.view(num_reqs, self.num_speculative_steps, -1)
        draft_candidate_ids = None
        if self.draft_backend == "dspark":
            logits = self.model.compute_draft_logits(sample_hidden_states)
        elif hasattr(getattr(self.model, "model", self.model), "candidate_selector") and hasattr(
            self.model, "compute_candidates"
        ):
            # DFlash2's selector head corrects the target-LM top-k lattice. Keep
            # the candidate IDs alongside the unary logits so the tree scorer
            # can evaluate edges against the actual frontier at each depth.
            draft_candidate_ids, logits = self.model.compute_candidates(sample_hidden_states)
        else:
            logits = self.model.compute_logits(sample_hidden_states)
        logits = logits.view(num_reqs, self.num_speculative_steps, -1)
        if draft_candidate_ids is not None:
            draft_candidate_ids = draft_candidate_ids.view(num_reqs, self.num_speculative_steps, -1)
        layout = self._load_layout_from_buffers(num_reqs)
        nqp = self.num_query_per_req
        root_token_ids = self.input_buffers.input_ids[: num_reqs * nqp].view(num_reqs, nqp)[:, 0]
        self.tree = self.tree_builder.build(
            logits,
            layout,
            root_token_ids=root_token_ids,
            draft_hidden=draft_hidden,
            draft_candidate_ids=draft_candidate_ids,
        )

    def _capture_tree_graphs(self) -> None:
        if self.tree_builder is None:
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

        from vllm_ascend.worker.v2.spec_decode.tree.layout import (
            ensure_finalize_scratch,
        )

        tree_vocab_size = self.model.config.draft_vocab_size if self.draft_backend == "dspark" else self.vocab_size
        self.tree_builder.prepare_for_capture(
            self.max_num_reqs,
            tree_vocab_size,
            self.num_speculative_steps,
            self.device,
        )
        self.tree_graph_stats.set_workspace_bytes(
            "tree", "shared", self.tree_graph_stats.tensor_bytes(self.tree_builder.workspace_tensors())
        )
        ensure_finalize_scratch(self.max_num_reqs, self.device)
        pool = current_platform.get_global_graph_pool()
        for num_reqs in sizes:
            try:
                validate_cudagraph_capturing_enabled()
                with torch.inference_mode():
                    self._run_tree_finalize(num_reqs)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, pool=pool):
                        self._run_tree_finalize(num_reqs)
                self._tree_graphs[num_reqs] = graph
                self.tree_graph_stats.record_capture("tree")
                logger.info("Captured tree ACLGraph scorer_backend=%s num_reqs=%s", self.draft_backend, num_reqs)
            except Exception as exc:
                if self.tree_builder.scorer.name == "dflash2":
                    raise RuntimeError(f"DFlash2 FULL tree construction capture failed for batch {num_reqs}") from exc
                logger.warning(
                    "Tree ACLGraph capture failed for scorer_backend=%s num_reqs=%s; eager fallback for this size. %s",
                    self.draft_backend,
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
        """Draft model forward only; tree construction uses a separate ACLGraph."""
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

        # Match the actual request count, independently of the padded batch
        # used by the draft-forward graph. Uncaptured sizes use eager build.
        graph = self._tree_graphs.get(num_reqs)
        self.tree_graph_stats.record_lookup("tree", graph is not None)
        if graph is not None:
            graph.replay()
            self.tree_graph_stats.record_replay("tree")
            self.tree = self._load_layout_from_buffers(num_reqs)
        else:
            self._run_tree_finalize(num_reqs)
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
