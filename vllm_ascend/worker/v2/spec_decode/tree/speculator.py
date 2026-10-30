from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.tree.builder import (
    create_tree_builder,
)
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout
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


def _dspark_swa_prefix(layer) -> str:
    attn = getattr(layer, "self_attn", None)
    dsa = getattr(attn, "dsa_attn", attn)
    swa = getattr(dsa, "swa_cache_layer", None)
    return str(getattr(swa, "prefix", "") or "")


def _gid_for_swa_prefix(prefix: str, groups) -> int | None:
    """KV-cache group id for a DSpark SWA prefix.

    DSA builders set ``cache_group_key = layer_names[0]``. A SWA name can
    also appear in a later group; last-wins membership then maps ``mtp.1``
    onto the 192+ pool. Prefer the group whose first name equals ``prefix``.
    """
    if not prefix:
        return None
    first_key = None
    member = None
    for gid, group in enumerate(groups):
        names = list(group.layer_names)
        if names and names[0] == prefix:
            first_key = gid
        elif member is None and prefix in names:
            member = gid
        elif member is None:
            for ln in names:
                if prefix.startswith(ln + ".") or ln.startswith(prefix + "."):
                    member = gid
                    break
    if first_key is not None:
        return first_key
    return member


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
        from vllm_ascend.worker.v2.spec_decode import dsv4_dspark_draft

        self._dsv4_dspark_draft = dsv4_dspark_draft(vllm_config)
        self.tree_shape = TreeShape(self.num_speculative_steps, self.budget, self.topk, self._dsv4_dspark_draft)
        self.tree_graph_stats = TreeGraphStats()
        if self._dsv4_dspark_draft and self.topk > 1:
            if not vllm_config.model_config.enforce_eager:
                if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL:
                    raise ValueError("DSV4 tree graph execution requires FULL mode")
            parallel = vllm_config.parallel_config
            for name in (
                "pipeline_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            ):
                if int(getattr(parallel, name, 1)) != 1:
                    raise ValueError(f"This V4 tree patch requires {name}=1")
            if getattr(vllm_config, "kv_transfer_config", None) is not None:
                raise ValueError("Tree branch commitment is not integrated with KV transfer")

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
        # DSpark combine_hidden_states / hc_head emit hf hidden_size, not the
        # HC-widened (hc_mult * H) buffer DFlash/MTP allocate via
        # get_hidden_size(). DSV4 hc_mult=4: dest [T, 4H] vs src [T, H].
        if self.draft_backend == "dspark":
            draft_hidden = int(getattr(draft_hf, "hidden_size", 0) or 0)
            if draft_hidden <= 0:
                draft_hidden = int(self.draft_model_config.get_hidden_size())
            self.hidden_size = draft_hidden
            self.hidden_states = torch.zeros(
                self.max_num_tokens,
                draft_hidden,
                dtype=self.dtype,
                device=device,
            )
        # Accepted-prefix hidden from the DSV4 commit forward. Next propose
        # must write DSpark KV from this prefix, not packed-path rows.
        self._dsv4_commit_n = None
        self._dsv4_commit_pos = None
        self._dsv4_commit_qsl = None
        self._dsv4_commit_aux = None
        self._dsv4_commit_hidden = None
        self._dsv4_chunk_prefix_len = None
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
        # DSV4 DSpark is MTP-structure: chain sampling is sequential Markov
        # over one parallel backbone pass. topk=1 is that chain; reuse the
        # DSpark sampler instead of beam log_softmax/topk.
        if self._dsv4_dspark_draft:
            self._step_cols = torch.arange(self.num_speculative_steps, dtype=torch.int32, device=device)
            self._anchor_idx = (
                torch.arange(self.max_num_reqs, dtype=torch.int64, device=device) * self.num_query_per_req
            )
            self._d2t_scatter_index = None
            self._draft_scatter_buf = None
            # Ascend DSV4 DSpark has no apply_markov_bias_gathered; use dense
            # sequential Markov (same as the working chain DSpark path).
            self._draft_topk = None
            self.draft_token_confidence_probs = torch.empty_like(self.draft_tokens, dtype=torch.float32)
            self.enable_adaptive_verification = bool(
                getattr(self.speculative_config, "enable_adaptive_verification", False)
            )
            self._sample_logits = DSparkSpeculator._sample_logits.__get__(self)
            self._sample_sequential = DSparkSpeculator._sample_sequential.__get__(self)

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

    def packed_dsv4_verify(self) -> bool:
        """Packed ori + skip-C4 target verify (DSV4 DSpark, topk>1)."""
        return self._dsv4_dspark_draft and self.topk > 1

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
        if self._dsv4_dspark_draft:
            from vllm.config import set_current_vllm_config

            with set_current_vllm_config(self.attn_vllm_config):
                super().set_attn(
                    model_state,
                    kv_cache_config,
                    block_tables,
                    target_input_buffers,
                    target_attn_groups,
                )
                self._install_dspark_all_group_context_kv(kv_cache_config)
                self.supports_decode_metadata_update = False
            return
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self.supports_decode_metadata_update = False

    def _bind_dspark_swa_group_idx(self, kv_cache_config: Any) -> None:
        """Pair each DSpark SWA prefix with a ``_context_slot_mappings`` row.

        DFlash ``_layer_group_idx`` follows ``get_draft_kv_cache_layer_names``.
        Rebuild it in ``layers.values()`` order so each layer's ``swa_cache``
        keeps the 96+ / 192+ row of its own kv-cache group.
        """
        inner = getattr(self.model, "model", self.model)
        layers = getattr(inner, "layers", None)
        groups = getattr(kv_cache_config, "kv_cache_groups", None)
        ids = list(getattr(self, "draft_kv_cache_group_ids", None) or [])
        if layers is None or not groups or not ids:
            return
        gid_to_idx = {gid: i for i, gid in enumerate(ids)}
        name_to_gidx: dict[str, int] = {}
        aligned: list[int] = []
        for layer in layers.values():
            prefix = _dspark_swa_prefix(layer)
            gid = _gid_for_swa_prefix(prefix, groups)
            if gid is None or gid not in gid_to_idx:
                raise RuntimeError(f"DSpark SWA layer has no context KV group: {prefix!r}")
            gidx = gid_to_idx[gid]
            name_to_gidx[prefix] = gidx
            aligned.append(gidx)
        self._dspark_swa_name_to_gidx = name_to_gidx
        if aligned:
            self._layer_group_idx = aligned

    def _dspark_slots_by_swa_name(self, n: int) -> dict[str, Any]:
        maps = getattr(self, "_context_slot_mappings", None)
        name_to_gidx = getattr(self, "_dspark_swa_name_to_gidx", None) or {}
        if maps is None or n <= 0:
            return {}
        out: dict[str, Any] = {}
        for prefix, gidx in name_to_gidx.items():
            if not 0 <= gidx < maps.shape[0] or n > maps.shape[1]:
                raise RuntimeError(f"DSpark SWA context slot row is invalid: {prefix!r}")
            out[prefix] = maps[gidx, :n]
        return out

    def _install_dspark_all_group_context_kv(self, kv_cache_config: Any) -> None:
        """Give each DSpark SWA cache the slot row for its own kv-cache group."""
        inner = getattr(self.model, "model", self.model)
        if not hasattr(inner, "_store_paged_kv") or not hasattr(inner, "layers"):
            return
        self._bind_dspark_swa_group_idx(kv_cache_config)
        if getattr(self, "_dspark_ctx_kv_wrapped", False):
            return
        orig = self.model.precompute_and_store_context_kv

        def _precompute(context_states, context_positions, context_slot_mapping=None):
            n = int(context_states.shape[0])
            bound = getattr(self, "_dspark_swa_name_to_gidx", None)
            inner._draft_kv_slots_by_name = self._dspark_slots_by_swa_name(n) if bound else None
            try:
                orig(context_states, context_positions, context_slot_mapping)
            finally:
                inner._draft_kv_slots_by_name = None

        self.model.precompute_and_store_context_kv = _precompute
        self._dspark_ctx_kv_wrapped = True

    def _dsv4_draft_is_prefilling(self, is_prefilling_np, num_reqs: int | None = None):
        """Preserve real request phases and clear padded draft rows."""
        n_req = int(num_reqs) if num_reqs is not None else int(self.max_num_reqs)
        out = np.zeros(self.max_num_reqs, dtype=bool)
        if is_prefilling_np is None:
            return torch.from_numpy(out)
        src = np.asarray(is_prefilling_np, dtype=bool)
        n = min(n_req, src.shape[0], out.shape[0])
        out[:n] = src[:n]
        return torch.from_numpy(out)

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        if not self._dsv4_dspark_draft:
            return super().build_draft_attn_metadatas(num_reqs_padded, seq_lens_cpu_upper_bound)
        from vllm_ascend.worker.v2.attn_utils import (
            build_attn_metadata_wrapper,
            build_draft_attn_metadata_factory,
        )

        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        with (
            build_attn_metadata_wrapper(),
            build_draft_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens_padded,
                self._dsv4_draft_is_prefilling(
                    self.input_batch.is_prefilling_np,
                    self.input_batch.num_reqs,
                ),
            ),
        ):
            attn_metadata = self._build_uniform_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                batch_desc=BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.FULL,
                    num_tokens=num_tokens_padded,
                    num_reqs=num_reqs_padded,
                ),
                num_query_per_req=self.num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
        return [attn_metadata]

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

    def _align_dspark_copy_buffer(self, width: int) -> None:
        """DFlash.propose copy_ dest must match combine(aux) / last_hidden."""
        buf = self.hidden_states
        if buf.shape[-1] == width:
            return
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            width,
            dtype=buf.dtype,
            device=buf.device,
        )
        if self._draft_hidden_buf.shape[-1] != width:
            self._draft_hidden_buf = torch.empty(
                self.max_num_reqs * self.num_query_per_req,
                width,
                dtype=buf.dtype,
                device=buf.device,
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
        if num_speculative_tokens is not None and num_speculative_tokens != self.num_speculative_steps:
            raise ValueError(
                "Tree speculation requires a fixed draft depth; dynamic speculative token counts are not supported."
            )
        from vllm_ascend.worker.v2.spec_decode.tree import chain_pack
        from vllm_ascend.worker.v2.spec_decode.tree.chain_pack import (
            clear_tree_chain_layout,
        )

        verification_layout = chain_pack._LAYOUT
        clear_tree_chain_layout()
        self._draft_graph_used = False
        try:
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
                    layout=verification_layout,
                    input_batch=input_batch,
                )
            self._tree_finalized = False
            # DSV4's HC inputs need width alignment. Qwen's combine_hidden_states
            # already projects target features to the fixed draft hidden width;
            # resizing its output buffer here would invalidate captured graphs.
            if self._dsv4_dspark_draft:
                copy_w = aux_hidden_states[0].shape[-1] if aux_hidden_states else last_hidden_states.shape[-1]
                self._align_dspark_copy_buffer(copy_w)
            if self._dsv4_dspark_draft:
                from vllm_ascend.worker.v2.attn_utils import (
                    build_attn_metadata_wrapper,
                    build_draft_attn_metadata_factory,
                )

                # Factory must wrap the Ascend ``build_attn_metadata`` installed by
                # the wrapper. Skip AscendDFlashSpeculator.propose so it cannot
                # reinstall the wrapper and drop the DSA positions/is_prefilling.
                self.input_batch = input_batch
                sync_state = dp_sync_state
                if dummy_run and skip_attn_for_dummy_run:
                    sync_state = None
                seq_np = None
                plen = self._dsv4_chunk_prefix_len
                if plen is not None:
                    seq_np = np.zeros(self.max_num_reqs, dtype=np.int32)
                    sl_np = getattr(self.input_buffers, "seq_lens_np", None)
                    nreq = input_batch.num_reqs
                    if sl_np is not None and sl_np.size >= nreq:
                        seq_np[:nreq] = sl_np[:nreq]
                    else:
                        seq_np[:nreq] = plen + int(self.num_query_per_req)
                with (
                    build_attn_metadata_wrapper(),
                    build_draft_attn_metadata_factory(
                        self.input_buffers.positions,
                        self.max_num_tokens,
                        self._dsv4_draft_is_prefilling(
                            input_batch.is_prefilling_np,
                            input_batch.num_reqs,
                        ),
                        seq_lens_np=seq_np,
                    ),
                ):
                    tokens = super(AscendDFlashSpeculator, self).propose(
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
                        sync_state,
                        dummy_run,
                        skip_attn_for_dummy_run,
                        mm_inputs,
                        is_profile=is_profile,
                        num_speculative_tokens=num_speculative_tokens,
                    )
            else:
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
            # FULL replay only runs draft forward; tree construction has its own graph.
            if not dummy_run:
                self.tree_graph_stats.record_lookup("draft", self._draft_graph_used)
                if self._draft_graph_used:
                    self.tree_graph_stats.record_replay("draft")
                self._finalize_tree(input_batch.num_reqs)
                if self._dsv4_dspark_draft and self.topk == 1:
                    # Storage capacity is the tree budget; the chain has only K
                    # draft tokens. Never publish the padded suffix to the runner.
                    return tokens[: input_batch.num_reqs, : self.num_speculative_steps]
                if self.tree_builder is not None:
                    # The target verifier must receive every node selected by the
                    # tree builder. ``super().propose`` returns the chain-shaped
                    # draft view, while the finalized layout contains budget nodes.
                    out = self.tree.tokens[: input_batch.num_reqs, : self.budget]
                    return out
            return tokens
        finally:
            self._dsv4_chunk_prefix_len = None

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

    def capture(self) -> None:
        logger.info("Capturing model for %s speculator...", self._speculator_name)
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
        if self.tree_kv_compact is not None and not self.packed_dsv4_verify():
            self.tree_kv_compact.capture(self._draft_capture_num_reqs())

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
        if (self._dsv4_dspark_draft and self.topk == 1) or self.tree_builder is None:
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

        tree_vocab_size = (
            self.model.config.draft_vocab_size
            if self.draft_backend == "dspark" and not self._dsv4_dspark_draft
            else self.vocab_size
        )
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
                if self.packed_dsv4_verify():
                    raise RuntimeError(f"DSV4 FULL tree construction capture failed for batch {num_reqs}") from exc
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
        if self._dsv4_dspark_draft and self.topk == 1:
            self._sample_sequential(num_reqs, self._draft_hidden_buf)
            self._materialize_chain_layout(num_reqs)
            self._tree_finalized = True
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

    def _materialize_chain_layout(self, num_reqs: int) -> None:
        """Pack sequential DSpark tokens as a depth-ordered chain tree."""
        spec = self.num_speculative_steps
        # ``TreeLayout.tokens`` is a view of ``draft_tokens``. finalize
        # ``fill_(-1)`` would wipe sequential samples if we passed that view.
        tokens = self.draft_tokens[:num_reqs, :spec].clone()
        device = tokens.device
        depths = torch.arange(1, spec + 1, dtype=torch.int32, device=device).unsqueeze(0).expand(num_reqs, -1)
        parent_ids = torch.arange(spec, dtype=torch.long, device=device).unsqueeze(0).expand(num_reqs, -1)
        self.tree = finalize_tree_layout(
            self._load_layout_from_buffers(num_reqs),
            tokens,
            depths,
            parent_ids,
            spec,
        )

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
