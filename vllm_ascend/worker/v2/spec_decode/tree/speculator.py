from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)

from vllm_ascend.worker.v2.spec_decode.tree.builder import (
    create_tree_builder,
)
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout

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


def _extra_context_slot_row(speculator, n: int):
    maps = getattr(speculator, "_context_slot_mappings", None)
    if maps is None or maps.ndim < 2 or maps.shape[0] < 2 or n <= 0:
        return None
    gidx_list = getattr(speculator, "_layer_group_idx", None) or []
    swa_row = int(gidx_list[0]) if gidx_list else 0
    extra_row = 0 if swa_row != 0 else 1
    if extra_row >= maps.shape[0]:
        return None
    return maps[extra_row, :n]


def _draft_causal_int(speculator, gid: int) -> int:
    causal = getattr(speculator, "_group_causal", None)
    if isinstance(causal, bool):
        return int(causal)
    if isinstance(causal, dict) and causal:
        if gid in causal:
            return int(bool(causal[gid]))
        return int(bool(next(iter(causal.values()))))
    return -1


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
        from vllm_ascend.worker.v2.spec_decode import dsv4_dspark_draft

        self._dsv4_dspark_draft = dsv4_dspark_draft(vllm_config)

        self.tree_builder = None
        self._domino_scorer = None
        self._domino_prefix_len = 0
        self._tree_finalized = True
        self.tree_kv_compact = None
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
        self._dsv4_chunk_bonus = -1
        self._dsv4_buf_seq0_pre = -1
        # Official-prefix rows only (prefill chunks + accepted commits).
        # Packed tree hidden never enters this buffer.
        self._dsv4_prefix_n = 0
        self._dsv4_prefix_num_reqs = 0
        self._dsv4_prefix_hidden = None
        self._dsv4_prefix_pos = None
        self._dsv4_prefix_qsl = None
        self._dsv4_prefix_aux = None
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
            self._step_cols = torch.arange(
                self.num_speculative_steps, dtype=torch.int32, device=device
            )
            self._anchor_idx = (
                torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
                * self.num_query_per_req
            )
            self._d2t_scatter_index = None
            self._draft_scatter_buf = None
            # Ascend DSV4 DSpark has no apply_markov_bias_gathered; use dense
            # sequential Markov (same as the working chain DSpark path).
            self._draft_topk = None
            self.draft_token_confidence_probs = torch.empty_like(
                self.draft_tokens, dtype=torch.float32
            )
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
            from vllm_ascend.models.qwen3_dspark import process_weight
            from vllm_ascend.utils import get_rotation_matrix, get_rotation_path

            model = load_dspark_model(target_model, self.vllm_config)
            rotation_path = get_rotation_path(self.vllm_config)
            if rotation_path is not None and hasattr(model.model, "fc"):
                rotation_weight = get_rotation_matrix(rotation_path)
                fc = model.model.fc
                with torch.no_grad():
                    fc.weight.data.copy_(
                        process_weight(fc.weight.data.cpu(), rotation_weight)
                    )
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
            return
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )

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
                continue
            gidx = gid_to_idx[gid]
            name_to_gidx[prefix] = gidx
            aligned.append(gidx)
        self._dspark_swa_name_to_gidx = name_to_gidx
        if aligned:
            self._layer_group_idx = aligned
        from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
            path_log,
        )

        gkeys = []
        for gid in ids:
            names = list(groups[gid].layer_names) if gid < len(groups) else []
            gkeys.append(names[0] if names else "")
        pfxs = []
        ratios = []
        for layer in layers.values():
            pfxs.append(_dspark_swa_prefix(layer).replace(".self_attn.swa_cache", ".swa"))
            attn = getattr(layer, "self_attn", None)
            ratios.append(int(getattr(attn, "compress_ratio", -1) or -1))
        path_log(
            "swa_bind pfx=%s ratio=%s gidx=%s ids=%s gkeys=%s",
            pfxs,
            ratios,
            aligned,
            ids,
            [g.replace(".self_attn.swa_cache", ".swa") for g in gkeys],
        )

    def _dspark_slots_by_swa_name(self, n: int) -> dict[str, Any]:
        maps = getattr(self, "_context_slot_mappings", None)
        name_to_gidx = getattr(self, "_dspark_swa_name_to_gidx", None) or {}
        if maps is None or n <= 0:
            return {}
        out: dict[str, Any] = {}
        for prefix, gidx in name_to_gidx.items():
            if 0 <= gidx < maps.shape[0]:
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
            inner._draft_kv_log_slot_b = _extra_context_slot_row(self, n)
            inner._draft_kv_log_extra_n = 0
            bound = getattr(self, "_dspark_swa_name_to_gidx", None)
            inner._draft_kv_slots_by_name = (
                self._dspark_slots_by_swa_name(n) if bound else None
            )
            inner._draft_kv_name_gidx = bound or {}
            orig(context_states, context_positions, context_slot_mapping)
            inner._draft_kv_log_slot_b = None
            inner._draft_kv_log_extra_n = 0
            inner._draft_kv_slots_by_name = None
            inner._draft_kv_name_gidx = None

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
            return super().build_draft_attn_metadatas(
                num_reqs_padded, seq_lens_cpu_upper_bound
            )
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
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens_padded=num_tokens_padded,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
        return [attn_metadata]

    def _build_draft_attn_metadata(self, *args, **kwargs):
        """After prepare_dflash_inputs: keep DSA seq_lens on the official prefix."""
        prefix = self._dsv4_chunk_prefix_len
        num_reqs = kwargs.get("num_reqs")
        if num_reqs is None and args:
            num_reqs = args[0]
        if prefix is not None and num_reqs:
            self._dsv4_buf_seq0_pre = self._dsv4_sync_draft_seq_lens(
                int(num_reqs), prefix
            )
        return super()._build_draft_attn_metadata(*args, **kwargs)

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

    def _dsv4_ensure_prefix_bufs(
        self,
        hidden: torch.Tensor,
        aux: list[torch.Tensor] | None,
        pos: torch.Tensor,
        *,
        allow_new_aux: bool,
    ) -> None:
        cap = self.max_num_tokens
        device = hidden.device
        need_h = (
            self._dsv4_prefix_hidden is None
            or self._dsv4_prefix_hidden.shape[-1] != hidden.shape[-1]
            or self._dsv4_prefix_hidden.dtype != hidden.dtype
        )
        if need_h:
            self._dsv4_prefix_hidden = torch.empty(
                cap, hidden.shape[-1], dtype=hidden.dtype, device=device
            )
            self._dsv4_prefix_pos = torch.empty(
                cap, dtype=pos.dtype, device=device
            )
            self._dsv4_prefix_qsl = torch.zeros(
                self.max_num_reqs + 1, dtype=torch.int32, device=device
            )
        if not aux:
            self._dsv4_prefix_aux = None
            return
        need_aux = (
            self._dsv4_prefix_aux is None
            or len(self._dsv4_prefix_aux) != len(aux)
            or self._dsv4_prefix_aux[0].shape[-1] != aux[0].shape[-1]
        )
        if not need_aux:
            return
        if allow_new_aux:
            self._dsv4_prefix_aux = [
                torch.empty(cap, a.shape[-1], dtype=a.dtype, device=device)
                for a in aux
            ]
        else:
            self._dsv4_prefix_aux = None

    def _dsv4_copy_chunk_into_prefix(
        self,
        dst_off: int,
        hidden: torch.Tensor,
        aux: list[torch.Tensor] | None,
        pos: torch.Tensor,
        n: int,
    ) -> None:
        self._dsv4_prefix_hidden[dst_off : dst_off + n].copy_(hidden[:n])
        self._dsv4_prefix_pos[dst_off : dst_off + n].copy_(
            pos[:n].to(dtype=self._dsv4_prefix_pos.dtype)
        )
        if aux and self._dsv4_prefix_aux:
            for dst, src in zip(self._dsv4_prefix_aux, aux):
                dst[dst_off : dst_off + n].copy_(src[:n])

    def _dsv4_prefix_merge(
        self,
        hidden: torch.Tensor,
        aux: list[torch.Tensor] | None,
        pos: torch.Tensor,
        qsl: torch.Tensor,
        num_reqs: int,
        replace: bool,
    ) -> bool:
        """Pack official-prefix rows. ``True`` when the buffer can be bound.

        ``num_reqs`` / chunk length are host ints from commit or ``qsl_np``.
        """
        chunk_n = hidden.shape[0]
        cap = self.max_num_tokens
        if chunk_n > cap:
            from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
                path_log,
            )

            path_log(
                "draft_kv prefix_cap n_tok=%s cap=%s fallback=chunk",
                chunk_n,
                cap,
            )
            self._dsv4_prefix_n = 0
            return False
        if (
            replace
            or self._dsv4_prefix_n == 0
            or num_reqs != self._dsv4_prefix_num_reqs
        ):
            self._dsv4_ensure_prefix_bufs(hidden, aux, pos, allow_new_aux=True)
            self._dsv4_copy_chunk_into_prefix(0, hidden, aux, pos, chunk_n)
            self._dsv4_prefix_qsl[: num_reqs + 1].copy_(
                qsl[: num_reqs + 1].to(dtype=self._dsv4_prefix_qsl.dtype)
            )
            self._dsv4_prefix_n = chunk_n
            self._dsv4_prefix_num_reqs = num_reqs
            return True
        new_n = self._dsv4_prefix_n + chunk_n
        if new_n > cap:
            from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
                path_log,
            )

            path_log(
                "draft_kv prefix_cap n_tok=%s cap=%s fallback=chunk",
                new_n,
                cap,
            )
            return False
        if (
            self._dsv4_prefix_hidden is not None
            and (
                self._dsv4_prefix_hidden.shape[-1] != hidden.shape[-1]
                or self._dsv4_prefix_hidden.dtype != hidden.dtype
            )
        ):
            from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
                path_log,
            )

            path_log(
                "draft_kv prefix_width have=%s chunk=%s fallback=chunk",
                self._dsv4_prefix_hidden.shape[-1],
                hidden.shape[-1],
            )
            return False
        self._dsv4_ensure_prefix_bufs(hidden, aux, pos, allow_new_aux=False)
        if num_reqs == 1:
            off = self._dsv4_prefix_n
            self._dsv4_copy_chunk_into_prefix(off, hidden, aux, pos, chunk_n)
            new_end = self._dsv4_prefix_qsl[1] + qsl[1].to(
                dtype=self._dsv4_prefix_qsl.dtype
            )
            self._dsv4_prefix_qsl[1].copy_(new_end)
            self._dsv4_prefix_n = new_n
            return True
        old_qsl = self._dsv4_prefix_qsl[: num_reqs + 1].cpu().numpy()  # D2H
        chunk_qsl = qsl[: num_reqs + 1].cpu().numpy()  # D2H
        packed_h = hidden.new_empty((new_n, hidden.shape[-1]))
        packed_p = pos.new_empty((new_n,))
        packed_aux = (
            [a.new_empty((new_n, a.shape[-1])) for a in aux] if aux else None
        )
        out_qsl = np.zeros(num_reqs + 1, dtype=np.int32)
        dst = 0
        for i in range(num_reqs):
            o0, o1 = int(old_qsl[i]), int(old_qsl[i + 1])
            c0, c1 = int(chunk_qsl[i]), int(chunk_qsl[i + 1])
            ol = o1 - o0
            cl = c1 - c0
            packed_h[dst : dst + ol].copy_(self._dsv4_prefix_hidden[o0:o1])
            packed_p[dst : dst + ol].copy_(self._dsv4_prefix_pos[o0:o1])
            if packed_aux and self._dsv4_prefix_aux:
                for d, s in zip(packed_aux, self._dsv4_prefix_aux):
                    d[dst : dst + ol].copy_(s[o0:o1])
            dst += ol
            packed_h[dst : dst + cl].copy_(hidden[c0:c1])
            packed_p[dst : dst + cl].copy_(pos[c0:c1])
            if packed_aux and aux:
                for d, s in zip(packed_aux, aux):
                    d[dst : dst + cl].copy_(s[c0:c1])
            dst += cl
            out_qsl[i + 1] = dst
        self._dsv4_prefix_hidden[:new_n].copy_(packed_h)
        self._dsv4_prefix_pos[:new_n].copy_(packed_p)
        if packed_aux and self._dsv4_prefix_aux:
            for d, s in zip(self._dsv4_prefix_aux, packed_aux):
                d[:new_n].copy_(s)
        self._dsv4_prefix_qsl[: num_reqs + 1].copy_(
            torch.from_numpy(out_qsl).to(
                device=self._dsv4_prefix_qsl.device,
                dtype=self._dsv4_prefix_qsl.dtype,
            )  # H2D
        )
        self._dsv4_prefix_n = new_n
        return True

    def _dsv4_absorb_official_prefix(
        self,
        hidden: torch.Tensor,
        aux: list[torch.Tensor] | None,
        input_batch: InputBatch,
    ) -> None:
        qsl_np = input_batch.query_start_loc_np
        n = min(int(qsl_np[-1]), hidden.shape[0], input_batch.positions.shape[0])
        if n <= 0:
            return
        num_reqs = input_batch.num_reqs
        computed = getattr(input_batch, "num_computed_tokens_np", None)
        replace = self._dsv4_prefix_n == 0
        if computed is not None and computed.size >= num_reqs:
            replace = replace or bool((computed[:num_reqs] == 0).all())
        self._dsv4_prefix_merge(
            hidden[:n],
            None if not aux else [a[:n] for a in aux],
            input_batch.positions[:n],
            input_batch.query_start_loc[: num_reqs + 1],
            num_reqs,
            replace,
        )

    def _dsv4_snapshot_batch(self, input_batch: InputBatch) -> dict[str, Any]:
        """References plus seq-len values mutated by the commit-chunk bind."""
        nreq = input_batch.num_reqs
        sl = input_batch.seq_lens
        ub = input_batch.seq_lens_cpu_upper_bound
        np_sl = input_batch.seq_lens_np
        cnp = getattr(input_batch, "num_computed_tokens_np", None)
        ccpu = getattr(input_batch, "num_computed_tokens_cpu", None)
        computed_cpu = None
        if ccpu is not None and torch.is_tensor(ccpu):
            computed_cpu = ccpu[:nreq].clone()
        elif ccpu is not None:
            computed_cpu = np.array(ccpu[:nreq], copy=True)
        return {
            "positions": input_batch.positions,
            "qsl": input_batch.query_start_loc,
            "qsl_np": input_batch.query_start_loc_np,
            "num_tokens": input_batch.num_tokens,
            "num_tokens_after_padding": input_batch.num_tokens_after_padding,
            "num_scheduled_tokens": input_batch.num_scheduled_tokens,
            "seq_lens": sl,
            "seq_lens_data": None if sl is None else sl[:nreq].clone(),
            "seq_lens_np": None if np_sl is None else np.array(np_sl[:nreq], copy=True),
            "seq_ub": ub,
            "seq_ub_data": None if ub is None else ub[:nreq].clone(),
            "max_query_len": getattr(input_batch, "max_query_len", None),
            "num_reqs": nreq,
            "computed_np": None if cnp is None else np.array(cnp[:nreq], copy=True),
            "computed_cpu": computed_cpu,
            "computed_cpu_is_tensor": torch.is_tensor(ccpu) if ccpu is not None else False,
        }

    def _dsv4_restore_batch(self, input_batch: InputBatch, snap: dict[str, Any]) -> None:
        nreq = snap["num_reqs"]
        input_batch.positions = snap["positions"]
        input_batch.query_start_loc = snap["qsl"]
        input_batch.query_start_loc_np = snap["qsl_np"]
        input_batch.num_tokens = snap["num_tokens"]
        input_batch.num_tokens_after_padding = snap["num_tokens_after_padding"]
        input_batch.num_scheduled_tokens = snap["num_scheduled_tokens"]
        input_batch.seq_lens = snap["seq_lens"]
        if snap["seq_lens_data"] is not None and input_batch.seq_lens is not None:
            input_batch.seq_lens[:nreq].copy_(snap["seq_lens_data"])
        if snap["seq_lens_np"] is not None and input_batch.seq_lens_np is not None:
            input_batch.seq_lens_np[:nreq] = snap["seq_lens_np"]
        input_batch.seq_lens_cpu_upper_bound = snap["seq_ub"]
        if snap["seq_ub_data"] is not None and input_batch.seq_lens_cpu_upper_bound is not None:
            input_batch.seq_lens_cpu_upper_bound[:nreq].copy_(snap["seq_ub_data"])
        if snap["max_query_len"] is not None:
            input_batch.max_query_len = snap["max_query_len"]
        if snap["computed_np"] is not None:
            cnp = getattr(input_batch, "num_computed_tokens_np", None)
            if cnp is not None:
                cnp[:nreq] = snap["computed_np"]
        if snap["computed_cpu"] is not None:
            ccpu = getattr(input_batch, "num_computed_tokens_cpu", None)
            if ccpu is not None and snap["computed_cpu_is_tensor"]:
                ccpu[:nreq].copy_(snap["computed_cpu"])
            elif ccpu is not None:
                ccpu[:nreq] = snap["computed_cpu"]

    def _dsv4_fill_computed(self, input_batch: InputBatch, num_reqs: int, prefix_len: int) -> None:
        """Bind host computed-token counts to the official prefix length."""
        cnp = getattr(input_batch, "num_computed_tokens_np", None)
        if cnp is not None and cnp.size >= num_reqs:
            cnp[:num_reqs] = prefix_len
        ccpu = getattr(input_batch, "num_computed_tokens_cpu", None)
        if ccpu is None:
            return
        if torch.is_tensor(ccpu):
            if ccpu.numel() >= num_reqs:
                ccpu[:num_reqs].fill_(prefix_len)
            return
        if len(ccpu) >= num_reqs:
            ccpu[:num_reqs] = prefix_len

    def _dsv4_sync_draft_seq_lens(self, num_reqs: int, prefix_len: int) -> int:
        """Write DSA seq_lens so ``prefix_lens = seq_lens - nqp = prefix_len``.

        Returns the pre-write ``input_buffers.seq_lens[0]`` (diag D2H).
        """
        from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import host_i0

        nqp = int(self.num_query_per_req)
        want = prefix_len + nqp
        bufs = self.input_buffers
        sl = getattr(bufs, "seq_lens", None)
        pre = host_i0(None if sl is None else sl[:num_reqs])
        if sl is not None and sl.numel() >= num_reqs:
            sl[:num_reqs].fill_(want)
        sl_np = getattr(bufs, "seq_lens_np", None)
        if sl_np is not None and sl_np.size >= num_reqs:
            sl_np[:num_reqs] = want
        sl_cpu = getattr(bufs, "seq_lens_cpu", None)
        if sl_cpu is not None and sl_cpu.numel() >= num_reqs:
            sl_cpu[:num_reqs].fill_(want)
        return pre

    def _dsv4_bind_chunk_batch(
        self,
        input_batch: InputBatch,
        pos: torch.Tensor,
        qsl: torch.Tensor,
        n: int,
        prefix_len: int,
    ) -> None:
        """Commit suffix only. ``pos``/``qsl`` stay off speculator query buffers.

        Draft attention still needs ``seq_lens = official prefix length`` so it
        reads the already-written 0..prefix-1 KV instead of leftover tree slots.
        ``num_computed_*`` is the leftover postprocess value (e.g. 85 after
        accept-5); bind it to ``prefix_len``. DSA SWA reads speculator
        ``input_buffers.seq_lens`` as total (prefix + query), not the batch
        prefix length.
        """
        num_reqs = qsl.shape[0] - 1
        input_batch.positions = pos[:n]
        input_batch.query_start_loc = qsl
        input_batch.query_start_loc_np = qsl.cpu().numpy()  # D2H
        input_batch.num_tokens = n
        input_batch.num_tokens_after_padding = n
        sched = np.diff(input_batch.query_start_loc_np).astype(np.int32, copy=False)
        input_batch.num_scheduled_tokens = sched
        input_batch.max_query_len = int(sched.max()) if sched.size else n
        sl = input_batch.seq_lens
        if sl is not None and sl.numel() >= num_reqs:
            sl[:num_reqs].fill_(prefix_len)
        if input_batch.seq_lens_np is not None and input_batch.seq_lens_np.size >= num_reqs:
            input_batch.seq_lens_np[:num_reqs] = prefix_len
        ub = input_batch.seq_lens_cpu_upper_bound
        if ub is not None and ub.numel() >= num_reqs:
            ub[:num_reqs].fill_(prefix_len)
        self._dsv4_fill_computed(input_batch, num_reqs, prefix_len)
        self._dsv4_sync_draft_seq_lens(num_reqs, prefix_len)
        self._dsv4_chunk_prefix_len = prefix_len

    def _dsv4_log_draft_meta(
        self,
        input_batch: InputBatch,
        tokens: torch.Tensor,
        source_hidden: torch.Tensor | None,
        source_aux: list[torch.Tensor] | None,
    ) -> None:
        """TP0 draft_meta after prepare_dflash + tree finalize. Diag D2H only.

        Context/query slots are the SWA layer's mapping (same tensors
        ``precompute_and_store_context_kv`` / draft query write use), not
        ``_context_slot_mappings[0]`` when that row is a different KV group.
        """
        from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
            host_i0,
            host_slot_head,
            log_draft_meta,
            log_draft_query,
            log_draft_source,
            tensor_rms,
            tensor_signature,
            tree_token_head,
        )

        nqp = int(self.num_query_per_req)
        n_ctx = int(input_batch.num_tokens)
        gidx_list = list(getattr(self, "_layer_group_idx", None) or [])
        layer_gidx_set = int(bool(gidx_list))
        layer_gidx = gidx_list
        primary = int(gidx_list[0]) if gidx_list else 0
        ids = list(getattr(self, "draft_kv_cache_group_ids", None) or [])
        ctx_map = getattr(self, "_context_slot_mappings", None)
        ngroups = len(ids)
        if not ngroups and ctx_map is not None and ctx_map.ndim > 1:
            ngroups = int(ctx_map.shape[0])
        gid = int(ids[primary]) if ids and primary < len(ids) else int(
            getattr(self, "draft_kv_cache_group_id", 0) or 0
        )
        ctx_row = None
        if ctx_map is not None and ctx_map.numel() > 0:
            if ctx_map.ndim > 1:
                row = primary if primary < ctx_map.shape[0] else 0
                ctx_row = ctx_map[row, :n_ctx]
            else:
                ctx_row = ctx_map[:n_ctx]
        qmap = None
        bt = getattr(self, "block_tables", None)
        if bt is not None:
            sm = getattr(bt, "slot_mappings", None)
            if sm is not None and sm.numel() > 0:
                qrow = sm[gid] if sm.ndim > 1 and gid < sm.shape[0] else sm
                qmap = qrow[:nqp]
        cnp = getattr(input_batch, "num_computed_tokens_np", None)
        sl_np = getattr(input_batch, "seq_lens_np", None)
        bufs = self.input_buffers
        qpos0 = host_i0(bufs.positions)
        prefix = self._dsv4_chunk_prefix_len
        if prefix is None:
            prefix = qpos0
        head = tree_token_head(tokens[: input_batch.num_reqs])
        log_draft_meta(
            prefix_len=prefix,
            num_computed=host_i0(cnp),
            batch_seq0=host_i0(sl_np if sl_np is not None else input_batch.seq_lens),
            buf_seq0_pre=self._dsv4_buf_seq0_pre,
            buf_seq0=host_i0(getattr(bufs, "seq_lens", None)),
            nqp=nqp,
            query0=host_i0(bufs.input_ids),
            bonus=self._dsv4_chunk_bonus,
            tree0=head[0] if head else -1,
            tree_head=head,
            qpos0=qpos0,
            ctx_slots=host_slot_head(ctx_row, min(n_ctx, 8)),
            qslots=host_slot_head(qmap, nqp),
            sequential=int(self._dsv4_dspark_draft and self.topk == 1),
            gid=gid,
            ngroups=ngroups,
            layer_gidx=layer_gidx,
            layer_gidx_set=layer_gidx_set,
            prefilling=host_i0(getattr(input_batch, "is_prefilling_np", None)),
            causal=_draft_causal_int(self, gid),
            has_prefill=int(bool(getattr(input_batch, "has_prefill", False))),
            force_prefill=0,
        )
        ctx_h = getattr(self, "hidden_states", None)
        ctx_h = None if ctx_h is None else ctx_h[:n_ctx]
        q_h = getattr(self, "_draft_hidden_buf", None)
        q_h = None if q_h is None else q_h[:nqp]
        n_head = min(8, n_ctx) if n_ctx > 0 else 0
        inner = getattr(self.model, "model", self.model)
        target_layer_ids = list(getattr(inner, "target_layer_ids", None) or [])
        log_draft_source(
            source_hidden,
            source_aux,
            n_ctx,
            target_layer_ids,
        )
        log_draft_query(
            qids=host_slot_head(bufs.input_ids, nqp),
            qpos=host_slot_head(bufs.positions, nqp),
            q_h_rms=tensor_rms(q_h),
            ctx_n=n_ctx,
            ctx_h_rms=tensor_rms(ctx_h),
            ctx_head_rms=tensor_rms(None if ctx_h is None else ctx_h[:n_head]),
            ctx_tail_rms=tensor_rms(None if ctx_h is None else ctx_h[-n_head:]),
            q_h_sig=tensor_signature(q_h),
            ctx_sig=tensor_signature(ctx_h),
            mask_id=int(getattr(self, "parallel_drafting_token_id", -1)),
        )

    def _apply_dsv4_commit_prefix(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        _aux_hidden_states: list[torch.Tensor] | None,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor]:
        """Inject accepted target-hidden rows into draft KV incrementally.

        The draft query owns only the speculative suffix slots. Accepted rows
        replace that suffix with target-derived KV; older official-prefix KV
        remains valid and does not need to be projected again.
        """
        n = self._dsv4_commit_n
        pos = self._dsv4_commit_pos
        qsl = self._dsv4_commit_qsl
        commit_aux = self._dsv4_commit_aux
        commit_hidden = self._dsv4_commit_hidden
        self._dsv4_commit_n = None
        self._dsv4_commit_pos = None
        self._dsv4_commit_qsl = None
        self._dsv4_commit_aux = None
        self._dsv4_commit_hidden = None
        chunk_hidden = (
            commit_hidden if commit_hidden is not None else last_hidden_states[:n]
        )
        chunk_aux = commit_aux
        num_reqs = qsl.shape[0] - 1
        merged = self._dsv4_prefix_merge(
            chunk_hidden, chunk_aux, pos, qsl, num_reqs, replace=False
        )
        num_rejected = torch.zeros_like(num_rejected)
        from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
            host_i0,
            path_log,
            pos_span,
        )

        pmax = int(pos[:n].amax().item())  # D2H
        prefix_len = pmax + 1
        pmin, _ = pos_span(pos, n)
        bonus = -1
        if last_sampled is not None and last_sampled.numel() > 0:
            idx = input_batch.idx_mapping
            req0 = int(idx[0].item()) if idx is not None and idx.numel() else 0  # D2H
            bonus = int(last_sampled[req0].item())  # D2H
        self._dsv4_chunk_bonus = bonus
        self._dsv4_bind_chunk_batch(input_batch, pos, qsl, n, prefix_len)
        cnp = getattr(input_batch, "num_computed_tokens_np", None)
        computed0 = host_i0(cnp)
        path_log(
            "propose draft_kv=commit_chunk n_tok=%s pos=%s..%s prefix_len=%s "
            "seq_bound=%s bonus=%s implied_buf=%s aux_layers=%s "
            "num_computed0=%s merged=%s skip_path_scatter=1",
            n,
            pmin,
            pmax,
            prefix_len,
            prefix_len,
            bonus,
            self._dsv4_prefix_n,
            0 if not chunk_aux else len(chunk_aux),
            computed0,
            int(merged),
        )
        return chunk_hidden, chunk_aux, num_rejected

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
        dp_sync: Any = None,
    ) -> torch.Tensor:
        snap = None
        try:
            if not dummy_run and self._dsv4_commit_n:
                snap = self._dsv4_snapshot_batch(input_batch)
                last_hidden_states, aux_hidden_states, num_rejected = (
                    self._apply_dsv4_commit_prefix(
                        input_batch,
                        last_hidden_states,
                        aux_hidden_states,
                        num_rejected,
                        last_sampled,
                    )
                )
            else:
                path_node_ids = getattr(input_batch, "path_node_ids", None)
                if (
                    self._dsv4_dspark_draft
                    and not dummy_run
                    and (path_node_ids is None or bool(input_batch.has_prefill))
                ):
                    self._dsv4_absorb_official_prefix(
                        last_hidden_states, aux_hidden_states, input_batch
                    )
                if path_node_ids is not None and not dummy_run:
                    from vllm_ascend.worker.v2.spec_decode.tree.timer import tree_time

                    node_row = getattr(input_batch, "tree_node_row", None)
                    if node_row is not None:
                        from vllm_ascend.worker.v2.spec_decode.tree.path_pack import (
                            compact_dsv4_path_hidden,
                        )

                        node_dim = 1 + (get_ascend_config().tree_spec_config.budget or 0)
                        with tree_time("compact_query_path"):
                            last_hidden_states, aux_hidden_states, pos, qsl = (
                                compact_dsv4_path_hidden(
                                    last_hidden_states,
                                    aux_hidden_states,
                                    input_batch.positions,
                                    node_row,
                                    path_node_ids,
                                    node_dim,
                                )
                            )
                        n = last_hidden_states.shape[0]
                        self.input_buffers.positions[:n].copy_(pos)
                        self.input_buffers.query_start_loc[: qsl.shape[0]].copy_(qsl)
                        input_batch.positions = self.input_buffers.positions[:n]
                        input_batch.query_start_loc = self.input_buffers.query_start_loc[
                            : qsl.shape[0]
                        ]
                        input_batch.query_start_loc_np = qsl.cpu().numpy()  # D2H
                        input_batch.num_tokens = n
                        input_batch.num_tokens_after_padding = n
                    else:
                        tree_q = 1 + (get_ascend_config().tree_spec_config.budget or 0)
                        # Skip when this step is not a full tree verify (e.g. 16 tokens
                        # padded to the 17-token FULL gear at max_seq_len).
                        if last_hidden_states.shape[0] >= path_node_ids.shape[0] * tree_q:
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
            if self.draft_backend == "dspark":
                copy_w = (
                    aux_hidden_states[0].shape[-1]
                    if aux_hidden_states
                    else last_hidden_states.shape[-1]
                )
                self._align_dspark_copy_buffer(copy_w)
            if self._dsv4_dspark_draft:
                from vllm_ascend.utils import vllm_version_is
                from vllm_ascend.worker.v2.attn_utils import (
                    build_attn_metadata_wrapper,
                    build_draft_attn_metadata_factory,
                )

                # Factory must wrap the Ascend ``build_attn_metadata`` installed by
                # the wrapper. Skip AscendDFlashSpeculator.propose so it cannot
                # reinstall the wrapper and drop the DSA positions/is_prefilling.
                self.input_batch = input_batch
                sync_state = num_tokens_across_dp if vllm_version_is("0.28.0") else dp_sync
                if dummy_run and skip_attn_for_dummy_run:
                    sync_state = None
                seq_np = None
                plen = self._dsv4_chunk_prefix_len
                if plen is not None:
                    seq_np = np.zeros(self.max_num_reqs, dtype=np.int32)
                    seq_np[: input_batch.num_reqs] = plen + int(self.num_query_per_req)
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
                    num_tokens_across_dp,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    mm_inputs,
                    is_profile=is_profile,
                    dp_sync=dp_sync,
                )
            # FULL replay only runs draft forward; prefix may replay a second graph.
            if not dummy_run:
                self._finalize_tree(input_batch.num_reqs)
                if self._dsv4_dspark_draft:
                    from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
                        host_i0,
                        log_tree_tokens,
                    )

                    tree = getattr(self, "tree", None)
                    tbuf = tree.tokens if tree is not None else tokens[: input_batch.num_reqs]
                    nnodes = host_i0(tree.num_nodes) if tree is not None else -1
                    log_tree_tokens(tbuf, num_nodes=nnodes)
                    self._dsv4_log_draft_meta(
                        input_batch,
                        tbuf,
                        last_hidden_states,
                        aux_hidden_states,
                    )
            return tokens
        finally:
            if snap is not None:
                self._dsv4_restore_batch(input_batch, snap)
            self._dsv4_chunk_prefix_len = None
            self._dsv4_chunk_bonus = -1
            self._dsv4_buf_seq0_pre = -1

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
        if self.tree_kv_compact is not None:
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

        if self._dsv4_dspark_draft and self.topk == 1:
            with tree_time("dspark_sequential_sample"):
                self._sample_sequential(num_reqs, self._draft_hidden_buf)
            self._materialize_chain_layout(num_reqs)
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
        # Drop leftover token ids from the previous propose before rewrite.
        self.draft_tokens[:num_reqs].fill_(-1)
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

    def _materialize_chain_layout(self, num_reqs: int) -> None:
        """Pack sequential DSpark tokens as a depth-ordered chain tree."""
        spec = self.num_speculative_steps
        # ``TreeLayout.tokens`` is a view of ``draft_tokens``. finalize
        # ``fill_(-1)`` would wipe sequential samples if we passed that view.
        tokens = self.draft_tokens[:num_reqs, :spec].clone()
        device = tokens.device
        depths = (
            torch.arange(1, spec + 1, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(num_reqs, -1)
        )
        parent_ids = (
            torch.arange(spec, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(num_reqs, -1)
        )
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
