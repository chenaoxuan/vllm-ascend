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


def _agent_dbg(location, message, data, hypothesis_id, limit=400):
    try:
        import importlib.util
        import sys

        mod = sys.modules.get("_agent_debug_trace")
        if mod is None:
            spec = importlib.util.spec_from_file_location(
                "_agent_debug_trace",
                "/home/specdec/spec260922/debug_trace.py",
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_agent_debug_trace"] = mod
            spec.loader.exec_module(mod)
        mod.dbg(location, message, data, hypothesis_id, limit=limit)
    except Exception:
        pass


def _plain_row(value, n: int = 16):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return value
    tensor = value.detach()
    if tensor.ndim == 0:
        if tensor.dtype.is_floating_point:
            return round(float(tensor.item()), 4)
        return int(tensor.item())
    if tensor.ndim > 1:
        tensor = tensor[0]
    tensor = tensor.reshape(-1)[:n]
    if tensor.dtype.is_floating_point:
        return [round(float(x), 4) for x in tensor.tolist()]
    return [int(x) for x in tensor.tolist()]


def _note_beam_life_ctx(spec, num_reqs, nqp, sample_hidden, root_token_ids) -> None:
    """Draft query the beam is about to expand. Req 0 only."""
    try:
        from vllm_ascend.worker.v2.spec_decode.tree.beam import note_life_ctx

        n_spec = spec.num_speculative_steps
        buf = spec.input_buffers
        batch = getattr(spec, "input_batch", None)
        hidden = sample_hidden.view(num_reqs, n_spec, -1)
        path = None if batch is None else getattr(batch, "path_node_ids", None)
        seq = getattr(buf, "seq_lens", None)
        ctx = {
            "q_ids": _plain_row(buf.input_ids[:nqp], nqp),
            "q_pos": _plain_row(buf.positions[:nqp], nqp),
            "sample_idx": _plain_row(spec.sample_indices[:n_spec], n_spec),
            "sample_pos": _plain_row(spec.sample_pos[:n_spec], n_spec),
            "seq": _plain_row(None if seq is None else seq[:num_reqs], 4),
            "root": _plain_row(root_token_ids, 1),
            "path": _plain_row(path, 8),
            "prefix_n": int(getattr(spec, "_dsv4_prefix_n", -1) or 0),
            "h_mean": [round(float(x), 4) for x in hidden[0].abs().mean(-1).tolist()],
        }
        note_life_ctx(ctx)
        _agent_dbg("tree/speculator.py:_finalize_tree", "life_ctx", ctx, "H6")
    except Exception:
        pass


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
            bound = getattr(self, "_dspark_swa_name_to_gidx", None)
            inner._draft_kv_slots_by_name = (
                self._dspark_slots_by_swa_name(n) if bound else None
            )
            # Draft KV is rewritten from the accepted-path hidden, one store
            # for every SWA layer. Log the positions so this stays visible
            # next to the target compress commit.
            if n > 0 and context_positions is not None and context_positions.numel() > 0:
                try:
                    from debug_trace import cmp_log

                    pos = context_positions[:n]
                    cmp_log(
                        {
                            "draft_kv": 1,
                            "n": n,
                            "pos0": int(pos[0].item()),
                            "pos_mid": int(pos[min(n - 1, 7)].item()),
                            "posn": int(pos[n - 1].item()),
                            "layers": len(inner._draft_kv_slots_by_name or {}),
                        }
                    )
                except Exception:
                    pass
            orig(context_states, context_positions, context_slot_mapping)
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
        """Merge official-prefix rows by absolute position.

        ``num_reqs`` / chunk length are host ints from commit or ``qsl_np``.
        """
        chunk_n = hidden.shape[0]
        cap = self.max_num_tokens
        if chunk_n > cap:
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
        if (
            self._dsv4_prefix_hidden is not None
            and (
                self._dsv4_prefix_hidden.shape[-1] != hidden.shape[-1]
                or self._dsv4_prefix_hidden.dtype != hidden.dtype
            )
        ):
            return False
        self._dsv4_ensure_prefix_bufs(hidden, aux, pos, allow_new_aux=False)
        old_qsl = self._dsv4_prefix_qsl[: num_reqs + 1].cpu().numpy()  # D2H
        chunk_qsl = qsl[: num_reqs + 1].cpu().numpy()  # D2H
        old_pos = self._dsv4_prefix_pos[: self._dsv4_prefix_n].cpu().numpy()  # D2H
        chunk_pos = pos[:chunk_n].cpu().numpy()  # D2H
        out_lens = np.empty(num_reqs, dtype=np.int32)
        old_keep: list[np.ndarray] = []
        chunk_order: list[np.ndarray] = []
        for i in range(num_reqs):
            o0, o1 = int(old_qsl[i]), int(old_qsl[i + 1])
            c0, c1 = int(chunk_qsl[i]), int(chunk_qsl[i + 1])
            old_req_pos = old_pos[o0:o1]
            chunk_req_pos = chunk_pos[c0:c1]
            keep = np.flatnonzero(~np.isin(old_req_pos, chunk_req_pos)) + o0
            merged_pos = np.concatenate((old_pos[keep], chunk_req_pos))
            order = np.argsort(merged_pos, kind="stable")
            old_keep.append(keep)
            chunk_order.append(order)
            out_lens[i] = order.size
        out_qsl = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(out_lens, out=out_qsl[1:])
        new_n = int(out_qsl[-1])
        if new_n > cap:
            return False
        packed_h = hidden.new_empty((new_n, hidden.shape[-1]))
        packed_p = pos.new_empty((new_n,))
        packed_aux = (
            [a.new_empty((new_n, a.shape[-1])) for a in aux]
            if aux and self._dsv4_prefix_aux
            else None
        )
        dst = 0
        for i in range(num_reqs):
            c0, c1 = int(chunk_qsl[i]), int(chunk_qsl[i + 1])
            keep = old_keep[i]
            keep_idx = torch.from_numpy(keep).to(
                device=hidden.device, dtype=torch.long
            )  # H2D
            merged_h = torch.cat(
                (
                    self._dsv4_prefix_hidden.index_select(0, keep_idx),
                    hidden[c0:c1],
                )
            )
            merged_p = torch.cat(
                (
                    self._dsv4_prefix_pos.index_select(0, keep_idx),
                    pos[c0:c1],
                )
            )
            order = torch.from_numpy(chunk_order[i]).to(
                device=hidden.device, dtype=torch.long
            )  # H2D
            length = int(out_lens[i])
            packed_h[dst : dst + length].copy_(merged_h.index_select(0, order))
            packed_p[dst : dst + length].copy_(merged_p.index_select(0, order))
            if packed_aux and self._dsv4_prefix_aux:
                for d, old, chunk in zip(
                    packed_aux, self._dsv4_prefix_aux, aux
                ):
                    merged = torch.cat(
                        (old.index_select(0, keep_idx), chunk[c0:c1])
                    )
                    d[dst : dst + length].copy_(merged.index_select(0, order))
            dst += length
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
        merged = self._dsv4_prefix_merge(
            hidden[:n],
            None if not aux else [a[:n] for a in aux],
            input_batch.positions[:n],
            input_batch.query_start_loc[: num_reqs + 1],
            num_reqs,
            replace,
        )
        if not merged:
            raise RuntimeError("Failed to merge the official DSV4 prefix")


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
        from vllm_ascend.worker.v2.spec_decode.tree.chain_pack import (
            clear_tree_chain_layout,
        )

        clear_tree_chain_layout()
        try:
            path_node_ids = input_batch.path_node_ids
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
                # #region agent log
                tree = self.tree
                _agent_dbg(
                    "tree/speculator.py:propose",
                    "propose_out",
                    {
                        "method": self.method,
                        "budget": self.budget,
                        "topk": self.topk,
                        "n_spec": self.num_speculative_steps,
                        "n_req": int(input_batch.num_reqs),
                        "hidden": int(last_hidden_states.shape[0]),
                        "tok_shape": list(tokens.shape),
                        "tok0": tokens,
                        "nodes": None if tree is None else tree.num_nodes,
                        "tree_tok0": None if tree is None else tree.tokens,
                        "dep0": None if tree is None else tree.depths,
                        "par0": None if tree is None else tree.parents,
                    },
                    "H4",
                )
                # #endregion
            return tokens
        finally:
            self._dsv4_chunk_prefix_len = None

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
        # Draft-query dump. Paused for the coverage run.
        if False and self.method == "beam":
            _note_beam_life_ctx(
                self, num_reqs, nqp, sample_hidden_states, root_token_ids
            )
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
        # #region agent log
        _agent_dbg(
            "tree/speculator.py:_finalize_tree",
            "tree_built",
            {
                "method": self.method,
                "branch": "build",
                "logits": list(logits.shape),
                "nodes": self.tree.num_nodes,
                "tok0": self.tree.tokens,
                "dep0": self.tree.depths,
                "par0": self.tree.parents,
                "child0": self.tree.first_child,
            },
            "H5",
        )
        # #endregion

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
