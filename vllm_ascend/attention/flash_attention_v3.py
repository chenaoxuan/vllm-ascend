# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dense Ascend attention using flash-attention-npu's device-side FA3 tiling.

Ported from vllm-project/vllm-ascend PR #20167 (flash_attention_v3.py),
adapted for this fork:
  - CUDAGraphMode import path (vllm.config.compilation on this base).
  - Backend name stays "CUSTOM" like the dense and fa_v1 backends: the
    registry resolves names against AttentionBackendEnum, and this fork
    selects this class by import path, not by name.
  - Sliding windows match upstream: causal layers look left, while
    non-causal draft layers use the same radius on both sides.
  - Layer scale assertions use getattr fallbacks (older AttentionLayer).
The tiling itself is unchanged from the PR: computed by an AICPU
``get_scheduler_metadata`` task on device lengths before the forward, cached
in graph-stable buffers keyed by (input addresses, token bound, causality),
and refreshed in place so replayed graphs keep their recorded addresses.
"""

from dataclasses import dataclass, field
from enum import Enum

import torch
from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata  # type: ignore[import-not-found]
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, notify_kv_cache_written
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window


class AscendFlashAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    # Both model runners use the attention layer's cache-update contract:
    # this backend writes K/V inside forward, before reading the paged cache.
    forward_includes_kv_cache_update = True
    supported_dtypes = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        # Same registry name as the dense Ascend backends on this base: the
        # platform selects this class by import path, not by name.
        return "CUSTOM"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        # Same paged layout as the dense Ascend backend on this base.
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
        return [128]

    @staticmethod
    def get_impl_cls() -> type["AscendFlashAttentionImpl"]:
        return AscendFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["AscendFlashAttentionMetadataBuilder"]:
        return AscendFlashAttentionMetadataBuilder


@dataclass
class AscendFlashAttentionMetadata(AttentionMetadata):
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    max_query_len: int
    causal: bool

    # spec -> tiling tensor consumed by each layer's forward. In graph mode this
    # aliases the builder's cache dictionary for the current graph_key; eager
    # execution owns a fresh dictionary. This field only references tiling storage.
    scheduler_metadata: dict[tuple, torch.Tensor] = field(default_factory=dict)


class AscendFlashAttentionMetadataBuilder(AttentionMetadataBuilder[AscendFlashAttentionMetadata]):
    # Per-request query offsets let one paged call handle mixed prefill/decode
    # and multi-token verification in the existing request order.
    reorder_batch_threshold = None

    # This builder refreshes the draft graph's device inputs (the FA3 tiling)
    # in place per replay via update_draft_decode_metadata, so the core
    # speculator can skip its eager draft metadata rebuild in FULL graph mode.
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.device = device
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.capture_sizes = set(vllm_config.compilation_config.cudagraph_capture_sizes or [])
        self.capture_query_len = None
        if vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
            speculative = vllm_config.speculative_config
            self.capture_query_len = 1 + (speculative.num_speculative_tokens if speculative else 0)
        self.block_size = kv_cache_spec.block_size
        # Deduplicate layer parameters so identical layers share one tiling call.
        self.scheduler_specs = set()
        for name in layer_names:
            impl = vllm_config.compilation_config.static_forward_context[name].impl
            self.scheduler_specs.add(
                (
                    impl.num_heads,
                    impl.num_kv_heads,
                    impl.head_size,
                    kv_cache_spec.dtype,
                    impl.scale,
                    impl.logits_soft_cap,
                    impl.sliding_window,
                )
            )
        # Two-level cache: scheduler_buffers[graph_key][spec] = tiling_tensor.
        # Outer key: (input-buffer addresses, captured token bound), identifying
        # the graph context. Inner key: the layer's operator spec. Retain these
        # tensors across builds and copy new tiling into them to preserve the
        # device addresses recorded by graph capture.
        self.scheduler_buffers: dict[tuple, dict[tuple, torch.Tensor]] = {}
        # graph_key -> the metadata object built for that context. The
        # per-replay in-place update recomputes each key's tiling from these
        # objects' tensor views (which alias the runner's stable buffers).
        self._graph_metadata: dict[tuple, AscendFlashAttentionMetadata] = {}

    @classmethod
    def get_cudagraph_support(cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec) -> AttentionCGSupport:
        return AttentionCGSupport.ALWAYS

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendFlashAttentionMetadata:
        common = common_attn_metadata
        num_reqs = common.num_reqs
        # The runner can append zero-length or dummy requests for graph padding.
        # Its CPU offsets identify the active prefix without a device sync.
        # Keep one empty request for dummy forwards: the operator requires B > 0.
        active_reqs = max(
            1,
            int(torch.searchsorted(common.query_start_loc_cpu[: num_reqs + 1], common.num_actual_tokens)),
        )
        # Device tiling limits reads to active_reqs, so padding rows need
        # neither copying nor sanitizing.
        query_start_loc = common.query_start_loc[: active_reqs + 1]
        seq_lens = common.seq_lens[:active_reqs]
        block_tables = common.block_table_tensor[:active_reqs]
        num_input_tokens = common.num_input_tokens or common.num_actual_tokens
        is_graph_capture_size = num_input_tokens in self.capture_sizes and (
            self.capture_query_len is None or common.max_query_len <= self.capture_query_len
        )
        # FA3 sizes split workspace from the per-request query bound. Prefill
        # can share a token bucket but must retain its own query length.
        max_query_len = (
            min(num_input_tokens, self.capture_query_len or num_input_tokens)
            if is_graph_capture_size
            else common.max_query_len
        )
        # Always retain tiling in the graph-keyed cache. A captured draft
        # ACLGraph records the tiling buffer's address, and this fork's
        # speculator rebuilds draft attention metadata eagerly every step
        # (no in-place update support) while capturing the draft graph at
        # token counts outside the runner's capture_sizes -- so an
        # unretained tiling tensor would be freed and its memory reused
        # between capture and replay, and the graph would read stale bytes
        # (device error 507057). graph_key bounds the cache (input buffer
        # addresses, token count, causality); each entry is a 2376-byte
        # tiling plus a small per-request reserve.
        source_key = (query_start_loc.data_ptr(), seq_lens.data_ptr(), block_tables.data_ptr())
        graph_key = (source_key, num_input_tokens, max_query_len, common.causal)
        # Select the inner spec -> tensor dictionary. Writing
        # scheduler_metadata[spec] also updates scheduler_buffers[graph_key].
        scheduler_metadata = self.scheduler_buffers.setdefault(graph_key, {})
        for spec in self.scheduler_specs:
            # Match the layer configuration used by forward. Causality is a
            # runtime property and is also part of the graph storage key.
            tiling = self._compute_tiling(spec, query_start_loc, seq_lens, block_tables, max_query_len, common.causal)
            if spec in scheduler_metadata:
                buffer = scheduler_metadata[spec]
                if tiling.numel() * tiling.element_size() > buffer.untyped_storage().nbytes():
                    raise RuntimeError("FA3 scheduler metadata exceeds its graph-stable buffer capacity")
                buffer.resize_(tiling.shape).copy_(tiling)
            else:
                # Scheduler metadata may append per-request tail entries.
                # Reserve that tail once so changing active batch size keeps
                # the address.
                reserve = self.max_num_reqs * common.seq_lens.element_size()
                buffer = tiling.new_empty(tiling.numel() + reserve)
                scheduler_metadata[spec] = buffer[: tiling.numel()].copy_(tiling)
        metadata = AscendFlashAttentionMetadata(
            num_actual_tokens=common.num_actual_tokens,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_tables=block_tables,
            slot_mapping=common.slot_mapping[: common.num_actual_tokens],
            max_query_len=max_query_len,
            causal=common.causal,
            scheduler_metadata=scheduler_metadata,
        )
        self._graph_metadata[graph_key] = metadata
        return metadata

    def _compute_tiling(
        self,
        spec: tuple,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        max_query_len: int,
        causal: bool,
    ) -> torch.Tensor:
        """AICPU tiling for one layer spec; consumes device lengths async."""
        num_heads, num_kv_heads, head_size, dtype, scale, softcap, sliding_window = spec
        return get_scheduler_metadata(
            batch_size=seq_lens.shape[0],
            max_seqlen_q=max_query_len,
            max_seqlen_k=block_tables.shape[1] * self.block_size,
            num_heads_q=num_heads,
            num_heads_kv=num_kv_heads,
            headdim=head_size,
            cache_seqlens=seq_lens,
            qkv_dtype=dtype,
            cu_seqlens_q=query_start_loc,
            page_size=self.block_size,
            causal=causal,
            window_size=self._window_size(sliding_window, causal),
            has_softcap=softcap != 0.0,
            num_splits=0,
        )

    def update_draft_decode_metadata(self, metadata: "AscendFlashAttentionMetadata | None" = None) -> None:
        """Refresh the draft graph's device inputs in place (per replay).

        The fork's speculator rebuilds draft attention metadata eagerly every
        step unless every draft group supports this in-place update. The
        runner owns seq_lens/query_start_loc/block_tables and refreshes them
        in place, and the captured graph reads those storages directly, so the
        only input needing an explicit refresh is the FA3 tiling: recompute it
        on the AICPU from the current length buffers and copy it into the
        retained graph-keyed buffers (whose addresses the graph recorded).
        """
        for graph_key, m in self._graph_metadata.items():
            scheduler_metadata = self.scheduler_buffers.get(graph_key)
            if not scheduler_metadata:
                continue
            for spec, buffer in scheduler_metadata.items():
                tiling = self._compute_tiling(
                    spec, m.query_start_loc, m.seq_lens, m.block_tables, m.max_query_len, m.causal
                )
                if tiling.numel() * tiling.element_size() > buffer.untyped_storage().nbytes():
                    raise RuntimeError("FA3 scheduler metadata exceeds its graph-stable buffer capacity")
                buffer.resize_(tiling.shape).copy_(tiling)

    @staticmethod
    def _window_size(sliding_window: int | None, causal: bool) -> tuple[int, int]:
        """Use symmetric windows for non-causal drafts, matching FIA."""
        if sliding_window is None:
            return (-1, -1)
        radius = sliding_window - 1
        return (radius, 0 if causal else radius)

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: Enum | None = None,
    ) -> AscendFlashAttentionMetadata:
        # Ascend speculative proposers use this entry point. The main model
        # runner uses the inherited build_for_cudagraph_capture, which also
        # calls build directly. This backend uses paged attention for all
        # scheduling states, so neither capture path needs attn_state.
        return self.build(0, common_attn_metadata)


class AscendFlashAttentionImpl(AttentionImpl[AscendFlashAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor = None,
        **kwargs,
    ):
        logger.info("Using Ascend FA3 attention backend for draft attention.")
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        # FA3 uses softcap=0.0 to disable logit capping. Explicit None from
        # vLLM must be normalized for both scheduler tiling and execution.
        self.logits_soft_cap = 0.0 if logits_soft_cap is None else logits_soft_cap

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        # The metadata builder updates FA3 inputs and device tiling in place
        # before replay; no host-side attention tasks need to be updated.
        pass

    def configure_tree_draft_window(self, attn_mask: torch.Tensor) -> None:
        """Accept the draft graph-capture window contract (dense-backend API).

        The DFlash2 speculator pre-stages a compressed window mask before
        capturing the draft ACLGraph (``_prepare_dflash2_graph_attention``).
        This backend already derives the equivalent causal or symmetric window from
        the layer's ``sliding_window``, so the mask is stored but never read:
        replay correctness rides on the address-pinned tiling buffers, not on
        this mask.
        """
        self._tree_draft_window_mask = attn_mask

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        # Also used by speculative proposers that insert context K/V separately.
        if self.kv_sharing_target_layer_name is not None:
            return
        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=kv_cache[0],
            value_cache=kv_cache[1],
            slot_mapping=slot_mapping,
        )
        notify_kv_cache_written(layer.layer_name)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AscendFlashAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        assert output_scale is None and output_block_scale is None
        assert getattr(layer, "_k_scale_float", 1.0) == 1.0 and getattr(layer, "_v_scale_float", 1.0) == 1.0
        # Both runners can profile the model before attention metadata exists.
        if attn_metadata is None:
            return output.fill_(0)
        if key is not None and value is not None:
            num_tokens = attn_metadata.num_actual_tokens
            self.do_kv_cache_update(
                layer,
                key[:num_tokens],
                value[:num_tokens],
                kv_cache,
                attn_metadata.slot_mapping[:num_tokens],
            )
        scheduler_key = (
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            query.dtype,
            self.scale,
            self.logits_soft_cap,
            self.sliding_window,
        )
        with attention_transfer_window():
            result = flash_attn_with_kvcache(
                query,
                kv_cache[0],
                kv_cache[1],
                cache_seqlens=attn_metadata.seq_lens,
                page_table=attn_metadata.block_tables,
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                window_size=AscendFlashAttentionMetadataBuilder._window_size(self.sliding_window, attn_metadata.causal),
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata[scheduler_key],
                num_splits=0,
            )
            output.copy_(result)
        return output
