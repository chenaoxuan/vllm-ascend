# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    NULL_BLOCK_ID,
    PAD_SLOT_ID,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ops.triton.fla.utils import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_final_chunk_indices,
    prepare_update_chunk_offsets,
)
from vllm_ascend.utils import vllm_version_is

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
        TreeVerifyPlan,
        TreeVerifyPlanBuffers,
    )

_GDN_CHUNK_SIZE = 64
# Keep this aligned with solve_tril.LARGE_BLOCK_T in ops/triton/fla/solve_tril.py.
_GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE = 608 * 2
_GDN_CUMSUM_WORKING_SET = 2**18
# Accepted-token counts are one-based. The zero-length query segment makes
# graph-padding rows no-ops, while 1 keeps their count valid for GDN kernels.
_GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT = 1


def _stable_argsort_for_npu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        # Boolean mask values are exactly representable in float32. Integer
        # argsort falls back to AiCPU on Ascend; keep this stable partition on
        # AiCore without changing the ordering of tokens within either group.
        tensor = tensor.to(torch.float32)
    return torch.argsort(tensor, stable=True)


def _remove_spec_graph_padding_queries(
    common_attn_metadata: CommonAttentionMetadata,
    num_decode_draft_tokens_cpu: torch.Tensor | None,
) -> CommonAttentionMetadata:
    """Remove inactive FIA graph rows from the GDN-local query view.

    FIA represents every padded graph request with a full K+1 query span.
    Those rows must be zero-length for the target's recurrent GDN state update.
    The shared MLA metadata remains unchanged.
    """
    seq_lens = common_attn_metadata.seq_lens_cpu_upper_bound
    if seq_lens is None or num_decode_draft_tokens_cpu is None:
        return common_attn_metadata
    num_reqs = common_attn_metadata.num_reqs
    draft_tokens = num_decode_draft_tokens_cpu[:num_reqs]
    if draft_tokens.numel() != num_reqs or not torch.any(draft_tokens > 0).item():
        return common_attn_metadata
    inactive = (seq_lens[:num_reqs] == 0) & (draft_tokens < 0)
    if not torch.any(inactive).item():
        return common_attn_metadata
    first_inactive = int(torch.nonzero(inactive, as_tuple=True)[0][0])
    if not torch.all(inactive[first_inactive:]).item():
        raise ValueError("GDN speculative graph padding must be a request suffix")
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu.clone()
    query_start_loc_cpu[first_inactive + 1 :] = query_start_loc_cpu[first_inactive]
    return common_attn_metadata.replace(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc_cpu.to(common_attn_metadata.query_start_loc.device),
        num_actual_tokens=int(query_start_loc_cpu[-1]),
    )


def _treat_single_token_prefills_with_state_as_decodes(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    """Use decode metadata for one-token stateful prompt chunks.

    A final one-token prompt chunk can replay the same fixed graph as an
    ordinary decode. Once recurrent state exists, both paths must construct
    identical GDN metadata so the graph consumes the current state indices.
    First-token prefills remain on the prefill path because they have no state
    to update.
    """
    is_prefilling = common_attn_metadata.is_prefilling
    seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
    if is_prefilling is None or seq_lens_cpu is None:
        return common_attn_metadata

    query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
    prefill_to_decode = is_prefilling & (query_lens_cpu == 1) & (seq_lens_cpu > 1)
    if not torch.any(prefill_to_decode).item():
        return common_attn_metadata

    is_prefilling = is_prefilling.clone()
    is_prefilling[prefill_to_decode] = False
    return common_attn_metadata.replace(is_prefilling=is_prefilling)


@dataclass
class GDNChunkedPrefillMetadata:
    cu_seqlens_host: tuple[int, ...]
    chunk_indices_chunk64_host: tuple[int, ...]
    chunk_indices_chunk64: torch.Tensor | None
    chunk_offsets_chunk64: torch.Tensor | None
    update_chunk_offsets_chunk64: torch.Tensor | None
    final_chunk_indices_chunk64: torch.Tensor | None
    chunk_indices_large_block: torch.Tensor | None
    block_indices_cumsum: torch.Tensor | None
    num_decodes: int = 0
    cu_seqlens_kern: tuple[int, ...] | None = None
    keep_meta: torch.Tensor | None = None


@dataclass
class GDNCausalConv1dMetadata:
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    initial_state_mode: torch.Tensor | None
    output_buffer: torch.Tensor | None = None


@dataclass
class GDNSpecCausalConv1dMetadata:
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    num_accepted_tokens: torch.Tensor
    output_buffer: torch.Tensor | None = None


@dataclass
class GDNPrefillMetadata:
    causal_conv1d: GDNCausalConv1dMetadata
    chunk: GDNChunkedPrefillMetadata


@dataclass
class GDNDecodeMetadata:
    causal_conv1d: GDNCausalConv1dMetadata
    actual_seq_lengths: torch.Tensor


def _tree_conv_width(kv_cache_spec: AttentionSpec, mamba_type, state_columns: int) -> int | None:
    """Short-conv kernel width of a GDN state spec (``None`` when not GDN).

    ``state_columns`` is the number of speculative state columns the conv-state
    shape carries: ``num_speculative_tokens`` while the spec still has the
    linear-chain layout, ``slots`` once ``attn_utils`` widened it to the
    draft-tree width.
    """
    shapes = getattr(kv_cache_spec, "shapes", None)
    if not shapes:
        return None
    # Lazy import: the tree spec-decode package imports this module.
    from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import tree_conv_width

    return tree_conv_width(mamba_type, tuple(shapes[0]), state_columns)


def _pack_tree_rows_to_tokens(
    table: torch.Tensor,
    row_widths: torch.Tensor,
    slots: int,
) -> torch.Tensor:
    """Pack a ``[rows, slots]`` tree table into token-major order.

    The recurrent kernel reads ``ssm_state_indices`` and ``init_state_indices``
    by global token offset (``ssmStateIndicesGm_.GetValue(seq_i)`` in
    ``csrc/attention/recurrent_gated_delta_rule``), while the tree tables are
    indexed as ``[row, node]``. Every row normally verifies the full
    ``1 + budget = slots`` tokens, but the scheduler truncates the draft of a
    request near ``max_model_len`` and the tree search can run out of candidates,
    so the rows have to be laid out back to back: token ``p`` of the step takes
    node ``p - start(row)`` of its row. The result keeps the ``[rows, slots]``
    shape because the op receives ``reshape(-1)``, and only the slots past the
    packed tokens hold ``NULL_BLOCK_ID`` -- the kernel never reaches them.
    """
    rows = table.shape[0]
    widths = row_widths.to(device=table.device, dtype=torch.long).reshape(-1)
    if widths.numel() != rows:
        raise ValueError(f"got {widths.numel()} row widths for {rows} tree rows.")
    starts = torch.cumsum(widths, dim=0) - widths
    ends = starts + widths
    positions = torch.arange(rows * slots, device=table.device)
    token_rows = (positions.unsqueeze(1) >= ends.unsqueeze(0)).sum(dim=1).clamp(max=rows - 1)
    token_cols = (positions - starts[token_rows]).clamp(max=slots - 1)
    packed = table.reshape(-1).index_select(0, token_rows * slots + token_cols)
    packed = packed.masked_fill(positions >= ends[-1], NULL_BLOCK_ID)
    return packed.view(rows, slots)


@dataclass
class GDNSpecDecodeMetadata:
    spec_causal_conv1d: GDNSpecCausalConv1dMetadata
    # 310P derives lengths in its recurrent wrapper and only consumes conv metadata.
    actual_seq_lengths: torch.Tensor | None
    # Draft-tree only: per-token state page the recurrent kernel loads its
    # initial state from, see spec_decode/tree/state_rollback.py. ``None`` keeps
    # the linear-chain behaviour of the kernel.
    init_state_indices: torch.Tensor | None = None
    # Draft-tree only: conv tap rows and committed-history source columns.
    conv_window_rows: torch.Tensor | None = None
    conv_prefix_src: torch.Tensor | None = None
    # Draft-tree only: chunked verify for trees wider than one kernel call.
    verify_plan: "TreeVerifyPlan | None" = None


def _build_actual_seq_lengths(
    query_start_loc: torch.Tensor,
    num_sequences: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    actual_seq_lengths = (
        torch.empty_like(query_start_loc[: num_sequences + 1]) if out is None else out[: num_sequences + 1]
    )
    actual_seq_lengths[:1].copy_(query_start_loc[:1])
    torch.sub(
        query_start_loc[1 : num_sequences + 1],
        query_start_loc[:num_sequences],
        out=actual_seq_lengths[1:],
    )
    return actual_seq_lengths


def _materialize_graph_request_tensor(
    buffer: torch.Tensor,
    values: torch.Tensor,
    actual_request_count: int,
    graph_request_count: int,
    padding_value: int | bool,
) -> torch.Tensor:
    """Write request-level values into a stable full-graph buffer."""
    if actual_request_count > graph_request_count:
        raise ValueError(
            f"actual_request_count ({actual_request_count}) exceeds graph_request_count ({graph_request_count})"
        )
    if graph_request_count > buffer.size(0):
        raise ValueError(f"graph_request_count ({graph_request_count}) exceeds buffer capacity ({buffer.size(0)})")
    if actual_request_count > values.size(0):
        raise ValueError(f"actual_request_count ({actual_request_count}) exceeds source rows ({values.size(0)})")

    materialized = buffer[:graph_request_count]
    materialized[:actual_request_count].copy_(
        values[:actual_request_count],
        non_blocking=True,
    )
    materialized[actual_request_count:].fill_(padding_value)
    return materialized


def _materialize_graph_query_start_loc(
    buffer: torch.Tensor,
    query_start_loc: torch.Tensor,
    actual_request_count: int,
    graph_request_count: int,
) -> torch.Tensor:
    """Pad request boundaries with zero-length dummy request segments."""
    if actual_request_count > graph_request_count:
        raise ValueError(
            f"actual_request_count ({actual_request_count}) exceeds graph_request_count ({graph_request_count})"
        )
    if graph_request_count + 1 > buffer.size(0):
        raise ValueError(
            f"graph request boundaries ({graph_request_count + 1}) exceed buffer capacity ({buffer.size(0)})"
        )
    if actual_request_count + 1 > query_start_loc.size(0):
        raise ValueError(
            f"actual request boundaries ({actual_request_count + 1}) exceed source rows ({query_start_loc.size(0)})"
        )
    materialized = buffer[: graph_request_count + 1]
    materialized[: actual_request_count + 1].copy_(
        query_start_loc[: actual_request_count + 1],
        non_blocking=True,
    )
    padding = materialized[actual_request_count + 1 :]
    if padding.numel() > 0:
        padding.copy_(
            materialized[actual_request_count : actual_request_count + 1].expand_as(padding),
            non_blocking=True,
        )
    return materialized


def _compact_empty_segments(cu_seqlens_host, initial_state, device=None):
    """Drop zero-length segments so AscendC fwd_h/fwd_o indexing lines up.

    Returns ``(cu_seqlens_kern, initial_state_kern, keep_meta)``:
    cu_seqlens / initial_state with empty segments removed, plus a bool
    mask (None when nothing was removed).  The compacted ``final_state``
    must be scattered back via ``keep_meta`` (empty segments keep their
    initial state).

    When *device* is given, ``keep_meta`` is moved to that device so that
    callers can index NPU tensors without an extra host→device sync.
    """
    if cu_seqlens_host is None:
        return None, initial_state, None
    cu = torch.tensor(cu_seqlens_host, dtype=torch.int64)
    keep = (cu[1:] - cu[:-1]) > 0
    if bool(keep.all()):
        return cu_seqlens_host, initial_state, None
    # Compute compact cu_seqlens while keep is still on CPU (cu is CPU-only).
    cu_kern = torch.cat([cu[:1], cu[1:][keep]]).tolist()
    # Move keep to device only for indexing device-side tensors.
    if device is not None:
        keep = keep.to(device)
    st_kern = initial_state[keep] if initial_state is not None else None
    return cu_kern, st_kern, keep


def _build_non_spec_chunked_prefill_metadata(
    builder,
    cu_seqlens_cpu: torch.Tensor,
    device: torch.device,
    build_device_metadata: bool = True,
) -> GDNChunkedPrefillMetadata:
    chunk_indices_chunk64 = prepare_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
    if build_device_metadata:
        hf_text_config = getattr(builder.vllm_config.model_config, "hf_text_config", None)
        linear_attn_config = getattr(hf_text_config, "linear_attn_config", None)
        if isinstance(linear_attn_config, dict) and linear_attn_config.get("num_heads") is not None:
            gdn_num_heads = linear_attn_config["num_heads"] // builder.vllm_config.parallel_config.tensor_parallel_size
        elif hf_text_config is not None and hasattr(hf_text_config, "linear_num_value_heads"):
            gdn_num_heads = (
                hf_text_config.linear_num_value_heads // builder.vllm_config.parallel_config.tensor_parallel_size
            )
        else:
            gdn_num_heads = builder.vllm_config.model_config.get_num_attention_heads(
                builder.vllm_config.parallel_config
            )
        cumsum_chunks = max(1, _GDN_CUMSUM_WORKING_SET // (gdn_num_heads * _GDN_CHUNK_SIZE))
        cumsum_chunk_size = 1 if cumsum_chunks <= 1 else 1 << (cumsum_chunks - 1).bit_length()
        chunk_offsets_chunk64 = prepare_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        update_chunk_offsets_chunk64 = prepare_update_chunk_offsets(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        final_chunk_indices_chunk64 = prepare_final_chunk_indices(cu_seqlens_cpu, _GDN_CHUNK_SIZE)
        chunk_indices_large_block = prepare_chunk_indices(
            cu_seqlens_cpu,
            _GDN_SOLVE_TRIL_LARGE_BLOCK_SIZE,
        )
        block_indices_cumsum = prepare_chunk_indices(cu_seqlens_cpu, cumsum_chunk_size)

    cu_seqlens_host = tuple(cu_seqlens_cpu.to(torch.int64).reshape(-1).tolist())
    # Pre-compute compact cu_seqlens for AscendC kernels so each layer
    # can reuse them instead of calling _compact_empty_segments again.
    cu_seqlens_kern, _, keep_meta = _compact_empty_segments(cu_seqlens_host, None, device=device)
    if keep_meta is None:
        cu_seqlens_kern = None
    else:
        cu_seqlens_kern = tuple(cu_seqlens_kern)

    if build_device_metadata:
        device_chunk_metadata = {
            "chunk_indices_chunk64": chunk_indices_chunk64.to(device=device, non_blocking=True),
            "chunk_offsets_chunk64": chunk_offsets_chunk64.to(device=device, non_blocking=True),
            "update_chunk_offsets_chunk64": update_chunk_offsets_chunk64.to(device=device, non_blocking=True),
            "final_chunk_indices_chunk64": final_chunk_indices_chunk64.to(device=device, non_blocking=True),
            "chunk_indices_large_block": chunk_indices_large_block.to(device=device, non_blocking=True),
            "block_indices_cumsum": block_indices_cumsum.to(device=device, non_blocking=True),
        }
    else:
        device_chunk_metadata = {
            "chunk_indices_chunk64": None,
            "chunk_offsets_chunk64": None,
            "update_chunk_offsets_chunk64": None,
            "final_chunk_indices_chunk64": None,
            "chunk_indices_large_block": None,
            "block_indices_cumsum": None,
        }

    return GDNChunkedPrefillMetadata(
        cu_seqlens_host=cu_seqlens_host,
        chunk_indices_chunk64_host=tuple(chunk_indices_chunk64.to(torch.int64).reshape(-1).tolist()),
        **device_chunk_metadata,
        cu_seqlens_kern=cu_seqlens_kern,
        keep_meta=keep_meta,
    )


class AscendGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    build_device_chunk_metadata = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Draft-tree state layout, filled in by _resize_spec_buffers_for_tree;
        # ``None`` means the linear MTP chain (default).
        self.spec_tree_slots: int | None = None
        self.spec_init_state_indices: torch.Tensor | None = None
        self.spec_conv_width: int | None = None
        self.spec_conv_window_rows: torch.Tensor | None = None
        self.spec_conv_prefix_src: torch.Tensor | None = None
        # Stable chunk tensors of a verify wider than one recurrent-op call, see
        # _build_tree_verify_plan; ``None`` while the tree fits a single call.
        self.spec_verify_plan_buffers: TreeVerifyPlanBuffers | None = None
        self._resize_spec_buffers_for_tree(device)
        sequence_index_capacity = max(
            self.vllm_config.scheduler_config.max_num_seqs,
            self.decode_cudagraph_max_bs,
        )

        self.spec_sequence_masks: torch.Tensor = torch.empty(
            (sequence_index_capacity,), dtype=torch.bool, device=device
        )

        self.spec_sequence_masks_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.bool,
            device="cpu",
            pin_memory=device.type != "cpu",
        )

        self.spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.non_spec_sequence_indices_cpu: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=device.type != "cpu",
        )
        self.spec_sequence_indices: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device=device,
        )
        self.non_spec_sequence_indices: torch.Tensor = torch.empty(
            (sequence_index_capacity,),
            dtype=torch.int64,
            device=device,
        )
        self.spec_actual_seq_lengths: torch.Tensor | None = None
        self.non_spec_actual_seq_lengths: torch.Tensor | None = None
        if self._USE_COMMON_KERNEL_METADATA:
            self.spec_actual_seq_lengths = torch.empty((sequence_index_capacity + 1,), dtype=torch.int32, device=device)
            self.non_spec_actual_seq_lengths = torch.empty(
                (sequence_index_capacity + 1,), dtype=torch.int32, device=device
            )

    def _resize_spec_buffers_for_tree(self, device: torch.device) -> None:
        """Follow the draft-tree state width in the spec-sized buffers.

        ``GDNAttentionMetadataBuilder.__init__`` sizes every spec-sized buffer
        with ``1 + num_speculative_tokens``, the number of queries of a linear
        draft chain, but a draft tree verifies ``1 + budget`` tokens per request
        (``tree_spec_config.budget >= num_speculative_tokens``) and caches one
        state per node. ``MambaSpec`` is widened to the same width by
        ``attn_utils.get_kv_cache_spec``. The tensors allocated below mirror the
        upstream constructor and have to stay in sync with it.
        """
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            tree_state_slot_count,
            validate_tree_state_slots,
        )

        slots = tree_state_slot_count(self.vllm_config)
        if slots is None:
            return
        mamba_type = getattr(self.kv_cache_spec, "mamba_type", None)
        if mamba_type is not None:
            validate_tree_state_slots(slots, mamba_type)
        self.spec_tree_slots = slots
        self.spec_init_state_indices = torch.empty(
            (self.decode_cudagraph_max_bs, slots), dtype=torch.int32, device=device
        )
        # ``attn_utils.get_kv_cache_spec`` widens the conv-state shape from the
        # linear-chain ``num_spec`` state columns to the tree's ``slots`` ones as
        # soon as the tree is wider than the chain, so the kernel width has to be
        # recovered with the column count the shape actually carries.
        state_columns = max(self.num_spec, slots)
        conv_width = _tree_conv_width(self.kv_cache_spec, mamba_type, state_columns)
        self.spec_conv_width = conv_width
        if conv_width is not None:
            # Row tables of the tree short conv, indexed by the spec-sized batch
            # and, for the windows, by its tokens; see _build_tree_conv_rows.
            self.spec_conv_window_rows = torch.empty(
                (self.decode_cudagraph_max_bs * slots, conv_width), dtype=torch.int32, device=device
            )
            self.spec_conv_prefix_src = torch.empty(
                (self.decode_cudagraph_max_bs, conv_width - 1), dtype=torch.int32, device=device
            )
        if slots == self.num_spec + 1:
            # The tree is exactly as wide as the draft chain, so the buffers the
            # upstream constructor allocated already carry its width. The verify
            # may still outgrow one op call, hence the chunk buffers first.
            self._allocate_tree_verify_plan_buffers(device, slots)
            return

        self.num_spec = slots - 1
        self.decode_cudagraph_max_bs = self.vllm_config.scheduler_config.max_num_seqs * slots
        max_capture_size = self.compilation_config.max_cudagraph_capture_size
        if max_capture_size is not None:
            self.decode_cudagraph_max_bs = min(self.decode_cudagraph_max_bs, max_capture_size)

        max_bs = self.decode_cudagraph_max_bs
        self.spec_state_indices_tensor = torch.empty((max_bs, slots), dtype=torch.int32, device=device)
        self.non_spec_state_indices_tensor = torch.empty((max_bs,), dtype=torch.int32, device=device)
        self.spec_token_indx = torch.empty((max_bs * slots,), dtype=torch.int32, device=device)
        self.non_spec_token_indx = torch.empty((max_bs * slots,), dtype=torch.int32, device=device)
        self.spec_query_start_loc = torch.empty((max_bs + 1,), dtype=torch.int32, device=device)
        self.non_spec_query_start_loc = torch.empty((max_bs + 1,), dtype=torch.int32, device=device)
        self.num_accepted_tokens = torch.empty((max_bs,), dtype=torch.int32, device=device)
        if self.use_spec_decode and self.reorder_batch_threshold != 1:
            # The threshold counts the queries of one verification batch, and a
            # tree verification always runs the root plus every node.
            self.reorder_batch_threshold = slots

        self._allocate_tree_verify_plan_buffers(device, slots)

    def _allocate_tree_verify_plan_buffers(self, device: torch.device, slots: int) -> None:
        """Stable chunk tensors of a verify wider than one recurrent-op call.

        A tree wider than the recurrent kernel's per-call width is verified in
        chunks of that width (see :meth:`_build_tree_verify_plan`). A full graph
        replays the recorded chunks, so they have to be the same addresses with
        fresh values every step: they are allocated here, once, for the graph's
        row capacity, exactly like the state tables. An eager verify rebuilds its
        chunk tensors on every step and needs no buffers of its own.
        """
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            GDN_RECURRENT_MAX_VERIFY_TOKENS,
            allocate_tree_verify_plan_buffers,
        )

        if slots <= GDN_RECURRENT_MAX_VERIFY_TOKENS or not self.use_full_cuda_graph:
            return
        self.spec_verify_plan_buffers = allocate_tree_verify_plan_buffers(slots, self.decode_cudagraph_max_bs, device)

    def _build_tree_verify_plan(
        self,
        row_widths: torch.Tensor,
        state_indices: torch.Tensor,
        init_state_indices: torch.Tensor | None,
        scratch_base: int,
    ) -> "TreeVerifyPlan | None":
        """Chunk plan of a verify step wider than one recurrent-op call.

        ``None`` -- the common case -- keeps the plain single-call path. A wider
        tree is cut into ``GDN_RECURRENT_MAX_VERIFY_TOKENS``-wide chunks of node
        columns which are chained through the state pages, so the op never
        verifies more tokens per request than its kernel buffers hold (above that
        the kernel returns *without updating the state*, i.e. silently stale).
        See ``spec_decode/tree/state_rollback.compute_tree_verify_plan``.

        A full graph replays the recorded chunks, so it needs the chunk tensors to
        be the same addresses with fresh values: the plan is copied into the
        stable buffers allocated by :meth:`_allocate_tree_verify_plan_buffers`.
        ``row_widths`` has one entry per op row, i.e. including the padded rows of
        a graph batch (they have length 0 and the kernel skips them).

        Both tables have to be in the packed token-major order of the step
        (``_pack_tree_rows_to_tokens``), where ``table.reshape(-1)[p]`` is token
        ``p`` of the step; the caller only builds a plan when that packing ran.
        """
        if init_state_indices is None or self.spec_tree_slots is None:
            return None
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            compute_tree_verify_plan,
        )

        plan = compute_tree_verify_plan(
            row_widths,
            state_indices,
            init_state_indices,
            scratch_base,
        )
        if plan is None:
            return None
        if self.use_full_cuda_graph:
            if self.spec_verify_plan_buffers is None:
                raise RuntimeError(
                    "a full CUDA graph replays the recorded verify chunks by address, "
                    "so a draft tree wider than one recurrent-op call needs the stable "
                    "chunk buffers of _allocate_tree_verify_plan_buffers."
                )
            plan = plan.copy_into(self.spec_verify_plan_buffers)
        return plan

    def _build_tree_init_state_indices(
        self,
        state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor,
        query_lens_cpu: torch.Tensor,
        spec_sequence_masks_cpu: torch.Tensor,
        tree_parents: torch.Tensor | None,
        tree_num_nodes: torch.Tensor | None,
        spec_sequence_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        """Per-token state page each draft-tree node starts from.

        The result keeps the ``[rows, slots]`` shape but is packed token-major,
        because the recurrent kernel indexes the table by global token offset.
        Returns ``None`` for the linear MTP chain, where the recurrent kernel
        carries the state of the previous token in registers.
        """
        if self.spec_tree_slots is None or state_indices is None:
            return None
        if tree_parents is None or tree_num_nodes is None:
            raise RuntimeError(
                "draft-tree GDN state rollback needs the tree layout; "
                "tree_parents/tree_num_nodes were not passed to build()."
            )
        slots = self.spec_tree_slots
        # A row may verify fewer than ``1 + budget`` tokens: the scheduler
        # truncates the draft of a request near ``max_model_len`` and the tree
        # search can run out of candidates. The tables are packed token-major
        # below, so only the per-row bounds matter here.
        spec_query_lens_cpu = query_lens_cpu[spec_sequence_masks_cpu]
        if bool(((spec_query_lens_cpu < 1) | (spec_query_lens_cpu > slots)).any()):
            raise ValueError(
                "draft-tree GDN state rollback needs 1 <= verified tokens <= "
                f"1 + budget = {slots} per speculative request, got "
                f"{spec_query_lens_cpu.tolist()}."
            )
        # Lazy import: the tree spec-decode package imports this module.
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            compute_tree_init_state_indices,
        )

        init_state_indices = compute_tree_init_state_indices(
            state_indices,
            num_accepted_tokens,
            torch.index_select(tree_parents, 0, spec_sequence_indices),
            torch.index_select(tree_num_nodes, 0, spec_sequence_indices),
        )
        return _pack_tree_rows_to_tokens(init_state_indices, spec_query_lens_cpu, slots)

    def _select_spec_state_indices(
        self,
        block_table_tensor: torch.Tensor,
        spec_sequence_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select the per-request GDN state pages for speculative rows.

        The chain path keeps its original ``num_spec + 1`` pages.  A tree
        verify owns one page per verified node, so its table must use the
        widened ``1 + budget`` capacity allocated by ``attn_utils``.  Using
        the chain width here silently aliases tree nodes and corrupts the
        recurrent state after the first branch.
        """
        if self.spec_tree_slots is None:
            return torch.index_select(
                block_table_tensor[:, : self.num_spec + 1],
                0,
                spec_sequence_indices,
            )
        return torch.index_select(
            block_table_tensor[:, : self.spec_tree_slots],
            0,
            spec_sequence_indices,
        )

    def _build_tree_conv_rows(
        self,
        num_accepted_tokens: torch.Tensor,
        tree_parents: torch.Tensor | None,
        tree_num_nodes: torch.Tensor | None,
        spec_sequence_indices: torch.Tensor,
        prev_path_node_ids: torch.Tensor | None = None,
        prev_num_sampled: torch.Tensor | None = None,
        row_widths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Conv tap table of every draft-tree node and the committed history.

        ``window_rows`` indexes ``cat([committed history, this step's
        activations])`` and ``prefix_src`` names the conv-state columns of the
        history the next step resumes from, both consumed by
        ``ops/gdn.tree_causal_conv1d``. ``(None, None)`` keeps the
        sliding-window custom operator of the linear MTP chain.
        """
        if self.spec_tree_slots is None or self.spec_conv_window_rows is None:
            return None, None
        if tree_parents is None or tree_num_nodes is None:
            raise RuntimeError(
                "draft-tree GDN state rollback needs the tree layout; "
                "tree_parents/tree_num_nodes were not passed to build()."
            )
        if prev_path_node_ids is None or prev_num_sampled is None:
            raise RuntimeError(
                "draft-tree GDN short conv needs the accepted path of the previous step; it was not passed to build()."
            )
        rows = num_accepted_tokens.shape[0]
        assert self.spec_conv_width is not None
        # Lazy import: the tree spec-decode package imports this module.
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
            compute_tree_conv_rows,
            compute_tree_prefix_source,
        )

        window_rows = compute_tree_conv_rows(
            torch.index_select(tree_parents, 0, spec_sequence_indices),
            torch.index_select(tree_num_nodes, 0, spec_sequence_indices),
            self.spec_conv_width,
            # The committed history of every request comes first, then this
            # step's activations, which are laid out request-major over the
            # tokens each request actually verifies.
            rows * (self.spec_conv_width - 1),
            row_widths=row_widths,
        )
        prefix_src = compute_tree_prefix_source(
            torch.index_select(prev_path_node_ids, 0, spec_sequence_indices),
            torch.index_select(prev_num_sampled, 0, spec_sequence_indices),
            self.spec_conv_width,
        )
        return window_rows, prefix_src

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        super()._init_reorder_batch_threshold(
            reorder_batch_threshold,
            supports_spec_as_decode,
            True,
        )
        if self.reorder_batch_threshold != 1:  # type: ignore
            speculative_config = self.vllm_config.speculative_config
            method = getattr(speculative_config, "method", None)
            num_spec = getattr(speculative_config, "num_speculative_tokens", None)
            if num_spec is not None and method in ("dflash", "dspark"):
                # The target-model verification query always contains the
                # base token plus N speculative tokens. DSpark's
                # sample_from_anchor only makes the draft-model forward use N
                # queries; it must not change target batch reordering.
                self.reorder_batch_threshold = 1 + num_spec

    def _copy_sequence_indices_to_device(
        self,
        spec_sequence_masks_cpu: torch.Tensor,
        num_spec_decodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = spec_sequence_masks_cpu.numel()
        num_non_spec_decodes = num_reqs - num_spec_decodes

        spec_indices_cpu = self.spec_sequence_indices_cpu[:num_spec_decodes]
        spec_indices_cpu.copy_(
            torch.nonzero(spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        spec_indices = self.spec_sequence_indices[:num_spec_decodes]
        spec_indices.copy_(spec_indices_cpu, non_blocking=True)

        non_spec_indices_cpu = self.non_spec_sequence_indices_cpu[:num_non_spec_decodes]
        non_spec_indices_cpu.copy_(
            torch.nonzero(~spec_sequence_masks_cpu, as_tuple=True)[0],
        )
        non_spec_indices = self.non_spec_sequence_indices[:num_non_spec_decodes]
        non_spec_indices.copy_(non_spec_indices_cpu, non_blocking=True)

        return spec_indices, non_spec_indices

    def _reset_spec_decode_graph_inputs(self, graph_batch_size: int) -> None:
        """Make a captured speculative branch a no-op for this replay."""
        self.spec_state_indices_tensor[:graph_batch_size].fill_(PAD_SLOT_ID)
        self.spec_query_start_loc[: graph_batch_size + 1].zero_()
        self.num_accepted_tokens[:graph_batch_size].zero_()
        if self.spec_actual_seq_lengths is not None:
            self.spec_actual_seq_lengths[: graph_batch_size + 1].zero_()
        if self.spec_init_state_indices is not None:
            self.spec_init_state_indices[:graph_batch_size].fill_(NULL_BLOCK_ID)
        if self.spec_verify_plan_buffers is not None:
            self.spec_verify_plan_buffers.zero_lengths()

    def _attach_non_spec_prefill_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        chunk_metadata: GDNChunkedPrefillMetadata | None,
        non_spec_cache_indices: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_prefill_metadata = None
        if not self._USE_COMMON_KERNEL_METADATA or attn_metadata.num_prefills <= 0:
            return attn_metadata

        if attn_metadata.non_spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.non_spec_query_start_loc for Ascend GDN non-spec prefill path.")
        if attn_metadata.prefill_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.prefill_query_start_loc for Ascend GDN non-spec prefill path.")
        if chunk_metadata is None:
            raise RuntimeError("Expected chunk metadata for Ascend GDN non-spec prefill path.")

        initial_state_mode = attn_metadata.has_initial_state
        if non_spec_cache_indices is None:
            raise RuntimeError("Expected non_spec_cache_indices for Ascend GDN prefill conv1d path.")
        prefill_num_rows = attn_metadata.non_spec_query_start_loc.size(0) - 1
        pcp_size = getattr(self.vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        if pcp_rank > 0 and attn_metadata.num_prefills > 0:
            prefill_seq_offset = max(0, prefill_num_rows - attn_metadata.num_prefills)
            initial_state_mode = initial_state_mode.clone()
            initial_state_mode[prefill_seq_offset:] = True
        attn_metadata.non_spec_prefill_metadata = GDNPrefillMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(
                query_start_loc=attn_metadata.non_spec_query_start_loc,
                cache_indices=non_spec_cache_indices[:prefill_num_rows],
                initial_state_mode=initial_state_mode,
            ),
            chunk=chunk_metadata,
        )
        return attn_metadata

    def _attach_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        init_state_indices: torch.Tensor | None = None,
        conv_window_rows: torch.Tensor | None = None,
        conv_prefix_src: torch.Tensor | None = None,
    ) -> GDNAttentionMetadata:
        attn_metadata.spec_decode_metadata = None
        if attn_metadata.spec_sequence_masks is None:
            return attn_metadata

        if attn_metadata.spec_query_start_loc is None:
            raise RuntimeError("Expected attn_metadata.spec_query_start_loc for Ascend GDN speculative path.")
        if attn_metadata.spec_state_indices_tensor is None:
            raise RuntimeError("Expected spec_state_indices_tensor for Ascend GDN speculative conv1d path.")
        if attn_metadata.num_accepted_tokens is None:
            raise RuntimeError("Expected num_accepted_tokens for Ascend GDN speculative conv1d path.")

        num_sequences = attn_metadata.num_spec_decodes
        actual_seq_lengths_buffer = None
        if self.use_full_cuda_graph and attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
            num_sequences = attn_metadata.spec_query_start_loc.size(0) - 1
            actual_seq_lengths_buffer = self.spec_actual_seq_lengths
        spec_num_rows = attn_metadata.spec_query_start_loc.size(0) - 1
        actual_seq_lengths = _build_actual_seq_lengths(
            attn_metadata.spec_query_start_loc,
            num_sequences,
            actual_seq_lengths_buffer,
        )
        # A wider-than-one-call tree is verified in chunks; the rows are the op's
        # rows, i.e. the padded graph batch under a full graph (a padded row has
        # query length 0 and the kernel skips it). The plan indexes the state
        # tables by packed token offset, so it is only built when the tree packing
        # ran -- ``build`` gates that on the tree conv rows.
        verify_plan = None
        if conv_window_rows is not None:
            verify_plan = self._build_tree_verify_plan(
                actual_seq_lengths[1:],
                attn_metadata.spec_state_indices_tensor[:spec_num_rows],
                None if init_state_indices is None else init_state_indices[:spec_num_rows],
                attn_metadata.num_actual_tokens,
            )

        attn_metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
            spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
                query_start_loc=attn_metadata.spec_query_start_loc,
                cache_indices=attn_metadata.spec_state_indices_tensor[:spec_num_rows],
                num_accepted_tokens=attn_metadata.num_accepted_tokens[:spec_num_rows],
            ),
            actual_seq_lengths=actual_seq_lengths if self._USE_COMMON_KERNEL_METADATA else None,
            init_state_indices=init_state_indices,
            conv_window_rows=conv_window_rows,
            conv_prefix_src=conv_prefix_src,
            verify_plan=verify_plan,
        )
        return attn_metadata

    def _attach_non_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        non_spec_cache_indices: torch.Tensor | None,
    ) -> GDNAttentionMetadata:
        attn_metadata.non_spec_decode_metadata = None
        if not self._USE_COMMON_KERNEL_METADATA:
            return attn_metadata
        if attn_metadata.num_decodes <= 0:
            return attn_metadata

        if attn_metadata.non_spec_query_start_loc is None:
            raise RuntimeError("Expected non-spec query_start_loc for Ascend GDN non-spec decode path.")
        if non_spec_cache_indices is None:
            raise RuntimeError("Expected non_spec_cache_indices for Ascend GDN decode conv1d path.")

        num_sequences = attn_metadata.num_decodes
        non_spec_num_rows = attn_metadata.non_spec_query_start_loc.size(0) - 1
        actual_seq_lengths_buffer = None
        if self.use_full_cuda_graph and attn_metadata.num_prefills == 0 and attn_metadata.num_spec_decodes == 0:
            num_sequences = attn_metadata.non_spec_query_start_loc.size(0) - 1
            actual_seq_lengths_buffer = self.non_spec_actual_seq_lengths

        attn_metadata.non_spec_decode_metadata = GDNDecodeMetadata(
            causal_conv1d=GDNCausalConv1dMetadata(
                query_start_loc=attn_metadata.non_spec_query_start_loc,
                cache_indices=non_spec_cache_indices[:non_spec_num_rows],
                initial_state_mode=None,
            ),
            actual_seq_lengths=_build_actual_seq_lengths(
                attn_metadata.non_spec_query_start_loc,
                num_sequences,
                actual_seq_lengths_buffer,
            ),
        )
        return attn_metadata

    _SPEC_GRAPH_PAD_SLOT_ID = NULL_BLOCK_ID
    # Common GDN/KDA kernels consume prebuilt chunks and recurrent lengths.
    # 310P consumes top-level fields and only shares the spec conv wrapper.
    _USE_COMMON_KERNEL_METADATA = True

    def _can_pad_spec_decode(self, graph_request_count: int, num_spec_decode_tokens: int) -> bool:
        return (
            graph_request_count <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        )

    def _pad_spec_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_request_count: int,
    ) -> None:
        """Materialize speculative decode metadata at graph request granularity."""
        actual_request_count = attn_metadata.num_spec_decodes
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        assert spec_state_indices is not None
        assert spec_sequence_masks is not None
        assert spec_query_start_loc is not None
        assert num_accepted_tokens is not None

        # Request and token buffers have different capacities under MTP.
        # Check token indices separately before modifying any captured inputs.
        assert attn_metadata.non_spec_token_indx is not None
        assert attn_metadata.spec_token_indx is not None
        if attn_metadata.spec_token_indx.numel() > self.spec_token_indx.numel():
            raise ValueError("Spec token indices exceed the graph token buffer capacity")
        if attn_metadata.non_spec_token_indx.numel() > self.non_spec_token_indx.numel():
            raise ValueError("Non-spec token indices exceed the graph token buffer capacity")

        attn_metadata.spec_state_indices_tensor = _materialize_graph_request_tensor(
            self.spec_state_indices_tensor,
            spec_state_indices,
            actual_request_count,
            graph_request_count,
            self._SPEC_GRAPH_PAD_SLOT_ID,
        )
        attn_metadata.spec_sequence_masks = _materialize_graph_request_tensor(
            self.spec_sequence_masks,
            spec_sequence_masks,
            actual_request_count,
            graph_request_count,
            False,
        )

        non_spec_token_count = attn_metadata.non_spec_token_indx.size(0)
        spec_token_count = attn_metadata.spec_token_indx.size(0)
        self.non_spec_token_indx[:non_spec_token_count].copy_(
            attn_metadata.non_spec_token_indx,
            non_blocking=True,
        )
        self.spec_token_indx[:spec_token_count].copy_(
            attn_metadata.spec_token_indx,
            non_blocking=True,
        )
        attn_metadata.non_spec_token_indx = self.non_spec_token_indx[:non_spec_token_count]
        attn_metadata.spec_token_indx = self.spec_token_indx[:spec_token_count]

        attn_metadata.spec_query_start_loc = _materialize_graph_query_start_loc(
            self.spec_query_start_loc,
            spec_query_start_loc,
            actual_request_count,
            graph_request_count,
        )
        attn_metadata.num_accepted_tokens = _materialize_graph_request_tensor(
            self.num_accepted_tokens,
            num_accepted_tokens,
            actual_request_count,
            graph_request_count,
            _GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT,
        )

    def _pad_decode_metadata(
        self,
        attn_metadata: GDNAttentionMetadata,
        graph_request_count: int,
    ) -> None:
        """Materialize non-speculative decode metadata at request granularity."""
        # Ordinary decode has one token per live request. Zero-length graph
        # rows (including idle DP batches) must not copy stale state indices.
        actual_request_count = attn_metadata.num_decode_tokens
        state_indices = attn_metadata.non_spec_state_indices_tensor
        query_start_loc = attn_metadata.non_spec_query_start_loc
        assert state_indices is not None
        assert query_start_loc is not None

        attn_metadata.non_spec_state_indices_tensor = _materialize_graph_request_tensor(
            self.non_spec_state_indices_tensor,
            state_indices,
            actual_request_count,
            graph_request_count,
            NULL_BLOCK_ID,
        )
        attn_metadata.non_spec_query_start_loc = _materialize_graph_query_start_loc(
            self.non_spec_query_start_loc,
            query_start_loc,
            actual_request_count,
            graph_request_count,
        )

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
        tree_parents: torch.Tensor | None = None,
        tree_num_nodes: torch.Tensor | None = None,
        prev_path_node_ids: torch.Tensor | None = None,
        prev_num_sampled: torch.Tensor | None = None,
    ) -> GDNAttentionMetadata:
        # Normalize FIA's full-width dummy queries before slicing requests:
        # its common token count may still include those inactive rows.
        if self.use_full_cuda_graph and self.use_spec_decode:
            common_attn_metadata = _remove_spec_graph_padding_queries(
                common_attn_metadata,
                num_decode_draft_tokens_cpu,
            )
        m = common_attn_metadata
        # Common metadata follows the padded graph shape. Build request-phase
        # metadata from the logical batch, then materialize graph-sized buffers.
        graph_request_count = m.num_reqs
        if num_actual_reqs is None:
            num_actual_reqs = graph_request_count
        elif not 0 <= num_actual_reqs <= graph_request_count:
            raise ValueError(
                f"num_actual_reqs ({num_actual_reqs}) must be between 0 and graph_request_count ({graph_request_count})"
            )

        if num_actual_reqs < graph_request_count:
            m = m.unpadded(m.num_actual_tokens, num_actual_reqs)
            if num_accepted_tokens is not None:
                num_accepted_tokens = num_accepted_tokens[:num_actual_reqs]
            if num_decode_draft_tokens_cpu is not None:
                num_decode_draft_tokens_cpu = num_decode_draft_tokens_cpu[:num_actual_reqs]
        m = _treat_single_token_prefills_with_state_as_decodes(m)

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        mamba_cache_mode = self.vllm_config.cache_config.mamba_cache_mode
        mamba_block_table = m.block_table_tensor
        # vLLM #58737 derives num_reqs from the block table and asserts that
        # seq_lens has the same number of rows. Ascend's unpadded() intentionally
        # keeps the padded block-table rows, so on vLLM main pass only the real
        # request rows. v0.31.0's align path already derives its output rows from
        # seq_lens, so the result is unchanged there.
        if (
            not vllm_version_is("0.31.0")
            and mamba_cache_mode != "none"
            and mamba_block_table.shape[0] != m.seq_lens.shape[0]
        ):
            mamba_block_table = mamba_block_table[: m.seq_lens.shape[0]]
        block_table_tensor = mamba_get_block_table_tensor(
            mamba_block_table,
            m.seq_lens,
            self.kv_cache_spec,
            mamba_cache_mode,
        )

        spec_sequence_masks_cpu: torch.Tensor | None = None
        spec_sequence_indices: torch.Tensor | None = None
        non_spec_sequence_indices: torch.Tensor | None = None
        non_spec_conv1d_cache_indices: torch.Tensor | None = None
        if not self.use_spec_decode or num_decode_draft_tokens_cpu is None:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            num_reqs = num_decode_draft_tokens_cpu.numel()
            spec_sequence_masks_cpu = self.spec_sequence_masks_cpu[:num_reqs]
            runtime_draft_tokens = num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
            if runtime_draft_tokens.sum().item() > 0:
                torch.ge(
                    num_decode_draft_tokens_cpu,
                    0,
                    out=spec_sequence_masks_cpu,
                )
            else:
                # Dynamic speculative decoding can be enabled while this batch
                # carries no draft tokens. Keep the normal prefill/decode split.
                spec_sequence_masks_cpu.zero_()
            num_spec_decodes = spec_sequence_masks_cpu.sum().item()
            if num_spec_decodes == 0:
                spec_sequence_masks = None
                spec_sequence_masks_cpu = None
            else:
                spec_sequence_masks = self.spec_sequence_masks[:num_reqs]
                spec_sequence_masks.copy_(spec_sequence_masks_cpu, non_blocking=True)
                spec_sequence_indices, non_spec_sequence_indices = self._copy_sequence_indices_to_device(
                    spec_sequence_masks_cpu,
                    num_spec_decodes,
                )

        spec_init_state_indices: torch.Tensor | None = None
        spec_conv_window_rows: torch.Tensor | None = None
        spec_conv_prefix_src: torch.Tensor | None = None
        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
                m,
                decode_threshold=1,
                treat_short_extends_as_decodes=False,
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            non_spec_state_indices_tensor = block_table_tensor[:, 0]
            non_spec_conv1d_cache_indices = block_table_tensor
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            assert spec_sequence_masks_cpu is not None
            assert spec_sequence_indices is not None
            assert non_spec_sequence_indices is not None

            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            num_prefills = non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
            num_decode_tokens = num_decodes
            num_prefill_tokens = non_spec_query_lens_cpu.sum().item() - num_decode_tokens
            num_spec_decode_tokens = query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens

            if num_decodes > 0 and num_spec_decodes > 0:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0

            if num_prefills == 0 and num_decodes == 0:
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec + 1),
                    query_start_loc_cpu[-1].item(),
                )
                spec_token_indx = torch.arange(
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                non_spec_token_indx = torch.empty(
                    0,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                spec_state_indices_tensor = self._select_spec_state_indices(
                    block_table_tensor,
                    spec_sequence_indices,
                )
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
            else:
                spec_token_masks = torch.repeat_interleave(
                    spec_sequence_masks,
                    query_lens,
                    output_size=query_start_loc_cpu[-1].item(),
                )
                index = _stable_argsort_for_npu(spec_token_masks)
                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                spec_state_indices_tensor = self._select_spec_state_indices(
                    block_table_tensor,
                    spec_sequence_indices,
                )
                non_spec_state_indices_tensor = torch.index_select(
                    block_table_tensor[:, 0],
                    0,
                    non_spec_sequence_indices,
                )
                non_spec_conv1d_cache_indices = non_spec_state_indices_tensor
                spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    spec_sequence_indices,
                )
                non_spec_query_lens = torch.index_select(
                    query_lens,
                    0,
                    non_spec_sequence_indices,
                )

                spec_query_start_loc = torch.zeros(
                    num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    spec_query_lens,
                    dim=0,
                    out=spec_query_start_loc[1:],
                )
                non_spec_query_start_loc = torch.zeros(
                    query_lens.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                )
                torch.cumsum(
                    non_spec_query_lens,
                    dim=0,
                    out=non_spec_query_start_loc[1:],
                )
                non_spec_query_start_loc_cpu = torch.zeros(
                    query_lens_cpu.size(0) - num_spec_decodes + 1,
                    dtype=torch.int32,
                )
                torch.cumsum(
                    query_lens_cpu[~spec_sequence_masks_cpu],
                    dim=0,
                    out=non_spec_query_start_loc_cpu[1:],
                )

            assert num_accepted_tokens is not None
            num_accepted_tokens = torch.index_select(
                num_accepted_tokens,
                0,
                spec_sequence_indices,
            )
            # Tokens each speculative request verifies; a row may be shorter
            # than the tree width, see _build_tree_init_state_indices.
            spec_row_widths = query_lens_cpu[spec_sequence_masks_cpu]
            spec_init_state_indices = self._build_tree_init_state_indices(
                spec_state_indices_tensor,
                num_accepted_tokens,
                query_lens_cpu,
                spec_sequence_masks_cpu,
                tree_parents,
                tree_num_nodes,
                spec_sequence_indices,
            )
            spec_conv_window_rows, spec_conv_prefix_src = self._build_tree_conv_rows(
                num_accepted_tokens,
                tree_parents,
                tree_num_nodes,
                spec_sequence_indices,
                prev_path_node_ids,
                prev_num_sampled,
                row_widths=spec_row_widths,
            )
            if spec_init_state_indices is not None and self.spec_conv_window_rows is not None:
                # Same packing for the state pages: the recurrent op consumes
                # them by global token offset, like the init pages and the conv
                # windows. The conv path keeps reading the table per row, taking
                # each request's running page at its token start
                # (ops/gdn.tree_causal_conv1d).
                spec_state_indices_tensor = _pack_tree_rows_to_tokens(
                    spec_state_indices_tensor,
                    spec_row_widths,
                    spec_state_indices_tensor.shape[1],
                )

        # A FULL graph retains captured speculative conv/recurrent tasks. Clear
        # their stable inputs on every no-spec replay so an idle or prefill
        # batch cannot mutate state belonging to the preceding request.
        if self.use_full_cuda_graph and self.use_spec_decode and num_spec_decodes == 0:
            self._reset_spec_decode_graph_inputs(graph_request_count)

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        prefill_query_start_loc: torch.Tensor | None = None
        prefill_query_start_loc_cpu: torch.Tensor | None = None
        prefill_state_indices: torch.Tensor | None = None
        prefill_has_initial_state: torch.Tensor | None = None
        non_spec_chunked_prefill_metadata: GDNChunkedPrefillMetadata | None = None
        if num_prefills > 0:
            if spec_sequence_masks is None and num_decodes > 0:
                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                assert non_spec_state_indices_tensor is not None
                prefill_query_start_loc = non_spec_query_start_loc[num_decodes:] - num_decode_tokens
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu[num_decodes:] - num_decode_tokens
                prefill_state_indices = non_spec_state_indices_tensor[num_decodes:]
            else:
                prefill_query_start_loc = non_spec_query_start_loc
                prefill_query_start_loc_cpu = non_spec_query_start_loc_cpu
                prefill_state_indices = non_spec_state_indices_tensor

            assert prefill_query_start_loc_cpu is not None
            if self._USE_COMMON_KERNEL_METADATA:
                non_spec_chunked_prefill_metadata = _build_non_spec_chunked_prefill_metadata(
                    self,
                    prefill_query_start_loc_cpu,
                    query_start_loc.device,
                    build_device_metadata=self.build_device_chunk_metadata,
                )
                # Common GDN/KDA callers also consume the upstream chunk fields.
                chunk_indices = non_spec_chunked_prefill_metadata.chunk_indices_chunk64
                chunk_offsets = non_spec_chunked_prefill_metadata.chunk_offsets_chunk64

        if num_prefills > 0:
            has_initial_state = self._build_prefill_has_initial_state(
                context_lens_tensor=m.compute_num_computed_tokens(),
                non_spec_sequence_indices=non_spec_sequence_indices,
            )
            if spec_sequence_masks is None and num_decodes > 0:
                prefill_has_initial_state = has_initial_state[num_decodes:]
            else:
                prefill_has_initial_state = has_initial_state
        else:
            has_initial_state = None

        assert not (num_decodes > 0 and num_spec_decodes > 0), (
            f"num_decodes: {num_decodes}, num_spec_decodes: {num_spec_decodes}"
        )

        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_state_indices=prefill_state_indices,
            prefill_has_initial_state=prefill_has_initial_state,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
        )
        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and self._can_pad_spec_decode(graph_request_count, num_spec_decode_tokens)
        ):
            self._pad_spec_decode_metadata(attn_metadata, graph_request_count)
            if spec_init_state_indices is not None:
                assert self.spec_init_state_indices is not None
                self.spec_init_state_indices[:num_spec_decodes].copy_(spec_init_state_indices, non_blocking=True)
                spec_init_state_indices = self.spec_init_state_indices[:graph_request_count]
                spec_init_state_indices[num_spec_decodes:].fill_(NULL_BLOCK_ID)
            if spec_conv_window_rows is not None:
                assert spec_conv_prefix_src is not None
                assert self.spec_conv_window_rows is not None and self.spec_conv_prefix_src is not None
                self.spec_conv_window_rows[: spec_conv_window_rows.size(0)].copy_(
                    spec_conv_window_rows, non_blocking=True
                )
                self.spec_conv_prefix_src[: spec_conv_prefix_src.size(0)].copy_(spec_conv_prefix_src, non_blocking=True)
                spec_conv_window_rows = self.spec_conv_window_rows[: graph_request_count * self.spec_tree_slots]
                spec_conv_prefix_src = self.spec_conv_prefix_src[:graph_request_count]
        elif (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and graph_request_count <= self.decode_cudagraph_max_bs
        ):
            self._pad_decode_metadata(attn_metadata, graph_request_count)
            non_spec_conv1d_cache_indices = attn_metadata.non_spec_state_indices_tensor
        # Attach once, using the final graph-stable Tensor references.
        self._attach_non_spec_prefill_metadata(
            attn_metadata, non_spec_chunked_prefill_metadata, non_spec_conv1d_cache_indices
        )
        self._attach_spec_decode_metadata(
            attn_metadata,
            spec_init_state_indices,
            spec_conv_window_rows,
            spec_conv_prefix_src,
        )
        self._attach_non_spec_decode_metadata(attn_metadata, non_spec_conv1d_cache_indices)
        return attn_metadata

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> GDNAttentionMetadata:
        if self.spec_tree_slots is None:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        m = common_attn_metadata
        device = m.query_start_loc.device
        tree_parents = torch.zeros((m.num_reqs, self.spec_tree_slots - 1), dtype=torch.int32, device=device)
        tree_num_nodes = torch.zeros((m.num_reqs,), dtype=torch.int32, device=device)
        prev_path_node_ids = torch.zeros_like(tree_parents)
        prev_num_sampled = torch.zeros((m.num_reqs,), dtype=torch.int32, device=device)
        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
        return self.build(
            0,
            m,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            tree_parents=tree_parents,
            tree_num_nodes=tree_num_nodes,
            prev_path_node_ids=prev_path_node_ids,
            prev_num_sampled=prev_num_sampled,
        )

    def _build_prefill_has_initial_state(
        self,
        *,
        context_lens_tensor: torch.Tensor,
        non_spec_sequence_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        has_initial_state = context_lens_tensor > 0
        if non_spec_sequence_indices is not None:
            has_initial_state = torch.index_select(
                has_initial_state,
                0,
                non_spec_sequence_indices,
            )
        return has_initial_state


class AscendGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        return AscendGDNAttentionMetadataBuilder


class AscendGDNHostMetadataBuilder(AscendGDNAttentionMetadataBuilder):
    """Build only the descriptors consumed by AscendC host-metadata paths."""

    build_device_chunk_metadata = False

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._conv_layer_name = layer_names[0] if layer_names else None
        self._spec_conv_indices: torch.Tensor | None = None
        self._non_spec_conv_indices: torch.Tensor | None = None
        self._conv_output_buffer: torch.Tensor | None = None

    def _ensure_conv_output_buffer(self) -> None:
        if self._conv_output_buffer is not None:
            return
        layers = getattr(self.compilation_config, "static_forward_context", {})
        layer = layers.get(self._conv_layer_name)
        if layer is None or not getattr(layer, "use_shared_conv_output", False):
            return
        # Decode needs at most K+1 tokens per request; prefill does not use this
        # buffer. Include graph padding without reserving the prefill token budget.
        token_capacity = max(
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1),
            self.decode_cudagraph_max_bs,
        )
        self._conv_output_buffer = torch.empty(
            (token_capacity, 3 * layer.local_projection_size),
            dtype=self.vllm_config.model_config.dtype,
            device=self.spec_sequence_masks.device,
        )

    def _ensure_conv_index_buffers(self) -> None:
        if self._spec_conv_indices is not None:
            return
        # Resolve the group's representative layer when building metadata.
        # Only opted-in models allocate the additional convolution buffers.
        layers = getattr(self.compilation_config, "static_forward_context", {})
        if not getattr(layers.get(self._conv_layer_name), "use_contiguous_conv_indices", False):
            return
        # Each group owns its buffers, with fixed addresses for graph replay.
        device = self.spec_sequence_masks.device
        self._spec_conv_indices = torch.empty(self.spec_sequence_masks.numel(), dtype=torch.int32, device=device)
        self._non_spec_conv_indices = torch.empty_like(self._spec_conv_indices)

    @staticmethod
    def _prepare_conv_indices(indices: torch.Tensor, buffer: torch.Tensor) -> torch.Tensor:
        # Mixed batches and ordinary graph decode already provide a contiguous
        # vector. Matrix sources are prepared once, including single-row MTP.
        if indices.ndim == 1 and indices.is_contiguous():
            return indices
        column = indices[:, 0] if indices.ndim == 2 else indices
        requests = column.numel()
        if requests > buffer.numel():
            raise ValueError("Convolution index count exceeds the cache-group buffer capacity")
        prepared = buffer[:requests]
        prepared.copy_(column, non_blocking=True)
        return prepared

    def _reset_spec_decode_graph_inputs(self, graph_batch_size: int) -> None:
        super()._reset_spec_decode_graph_inputs(graph_batch_size)
        if self._spec_conv_indices is not None:
            self._spec_conv_indices[:graph_batch_size].fill_(PAD_SLOT_ID)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
        num_actual_reqs: int | None = None,
        tree_parents: torch.Tensor | None = None,
        tree_num_nodes: torch.Tensor | None = None,
        prev_path_node_ids: torch.Tensor | None = None,
        prev_num_sampled: torch.Tensor | None = None,
    ) -> GDNAttentionMetadata:
        self._ensure_conv_index_buffers()
        self._ensure_conv_output_buffer()
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build,
            num_actual_reqs=num_actual_reqs,
            tree_parents=tree_parents,
            tree_num_nodes=tree_num_nodes,
            prev_path_node_ids=prev_path_node_ids,
            prev_num_sampled=prev_num_sampled,
        )
        if self._conv_output_buffer is not None:
            # Layers in this group consume the same decode rows in sequence.
            # Clear once per step, including inactive captured branches: rows
            # that were active on the previous replay can now be padding.
            self._conv_output_buffer.zero_()
            if metadata.spec_decode_metadata is not None:
                metadata.spec_decode_metadata.spec_causal_conv1d.output_buffer = self._conv_output_buffer
            if metadata.non_spec_decode_metadata is not None:
                metadata.non_spec_decode_metadata.causal_conv1d.output_buffer = self._conv_output_buffer
        if self._spec_conv_indices is None:
            return metadata
        assert self._non_spec_conv_indices is not None
        # Parent build has finalized request counts and graph padding. Keep
        # recurrent indices intact and only replace the convolution view.
        if metadata.spec_decode_metadata is not None:
            conv = metadata.spec_decode_metadata.spec_causal_conv1d
            conv.cache_indices = self._prepare_conv_indices(conv.cache_indices, self._spec_conv_indices)
        non_spec_phases = (metadata.non_spec_prefill_metadata, metadata.non_spec_decode_metadata)
        conv_metadata = [phase.causal_conv1d for phase in non_spec_phases if phase is not None]
        if conv_metadata:
            # Both phases reference prefixes of the same non-spec index table.
            source = max(conv_metadata, key=lambda conv: conv.cache_indices.shape[0])
            prepared = self._prepare_conv_indices(source.cache_indices, self._non_spec_conv_indices)
            for conv in conv_metadata:
                conv.cache_indices = prepared[: conv.cache_indices.shape[0]]
        return metadata


class AscendGDNHostMetadataBackend(AscendGDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendGDNHostMetadataBuilder]:
        return AscendGDNHostMetadataBuilder


class AscendGDNFusedAttentionBackend(AscendGDNAttentionBackend):
    """Select host-only metadata when Qwen can use a fused prefill op."""

    @staticmethod
    def get_builder_cls() -> type[AscendGDNAttentionMetadataBuilder]:
        # Import lazily to avoid the gdn <-> builder import cycle at module
        # initialization.  The probe result is cached by the attention class.
        from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention

        if get_pcp_group().world_size == 1 and AscendGatedDeltaNetAttention._supports_host_metadata_prefill():
            return AscendGDNHostMetadataBuilder
        return AscendGDNAttentionMetadataBuilder
