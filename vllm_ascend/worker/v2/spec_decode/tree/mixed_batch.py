# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-owned phase views for one mixed DSV4 target invocation."""

from copy import copy
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TreePhase:
    request_indices: torch.Tensor
    token_indices: torch.Tensor
    query_start_loc: torch.Tensor
    request_indices_cpu: torch.Tensor
    token_indices_cpu: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    omitted_tokens: torch.Tensor
    omitted_tokens_cpu: torch.Tensor
    prefilling: bool

    @property
    def num_reqs(self):
        return self.request_indices.shape[0]

    @property
    def num_tokens(self):
        return self.token_indices.shape[0]

    def common_metadata(self, original):
        """Select raw inputs; the ordinary builder regenerates all tiling data."""
        result = copy(original)
        req = self.request_indices
        cpu_req = self.request_indices_cpu
        for name in (
            "seq_lens", "seq_lens_cpu", "seq_lens_cpu_upper_bound",
            "_seq_lens_cpu", "num_computed_tokens_cpu", "_num_computed_tokens_cpu",
            "is_prefilling", "req_ids_tensor", "req_topk_buffer_slots",
            "req_topk_buffer_active", "replay_start", "encoder_seq_lens",
            "tree_visibility",
        ):
            value = getattr(original, name, None)
            if value is not None:
                indices = cpu_req if value.device.type == "cpu" else req
                setattr(result, name, value.index_select(0, indices))
        for name in ("encoder_seq_lens_cpu", "req_idx"):
            value = getattr(original, name, None)
            if value is not None:
                setattr(result, name, value[cpu_req.numpy()])
        if isinstance(original.causal, torch.Tensor):
            result.causal = original.causal.index_select(0, req)
        # Derived caches describe the original batch, so rebuild from the phase
        # query boundaries and sequence lengths on first use.
        result._num_computed_tokens_cache = None
        result._token_to_req_indices_cache = None
        # A prompt-tail admission can contain placeholder drafts. The linear
        # prefill path consumes only known context, never those placeholder rows.
        result.seq_lens = result.seq_lens - self.omitted_tokens
        for name in ("seq_lens_cpu", "seq_lens_cpu_upper_bound", "_seq_lens_cpu"):
            value = getattr(result, name, None)
            if value is not None:
                setattr(result, name, value - self.omitted_tokens_cpu)
        result.block_table_tensor = original.block_table_tensor.index_select(0, req)
        for name in ("slot_mapping", "positions", "slot_positions"):
            value = getattr(original, name, None)
            if value is not None:
                setattr(result, name, value.index_select(0, self.token_indices))
        if original.positions_cpu is not None:
            result.positions_cpu = original.positions_cpu.index_select(0, self.token_indices_cpu)
        if original.token_to_req is not None:
            result.token_to_req = torch.repeat_interleave(
                torch.arange(self.num_reqs, dtype=original.token_to_req.dtype, device=req.device),
                torch.diff(self.query_start_loc), output_size=self.num_tokens,
            )
        result.query_start_loc = self.query_start_loc
        result.query_start_loc_cpu = self.query_start_loc_cpu
        result.num_reqs = self.num_reqs
        result.num_actual_tokens = result.num_input_tokens = self.num_tokens
        result.max_query_len = int(torch.diff(self.query_start_loc_cpu).max())
        result.max_seq_len = int(result.seq_lens_cpu_upper_bound.max())
        result.graph_pad_size = -1
        result.for_cudagraph_capture = False
        result.actual_seq_lengths_q = self.query_start_loc_cpu[1:].tolist()
        if self.prefilling:
            result.tree_visibility = None
        if getattr(original, "mm_req_doc_ranges", None) is not None:
            result.mm_req_doc_ranges = {
                new: original.mm_req_doc_ranges[old]
                for new, old in enumerate(cpu_req.tolist())
                if old in original.mm_req_doc_ranges
            }
        return result


@dataclass(frozen=True)
class TreeMixedBatch:
    decode: TreePhase
    prefill: TreePhase
    decode_mask: torch.Tensor
    host_uploads: list[torch.Tensor]

    @classmethod
    def from_input_batch(cls, batch, device):
        from vllm_ascend.worker.v2.spec_decode.tree.chain_pack import _upload_metadata

        # All decisions use the runner's existing CPU metadata. Device values
        # are only gathered later using these immutable per-round indices.
        n = batch.num_reqs
        qsl = batch.query_start_loc_np[:n + 1]
        phases = batch.is_prefilling_np[:n]
        uploads = []

        def build(prefilling):
            request_ids = [i for i in range(n) if bool(phases[i]) == prefilling]
            token_ids, boundaries, omitted = [], [0], []
            for i in request_ids:
                begin, end = int(qsl[i]), int(qsl[i + 1])
                length = end - begin
                if prefilling:
                    remaining = int(batch.prefill_len_np[i] - batch.num_computed_prefill_tokens_np[i])
                    length = min(length, remaining)
                if length <= 0:
                    raise RuntimeError("Mixed tree phase contains an empty request")
                token_ids.extend(range(begin, begin + length))
                boundaries.append(boundaries[-1] + length)
                omitted.append(end - begin - length)
            req, tok, cu, drop = _upload_metadata(
                device,
                (request_ids, torch.long, (len(request_ids),)),
                (token_ids, torch.long, (len(token_ids),)),
                (boundaries, torch.int32, (len(boundaries),)),
                (omitted, torch.int32, (len(omitted),)),
                host_buffers=uploads,
            )
            return TreePhase(
                req, tok, cu,
                torch.tensor(request_ids, dtype=torch.long, device="cpu"),
                torch.tensor(token_ids, dtype=torch.long, device="cpu"),
                torch.tensor(boundaries, dtype=torch.int32, device="cpu"),
                drop, torch.tensor(omitted, dtype=torch.int32, device="cpu"), prefilling,
            )

        decode, prefill = build(False), build(True)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask.index_fill_(0, decode.request_indices, True)
        return cls(decode, prefill, mask, uploads)
