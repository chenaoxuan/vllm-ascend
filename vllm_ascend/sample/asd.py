# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from Mterzzl/vllm-ascend-asd at 3d47910.

import ast
import struct
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from vllm.config import SpeculativeConfig
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

from vllm_ascend import envs
from vllm_ascend.ops.triton.spec_decode.asd import select_asd_prefix


def validate_asd_config(config: SpeculativeConfig) -> None:
    if config.rejection_sample_method not in ("strict", "standard"):
        raise ValueError("ASD requires greedy verification")
    tree = getattr(config, "speculative_token_tree", None)
    if tree is not None:
        try:
            paths = ast.literal_eval(tree) if isinstance(tree, str) else tree
            paths = sorted((tuple(path) for path in paths), key=len)
            if not paths or any(path != paths[-1][:depth] or len(path) != depth for depth, path in enumerate(paths, 1)):
                raise ValueError
        except (SyntaxError, TypeError, ValueError) as exc:
            raise ValueError("ASD requires a single contiguous draft chain") from exc


class ASDVerifier:
    """Keep request identity on the host and cumulative regret on the device."""

    def __init__(self, capacity: int, device: torch.device, vocab_size: int):
        if not HAS_TRITON:
            raise ValueError("ASD requires Triton-Ascend")
        self.budget = envs.VLLM_ASCEND_ASD_RISK_BUDGET
        self.gate = envs.VLLM_ASCEND_ASD_MAX_REGRET_PER_VALUE
        self.mismatch_cap = envs.VLLM_ASCEND_ASD_MAX_RELAXED_TOKENS_PER_BLOCK
        for name, value in (("risk budget", self.budget), ("regret gate", self.gate)):
            if not 0 <= value <= torch.finfo(torch.float32).max:
                raise ValueError(f"ASD {name} must be finite and nonnegative in FP32")
            if value > 0 and struct.unpack("f", struct.pack("f", value))[0] == 0:
                raise ValueError(f"ASD {name} underflows in FP32")
        if self.mismatch_cap < 0:
            raise ValueError("ASD mismatch cap must be nonnegative")
        self.vocab_size = vocab_size
        self.spent = torch.empty(max(1, capacity), dtype=torch.float32, device=device)
        self.slots: dict[str, int] = {}
        self.free_slots = list(reversed(range(self.spent.numel())))
        logger.info("ASD enabled (budget=%s, gate=%s, cap=%s)", self.budget, self.gate, self.mismatch_cap)

    def drop_finished(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            slot = self.slots.pop(request_id, None)
            if slot is not None:
                self.free_slots.append(slot)

    def _prepare(self, request_ids: Sequence[str], discarded_rows: Iterable[int]) -> torch.Tensor:
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("ASD requires unique request IDs in each batch")
        discarded = set(discarded_rows)
        rows = []
        for row, request_id in enumerate(request_ids):
            fresh = request_id not in self.slots
            if fresh:
                if not self.free_slots:
                    size = self.spent.numel()
                    self.spent = torch.cat((self.spent, self.spent.new_empty((size,))))
                    self.free_slots.extend(reversed(range(size, 2 * size)))
                self.slots[request_id] = self.free_slots.pop()
            rows.append((self.slots[request_id], fresh, row not in discarded))
        metadata = np.asarray(rows, dtype=np.int32).reshape(-1, 3)
        return async_copy_to_gpu(metadata, device=self.spent.device)

    def verify(
        self,
        metadata: SpecDecodeMetadata,
        target_logits: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        request_ids: Sequence[str],
        discarded_rows: Iterable[int],
    ) -> torch.Tensor:
        if target_logits.size(1) < self.vocab_size:
            raise ValueError("ASD requires full-vocabulary target logits")
        if len(request_ids) != metadata.cu_num_draft_tokens.numel():
            raise ValueError("ASD request and speculative block counts differ")
        return select_asd_prefix(
            metadata.draft_token_ids,
            target_logits,
            bonus_token_ids,
            metadata.cu_num_draft_tokens,
            self._prepare(request_ids, discarded_rows),
            self.spent,
            metadata.max_spec_len,
            budget=self.budget,
            gate=self.gate,
            mismatch_cap=self.mismatch_cap,
        )
