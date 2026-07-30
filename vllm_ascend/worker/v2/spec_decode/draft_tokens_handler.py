#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

"""Draft token hand-off with optional per-request length truncation.

When ``VLLM_ASCEND_DYNAMIC_DRAFT_TOKENS`` is enabled, each request keeps a
random prefix of its draft tokens (length in ``[1, N]``). Truncated lists are
published to the scheduler via ``update_draft_token_ids``, so unused draft
positions are not scheduled into the target forward.

Intended for ``cudagraph_mode=FULL`` (mixed batches). ``FULL_DECODE_ONLY`` is
not adapted: per-request variable lengths break uniform decode matching.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

from vllm_ascend import envs


class AscendDraftTokensHandler(DraftTokensHandler):
    """Ascend draft-token hand-off; uses ``torch.npu`` streams/events only."""

    def __init__(self, device: torch.device | None = None):
        # Do not call DraftTokensHandler.__init__: it constructs torch.cuda.Stream/Event.
        self.device = device
        self.copy_stream = torch.npu.Stream()
        self.copy_event = torch.npu.Event()
        self.req_ids: list[str] = []
        self.draft_tokens_np: np.ndarray | None = None
        self.num_draft_tokens: int = 0
        self._dynamic_enabled = envs.VLLM_ASCEND_DYNAMIC_DRAFT_TOKENS
        self._draft_lens: list[int] = []
        self._draft_cpu: torch.Tensor | None = None
        self._copy_pending = False

    def _ensure_cpu_buffer(self, num_reqs: int, num_cols: int) -> torch.Tensor:
        buf = self._draft_cpu
        if buf is None or buf.shape[0] < num_reqs or buf.shape[1] < num_cols:
            buf = torch.empty(
                (num_reqs, num_cols),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            self._draft_cpu = buf
        return buf

    def _copy_drafts_to_cpu(self, draft_tokens: torch.Tensor, num_reqs: int) -> None:
        # Match NPUModelRunner D2H style: pinned CPU buf + npu copy stream + Event.
        num_cols = int(draft_tokens.shape[1])
        cpu_buf = self._ensure_cpu_buffer(num_reqs, num_cols)
        current_stream = torch.npu.current_stream()
        self.copy_stream.wait_stream(current_stream)
        with torch.npu.stream(self.copy_stream):
            cpu_buf[:num_reqs, :num_cols].copy_(
                draft_tokens[:num_reqs],
                non_blocking=True,
            )
            self.copy_event.record()
        self.draft_tokens_np = None
        self._copy_pending = True

    def _materialize_np(self) -> np.ndarray:
        if self.draft_tokens_np is not None:
            return self.draft_tokens_np
        assert self._copy_pending and self._draft_cpu is not None
        self.copy_event.synchronize()
        num_reqs = len(self.req_ids)
        self.draft_tokens_np = self._draft_cpu[:num_reqs, : self.num_draft_tokens].numpy()
        self._copy_pending = False
        return self.draft_tokens_np

    def set_draft_tokens(self, input_batch: InputBatch, draft_tokens: torch.Tensor) -> None:
        self.req_ids = input_batch.req_ids
        num_reqs = len(self.req_ids)
        max_n = int(draft_tokens.shape[1])
        self.num_draft_tokens = max_n
        self._copy_pending = False
        self.draft_tokens_np = None

        if not self._dynamic_enabled:
            self._draft_lens = []
            if not input_batch.has_structured_output_reqs:
                return
            self._copy_drafts_to_cpu(draft_tokens, num_reqs)
            return

        # Placeholder policy: independent length in [1, N] per request (FULL).
        self._draft_lens = [random.randint(1, max_n) if max_n > 0 else 0 for _ in range(num_reqs)]
        self._copy_drafts_to_cpu(draft_tokens, num_reqs)

    def get_draft_tokens(self) -> DraftTokenIds | None:
        if self._dynamic_enabled and self._draft_lens:
            full = self._materialize_np()
            draft_token_ids = [
                np.asarray(full[i, :length], dtype=np.int64).tolist() for i, length in enumerate(self._draft_lens)
            ]
            return DraftTokenIds(self.req_ids, draft_token_ids)

        if self._copy_pending or self.draft_tokens_np is not None:
            return DraftTokenIds(self.req_ids, self._materialize_np().tolist())

        # Sync path without structured outputs: length-only placeholders.
        draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]
        return DraftTokenIds(self.req_ids, draft_token_ids)
