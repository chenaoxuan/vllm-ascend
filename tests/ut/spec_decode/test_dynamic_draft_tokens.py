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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_tokens_handler import AscendDraftTokensHandler


def _make_input_batch(req_ids: list[str], structured: bool = False) -> SimpleNamespace:
    return SimpleNamespace(req_ids=req_ids, has_structured_output_reqs=structured)


@pytest.fixture
def handler_cpu(monkeypatch):
    """Build handler without requiring a real NPU runtime."""
    with (
        patch("torch.npu.Stream", return_value=MagicMock()),
        patch("torch.npu.Event", return_value=MagicMock()),
    ):
        h = AscendDraftTokensHandler(device=torch.device("cpu"))
    h.copy_stream = MagicMock()
    h.copy_event = MagicMock()
    return h


class TestAscendDraftTokensHandler:
    def test_disabled_matches_fixed_placeholder_length(self, handler_cpu, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.worker.v2.spec_decode.draft_tokens_handler.envs.VLLM_ASCEND_DYNAMIC_DRAFT_TOKENS",
            False,
        )
        handler = handler_cpu
        handler._dynamic_enabled = False
        draft = torch.arange(6, dtype=torch.int64).view(2, 3)
        handler.set_draft_tokens(_make_input_batch(["r0", "r1"]), draft)
        out = handler.get_draft_tokens()
        assert out is not None
        assert out.req_ids == ["r0", "r1"]
        assert out.draft_token_ids == [[-1, -1, -1], [-1, -1, -1]]

    def test_enabled_returns_truncated_prefix(self, handler_cpu, monkeypatch):
        monkeypatch.setattr(
            "vllm_ascend.worker.v2.spec_decode.draft_tokens_handler.envs.VLLM_ASCEND_DYNAMIC_DRAFT_TOKENS",
            True,
        )
        handler = handler_cpu
        handler._dynamic_enabled = True

        draft = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64)
        cpu_buf = draft.detach().cpu().contiguous()

        def _fake_copy(dst, src, non_blocking=False):
            dst.copy_(src)

        with (
            patch(
                "vllm_ascend.worker.v2.spec_decode.draft_tokens_handler.torch.npu.current_stream",
                return_value=MagicMock(),
            ),
            patch(
                "vllm_ascend.worker.v2.spec_decode.draft_tokens_handler.torch.npu.stream",
                return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()),
            ),
            patch("random.randint", side_effect=[1, 3]),
            patch.object(handler, "_ensure_cpu_buffer", return_value=cpu_buf),
            patch.object(cpu_buf, "copy_", side_effect=lambda *a, **k: None),
        ):
            # Seed CPU buffer with draft values so materialize sees them.
            handler._draft_cpu = cpu_buf
            handler.set_draft_tokens(_make_input_batch(["r0", "r1"]), draft)
            # Bypass async copy; pretend copy already landed.
            handler._copy_pending = False
            handler.draft_tokens_np = cpu_buf.numpy()
            out = handler.get_draft_tokens()

        assert out is not None
        assert out.draft_token_ids == [[10], [20, 21, 22]]
        assert all(1 <= len(row) <= 3 for row in out.draft_token_ids)


@pytest.mark.parametrize("length", [1, 2, 3])
def test_truncation_preserves_prefix(length, handler_cpu, monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.draft_tokens_handler.envs.VLLM_ASCEND_DYNAMIC_DRAFT_TOKENS",
        True,
    )
    handler = handler_cpu
    handler._dynamic_enabled = True
    handler.req_ids = ["r0"]
    handler._draft_lens = [length]
    handler.num_draft_tokens = 3
    handler.draft_tokens_np = np.array([[7, 8, 9]], dtype=np.int64)
    handler._copy_pending = False

    out = handler.get_draft_tokens()
    assert out is not None
    assert out.draft_token_ids == [[7, 8, 9][:length]]
