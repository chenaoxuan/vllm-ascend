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

"""Publish variable-length draft tokens to the scheduler (model runner v2).

Upstream async scheduling installs fixed-length ``[-1] * N`` placeholders in
``AsyncScheduler._update_after_schedule`` and skips ``update_draft_token_ids``
in ``EngineCore.post_step``. That forces every request to schedule ``N`` draft
positions into the target forward.

When ``VLLM_ASCEND_DYNAMIC_DRAFT_TOKENS`` is on:
  * skip the fixed async placeholders so lengths come from the handler;
  * call ``update_draft_token_ids`` even under async scheduling after propose.
"""

from __future__ import annotations

from functools import wraps

from vllm_ascend import envs


def _dynamic_draft_enabled() -> bool:
    return bool(envs.VLLM_ASCEND_DYNAMIC_DRAFT_TOKENS)


def _patch_async_scheduler() -> None:
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if getattr(
        AsyncScheduler._update_after_schedule,
        "_vllm_ascend_dynamic_draft_patched",
        False,
    ):
        return

    original = AsyncScheduler._update_after_schedule

    @wraps(original)
    def _patched(self, scheduler_output):
        if not _dynamic_draft_enabled():
            return original(self, scheduler_output)

        # Run parent Scheduler._update_after_schedule only (skip AsyncScheduler's
        # fixed-length placeholder install). Lengths are set later via
        # EngineCore.post_step -> update_draft_token_ids.
        from vllm.v1.core.sched.scheduler import Scheduler

        Scheduler._update_after_schedule(self, scheduler_output)

        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue
            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = len(
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ())
            )
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Leave spec_token_ids empty; post_step fills truncated drafts.
            request.spec_token_ids = []
            if self.use_v2_model_runner:
                request.next_decode_eligible_step = self.current_step + self.pp_size

    _patched._vllm_ascend_dynamic_draft_patched = True  # type: ignore[attr-defined]
    AsyncScheduler._update_after_schedule = _patched


def _patch_engine_core_post_step() -> None:
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore.post_step, "_vllm_ascend_dynamic_draft_patched", False):
        return

    original = EngineCore.post_step

    @wraps(original)
    def _patched(self, model_executed: bool) -> None:
        if (
            _dynamic_draft_enabled()
            and getattr(self, "check_for_draft_tokens", False)
            and model_executed
        ):
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)
            return
        return original(self, model_executed)

    _patched._vllm_ascend_dynamic_draft_patched = True  # type: ignore[attr-defined]
    EngineCore.post_step = _patched


_patch_async_scheduler()
_patch_engine_core_post_step()
