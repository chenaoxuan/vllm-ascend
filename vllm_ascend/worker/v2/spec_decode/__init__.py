# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/__init__.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
import torch
from vllm.config import VllmConfig


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
):
    """Override GPU init_speculator for Ascend NPUs.

    DFlash uses ``AscendTreeSpeculator`` when
    ``additional_config.tree_spec_config.enabled`` is true.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.use_dspark():
        from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
            AscendDSparkSpeculator,
        )

        return AscendDSparkSpeculator(vllm_config, device)
    if speculative_config.use_dflash():
        if dflash_tree_spec_enabled(vllm_config):
            from vllm_ascend.worker.v2.spec_decode.tree.speculator import (
                AscendTreeSpeculator,
            )

            return AscendTreeSpeculator(vllm_config, device)
        from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
            AscendDFlashSpeculator,
        )

        return AscendDFlashSpeculator(vllm_config, device)
    if (
        speculative_config.method == "mtp"
        and not speculative_config.use_gemma4_mtp()
        and not speculative_config.use_step3p5_mtp()
    ):
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
            AscendMTPSpeculator,
        )

        return AscendMTPSpeculator(vllm_config, device)
    if speculative_config.use_eagle():
        from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

        return AscendEagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")


def dflash_tree_spec_enabled(vllm_config: VllmConfig=None) -> bool:
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().tree_spec_config.enabled)
    except RuntimeError:
        additional_config = getattr(vllm_config, "additional_config", {})
        tree_cfg = additional_config.get("tree_spec_config") or {}
        return bool(tree_cfg.get("enabled", False))


def tree_spec_budget(
    num_speculative_tokens: int | None = None,
    vllm_config: VllmConfig | None = None,
) -> int | None:
    """Draft-node budget excluding the already-accepted root.

    ``None`` means tree spec is off, or the config left budget unset and
    ``num_speculative_tokens`` was not given. When tree spec is on and budget
    is unset, pass ``num_speculative_tokens`` to get the chain-length default.
    """
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        tree_cfg = get_ascend_config().tree_spec_config
        if not tree_cfg.enabled:
            return None
        if tree_cfg.budget is not None:
            return tree_cfg.budget
        return num_speculative_tokens
    except RuntimeError:
        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        tree_cfg = additional_config.get("tree_spec_config") or {}
        if not tree_cfg.get("enabled", False):
            return None
        budget = tree_cfg.get("budget")
        if budget is not None:
            return int(budget)
        return num_speculative_tokens


def dflash_tree_decode_query_len(
    num_speculative_tokens: int,
    vllm_config: VllmConfig | None = None,
) -> int:
    """Target-verify query length: root plus packed draft nodes.

    Chain DFlash forwards ``1 + num_speculative_tokens``. Tree spec packs
    ``budget`` non-root nodes, so the query is ``1 + budget`` when budget is
    set. Using spec-depth here classifies a budget=12 verify (13 tokens) as
    prefill against decode_threshold=9 and drops the tree mask.
    """
    budget = tree_spec_budget(num_speculative_tokens, vllm_config)
    if budget is None:
        return 1 + num_speculative_tokens
    return 1 + budget
