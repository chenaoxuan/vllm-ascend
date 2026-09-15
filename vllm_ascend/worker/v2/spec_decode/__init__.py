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

from vllm_ascend.utils import vllm_version_is


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
):
    """Override GPU init_speculator for Ascend NPUs.

    DFlash (``priority`` / ``prefix``) and DSpark (``beam``, Qwen3 or
    DeepSeek-V4 ``DSparkDraftModel``) use ``AscendTreeSpeculator`` when
    ``tree_spec_config.enabled`` is true.
    Method/backend pairing is validated inside the tree host.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "extract_hidden_states":
        # vLLM #49811 adds this MRV2 speculator on main only. The release
        # configuration rejects this method for MRV2; keep direct calls explicit.
        if vllm_version_is("0.28.0"):
            raise NotImplementedError("extract_hidden_states is not supported by Model Runner V2 in vLLM 0.28.0.")
        # No Ascend-specific behavior beyond update_stream assignment in
        # NPUModelRunner; reuse upstream ExtractHiddenStatesSpeculator as-is.
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(vllm_config, device)
    if speculative_config.use_dspark():
        if dflash_tree_spec_enabled(vllm_config) and _is_tree_dspark(
            speculative_config
        ):
            from vllm_ascend.worker.v2.spec_decode.tree.speculator import (
                AscendTreeSpeculator,
            )

            return AscendTreeSpeculator(vllm_config, device)
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
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
                AscendDFlash2Speculator,
            )

            return AscendDFlash2Speculator(vllm_config, device)
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


def _is_tree_dspark(speculative_config) -> bool:
    draft = getattr(speculative_config, "draft_model_config", None)
    arches = getattr(draft, "architectures", None) or ()
    if "Qwen3DSparkModel" in arches or "DSparkDraftModel" in arches:
        return True
    hf = getattr(draft, "hf_config", None)
    return getattr(hf, "model_type", None) == "deepseek_v4"


def dflash_tree_spec_enabled(vllm_config: VllmConfig=None) -> bool:
    from vllm_ascend.attention.tree_spec import tree_spec_enabled

    return tree_spec_enabled(vllm_config)
