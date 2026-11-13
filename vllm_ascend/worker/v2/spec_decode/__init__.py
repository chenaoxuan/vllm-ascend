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
import copy
from typing import TYPE_CHECKING

import torch
from vllm.config import CompilationMode, VllmConfig

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.states import RequestState


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    req_states: "RequestState | None" = None,
):
    """Override GPU init_speculator for Ascend NPUs.

    DFlash and Qwen3 DSpark use
    ``AscendTreeSpeculator`` when ``tree_spec_config.enabled`` is true.
    The tree host selects its scorer from the loaded draft model.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.enforce_eager:
        vllm_config = copy.copy(vllm_config)
        vllm_config.compilation_config = copy.copy(vllm_config.compilation_config)
        vllm_config.compilation_config.mode = CompilationMode.NONE
    if speculative_config.method in ("ngram", "ngram_gpu"):
        if req_states is None:
            raise ValueError("MRV2 n-gram speculative decoding requires request states")
        # Older vLLM pins lack this module; MRV2 n-gram requires upstream #40704.
        from vllm.v1.worker.gpu.spec_decode.ngram.speculator import (  # type: ignore[import-not-found]
            NgramGPUSpeculator,
        )

        # MRV2 postprocess has already updated history. The MRV1 NPU kernel
        # appends sampled tokens itself and cannot be used here unchanged.
        return NgramGPUSpeculator(vllm_config, device, req_states)
    if speculative_config.method == "extract_hidden_states":
        # No Ascend-specific behavior beyond update_stream assignment in
        # NPUModelRunner; reuse upstream ExtractHiddenStatesSpeculator as-is.
        from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
            ExtractHiddenStatesSpeculator,
        )

        return ExtractHiddenStatesSpeculator(vllm_config, device)
    if speculative_config.use_dspark():
        if dflash_tree_spec_enabled(vllm_config) and _is_tree_dspark(speculative_config):
            from vllm_ascend.worker.v2.spec_decode.tree.speculator import (
                AscendTreeSpeculator,
            )

            return AscendTreeSpeculator(vllm_config, device)
        if dflash_tree_spec_enabled(vllm_config):
            raise ValueError("Tree DSpark supports only Qwen3 and Qwen3Omni draft models")
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
        if "LiLiCorrDraftModel" in speculative_config.draft_model_config.architectures:
            from vllm_ascend.worker.v2.spec_decode.lilicorr.speculator import (
                AscendLiLiCorrSpeculator,
            )

            return AscendLiLiCorrSpeculator(vllm_config, device)
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
                AscendDFlash2Speculator,
            )

            return AscendDFlash2Speculator(vllm_config, device)
        from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
            AscendDFlashSpeculator,
        )

        return AscendDFlashSpeculator(vllm_config, device)
    if speculative_config.use_gemma4_mtp():
        from vllm_ascend.worker.v2.spec_decode.gemma4.speculator import (
            AscendGemma4Speculator,
        )

        return AscendGemma4Speculator(vllm_config, device)
    if speculative_config.method == "mtp" and not speculative_config.use_step3p5_mtp():
        from vllm_ascend.worker.v2.spec_decode.mtp.speculator import (
            AscendMTPSpeculator,
        )

        return AscendMTPSpeculator(vllm_config, device)
    if speculative_config.use_multi_module_mtp():
        from vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator import AscendMultiModuleMTPSpeculator

        return AscendMultiModuleMTPSpeculator(vllm_config, device)
    if speculative_config.use_eagle():
        from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

        return AscendEagleSpeculator(vllm_config, device)
    if speculative_config.uses_draft_model():
        from vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator import (
            AscendStandaloneARSpeculator,
        )

        return AscendStandaloneARSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")


def _config_arches(config) -> tuple:
    if config is None:
        return ()
    arches = tuple(getattr(config, "architectures", None) or ())
    hf = getattr(config, "hf_config", None)
    hf_arches = tuple(getattr(hf, "architectures", None) or ()) if hf is not None else ()
    if not hf_arches:
        return arches
    if not arches:
        return hf_arches
    return arches + tuple(a for a in hf_arches if a not in arches)


def _is_tree_dspark(speculative_config) -> bool:
    """True when a DSpark drafter should host tree spec (beam / prefix / ...)."""
    draft = getattr(speculative_config, "draft_model_config", None)
    arches = _config_arches(draft)
    return any(a in arches for a in ("Qwen3DSparkModel", "Qwen3OmniDSparkModel"))


def tree_target_query_len(vllm_config: VllmConfig) -> int | None:
    """Target capacity from the same host shape contract used by tree proposals."""
    from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape

    shape = TreeShape.from_config(vllm_config)
    return shape.target_query_len if shape is not None else None


def dflash_tree_spec_enabled(vllm_config: VllmConfig | None = None) -> bool:
    from vllm_ascend.attention.tree_spec import tree_spec_enabled

    return tree_spec_enabled(vllm_config)
