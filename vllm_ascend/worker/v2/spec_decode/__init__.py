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

import torch
from vllm.config import CompilationMode, VllmConfig
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
):
    """Override GPU init_speculator for Ascend NPUs.

    DFlash and DSpark (Qwen3, DeepSeek-V4 ``DSparkDraftModel``, or
    ``DeepSeekV4MTPModel``) use
    ``AscendTreeSpeculator`` when ``tree_spec_config.enabled`` is true.
    The tree host selects its scorer from the loaded draft model.
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.enforce_eager:
        vllm_config = copy.copy(vllm_config)
        vllm_config.compilation_config = copy.copy(vllm_config.compilation_config)
        vllm_config.compilation_config.mode = CompilationMode.NONE
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
            draft = getattr(speculative_config, "draft_model_config", None)
            logger.warning(
                "tree_spec_config.enabled but using plain DSpark speculator (draft arches=%s model_type=%s)",
                _config_arches(draft),
                _config_model_type(draft),
            )
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
    if speculative_config.use_eagle():
        from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

        return AscendEagleSpeculator(vllm_config, device)
    if speculative_config.uses_draft_model():
        from vllm_ascend.worker.v2.spec_decode.standalone_ar.speculator import (
            AscendStandaloneARSpeculator,
        )

        return AscendStandaloneARSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")


_DSV4_DSPARK_ARCHES = ("DSparkDraftModel", "DeepSeekV4MTPModel")
_NON_DSV4_DSPARK_ARCHES = (
    "Qwen3DSparkModel",
    "Qwen3OmniDSparkModel",
    "Gemma4DSparkModel",
    "K3DSparkModel",
)
_DSV4_TARGET_ARCHES = (
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForConditionalGeneration",
    "DeepSeekV4MTPModel",
    "DSparkDraftModel",
)


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


def _config_model_type(config) -> str:
    hf = getattr(config, "hf_config", None) if config is not None else None
    return str(getattr(hf, "model_type", None) or "")


def _is_dsv4_model_type(model_type: str) -> bool:
    return model_type in ("deepseek_v4", "deepseek_mtp") or model_type.startswith("deepseek_v4")


def _is_dsv4_config(config) -> bool:
    if config is None:
        return False
    arches = _config_arches(config)
    if any(a in _DSV4_TARGET_ARCHES or a in _DSV4_DSPARK_ARCHES for a in arches):
        return True
    return _is_dsv4_model_type(_config_model_type(config))


def _is_tree_dspark(speculative_config) -> bool:
    """True when a DSpark drafter should host tree spec (beam / prefix / ...)."""
    draft = getattr(speculative_config, "draft_model_config", None)
    arches = _config_arches(draft)
    if any(a in arches for a in ("Qwen3DSparkModel", "Qwen3OmniDSparkModel")):
        return True
    if any(a in _DSV4_DSPARK_ARCHES for a in arches):
        return True
    if _is_dsv4_model_type(_config_model_type(draft)):
        return True
    # method=dspark + DSV4 target: hf_config_override may leave the draft as
    # DeepSeekV4MTPModel / deepseek_mtp while use_dspark() is already true.
    return _is_dsv4_config(getattr(speculative_config, "target_model_config", None))


def dsv4_dspark_draft(
    vllm_config: VllmConfig | None = None,
    speculative_config=None,
) -> bool:
    """True for DeepSeek V4 DSpark (MTP-structure), not Qwen3/Gemma/K3 DSpark."""
    spec = speculative_config
    if spec is None and vllm_config is not None:
        spec = vllm_config.speculative_config
    if spec is None or not spec.use_dspark():
        return False
    draft = getattr(spec, "draft_model_config", None)
    arches = _config_arches(draft)
    if any(a in _NON_DSV4_DSPARK_ARCHES for a in arches):
        return False
    if any(a in _DSV4_DSPARK_ARCHES for a in arches):
        return True
    if _is_dsv4_model_type(_config_model_type(draft)):
        return True
    return _is_dsv4_config(getattr(spec, "target_model_config", None))


def tree_target_query_len(vllm_config: VllmConfig) -> int | None:
    """Target capacity from the same host shape contract used by tree proposals."""
    from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape

    shape = TreeShape.from_config(vllm_config)
    return shape.target_query_len if shape is not None else None


def dflash_tree_spec_enabled(vllm_config: VllmConfig | None = None) -> bool:
    from vllm_ascend.attention.tree_spec import tree_spec_enabled

    return tree_spec_enabled(vllm_config)
