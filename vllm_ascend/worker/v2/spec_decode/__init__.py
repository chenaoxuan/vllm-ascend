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
from vllm.logger import init_logger

from vllm_ascend.utils import vllm_version_is

logger = init_logger("vllm." + __name__)


def _agent_dbg(location, message, data, hypothesis_id, limit=400):
    try:
        import importlib.util
        import sys

        mod = sys.modules.get("_agent_debug_trace")
        if mod is None:
            spec = importlib.util.spec_from_file_location(
                "_agent_debug_trace",
                "/home/specdec/spec260922/debug_trace.py",
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_agent_debug_trace"] = mod
            spec.loader.exec_module(mod)
        mod.dbg(location, message, data, hypothesis_id, limit=limit)
    except Exception:
        pass


def init_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
):
    """Override GPU init_speculator for Ascend NPUs.

    DFlash (``priority`` / ``prefix``) and DSpark (``beam``, Qwen3,
    DeepSeek-V4 ``DSparkDraftModel``, or ``DeepSeekV4MTPModel``) use
    ``AscendTreeSpeculator`` when ``tree_spec_config.enabled`` is true.
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

            # #region agent log
            _agent_dbg(
                "spec_decode/__init__.py:init_speculator",
                "choose_speculator",
                {
                    "cls": "AscendTreeSpeculator",
                    "backend": "dspark",
                    "method": speculative_config.method,
                    "n_spec": speculative_config.num_speculative_tokens,
                },
                "H1",
            )
            # #endregion
            return AscendTreeSpeculator(vllm_config, device)
        if dflash_tree_spec_enabled(vllm_config):
            draft = getattr(speculative_config, "draft_model_config", None)
            # #region agent log
            _agent_dbg(
                "spec_decode/__init__.py:init_speculator",
                "choose_speculator",
                {
                    "cls": "AscendDSparkSpeculator",
                    "reason": "tree_enabled_but_not_tree_dspark",
                    "method": speculative_config.method,
                    "arches": list(_config_arches(draft)),
                },
                "H1",
            )
            # #endregion
            logger.warning(
                "tree_spec_config.enabled but using plain DSpark speculator "
                "(draft arches=%s model_type=%s)",
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

            # #region agent log
            _agent_dbg(
                "spec_decode/__init__.py:init_speculator",
                "choose_speculator",
                {
                    "cls": "AscendTreeSpeculator",
                    "backend": "dflash",
                    "method": speculative_config.method,
                    "n_spec": speculative_config.num_speculative_tokens,
                },
                "H1",
            )
            # #endregion
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
    return model_type in ("deepseek_v4", "deepseek_mtp") or model_type.startswith(
        "deepseek_v4"
    )


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


def tree_target_query_len(vllm_config: VllmConfig | None = None) -> int | None:
    """Scheduler / MC2 / dummy-run query width per request.

    Qwen3 packed tree and DSV4 ``topk>1`` are ``1+budget`` (must not inflate
    MC2 profile dummy past the 512-token cap). DSV4 ``topk=1`` is a chain of
    ``1+spec``.
    """
    if not dflash_tree_spec_enabled(vllm_config):
        return None
    spec = None
    if vllm_config is not None:
        spec = vllm_config.speculative_config
    from vllm_ascend.ascend_config import get_ascend_config

    tree_cfg = get_ascend_config().tree_spec_config
    if dsv4_dspark_draft(vllm_config, spec) and int(tree_cfg.topk or 0) <= 1:
        n = int(getattr(spec, "num_speculative_tokens", 0) or 0)
        return 1 + n if n > 0 else None
    return 1 + int(tree_cfg.budget)


def dflash_tree_spec_enabled(vllm_config: VllmConfig=None) -> bool:
    from vllm_ascend.attention.tree_spec import tree_spec_enabled

    return tree_spec_enabled(vllm_config)
