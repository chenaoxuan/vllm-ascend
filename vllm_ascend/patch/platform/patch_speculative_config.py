from transformers import DeepseekV2Config, PretrainedConfig
from vllm.config.speculative import SpeculativeConfig

_orig_post_init = SpeculativeConfig.__post_init__
_orig_hf_config_override = SpeculativeConfig.hf_config_override


# Transformers 5.14 inherited a hidden_size % num_heads check from Llama in
# DeepseekV2Config. K3 MLA has independent projection/head dimensions (e.g.
# hidden_size=7168, num_heads=96), so that MHA constraint does not apply.
# strict stores unbound validators; patch that entry, not all config validation.
if hasattr(DeepseekV2Config, "__class_validators__"):
    _orig_validate_architecture = DeepseekV2Config.validate_architecture

    def _validate_dspark_architecture(config):
        if config.model_type != "k3_dspark":
            _orig_validate_architecture(config)

    DeepseekV2Config.__class_validators__ = [
        _validate_dspark_architecture if validator is _orig_validate_architecture else validator
        for validator in DeepseekV2Config.__class_validators__
    ]


def _normalize_legacy_qwen3_dspark_config(hf_config: PretrainedConfig) -> PretrainedConfig:
    hf_config = _orig_hf_config_override(hf_config)
    architectures = hf_config.architectures or ()
    if hf_config.model_type == "qwen3" and "DSparkDraftModel" in architectures:
        dflash_config = hf_config.dflash_config
        hf_config.update(
            {
                "architectures": ["Qwen3DSparkModel"],
                "mask_token_id": dflash_config["mask_token_id"],
                "target_layer_ids": dflash_config["target_layer_ids"],
            }
        )
    return hf_config


def _draft_path_looks_like_dflash(speculative_config) -> bool:
    model = getattr(speculative_config, "model", None)
    if not isinstance(model, str):
        return False
    name = model.lower()
    return "domino" in name or "dflash" in name


def _hf_looks_like_dflash(hf_config) -> bool:
    if hf_config is None:
        return False
    arches = getattr(hf_config, "architectures", None) or ()
    if any(str(arch).startswith("DFlash") for arch in arches):
        return True
    dflash_cfg = getattr(hf_config, "dflash_config", None)
    if isinstance(dflash_cfg, dict):
        return dflash_cfg.get("projector_type") == "domino"
    return getattr(dflash_cfg, "projector_type", None) == "domino"


def _promote_dflash_draft(self) -> None:
    """Turn a leftover ``draft_model`` method into DFlash after config load.

    Upstream only auto-detects DFlash when the repo name contains ``dflash``.
    Domino checkpoints are ``DFlashDraftModel`` / ``projector_type=domino``
    (e.g. Huang2020/Qwen3-8B-Domino-b16) and would otherwise stay draft_model.
    """
    if self.method != "draft_model":
        return
    draft = getattr(self, "draft_model_config", None)
    hf = getattr(draft, "hf_config", None) if draft is not None else None
    draft_arches = getattr(draft, "architectures", None) or () if draft is not None else ()
    if not _hf_looks_like_dflash(hf) and not any(
        str(arch).startswith("DFlash") for arch in draft_arches
    ):
        return
    self.method = "dflash"
    from vllm.transformers_utils.configs.eagle import EAGLEConfig
    from vllm.transformers_utils.configs.speculators import SpeculatorsConfig

    if hf is not None and not isinstance(hf, (EAGLEConfig, SpeculatorsConfig)):
        self.draft_model_config.hf_config = EAGLEConfig(
            hf, method="dflash", model_type="eagle"
        )
        self.update_arch_()
    self.parallel_drafting = True


def _dspark_post_init(self):
    # Upstream treats an unset method as draft_model before it can see the
    # checkpoint architecture. Name-based DFlash/Domino must be set first so
    # EAGLEConfig wrapping still runs inside the original post_init.
    if self.method is None and _draft_path_looks_like_dflash(self):
        self.method = "dflash"
    _orig_post_init(self)
    _promote_dflash_draft(self)
    if self.use_dspark():
        draft_model_config = getattr(self, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        # deepseek v4 dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "dspark_noise_token_id", None)  # type: ignore
        # gqa backend dspark
        if getattr(draft_hf_config, "ptd_token_id", None) is None:  # type: ignore
            draft_hf_config.ptd_token_id = getattr(draft_hf_config, "mask_token_id", None)  # type: ignore


SpeculativeConfig.hf_config_override = staticmethod(_normalize_legacy_qwen3_dspark_config)
SpeculativeConfig.__post_init__ = _dspark_post_init
