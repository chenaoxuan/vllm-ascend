from vllm.logger import init_logger
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

import vllm_ascend.envs as envs_ascend

logger = init_logger(__name__)

ori_init_parallel_drafting_params = SpecDecodeBaseProposer._init_parallel_drafting_params


def new_init_parallel_drafting_params(self):
    spec_config = self.vllm_config.speculative_config
    model_hf_config = self.draft_model_config.hf_config
    if spec_config.method == "dspark" and hasattr(model_hf_config, "mask_token_id"):
        self.parallel_drafting_token_id = model_hf_config.mask_token_id
    else:
        ori_init_parallel_drafting_params(self)


SpecDecodeBaseProposer._init_parallel_drafting_params = new_init_parallel_drafting_params


# ---------------------------------------------------------------------------
# Confidence head wiring for the DSpark draft model.
#
# The upstream ``Qwen3DSparkForCausalLM`` defines the Markov head only and
# explicitly skips ``confidence_head.*`` weights. Since we may only modify
# vllm-ascend, we patch the model class here to (1) attach the confidence head
# submodule, (2) load its checkpoint weights (degrading gracefully to
# markov-only when the checkpoint omits them), and (3) expose
# ``compute_confidence`` for the proposer's draft loop.
# ---------------------------------------------------------------------------
try:
    from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
    from vllm.model_executor.models.utils import (
        AutoWeightsLoader,
        process_eagle_weight,
    )

    from vllm_ascend.spec_decode.dspark_confidence import build_confidence_head

    _ori_dspark_model_init = Qwen3DSparkForCausalLM.__init__

    def _patched_dspark_model_init(self, *, vllm_config, prefix: str = ""):
        _ori_dspark_model_init(self, vllm_config=vllm_config, prefix=prefix)
        if not envs_ascend.VLLM_ASCEND_DSPARK_ENABLE_CONFIDENCE:
            return
        try:
            head = build_confidence_head(self.config)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to build DSpark confidence head: %s", e)
            head = None
        if head is not None:
            # Placed under ``self.model`` so its checkpoint weight name
            # ``confidence_head.proj.*`` (prefixed with ``model.`` by
            # load_weights) resolves to this submodule.
            self.model.confidence_head = head

    def _patched_dspark_load_weights(self, weights):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        includes_draft_id_mapping = False
        includes_confidence = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            if "confidence_head" in name:
                includes_confidence = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = ["mask_embedding"]
        head = getattr(self.model, "confidence_head", None)
        if head is not None and not includes_confidence:
            # Checkpoint has no confidence head: degrade to markov-only so the
            # existing lossless behavior is preserved.
            logger.warning(
                "DSpark confidence head enabled but checkpoint has no "
                "confidence_head.* weights; disabling the confidence head "
                "(falling back to markov-only drafting)."
            )
            del self.model.confidence_head
            skip_substrs.append("confidence_head")
        elif head is None:
            skip_substrs.append("confidence_head")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()

    def _dspark_compute_confidence(self, draft_hidden, markov_embed_stack):
        """Calibrated per-position acceptance probability, or ``None``.

        ``draft_hidden``: ``[num_reqs, gamma, hidden]``
        ``markov_embed_stack``: ``[num_reqs, gamma, markov_rank]``
        returns ``[num_reqs, gamma]`` in ``[0, 1]``.
        """
        head = getattr(self.model, "confidence_head", None)
        if head is None:
            return None
        raw = head(draft_hidden, markov_embed_stack)
        return head.apply_sts(raw)

    Qwen3DSparkForCausalLM.__init__ = _patched_dspark_model_init
    Qwen3DSparkForCausalLM.load_weights = _patched_dspark_load_weights
    Qwen3DSparkForCausalLM.compute_confidence = _dspark_compute_confidence
except ImportError:
    # DSpark draft model is unavailable in this vllm build; nothing to patch.
    logger.debug("Qwen3DSparkForCausalLM not found; skip confidence head patch.")
