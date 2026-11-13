from typing import Any

import torch
from vllm.config import VllmConfig

from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def align_up(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def tree_spec_enabled(vllm_config: VllmConfig | None = None) -> bool:
    """True when ``additional_config.tree_spec_config.enabled`` is set."""
    if vllm_config is not None:
        tree_config = (getattr(vllm_config, "additional_config", None) or {}).get("tree_spec_config") or {}
        return isinstance(tree_config, dict) and bool(tree_config.get("enabled", False))
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().tree_spec_config.enabled)
    except RuntimeError:
        return False


def tree_query_len() -> int | None:
    """Storage/visibility capacity B+1, independent of scheduled target rows.

    Target capture and dispatch use TreeShape.target_query_len. This helper
    sizes root-inclusive visibility storage, including for a truncated tree.
    """
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        tree_cfg = get_ascend_config().tree_spec_config
        if not tree_cfg.enabled:
            return None
        return 1 + int(tree_cfg.budget)
    except (RuntimeError, AssertionError):
        return None


def tree_decode_threshold(vllm_config: VllmConfig | None = None) -> int:
    """Target decode capacity including the root token."""
    if vllm_config is not None:
        from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape

        shape = TreeShape.from_config(vllm_config)
        if shape is not None:
            return shape.target_query_len
        spec = vllm_config.speculative_config
        return 1 + int(spec.num_speculative_tokens) if spec is not None else 1
    query_len = tree_query_len()
    return query_len if query_len is not None else 1


def tree_treat_short_extends_as_decodes(
    common_attn_metadata,
    *,
    tree_enabled: bool,
    extra_disable: bool = False,
) -> bool:
    """Short prefills stay prefills when tree target metadata is available.

    Draft propose does not populate ``is_prefilling``; keep the default
    short-extend-as-decode path there so split does not require the flag.
    """
    if extra_disable:
        return False
    if not tree_enabled:
        return True
    return getattr(common_attn_metadata, "is_prefilling", None) is None


def dummy_tree_visibility(
    num_decode: int,
    device: torch.device,
    budget: int | None = None,
) -> torch.Tensor:
    """Identity visibility for FULL capture when the dummy batch has no tree."""
    if budget is None:
        query_len = tree_query_len()
        budget = query_len - 1 if query_len is not None else 1
    vis = torch.eye(budget, dtype=torch.bool, device=device)
    return vis.unsqueeze(0).expand(num_decode, -1, -1).contiguous()


def tree_verify_shape(num_tokens: int) -> tuple[int, int] | None:
    """``(num_decode, 1+budget)`` when the batch is tree-verify shaped."""
    tree_q = tree_query_len()
    if tree_q is None or tree_q <= 0 or num_tokens < tree_q or num_tokens % tree_q != 0:
        return None
    n_dec = num_tokens // tree_q
    if n_dec <= 0:
        return None
    return n_dec, tree_q


def dummy_tree_mask_for_capture(num_tokens: int, num_reqs: int) -> bool:
    """Dummy FULL capture only needs the tree layout for ``k × (1+budget)``."""
    shaped = tree_verify_shape(num_tokens)
    return shaped is not None and shaped[0] == num_reqs


def seq_lens_are_tree_query(seq_lens: torch.Tensor | None, num_decode: int) -> bool:
    """Dummy FULL capture sets seq_len == query_len; tree verify is 1+budget."""
    tree_q = tree_query_len()
    if tree_q is None or seq_lens is None or num_decode <= 0:
        return False
    sl = seq_lens[:num_decode]
    if sl.numel() != num_decode or sl.device.type != "cpu":
        return False
    return all(v == tree_q for v in sl.tolist())


def is_draft_forward() -> bool:
    """True only inside a draft forward that already has a forward context.

    Graph ``prepare_attn`` runs before ``set_forward_context``; treat that as
    target so dummy tree visibility can still be injected.
    """
    from vllm.forward_context import is_forward_context_available

    return is_forward_context_available() and bool(_EXTRA_CTX.is_draft_model)


def need_dummy_tree_visibility_for_capture() -> bool:
    """Target FULL capture only; draft decode must keep its own mask/indices."""
    from vllm.forward_context import is_forward_context_available

    if not is_forward_context_available():
        return False
    if not _EXTRA_CTX.capturing or is_draft_forward():
        return False
    return tree_query_len() is not None


def is_target_tree_step(common_attn_metadata) -> bool:
    """True when the current target forward should apply tree visibility."""
    if not tree_spec_enabled() or is_draft_forward():
        return False
    if common_attn_metadata.tree_visibility is not None:
        return True
    return need_dummy_tree_visibility_for_capture()


class TreeSpecAttnAdapter:
    """Target-side tree metadata. Kernel dispatch stays in each backend."""

    def configure_builder(self, builder, vllm_config: VllmConfig, device: torch.device) -> None:
        return

    def build_target_inputs(self, builder, common_attn_metadata, *, num_decodes: int) -> Any:
        return None


class GqaTreeSpecAdapter(TreeSpecAttnAdapter):
    """FIA 4D/3D custom mask is filled later in AttentionMaskBuilder."""

    def configure_builder(self, builder, vllm_config: VllmConfig, device: torch.device) -> None:
        query_len = tree_decode_threshold(vllm_config)
        builder.attn_mask_builder.configure_tree_mask(
            max_num_decode=int(vllm_config.scheduler_config.max_num_seqs),
            query_len=query_len,
            kv_len=align_up(int(vllm_config.model_config.max_model_len), 128),
        )
