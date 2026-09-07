from vllm.triton_utils import HAS_TRITON


def use_tree_triton() -> bool:
    """Whether tree-spec hot paths should call NPU Triton kernels.

    Gate: ``tree_spec_config.enable_triton`` and ``HAS_TRITON``.
    """
    if not HAS_TRITON:
        return False
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().tree_spec_config.enable_triton)
    except RuntimeError:
        return False
