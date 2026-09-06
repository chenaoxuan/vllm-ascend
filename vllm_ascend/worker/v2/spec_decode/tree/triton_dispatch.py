import os

import torch
from vllm.triton_utils import HAS_TRITON


def use_tree_triton(device: torch.device | str | None = None) -> bool:
    """Whether tree-spec hot paths should call NPU Triton kernels."""
    if os.getenv("VLLM_ASCEND_TREE_SPEC_TRITON", "1").lower() in (
        "0",
        "false",
        "no",
    ):
        return False
    if not HAS_TRITON:
        return False
    if device is None:
        return True
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    return dev.type == "npu"
