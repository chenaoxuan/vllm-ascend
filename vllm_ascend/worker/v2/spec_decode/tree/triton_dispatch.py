import os

import torch
from vllm.triton_utils import HAS_TRITON

# Names accepted by ``tree_spec_config.triton_ops``.
TREE_TRITON_OPS: tuple[str, ...] = (
    "attention_mask",
    "finalize_layout",
    "greedy_reject",
    "kv_compact",
)


def _env_triton_disabled() -> bool:
    return os.getenv("VLLM_ASCEND_TREE_SPEC_TRITON", "1").lower() in (
        "0",
        "false",
        "no",
    )


def _device_ok(device: torch.device | str | None) -> bool:
    if device is None:
        return True
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    return dev.type == "npu"


def _configured_triton_ops() -> list[str] | None:
    """Return whitelist, or ``None`` when unset (all ops allowed)."""
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return get_ascend_config().tree_spec_config.triton_ops
    except RuntimeError:
        return None


def tree_triton_base_enabled(device: torch.device | str | None = None) -> bool:
    """Env + HAS_TRITON + device gate (ignores per-op whitelist)."""
    if _env_triton_disabled():
        return False
    if not HAS_TRITON:
        return False
    return _device_ok(device)


def use_tree_triton(
    op: str,
    device: torch.device | str | None = None,
) -> bool:
    """Whether one tree-spec hot path should call its NPU Triton kernel.

    ``tree_spec_config.triton_ops``:
    - ``None`` (default): all ops on when base gate passes (legacy).
    - ``[]``: all torch.
    - non-empty: only listed op names use Triton.
    """
    if not tree_triton_base_enabled(device):
        return False
    ops = _configured_triton_ops()
    if ops is None:
        return True
    return op in ops
