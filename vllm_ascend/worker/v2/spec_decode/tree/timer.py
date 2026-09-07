import atexit
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch

TREE_TIMER_SEGMENTS: tuple[str, ...] = (
    "mask",
    "target_fia",
    "greedy_tree_reject",
    "kv_query_compact",
    "draft_forward",
    "prefix_tree_builder",
    "finalize",
)

_ENABLED = False
_BACKEND = "torch"
_WARMUP_STEPS = 2
_STEP = 0
_RECORDING = False
_SAMPLES: dict[str, list[float]] = defaultdict(list)
_META: dict[str, object] = {}
_REGISTERED = False


def _env_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_TREE_SPEC_TIMER", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def configure_tree_timer(
    *,
    enabled: bool | None = None,
    backend: str = "torch",
    warmup_steps: int = 2,
    meta: dict[str, object] | None = None,
) -> None:
    """Enable/disable the tree-spec segment timer.

    ``enabled=None`` keeps current state or turns on via
    ``VLLM_ASCEND_TREE_SPEC_TIMER``. ``backend`` is printed as ``torch`` or
    ``triton``; triton skips ``torch.npu.synchronize`` (Ascend segfault).
    """
    global _ENABLED, _BACKEND, _WARMUP_STEPS, _REGISTERED
    if enabled is None:
        enabled = _ENABLED or _env_enabled()
    else:
        enabled = bool(enabled) or _env_enabled()
    _ENABLED = bool(enabled)
    _BACKEND = backend
    _WARMUP_STEPS = max(0, int(warmup_steps))
    if meta:
        _META.update(meta)
    if _ENABLED and not _REGISTERED:
        atexit.register(print_tree_timer_report)
        _REGISTERED = True


def set_tree_timer_backend(backend: str) -> None:
    """Update printed backend label (``torch`` / ``triton``)."""
    global _BACKEND
    _BACKEND = backend


def tree_timer_enabled() -> bool:
    return _ENABLED or _env_enabled()


def tree_timer_begin_step() -> None:
    """Mark the start of one decode step (verify + propose)."""
    global _STEP, _RECORDING
    if not tree_timer_enabled():
        return
    _STEP += 1
    _RECORDING = _STEP > _WARMUP_STEPS


def _device_sync_safe() -> bool:
    """Full ``torch.npu.synchronize`` after Ascend Triton launches can segfault.

    Torch/eager segments still fence; triton segments use host wall-clock only.
    """
    return _BACKEND != "triton"


def _sync() -> None:
    if not _device_sync_safe():
        return
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        try:
            torch.npu.synchronize()
            return
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def tree_time(segment: str) -> Iterator[None]:
    """Time one segment; device fence only for non-triton backend."""
    if not tree_timer_enabled():
        yield
        return
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        if _RECORDING:
            _SAMPLES[segment].append((time.perf_counter() - t0) * 1000.0)


def print_tree_timer_report() -> None:
    """Print aggregated segment timings to stdout (no file)."""
    if not (_ENABLED or _env_enabled()):
        return
    sync_mode = "device" if _device_sync_safe() else "host_only"
    lines = [
        "========== tree-spec timer report ==========",
        f"backend={_BACKEND} sync={sync_mode} steps={_STEP} warmup={_WARMUP_STEPS}",
    ]
    if _META:
        meta_str = " ".join(f"{k}={v}" for k, v in sorted(_META.items()))
        lines.append(f"config: {meta_str}")
    any_data = False
    for name in TREE_TIMER_SEGMENTS:
        vals = list(_SAMPLES.get(name, []))
        if not vals:
            lines.append(f"  {name}: n=0")
            continue
        any_data = True
        n = len(vals)
        mean = sum(vals) / n
        lines.append(
            f"  {name}: n={n} mean_ms={mean:.3f} "
            f"min_ms={min(vals):.3f} max_ms={max(vals):.3f}"
        )
    if not any_data:
        lines.append("  (no samples after warmup)")
    lines.append("===========================================")
    print("\n".join(lines), flush=True)
