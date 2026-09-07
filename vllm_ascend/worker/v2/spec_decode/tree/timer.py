import atexit
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch

TREE_TIMER_SEGMENTS: tuple[str, ...] = (
    "build_draft_tree",
    "build_attn_mask",
    "target_fia_forward",
    "rejection_sample",
    "compact_kv_path",
    "compact_query_path",
    "draft_model_forward",
)

_ENABLED = False
_BACKEND = "torch"
_WARMUP_STEPS = 2
_STEP = 0
_RECORDING = False
_SAMPLES: dict[str, list[float]] = defaultdict(list)
_META: dict[str, object] = {}
_REGISTERED = False


def configure_tree_timer(
    *,
    enabled: bool = False,
    backend: str = "torch",
    warmup_steps: int = 2,
    meta: dict[str, object] | None = None,
) -> None:
    """Enable/disable the tree-spec segment timer.

    ``backend`` is printed as ``torch`` or ``triton`` (implementation path).
    Segments always fence with ``torch.npu.synchronize`` so samples include
    host wall-clock and device compute wait.
    """
    global _ENABLED, _BACKEND, _WARMUP_STEPS, _REGISTERED
    _ENABLED = bool(enabled)
    _BACKEND = backend
    _WARMUP_STEPS = max(0, int(warmup_steps))
    if meta:
        _META.update(meta)
    if _ENABLED and not _REGISTERED:
        atexit.register(print_tree_timer_report)
        _REGISTERED = True


def tree_timer_enabled() -> bool:
    return _ENABLED


def tree_timer_begin_step() -> None:
    """Mark the start of one decode step (verify + propose)."""
    global _STEP, _RECORDING
    if not tree_timer_enabled():
        return
    _STEP += 1
    _RECORDING = _STEP > _WARMUP_STEPS


def _sync() -> None:
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
    """Time one segment with device synchronize; no-op when disabled."""
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
    if not _ENABLED:
        return
    lines = [
        "========== tree-spec timer report ==========",
        f"backend={_BACKEND} sync=device steps={_STEP} warmup={_WARMUP_STEPS}",
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
