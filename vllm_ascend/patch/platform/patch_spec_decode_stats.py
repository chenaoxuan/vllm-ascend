from functools import wraps

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode import dsv4_dspark_draft


def _observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
    """Count drafts even when num_draft_tokens exceeds num_spec_tokens.

    Tree spec can schedule ``budget`` nodes while per-position vectors stay
    length ``num_speculative_tokens`` (chain depth) for logging/Prometheus.
    Scalar totals still include every draft node.
    """
    self.num_drafts += 1
    self.num_draft_tokens += num_draft_tokens
    self.num_accepted_tokens += num_accepted_tokens
    n_acc = min(num_accepted_tokens, len(self.num_accepted_tokens_per_pos))
    n_draft = min(num_draft_tokens, len(self.num_draft_tokens_per_pos))
    for i in range(n_acc):
        self.num_accepted_tokens_per_pos[i] += 1
    for i in range(n_draft):
        self.num_draft_tokens_per_pos[i] += 1


SpecDecodingStats.observe_draft = _observe_draft


def _tree_verify_width(scheduler) -> int:
    vllm_config = scheduler.vllm_config
    tree_config = get_ascend_config().tree_spec_config
    if (
        vllm_config.model_config.enforce_eager
        and dsv4_dspark_draft(vllm_config)
        and int(tree_config.topk or 0) > 1
    ):
        return max(scheduler.num_spec_tokens, int(tree_config.budget))
    return scheduler.num_spec_tokens


def _patch_tree_verify_width(scheduler_cls) -> None:
    original_schedule = scheduler_cls.schedule
    if getattr(original_schedule, "_ascend_tree_verify_width", False):
        return

    @wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        draft_depth = self.num_spec_tokens
        verify_width = _tree_verify_width(self)
        if verify_width == draft_depth:
            return original_schedule(self, *args, **kwargs)
        try:
            self.num_spec_tokens = verify_width
            output = original_schedule(self, *args, **kwargs)
        finally:
            self.num_spec_tokens = draft_depth
        output.num_spec_tokens_to_schedule = draft_depth
        async_placeholder = getattr(self, "_spec_token_placeholders", None)
        if async_placeholder is not None and len(async_placeholder) != draft_depth:
            depth_placeholder = [-1] * draft_depth
            self._spec_token_placeholders = depth_placeholder
            for req_id in output.num_scheduled_tokens:
                request = self.requests[req_id]
                if request.spec_token_ids is async_placeholder:
                    request.spec_token_ids = depth_placeholder
        return output

    schedule._ascend_tree_verify_width = True
    scheduler_cls.schedule = schedule


_patch_tree_verify_width(Scheduler)
