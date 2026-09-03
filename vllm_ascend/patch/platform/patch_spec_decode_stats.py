from vllm.v1.spec_decode.metrics import SpecDecodingStats


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
