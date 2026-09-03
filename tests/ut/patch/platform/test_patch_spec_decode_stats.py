from vllm.v1.spec_decode.metrics import SpecDecodingStats

import vllm_ascend.patch.platform.patch_spec_decode_stats  # noqa: F401


def test_observe_draft_allows_tree_budget_longer_than_spec_len():
    stats = SpecDecodingStats.new(3)
    stats.observe_draft(num_draft_tokens=8, num_accepted_tokens=2)

    assert stats.num_drafts == 1
    assert stats.num_draft_tokens == 8
    assert stats.num_accepted_tokens == 2
    assert len(stats.num_accepted_tokens_per_pos) == 3
    assert len(stats.num_draft_tokens_per_pos) == 3
    assert stats.num_accepted_tokens_per_pos == [1, 1, 0]
    assert stats.num_draft_tokens_per_pos == [1, 1, 1]
