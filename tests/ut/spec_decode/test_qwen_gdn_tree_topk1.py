# SPDX-License-Identifier: Apache-2.0
"""A GDN single-branch tree keeps the accepted path and conv history."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState
from vllm_ascend.worker.v2.spec_decode.tree.builder import create_tree_builder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout
from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import TreeRejectionSampler
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    compute_tree_prefix_source,
    compute_tree_resume_column,
    tree_row_mask,
)

DEPTH = 15


def config(tree=True):
    return SimpleNamespace(
        use_v2_model_runner=True,
        additional_config={"tree_spec_config": {"enabled": tree, "budget": DEPTH, "topk": 1}},
        cache_config=SimpleNamespace(mamba_cache_mode="all"),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=DEPTH,
            enable_adaptive_verification=False,
            rejection_sample_method="standard",
        ),
    )


def ascend_config(topk=1):
    return SimpleNamespace(
        tree_spec_config=SimpleNamespace(enabled=True, budget=DEPTH, topk=topk, gdn_state_update="snapshot")
    )


def sampler_interface(device="cpu"):
    return SimpleNamespace(
        compute_nans=False,
        req_states=SimpleNamespace(
            prefill_len=SimpleNamespace(gpu=torch.tensor([10], device=device, dtype=torch.int32))
        ),
    )


def linear_tree(device="cpu"):
    return TreeLayout(
        tokens=torch.arange(11, 11 + DEPTH, device=device, dtype=torch.long)[None],
        parents=torch.arange(DEPTH, device=device, dtype=torch.long)[None],
        depths=torch.arange(1, DEPTH + 1, device=device, dtype=torch.long)[None],
        num_nodes=torch.tensor([DEPTH], device=device, dtype=torch.int32),
        visibility=torch.ones((1, DEPTH, DEPTH), device=device, dtype=torch.bool).tril(),
        first_child=torch.tensor([list(range(1, DEPTH + 1)) + [-1]], device=device, dtype=torch.int32),
        next_sibling=torch.full((1, DEPTH + 1), -1, device=device, dtype=torch.int32),
    )


def batch_and_logits(accepted=DEPTH, *, mode="decode", device="cpu"):
    tree = linear_tree(device)
    rows = DEPTH + 1 if mode == "decode" else 1
    targets = torch.full((rows,), 50, device=device, dtype=torch.long)
    if mode == "decode" and accepted:
        targets[:accepted] = tree.tokens[0, :accepted]
    logits = torch.full((rows, 64), -10.0, device=device)
    logits.scatter_(1, targets[:, None], 10.0)
    qlen = DEPTH + 1 if mode == "decode" else 24
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([0], device=device, dtype=torch.long),
        cu_num_logits=torch.tensor([0, rows], device=device, dtype=torch.int32),
        query_start_loc=torch.tensor([0, qlen], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([5 if mode == "chunked_prefill" else 32], device=device, dtype=torch.int32),
        tree_tokens=tree.tokens,
        tree_parents=tree.parents,
        tree_depths=tree.depths,
        tree_num_nodes=tree.num_nodes,
        tree_visibility=tree.visibility,
        tree_first_child=tree.first_child,
        tree_next_sibling=tree.next_sibling,
        path_node_ids=None,
        tree_acceptance=None,
    )
    return batch, logits, targets


def make_sampler(device="cpu", preserve=True):
    return TreeRejectionSampler(
        sampler_interface(device),
        config().speculative_config,
        torch.device(device),
        preserve_tree_path=preserve,
    )


def sample_case(accepted=DEPTH, *, mode="decode", device="cpu", preserve=True):
    batch, logits, targets = batch_and_logits(accepted, mode=mode, device=device)
    sampler = make_sampler(device, preserve)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler.get_ascend_config", return_value=ascend_config()
    ):
        output = sampler(logits, batch)
    return sampler, batch, output, targets


@pytest.mark.parametrize("accepted", [0, 1, 4, DEPTH])
def test_gdn_topk1_records_real_path_and_advances_cursor_and_conv_prefix(accepted):
    sampler, batch, output, targets = sample_case(accepted)
    assert sampler.acceptance is batch.tree_acceptance
    assert sampler.path_node_ids is batch.path_node_ids
    assert output.num_sampled.tolist() == [accepted + 1]
    assert output.num_rejected.tolist() == [DEPTH - accepted]
    assert sampler.acceptance.num_accepted.tolist() == [accepted]
    assert sampler.acceptance.endpoint_node.tolist() == [accepted]
    assert sampler.acceptance.active.tolist() == [True]
    expected_path = torch.full((1, DEPTH), -1, dtype=torch.long)
    expected_path[0, :accepted] = torch.arange(1, accepted + 1)
    torch.testing.assert_close(sampler.path_node_ids, expected_path)
    torch.testing.assert_close(output.sampled_token_ids[0, : accepted + 1], targets[: accepted + 1])
    assert output.sampled_token_ids[0, accepted + 1 :].eq(-1).all()
    cursor = compute_tree_resume_column(sampler.path_node_ids, output.num_sampled)
    assert cursor.tolist() == [accepted + 1]
    prefix = compute_tree_prefix_source(sampler.path_node_ids, output.num_sampled, 4)
    assert prefix.tolist() == [[accepted + 1, accepted + 2, accepted + 3]]
    if accepted == DEPTH:
        assert cursor.tolist() == [16]
        assert prefix.tolist() == [[16, 17, 18]]


@pytest.mark.parametrize(
    "mode,expected_count,active",
    [
        ("prefill", 1, True),
        ("chunked_prefill", 0, False),
    ],
)
def test_prefill_does_not_leave_a_tree_conv_reanchor(mode, expected_count, active):
    sampler, _, output, _ = sample_case(mode=mode)
    assert output.num_sampled.tolist() == [expected_count]
    assert sampler.acceptance.active.tolist() == [active]
    assert sampler.path_node_ids.eq(-1).all()
    assert tree_row_mask(output.num_sampled, output.num_rejected).tolist() == [False]


def test_model_state_factory_preserves_only_gdn_tree_single_branch_paths():
    for state_type, enabled, wanted in (
        (AscendMambaHybridModelState, True, True),
        (AscendModelState, True, False),
        (AscendMambaHybridModelState, False, None),
        (AscendModelState, False, None),
    ):
        state = state_type.__new__(state_type)
        state.vllm_config = config(enabled)
        state.device = torch.device("cpu")
        result = AscendModelState.custom_sampler(state, sampler_interface())
        if wanted is None:
            assert result is None
        else:
            assert isinstance(result[1], TreeRejectionSampler)
            assert result[1].preserve_tree_path is wanted


def test_other_topk1_models_still_call_upstream_chain_sampler():
    sampler = make_sampler(preserve=False)
    batch, logits, _ = batch_and_logits()
    marker = object()
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler.get_ascend_config", return_value=ascend_config()
        ),
        patch.object(RejectionSampler, "__call__", return_value=marker) as upstream,
    ):
        assert sampler(logits, batch) is marker
    upstream.assert_called_once()
    assert sampler.acceptance is None and sampler.path_node_ids is None
    assert batch.tree_acceptance is None and batch.path_node_ids is None


def test_gdn_topk1_does_not_call_upstream_fallback():
    batch, logits, _ = batch_and_logits()
    sampler = make_sampler()
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler.get_ascend_config", return_value=ascend_config()
        ),
        patch.object(RejectionSampler, "__call__", side_effect=AssertionError("lost GDN tree path")),
    ):
        sampler(logits, batch)
    assert sampler.acceptance.num_sampled.tolist() == [16]


def test_topk8_sampler_algorithm_is_unchanged_by_path_flag():
    results = []
    for flag in (False, True):
        sampler = make_sampler(preserve=flag)
        batch, logits, _ = batch_and_logits(4)
        with patch(
            "vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler.get_ascend_config", return_value=ascend_config(8)
        ):
            output = sampler(logits, batch)
        results.append((output.sampled_token_ids, sampler.path_node_ids, output.num_sampled))
    for left, right in zip(*results):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_depth15_topk1_builder_is_a_single_chain_of_native_selector_argmax(dtype):
    from vllm.model_executor.models.qwen3_dflash2 import _score_edges

    generator = torch.Generator().manual_seed(177)
    selector = SimpleNamespace(
        predecessor_codebook=torch.randn(64, 8, generator=generator).to(dtype),
        successor_codebook=torch.randn(64, 8, generator=generator).to(dtype),
        hidden_projection=torch.nn.Identity(),
        top_k=16,
    )
    draft = SimpleNamespace(model=SimpleNamespace(candidate_selector=selector))
    candidates = torch.stack([torch.randperm(64, generator=generator)[:16] for _ in range(DEPTH)])[None]
    logits = torch.randn((1, DEPTH, 16), generator=generator)
    hidden = torch.randn((1, DEPTH, 8), generator=generator).to(dtype)
    roots = torch.tensor([63])
    scores = _score_edges(
        selector.predecessor_codebook,
        selector.successor_codebook,
        candidates,
        logits,
        hidden,
        roots,
        selector.top_k,
    )
    previous = 0
    expected = []
    for step in range(DEPTH):
        previous = int(scores[0, step, previous].argmax())
        expected.append(int(candidates[0, step, previous]))
    builder = create_tree_builder(DEPTH, 1, "dflash", draft_model=draft)
    layout = linear_tree()
    tree = builder.build(
        logits,
        layout,
        root_token_ids=roots,
        draft_hidden=hidden,
        draft_candidate_ids=candidates,
    )
    assert builder._tokens_buf.shape == (1, DEPTH)
    assert tree.tokens.tolist() == [expected]
    assert tree.parents.tolist() == [list(range(DEPTH))]
    assert tree.depths.tolist() == [list(range(1, DEPTH + 1))]
    torch.testing.assert_close(tree.visibility, torch.ones_like(tree.visibility).tril())


def test_actual_sampler_path_survives_snapshot_postprocess_and_later_prefill_clear():
    from vllm_ascend.worker.v2.model_states import mamba_hybrid

    sampler, _, output, _ = sample_case()
    state = AscendMambaHybridModelState.__new__(AscendMambaHybridModelState)
    state.vllm_config = config()
    state.num_accepted_tokens_gpu = torch.zeros(1, dtype=torch.int32)
    state._tree_prev_path = None
    state._tree_prev_num_sampled = None
    captured_cursors = []

    def cpu_io(previous_path, previous_sampled, mapping, path, sampled, tree_rows):
        # Test IO boundary only; actual masked kernel is covered by NPU tests.
        for row, slot in enumerate(mapping.tolist()):
            if slot >= 0:
                if path is not None and sampled is not None:
                    previous_path[slot, : path.shape[1]] = path[row].int()
                    previous_sampled[slot] = sampled[row] if tree_rows is None or tree_rows[row] else 0
                else:
                    previous_sampled[slot] = 0

    def upstream(_, mapping, count, computed=None):
        captured_cursors.append(count.clone())

    with (
        patch("vllm_ascend.ascend_config.get_ascend_config", return_value=ascend_config()),
        patch.object(mamba_hybrid, "commit_tree_snapshot_metadata", side_effect=cpu_io),
        patch.object(MambaHybridModelState, "postprocess_state", new=upstream),
    ):
        state.stage_tree_resume_column(
            compute_tree_resume_column(sampler.path_node_ids, output.num_sampled),
            sampler.path_node_ids,
            output.num_sampled,
            tree_row_mask(output.num_sampled, output.num_rejected),
        )
        state.postprocess_state(torch.tensor([0]), output.num_sampled)
        assert state._tree_prev_num_sampled.tolist() == [16]
        assert state._tree_prev_path.tolist() == [list(range(1, 16))]
        assert compute_tree_prefix_source(state._tree_prev_path, state._tree_prev_num_sampled, 4).tolist() == [
            [16, 17, 18]
        ]
        prefill_sampler, _, prefill, _ = sample_case(mode="prefill")
        state.stage_tree_resume_column(
            compute_tree_resume_column(prefill_sampler.path_node_ids, prefill.num_sampled),
            prefill_sampler.path_node_ids,
            prefill.num_sampled,
            tree_row_mask(prefill.num_sampled, prefill.num_rejected),
        )
        state.postprocess_state(torch.tensor([0]), prefill.num_sampled)
        assert state._tree_prev_num_sampled.tolist() == [0]
        assert compute_tree_prefix_source(state._tree_prev_path, state._tree_prev_num_sampled, 4).tolist() == [
            [0, 1, 2]
        ]
    assert [value.tolist() for value in captured_cursors] == [[16], [1]]
