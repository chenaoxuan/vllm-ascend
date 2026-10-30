# SPDX-License-Identifier: Apache-2.0
"""DFlash2 edge normalization must preserve the tree builder's path contract."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.tree.builder import create_tree_builder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout
from vllm_ascend.worker.v2.spec_decode.tree.scoring import DFlash2Scorer


def model():
    return SimpleNamespace(
        model=SimpleNamespace(
            candidate_selector=SimpleNamespace(
                predecessor_codebook=torch.zeros(64, 2),
                successor_codebook=torch.zeros(64, 2),
                hidden_projection=torch.nn.Identity(),
                top_k=16,
            )
        )
    )


def layout():
    return TreeLayout(
        tokens=torch.empty((1, 16), dtype=torch.long),
        parents=torch.empty((1, 16), dtype=torch.long),
        depths=torch.empty((1, 16), dtype=torch.long),
        num_nodes=torch.empty(1, dtype=torch.int32),
        visibility=torch.empty((1, 16, 16), dtype=torch.bool),
        first_child=torch.empty((1, 17), dtype=torch.int32),
        next_sibling=torch.empty((1, 17), dtype=torch.int32),
    )


def build(offset):
    builder = create_tree_builder(16, 8, "dflash", draft_model=model())
    logits = torch.arange(16, 0, -1, dtype=torch.float32).reshape(1, 1, 16).expand(1, 7, 16)
    logits = logits + offset
    candidates = torch.arange(1, 17).reshape(1, 1, 16).expand(1, 7, 16)
    return builder.build(
        logits,
        layout(),
        root_token_ids=torch.zeros(1, dtype=torch.long),
        draft_hidden=torch.ones(1, 7, 2),
        draft_candidate_ids=candidates,
    )


def test_dflash2_pruning_keeps_ancestors_and_consistent_depths():
    tree = build(torch.full((1, 7, 1), 50.0))
    assert tree.num_nodes.tolist() == [16]
    for column in range(16):
        parent = int(tree.parents[0, column])
        parent_depth = 0 if parent == 0 else int(tree.depths[0, parent - 1])
        assert 0 <= parent <= column
        assert int(tree.depths[0, column]) == parent_depth + 1


def test_dflash2_tree_is_invariant_to_per_depth_logit_offsets():
    tree = build(torch.zeros(1, 7, 1))
    shifted = build(torch.arange(7).reshape(1, 7, 1) * 100.0 + 20.0)
    for name in ("tokens", "parents", "depths", "visibility"):
        torch.testing.assert_close(getattr(tree, name), getattr(shifted, name))


def test_dflash2_edge_scores_are_normalized_without_changing_argmax():
    scorer = DFlash2Scorer(model())
    state = SimpleNamespace(frontier=torch.zeros(1, 1, dtype=torch.long), tokens=torch.zeros(1, 16, dtype=torch.long))
    logits = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16) + 80
    candidates = torch.arange(1, 17).reshape(1, 1, 16)
    ctx = scorer.prepare(state, logits, torch.zeros(1, dtype=torch.long), torch.ones(1, 1, 2), candidate_ids=candidates)
    edges = scorer.score(state, ctx, 0, False).scores
    torch.testing.assert_close(edges, logits.log_softmax(-1))
    torch.testing.assert_close(edges.exp().sum(-1), torch.ones(1, 1))
    assert edges.argmax(-1).item() == logits.argmax(-1).item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dflash2_matches_native_selector_for_reordered_frontier(dtype):
    from vllm.model_executor.models.qwen3_dflash2 import _score_edges

    generator = torch.Generator().manual_seed(1821)
    selector = model().model.candidate_selector
    selector.predecessor_codebook = torch.randn(64, 2, generator=generator).to(dtype)
    selector.successor_codebook = torch.randn(64, 2, generator=generator).to(dtype)
    draft = SimpleNamespace(model=SimpleNamespace(candidate_selector=selector))
    candidates = torch.stack((torch.arange(1, 17), torch.arange(17, 33)))[None]
    logits = torch.randn(1, 2, 16, generator=generator)
    hidden = torch.randn(1, 2, 2, generator=generator).to(dtype)
    roots = torch.tensor([43])
    native = (
        _score_edges(
            selector.predecessor_codebook,
            selector.successor_codebook,
            candidates,
            logits,
            hidden,
            roots,
            selector.top_k,
        )
        .float()
        .log_softmax(-1)
    )
    state = SimpleNamespace(
        frontier=torch.zeros(1, 1, dtype=torch.long),
        tokens=torch.zeros(1, 16, dtype=torch.long),
    )
    scorer = DFlash2Scorer(draft)
    ctx = scorer.prepare(state, logits, roots, hidden, candidate_ids=candidates)
    torch.testing.assert_close(scorer.score(state, ctx, 0, False).scores, native[:, 0, :1])
    # Beam order is unrelated to the candidate lattice order. Resolve the
    # predecessor using each frontier node's actual token, including siblings.
    slots = torch.tensor([9, 2, 15])
    state.frontier = torch.tensor([[3, 1, 8]])
    state.tokens[0, state.frontier[0] - 1] = candidates[0, 0, slots]
    expected = native[:, 1].index_select(1, slots)
    torch.testing.assert_close(scorer.score(state, ctx, 1, False).scores, expected)
