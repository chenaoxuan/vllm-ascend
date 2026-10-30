# SPDX-License-Identifier: Apache-2.0
"""DFlash2 beam pools agree with tree260914 at 30d69adaf (BS1)."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder, create_tree_builder
from vllm_ascend.worker.v2.spec_decode.tree.dflash2_builder import DFlash2BeamTreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.layout import (
    TreeLayout,
    finalize_tree_layout,
    finalize_tree_layout_from_state,
)
from vllm_ascend.worker.v2.spec_decode.tree.scoring import CandidateLayer

BUDGET = 16
TOPK = 8
DEPTH = 7
SELECTOR_TOPK = 16


def _layout(device="cpu"):
    return TreeLayout(
        tokens=torch.empty((1, BUDGET), device=device, dtype=torch.long),
        parents=torch.empty((1, BUDGET), device=device, dtype=torch.long),
        depths=torch.empty((1, BUDGET), device=device, dtype=torch.long),
        num_nodes=torch.empty(1, device=device, dtype=torch.int32),
        visibility=torch.empty((1, BUDGET, BUDGET), device=device, dtype=torch.bool),
        first_child=torch.empty((1, BUDGET + 1), device=device, dtype=torch.int32),
        next_sibling=torch.empty((1, BUDGET + 1), device=device, dtype=torch.int32),
    )


def _fixture(kind):
    candidates = torch.arange(1, DEPTH * SELECTOR_TOPK + 1).reshape(1, DEPTH, SELECTOR_TOPK)
    scores = torch.zeros((1, DEPTH, SELECTOR_TOPK, SELECTOR_TOPK))
    if kind == "wide":
        # Two useful root branches, each with eight useful children. All
        # subsequent transitions are diffuse, so both branches belong in the
        # final budget even though only one survives the next beam frontier.
        root = torch.full((SELECTOR_TOPK,), -20.0)
        root[:2] = torch.tensor([0.6, 0.4]).log()
        scores[:, 0] = root
        scores[:, 1, :2, TOPK:] = -20.0
    elif kind == "deep":
        generator = torch.Generator().manual_seed(1729)
        scores = torch.randn(scores.shape, generator=generator) * 0.1
        scores[..., 0] += 10.0
    elif kind != "ties":
        raise ValueError(kind)
    return candidates, scores, torch.tensor([999])


def _reference_beam(candidate_ids, selector_scores, roots):
    """Independent CPU oracle for the reference beam's pool/frontier rules."""
    pool = []
    frontier = [(int(roots[0]), 0.0, -1)]
    for depth in range(DEPTH):
        rows = []
        for token, _, _ in frontier:
            previous = 0 if depth == 0 else candidate_ids[0, depth - 1].tolist().index(token)
            rows.append(selector_scores[0, depth, previous])
        log_probs = torch.stack(rows).float().log_softmax(-1)
        values, slots = log_probs.topk(TOPK, dim=-1)
        offset = len(pool)
        candidate_scores = []
        for parent, (_, parent_score, parent_pool) in enumerate(frontier):
            for child in range(TOPK):
                score = float(torch.tensor(parent_score, dtype=values.dtype) + values[parent, child])
                token = int(candidate_ids[0, depth, slots[parent, child]])
                pool.append((token, score, parent_pool, depth + 1))
                candidate_scores.append(score)
        selected = torch.tensor(candidate_scores).topk(TOPK).indices.tolist()
        frontier = [(pool[offset + index][0], pool[offset + index][1], offset + index) for index in selected]

    # Pool order is depth-major; stable score sorting therefore resolves ties
    # ancestor-first just like the reference's depth-sort plus stable argsort.
    kept = sorted(sorted(range(len(pool)), key=lambda index: -pool[index][1])[:BUDGET])
    remap = {old: new + 1 for new, old in enumerate(kept)}
    return (
        torch.tensor([[pool[index][0] for index in kept]]),
        torch.tensor([[0 if pool[index][2] < 0 else remap[pool[index][2]] for index in kept]]),
        torch.tensor([[pool[index][3] for index in kept]]),
    )


class _LatticeScorer:
    """Supply normalized selector rows while leaving beam construction real."""

    default_depth_penalty = 0.0

    def __init__(self, selector_scores):
        self.selector_scores = selector_scores

    def reserve(self, num_reqs, num_nodes, device):
        pass

    def workspace_tensors(self):
        return ()

    def prepare(self, state, logits, roots, hidden, *, keep_origin_chain, candidate_ids):
        return SimpleNamespace(candidates=candidate_ids, roots=roots)

    def score(self, state, ctx, depth_slot, keep_origin_chain):
        parent_tokens = state.tokens.gather(1, (state.frontier - 1).clamp_min(0))
        parent_tokens = torch.where(state.frontier == 0, ctx.roots[:, None], parent_tokens)
        if depth_slot == 0:
            predecessor = torch.zeros_like(parent_tokens)
        else:
            predecessor = (parent_tokens[..., None] == ctx.candidates[:, depth_slot - 1, None]).long().argmax(-1)
        raw_scores = self.selector_scores[:, depth_slot].gather(1, predecessor[..., None].expand(-1, -1, SELECTOR_TOPK))
        return CandidateLayer(
            raw_scores.float().log_softmax(-1),
            ctx.candidates[:, depth_slot, None].expand(-1, state.width, -1),
            raw_scores.argmax(-1),
        )

    def update(self, state, ctx, num_added):
        pass


def _build(kind, *, keep_origin_chain=False, builder_type=DFlash2BeamTreeBuilder):
    candidates, scores, roots = _fixture(kind)
    builder = builder_type(BUDGET, TOPK, _LatticeScorer(scores), keep_origin_chain=keep_origin_chain)
    tree = builder.build(
        torch.zeros_like(candidates, dtype=torch.float32),
        _layout(),
        root_token_ids=roots,
        draft_candidate_ids=candidates,
    )
    return tree, builder


def _assert_ancestors(tree):
    assert tree.num_nodes.tolist() == [BUDGET]
    for column in range(BUDGET):
        parent = int(tree.parents[0, column])
        parent_depth = 0 if parent == 0 else int(tree.depths[0, parent - 1])
        assert 0 <= parent <= column
        assert int(tree.depths[0, column]) == parent_depth + 1
        ancestors = {column}
        while parent:
            ancestors.add(parent - 1)
            parent = int(tree.parents[0, parent - 1])
        visible = set(tree.visibility[0, column].nonzero().flatten().tolist())
        assert visible == ancestors


@pytest.mark.parametrize("kind", ["wide", "deep", "ties"])
def test_dflash2_matches_reference_beam_budget16_topk8_depth7(kind):
    candidates, scores, roots = _fixture(kind)
    expected = _reference_beam(candidates, scores, roots)
    tree, builder = _build(kind)
    for actual, reference in zip((tree.tokens, tree.parents, tree.depths), expected):
        torch.testing.assert_close(actual, reference)
    assert builder._tokens_buf.shape == (1, TOPK + (DEPTH - 1) * TOPK * TOPK)
    _assert_ancestors(tree)


def test_dflash2_keeps_useful_nodes_outside_next_frontier():
    tree, _ = _build("wide")
    assert tree.depths.eq(1).sum() == 2
    assert tree.depths.eq(2).sum() == 14
    # The former common builder stores only eight candidates at each depth.
    old_tree, _ = _build("wide", builder_type=TreeBuilder)
    assert old_tree.depths.eq(2).sum() == TOPK
    assert old_tree.depths.gt(2).any()


def test_dflash2_ties_keep_shallower_ancestors_first():
    tree, _ = _build("ties")
    assert tree.depths.tolist() == [[1] * TOPK + [2] * TOPK]
    parents = tree.parents[:, TOPK:]
    assert parents.ge(1).all() and parents.le(TOPK).all()
    assert tree.depths.gather(1, parents - 1).eq(1).all()
    candidates, scores, roots = _fixture("ties")
    reference_parents = _reference_beam(candidates, scores, roots)[1]
    torch.testing.assert_close(parents, reference_parents[:, TOPK:])
    _assert_ancestors(tree)


def test_dflash2_origin_chain_keeps_all_seven_depths():
    tree, _ = _build("deep", keep_origin_chain=True)
    assert int(tree.depths.max()) == DEPTH
    _assert_ancestors(tree)


def test_only_dflash2_uses_the_reference_beam_builder():
    selector = SimpleNamespace(
        predecessor_codebook=torch.zeros(128, 2),
        successor_codebook=torch.zeros(128, 2),
        hidden_projection=torch.nn.Identity(),
        top_k=SELECTOR_TOPK,
    )
    model = SimpleNamespace(model=SimpleNamespace(candidate_selector=selector))
    assert isinstance(create_tree_builder(BUDGET, TOPK, "dflash", draft_model=model), DFlash2BeamTreeBuilder)
    assert type(create_tree_builder(BUDGET, TOPK, "dflash")) is TreeBuilder
    assert type(create_tree_builder(BUDGET, TOPK, "dspark")) is TreeBuilder


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU; run separately with validation device locks",
)
@pytest.mark.parametrize("kind", ["wide", "deep", "ties"])
def test_dflash2_npu_pool392_finalization_matches_reference(kind):
    init_device_properties_triton()
    # Keep beam arithmetic on CPU to isolate the production NPU prune/link
    # kernels, including their 512-lane padded candidate pool, from topk ties.
    candidates, scores, roots = _fixture(kind)
    tokens, parents, depths = _reference_beam(candidates, scores, roots)
    expected = finalize_tree_layout(_layout(), tokens, depths, parents, BUDGET, force_torch=True)
    _assert_ancestors(expected)
    _, builder = _build(kind)
    assert builder._tokens_buf.shape[1] == 392
    actual = finalize_tree_layout_from_state(
        _layout("npu"),
        builder._path_scores_buf[:, 1:].to("npu"),
        builder._tokens_buf.to("npu"),
        builder._depths_buf.to("npu"),
        builder._parents_buf.to("npu"),
        BUDGET,
    )
    torch.npu.synchronize()
    for name in ("tokens", "parents", "depths", "num_nodes", "visibility", "first_child", "next_sibling"):
        torch.testing.assert_close(
            getattr(actual, name).cpu(),
            getattr(expected, name),
            rtol=0,
            atol=0,
            msg=lambda message, name=name: f"{name}: {message}",
        )
