"""CPU unit tests for DFlash draft-tree construction."""

from unittest.mock import MagicMock

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.tree.speculator import AscendTreeSpeculator
from vllm_ascend.worker.v2.spec_decode.tree.utils import (
    TreeLayout,
    build_trees,
    empty_tree_layout,
)


def _build_trees(logits: torch.Tensor, budget: int, topk: int) -> TreeLayout:
    out = empty_tree_layout(logits.shape[0], budget, device=logits.device)
    return build_trees(logits, budget, topk, out)


def test_batch_spine_and_sibling_trees() -> None:
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            [
                [3.0, 2.9, 0.0],
                [-20.0, -20.0, -20.0],
                [-20.0, -20.0, -20.0],
            ],
        ]
    )
    out = empty_tree_layout(2, 3, device=logits.device)
    layout = build_trees(logits, budget=3, topk=3, out=out)

    assert layout is out
    assert layout.num_nodes.tolist() == [3, 3]

    assert layout.tokens[0].tolist() == [0, 1, 2]
    assert layout.depths[0].tolist() == [1, 2, 3]
    assert layout.parents[0].tolist() == [0, 1, 2]
    assert layout.visibility[0].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]
    assert layout.first_child[0, :4].tolist() == [1, 2, 3, -1]
    assert layout.next_sibling[0, :4].tolist() == [-1, -1, -1, -1]

    assert layout.tokens[1].tolist() == [0, 1, 0]
    assert layout.depths[1].tolist() == [1, 1, 2]
    assert layout.parents[1].tolist() == [0, 0, 1]
    assert layout.visibility[1].tolist() == [
        [True, False, False],
        [False, True, False],
        [True, False, True],
    ]
    # Newest child is prepended: node 2 then sibling 1 under root.
    assert layout.first_child[1, :4].tolist() == [2, 3, -1, -1]
    assert layout.next_sibling[1, :4].tolist() == [-1, -1, 1, -1]


def test_topk_is_independent_of_budget() -> None:
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ]
    )
    spine = _build_trees(logits, budget=8, topk=1)
    filled = _build_trees(logits, budget=8, topk=3)

    assert spine.num_nodes[0].item() == 3
    assert spine.tokens[0, :3].tolist() == [0, 1, 2]
    assert spine.parents[0, :3].tolist() == [0, 1, 2]
    assert spine.tokens[0, 3:].tolist() == [-1] * 5
    assert spine.parents[0, 3:].tolist() == [-1] * 5
    assert filled.num_nodes[0].item() == 8


def test_random_trees_are_well_formed() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 5, 17)
    layout = _build_trees(logits, budget=8, topk=8)
    for req_idx in range(4):
        node_count = int(layout.num_nodes[req_idx].item())
        assert 0 <= node_count <= 8
        parents = layout.parents[req_idx, :node_count].tolist()
        for child_arr, parent_id in enumerate(parents):
            child_idx = child_arr + 1
            assert 0 <= parent_id < child_idx
        vis = layout.visibility[req_idx, :node_count, :node_count]
        for child_arr in range(node_count):
            assert bool(vis[child_arr, child_arr].item())
            parent_id = parents[child_arr]
            if parent_id == 0:
                if child_arr > 0:
                    assert not vis[child_arr, :child_arr].any()
            else:
                parent_arr = parent_id - 1
                assert torch.equal(vis[child_arr, :child_arr], vis[parent_arr, :child_arr])
        seen = set()
        for parent_id in range(node_count + 1):
            child = int(layout.first_child[req_idx, parent_id].item())
            while child != -1:
                assert child not in seen
                seen.add(child)
                assert int(layout.parents[req_idx, child - 1].item()) == parent_id
                child = int(layout.next_sibling[req_idx, child].item())
        assert seen == set(range(1, node_count + 1))


def _make_tree_speculator(num_reqs: int = 1, steps: int = 3, budget: int = 3) -> AscendTreeSpeculator:
    spec = AscendTreeSpeculator.__new__(AscendTreeSpeculator)
    spec.num_speculative_steps = steps
    spec.budget = budget
    spec.topk = None
    spec.max_num_reqs = num_reqs
    spec.device = torch.device("cpu")
    spec.draft_tokens = torch.full((num_reqs, max(steps, budget)), -1, dtype=torch.int64)
    spec.tree_parents = torch.full((num_reqs, budget), 7, dtype=torch.int32)
    spec.tree_depths = torch.full((num_reqs, budget), 7, dtype=torch.int32)
    spec.tree_num_nodes = torch.zeros(num_reqs, dtype=torch.int32)
    spec.tree_visibility = torch.zeros((num_reqs, budget, budget), dtype=torch.bool)
    spec.tree_first_child = torch.full((num_reqs, budget + 1), -1, dtype=torch.int32)
    spec.tree_next_sibling = torch.full((num_reqs, budget + 1), -1, dtype=torch.int32)
    spec.sample_indices = torch.arange(num_reqs * steps, dtype=torch.int64)
    spec.model = MagicMock()
    spec.tree = None
    return spec


def test_generate_draft_stores_tree_layout() -> None:
    spec = _make_tree_speculator(steps=3, budget=8)
    spec.topk = 1
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )
    spec._run_model = MagicMock(return_value=torch.zeros(3, 4))
    spec.model.compute_logits = MagicMock(return_value=logits)

    layout = spec._generate_draft(
        num_reqs=1,
        num_tokens_padded=4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert spec.draft_tokens[0].tolist() == [0, 1, 2] + [-1] * 5
    assert spec.tree_depths[0, :3].tolist() == [1, 2, 3]
    assert spec.tree_parents[0, :3].tolist() == [0, 1, 2]
    assert spec.tree_parents[0, 3:].tolist() == [-1] * 5
    assert spec.tree_num_nodes[0].item() == 3
    assert layout is spec.get_tree()
    assert layout.tokens[0, :3].tolist() == [0, 1, 2]
    assert layout.tokens[0, 3:].tolist() == [-1] * 5
    assert layout.parents[0, :3].tolist() == [0, 1, 2]
    assert layout.num_nodes[0].item() == 3
    assert layout.first_child[0, :4].tolist() == [1, 2, 3, -1]
    assert layout.first_child.data_ptr() == spec.tree_first_child.data_ptr()
    spec.model.compute_logits.assert_called_once()
    spec._run_model.assert_called_once()
