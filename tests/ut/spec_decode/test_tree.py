"""CPU unit tests for DFlash draft-tree construction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.ascend_config import TreeSpecConfig
from vllm_ascend.worker.v2.spec_decode import _dflash_tree_spec_enabled, init_speculator
from vllm_ascend.worker.v2.spec_decode.tree.speculator import (
    AscendTreeSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.tree.utils import build_trees


def test_empty_budget_returns_root_only() -> None:
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    layout = build_trees(logits, budget=0)

    assert layout.tokens.shape == (1, 0)
    assert layout.num_nodes.tolist() == [1]
    assert layout.parents.tolist() == [[-1]]
    assert layout.visibility.tolist() == [[[True]]]
    assert layout.child_maps == [[{}]]


def test_peaked_logits_follow_greedy_spine() -> None:
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ]
    )
    layout = build_trees(logits, budget=3)

    assert layout.tokens[0].tolist() == [0, 1, 2]
    assert layout.depths[0].tolist() == [1, 2, 3]
    assert layout.parents[0].tolist() == [-1, 0, 1, 2]
    assert layout.num_nodes[0].item() == 4
    assert layout.visibility[0].tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]
    assert layout.child_maps[0][0][0] == 1
    assert layout.child_maps[0][1][1] == 2
    assert layout.child_maps[0][2][2] == 3


def test_close_siblings_outrank_weak_children() -> None:
    logits = torch.tensor(
        [
            [
                [3.0, 2.9, 0.0],
                [-20.0, -20.0, -20.0],
            ]
        ]
    )
    layout = build_trees(logits, budget=2)

    assert layout.tokens[0].tolist() == [0, 1]
    assert layout.depths[0].tolist() == [1, 1]
    assert layout.parents[0].tolist() == [-1, 0, 0]
    assert layout.visibility[0].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, False, True],
    ]
    assert layout.child_maps[0][0] == {0: 1, 1: 2}


def test_batch_matches_per_request() -> None:
    logits = torch.tensor(
        [
            [[10.0, 0.0], [0.0, 10.0]],
            [[3.0, 2.9], [-20.0, -20.0]],
        ]
    )
    batched = build_trees(logits, budget=2)
    first = build_trees(logits[0:1], budget=2)
    second = build_trees(logits[1:2], budget=2)

    torch.testing.assert_close(batched.tokens[0], first.tokens[0])
    torch.testing.assert_close(batched.tokens[1], second.tokens[0])
    torch.testing.assert_close(batched.parents[0], first.parents[0])
    torch.testing.assert_close(batched.parents[1], second.parents[0])
    assert batched.num_nodes.tolist() == [3, 3]


def test_random_trees_are_well_formed() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 5, 17)
    layout = build_trees(logits, budget=8)
    for req_idx in range(4):
        node_count = int(layout.num_nodes[req_idx].item())
        assert 1 <= node_count <= 9
        parents = layout.parents[req_idx, :node_count].tolist()
        assert parents[0] == -1
        for child_idx, parent_idx in enumerate(parents[1:], start=1):
            assert 0 <= parent_idx < child_idx
        vis = layout.visibility[req_idx, :node_count, :node_count]
        assert bool(vis[0, 0].item())
        for child_idx in range(1, node_count):
            parent_idx = parents[child_idx]
            assert torch.equal(vis[child_idx, :child_idx], vis[parent_idx, :child_idx])
            assert bool(vis[child_idx, child_idx].item())


def test_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="shape"):
        build_trees(torch.zeros(2, 4), budget=2)


def test_tree_spec_config_defaults() -> None:
    cfg = TreeSpecConfig()
    assert cfg.enabled is False
    assert cfg.max_nodes is None


def test_tree_spec_config_validates_max_nodes() -> None:
    with pytest.raises(ValueError, match="max_nodes"):
        TreeSpecConfig({"enabled": True, "max_nodes": -1})
    with pytest.raises(TypeError, match="max_nodes"):
        TreeSpecConfig({"max_nodes": True})
    with pytest.raises(TypeError, match="must be a dict"):
        TreeSpecConfig("enabled")  # type: ignore[arg-type]


def test_dflash_tree_spec_enabled_reads_additional_config() -> None:
    vllm_config = SimpleNamespace(additional_config={"tree_spec_config": {"enabled": True}})
    with patch(
        "vllm_ascend.ascend_config.get_ascend_config",
        side_effect=RuntimeError("not initialized"),
    ):
        assert _dflash_tree_spec_enabled(vllm_config) is True


def test_init_speculator_routes_to_tree_class() -> None:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            use_dspark=lambda: False,
            use_dflash=lambda: True,
        ),
        additional_config={"tree_spec_config": {"enabled": True}},
    )
    tree_spec = object()
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode._dflash_tree_spec_enabled",
            return_value=True,
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.tree.speculator.AscendTreeSpeculator",
            return_value=tree_spec,
        ) as ctor,
    ):
        result = init_speculator(vllm_config, torch.device("cpu"))  # type: ignore[arg-type]
    assert result is tree_spec
    ctor.assert_called_once()


def test_init_speculator_keeps_linear_dflash_by_default() -> None:
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            use_dspark=lambda: False,
            use_dflash=lambda: True,
        ),
        additional_config={},
    )
    linear_spec = object()
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode._dflash_tree_spec_enabled",
            return_value=False,
        ),
        patch(
            "vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator",
            return_value=linear_spec,
        ) as ctor,
    ):
        result = init_speculator(vllm_config, torch.device("cpu"))  # type: ignore[arg-type]
    assert result is linear_spec
    ctor.assert_called_once()


def _make_tree_speculator(num_reqs: int = 1, steps: int = 3, budget: int = 3) -> AscendTreeSpeculator:
    spec = AscendTreeSpeculator.__new__(AscendTreeSpeculator)
    spec.num_speculative_steps = steps
    spec.tree_budget = budget
    spec.max_num_reqs = num_reqs
    spec.device = torch.device("cpu")
    spec.draft_tokens = torch.full((num_reqs, steps), -1, dtype=torch.int64)
    spec.tree_parents = torch.full((num_reqs, steps + 1), 7, dtype=torch.int32)
    spec.tree_depths = torch.full((num_reqs, steps + 1), 7, dtype=torch.int32)
    spec.tree_num_nodes = torch.zeros(num_reqs, dtype=torch.int32)
    spec.tree_visibility = torch.zeros((num_reqs, steps + 1, steps + 1), dtype=torch.bool)
    spec.tree_child_maps = [[{}] for _ in range(num_reqs)]
    spec.sample_indices = torch.arange(num_reqs * steps, dtype=torch.int64)
    spec.model = MagicMock()
    spec.tree = None
    return spec


def test_generate_draft_stores_tree_layout() -> None:
    spec = _make_tree_speculator()
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )
    hidden = torch.zeros(3, 4)
    spec._run_model = MagicMock(return_value=hidden)
    spec.model.compute_logits = MagicMock(return_value=logits)

    layout = spec._generate_draft(
        num_reqs=1,
        num_tokens_padded=4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert spec.draft_tokens[0].tolist() == [0, 1, 2]
    assert spec.tree_depths[0].tolist() == [0, 1, 2, 3]
    assert spec.tree_parents[0].tolist() == [-1, 0, 1, 2]
    assert spec.tree_num_nodes[0].item() == 4
    assert layout is spec.get_tree()
    assert layout.tokens[0].tolist() == [0, 1, 2]
    assert layout.parents[0].tolist() == [-1, 0, 1, 2]
    assert layout.num_nodes[0].item() == 4
    spec.model.compute_logits.assert_called_once()
    spec._run_model.assert_called_once()


def test_generate_draft_pads_unused_nodes() -> None:
    spec = _make_tree_speculator(steps=3, budget=2)
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

    assert spec.draft_tokens[0].tolist() == [0, 1, -1]
    assert spec.tree_num_nodes[0].item() == 3
    assert spec.tree_parents[0].tolist() == [-1, 0, 1, -1]
    assert layout is spec.get_tree()
    assert layout.tokens[0].tolist() == [0, 1]
    assert layout.parents[0].tolist() == [-1, 0, 1]
    assert layout.num_nodes[0].item() == 3
