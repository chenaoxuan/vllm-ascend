"""CPU UT: torch golden for each tree Triton op + enable_triton gate."""

from unittest.mock import patch

import torch

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder, align_up
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import (
    compact_tree_kv_along_path_torch,
)
from vllm_ascend.worker.v2.spec_decode.tree.layout import (
    empty_tree_layout,
    finalize_tree_layout_torch,
)
from vllm_ascend.worker.v2.spec_decode.tree.prefix import expand_prefix_layer
from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import (
    greedy_tree_reject,
    greedy_tree_reject_torch,
)
from vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch import use_tree_triton


def _tree_triton_kernels_runnable() -> bool:
    from vllm.triton_utils import HAS_TRITON

    if not HAS_TRITON:
        return False
    try:
        from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

        get_vectorcore_num()
    except (AssertionError, RuntimeError):
        return False
    return True


def _expand_args_live1():
    top_scores = torch.tensor(
        [
            [[0.5, 0.1], [9.0, 8.0]],
            [[0.4, -0.2], [7.0, 6.0]],
        ]
    )
    cand_ids = torch.tensor(
        [
            [[10, 11], [90, 91]],
            [[20, 21], [92, 93]],
        ],
        dtype=torch.long,
    )
    path_scores = torch.zeros(2, 5)
    frontier = torch.zeros(2, 2, dtype=torch.long)
    tokens = torch.full((2, 4), -1, dtype=torch.long)
    depths = torch.zeros(2, 4, dtype=torch.long)
    parents = torch.zeros(2, 5, dtype=torch.long)
    parent_pos = torch.zeros(2, 2, dtype=torch.long)
    return (
        top_scores,
        cand_ids,
        path_scores,
        frontier,
        tokens,
        depths,
        parents,
        parent_pos,
    )


def test_greedy_tree_reject_torch_golden_branch_sibling():
    """Branching tree: sibling walk, partial accept, path ids, bonus."""
    budget, spec_len, vocab = 8, 3, 16
    tree = empty_tree_layout(2, budget, device="cpu")
    # req0: spine 0→1→2 fully accepted + bonus
    tree.tokens[0, :3] = torch.tensor([0, 1, 2])
    tree.parents[0, :3] = torch.tensor([0, 1, 2])
    tree.num_nodes[0] = 3
    tree.first_child[0, :4] = torch.tensor([1, 2, 3, -1])
    # req1: root children 1(tok0) and 2(tok1); accept 1 then reject at depth1
    tree.tokens[1, :3] = torch.tensor([0, 1, 5])
    tree.parents[1, :3] = torch.tensor([0, 0, 1])
    tree.num_nodes[1] = 3
    tree.first_child[1, 0] = 1
    tree.next_sibling[1, 1] = 2
    tree.first_child[1, 1] = 3

    target_ids = torch.zeros(2, budget + 1, dtype=torch.long)
    target_ids[0, :4] = torch.tensor([0, 1, 2, 9])
    # At root pick tok1 → sibling node 2; then no child matches (bonus not written).
    target_ids[1, :3] = torch.tensor([1, 0, 7])
    logits = torch.zeros(2, budget + 1, vocab)
    logits.scatter_(-1, target_ids.unsqueeze(-1), 5.0)

    path = torch.full((2, spec_len), -1, dtype=torch.long)
    sampled = torch.full((2, spec_len + 1), -1, dtype=torch.long)
    out = greedy_tree_reject_torch(tree, logits, spec_len, path, sampled)
    assert out[0, :4].tolist() == [0, 1, 2, 9]
    assert path[0].tolist() == [1, 2, 3]
    assert out[1, :2].tolist() == [1, 7]
    assert path[1].tolist() == [2, -1, -1]

    path2 = torch.full((2, spec_len), -1, dtype=torch.long)
    sampled2 = torch.full((2, spec_len + 1), -1, dtype=torch.long)
    out2 = greedy_tree_reject(tree, logits, spec_len, path2, sampled2)
    assert torch.equal(out, out2)
    assert torch.equal(path, path2)


def test_finalize_tree_layout_torch_sibling_and_visibility():
    budget = 4
    out = empty_tree_layout(1, budget, device="cpu")
    tokens = torch.tensor([[10, 11, 12]], dtype=torch.long)
    depths = torch.tensor([[1, 2, 2]], dtype=torch.long)
    parents = torch.tensor([[0, 1, 1]], dtype=torch.long)
    finalize_tree_layout_torch(out, tokens, depths, parents, 3)
    assert out.first_child[0, 0].item() == 1
    assert out.first_child[0, 1].item() == 3  # last child wins (prepending)
    assert out.next_sibling[0, 3].item() == 2
    assert bool(out.visibility[0, 0, 0])
    assert bool(out.visibility[0, 1, 0]) and bool(out.visibility[0, 1, 1])


def test_compact_tree_kv_along_path_torch_non_prefix_overlap():
    """Non-prefix path + overlapping src/dst gather-then-scatter."""
    num_computed = 10
    block_size = 16
    cache = torch.arange(2 * block_size, dtype=torch.float32).view(2, block_size, 1)
    before = cache.clone()
    block_table = torch.zeros((1, 4), dtype=torch.int32)
    block_table[0, 1] = 1
    idx_mapping = torch.zeros(1, dtype=torch.int32)
    computed = torch.tensor([num_computed], dtype=torch.int32)

    compact_tree_kv_along_path_torch(
        [cache],
        block_table,
        block_size,
        idx_mapping,
        computed,
        torch.tensor([[2, -1, -1]], dtype=torch.long),
    )
    # node 2 at logical 12 → dest depth-1 slot 11
    assert cache[0, 11].tolist() == before[0, 12].tolist()
    assert cache[0, 12].tolist() == before[0, 12].tolist()

    cache = before.clone()
    compact_tree_kv_along_path_torch(
        [cache],
        block_table,
        block_size,
        idx_mapping,
        computed,
        torch.tensor([[2, 3, -1]], dtype=torch.long),
    )
    assert cache[0, 11].tolist() == before[0, 12].tolist()
    assert cache[0, 12].tolist() == before[0, 13].tolist()


def test_tree_attention_mask_torch_builder():
    # AttentionMaskBuilder is @singleton-wrapped; call via instance, not class attr.
    builder = AttentionMaskBuilder(torch.device("cpu"))
    vis = torch.zeros(1, 2, 2, dtype=torch.bool)
    vis[0, 0, 0] = True
    vis[0, 1, 0] = True
    vis[0, 1, 1] = True
    seq_lens = torch.tensor([10], dtype=torch.int32)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch.use_tree_triton",
        return_value=False,
    ):
        mask = builder._get_tree_attention_mask_impl(vis, seq_lens, 1)
    assert mask.shape == (1, 1, 3, align_up(10))
    prev = 10 - 3
    assert not bool(mask[0, 0, 1, prev + 1])


def test_use_tree_triton_enable_flag():
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch.HAS_TRITON",
            True,
        ),
        patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=type(
                "C",
                (),
                {"tree_spec_config": type("T", (), {"enable_triton": True})()},
            )(),
        ),
    ):
        assert use_tree_triton()
    with (
        patch(
            "vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch.HAS_TRITON",
            True,
        ),
        patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=type(
                "C",
                (),
                {"tree_spec_config": type("T", (), {"enable_triton": False})()},
            )(),
        ),
    ):
        assert not use_tree_triton()
    with patch(
        "vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch.HAS_TRITON",
        False,
    ):
        assert not use_tree_triton()


def test_expand_prefix_layer_torch_golden_live1():
    """Layer-1 live=1: invalid parent scores must not win; optional kernel."""
    args = _expand_args_live1()
    (
        top_scores,
        cand_ids,
        path_scores,
        frontier,
        tokens,
        depths,
        parents,
        parent_pos,
    ) = args
    with patch(
        "vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch.use_tree_triton",
        return_value=False,
    ):
        expand_prefix_layer(
            top_scores,
            cand_ids,
            path_scores,
            frontier,
            tokens,
            depths,
            parents,
            parent_pos,
            live=1,
            num_nodes=0,
            child_depth=1,
        )
    assert tokens[:, :2].tolist() == [[10, 11], [20, 21]]
    assert depths[:, :2].tolist() == [[1, 1], [1, 1]]
    assert parents[:, 1:3].tolist() == [[0, 0], [0, 0]]
    assert torch.allclose(
        path_scores[:, 1:3],
        torch.tensor([[0.5, 0.1], [0.4, -0.2]]),
    )
    assert parent_pos.tolist() == [[0, 0], [0, 0]]

    top_scores2 = torch.tensor(
        [
            [[1.0, 0.0], [0.8, -1.0]],
            [[0.2, -0.5], [0.3, 0.1]],
        ]
    )
    cand_ids2 = torch.tensor(
        [
            [[30, 31], [32, 33]],
            [[40, 41], [42, 43]],
        ],
        dtype=torch.long,
    )
    frontier2 = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
    with patch(
        "vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch.use_tree_triton",
        return_value=False,
    ):
        expand_prefix_layer(
            top_scores2,
            cand_ids2,
            path_scores,
            frontier2,
            tokens,
            depths,
            parents,
            parent_pos,
            live=2,
            num_nodes=2,
            child_depth=2,
        )
    # req0: 0.5+1.0=1.5 (tok30), 0.1+0.8=0.9 (tok32)
    # req1: 0.4+0.2=0.6 (tok40), -0.2+0.3=0.1 (tok42)
    assert tokens[:, 2:4].tolist() == [[30, 32], [40, 42]]
    assert parents[:, 3:5].tolist() == [[1, 2], [1, 2]]
    assert parent_pos.tolist() == [[0, 1], [0, 1]]

    if not _tree_triton_kernels_runnable():
        return
    from vllm_ascend.ops.triton.spec_decode.tree.prefix_expand import (
        expand_prefix_layer_triton,
    )

    t_args = _expand_args_live1()
    expand_prefix_layer_triton(
        t_args[0],
        t_args[1],
        t_args[2],
        t_args[3],
        t_args[4],
        t_args[5],
        t_args[6],
        t_args[7],
        live=1,
        num_nodes=0,
        child_depth=1,
    )
    assert t_args[4][:, :2].tolist() == [[10, 11], [20, 21]]
    assert t_args[7].tolist() == [[0, 0], [0, 0]]

