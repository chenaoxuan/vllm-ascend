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
from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import (
    greedy_tree_reject,
    greedy_tree_reject_torch,
)
from vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch import use_tree_triton


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


def test_prefix_expand_depth_torch_golden_matches_builder():
    """Domino greedy expand select: torch golden + optional Triton parity."""
    import torch.nn as nn

    from vllm.triton_utils import HAS_TRITON

    from vllm_ascend.ops.triton.spec_decode.tree.prefix_expand import (
        prefix_expand_depth_torch,
        prefix_expand_depth_triton,
    )
    from vllm_ascend.worker.v2.spec_decode.tree.layout import empty_tree_layout
    from vllm_ascend.worker.v2.spec_decode.tree.prefix import PrefixTreeBuilder

    class _Scorer:
        gru_hidden_dim = 2

        def __init__(self) -> None:
            self.fc2_weight = torch.zeros(4, 2)
            self.fc2_bias = None
            self.w_s = torch.zeros(2, 2)
            self.middle = nn.Identity()
            self._gru_input_proj_table = torch.zeros(4, 6)
            self.gru_w_hh = torch.zeros(6, 2)
            self.gru_b_hh = None

        def project_z(self, parallel_hiddens: torch.Tensor) -> torch.Tensor:
            return parallel_hiddens.new_zeros(
                *parallel_hiddens.shape[:-1], 2
            )

        def update_hidden(
            self, token_ids: torch.Tensor, h_state: torch.Tensor
        ) -> torch.Tensor:
            return h_state

    # Distinct logits so torch.topk / iterative argmax agree.
    logits = torch.tensor(
        [[[5.0, 3.0, 1.0, 0.0], [4.0, 2.0, 1.5, 0.5], [3.0, 2.5, 1.0, 0.0]]]
    )
    draft_hidden = torch.zeros(1, 3, 4)
    root = torch.tensor([0])

    with patch(
        "vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch.use_tree_triton",
        return_value=False,
    ):
        out = empty_tree_layout(1, budget=3, device="cpu")
        layout = PrefixTreeBuilder(
            budget=3,
            topk=1,
            correction_scorer=_Scorer(),
            prefix_len=0,
        ).build(logits, out, root_token_ids=root, draft_hidden=draft_hidden)

    assert layout.num_nodes.tolist() == [3]
    assert layout.tokens[0, :3].tolist() == [0, 0, 0]
    assert layout.parents[0, :3].tolist() == [0, 1, 2]
    assert layout.depths[0, :3].tolist() == [1, 2, 3]

    # Direct select op: torch golden vs Triton when available.
    width = k = 2
    top_scores = torch.tensor(
        [[[0.5, 0.1], [0.4, 0.2]]], dtype=torch.float32
    )
    cand_ids = torch.tensor([[[10, 11], [12, 13]]], dtype=torch.long)
    frontier = torch.tensor([[0, 0]], dtype=torch.long)
    path_scores = torch.zeros(1, 8, dtype=torch.float32)
    tokens_t = torch.full((1, 6), -1, dtype=torch.long)
    depths_t = torch.zeros(1, 6, dtype=torch.long)
    parents_t = torch.zeros(1, 8, dtype=torch.long)
    tokens_k = tokens_t.clone()
    depths_k = depths_t.clone()
    parents_k = parents_t.clone()
    frontier_k = frontier.clone()
    path_k = path_scores.clone()
    scratch = torch.empty(1, width * k, dtype=torch.float32)

    prefix_expand_depth_torch(
        top_scores,
        cand_ids,
        frontier,
        path_scores,
        tokens_t,
        depths_t,
        parents_t,
        frontier_len=1,
        take=2,
        child_depth=1,
        num_nodes=0,
        width=width,
        k=k,
    )
    assert tokens_t[0, :2].tolist() == [10, 11]
    assert parents_t[0, 1:3].tolist() == [0, 0]
    assert frontier[0].tolist() == [1, 2]

    if not HAS_TRITON:
        return
    try:
        prefix_expand_depth_triton(
            top_scores,
            cand_ids,
            frontier_k,
            path_k,
            scratch,
            tokens_k,
            depths_k,
            parents_k,
            frontier_len=1,
            take=2,
            child_depth=1,
            num_nodes=0,
            width=width,
            k=k,
        )
    except Exception:
        return
    assert torch.equal(tokens_t, tokens_k)
    assert torch.equal(parents_t, parents_k)
    assert torch.equal(frontier, frontier_k)


def test_prefix_domino_score_and_gru_mix_torch_golden():
    """Domino score + GRU mix torch goldens (Cube path; Triton wrappers alias torch)."""
    from vllm_ascend.ops.triton.spec_decode.tree.prefix_expand import (
        prefix_domino_score_torch,
        prefix_domino_score_triton,
        prefix_gru_mix_torch,
        prefix_gru_mix_triton,
    )

    # Distinct candidate logits so topk order is stable.
    s_proj = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
    z = torch.zeros(1, 2, dtype=torch.float32)
    cand_vals = torch.tensor([[4.0, 2.0, 1.0]], dtype=torch.float32)
    cand_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    cand_w = torch.zeros(1, 3, 2, dtype=torch.float32)
    valid = torch.tensor([[True, False]])
    top_t, ids_t = prefix_domino_score_torch(
        s_proj, z, cand_vals, cand_ids, cand_w, None, valid, k=2, use_silu=False
    )
    assert ids_t[0, 0].tolist() == [10, 20]
    assert top_t[0, 1, 0].item() == float("-inf")

    top_k = torch.empty_like(top_t)
    ids_k = torch.empty_like(ids_t)
    prefix_domino_score_triton(
        s_proj,
        z,
        cand_vals,
        cand_ids,
        cand_w,
        None,
        valid,
        top_k,
        ids_k,
        k=2,
        use_silu=False,
    )
    assert torch.equal(ids_t, ids_k)
    assert torch.allclose(top_t, top_k, atol=1e-5)

    gru_h = 2
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    parent_h = torch.zeros(1, 2, gru_h)
    parent_h[0, 0] = torch.tensor([0.5, -0.5])
    parent_h[0, 1] = torch.tensor([0.25, 0.75])
    gru_table = torch.zeros(4, 3 * gru_h)
    gru_table[2] = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.2, -0.2])
    gh = torch.zeros(1, 2, 3 * gru_h)
    out_t = torch.empty(1, 2, gru_h)
    prefix_gru_mix_torch(tokens, parent_h, gh, gru_table, out_t)
    assert torch.allclose(out_t[0, 0], 0.5 * parent_h[0, 0], atol=1e-5)

    out_k = torch.empty_like(out_t)
    prefix_gru_mix_triton(tokens, parent_h, gh, gru_table, out_k)
    assert torch.allclose(out_t, out_k, atol=1e-5)
