"""CPU unit tests for DFlash draft-tree construction."""

import torch

from vllm_ascend.worker.v2.input_batch import prepare_tree_spec_pos_seq_lens
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import (
    compact_tree_kv_along_path,
    compact_tree_query_along_path,
    mask_rejected_dflash_context_slots,
)
from vllm_ascend.worker.v2.spec_decode.tree.utils import (
    TreeLayout,
    build_best_first_trees,
    build_beam_trees,
    empty_tree_layout,
)


def _build_trees(logits: torch.Tensor, budget: int, topk: int) -> TreeLayout:
    out = empty_tree_layout(logits.shape[0], budget, device=logits.device)
    return build_best_first_trees(logits, budget, topk, out)


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


# Test build_beam_trees
def _make_zero_bias_draft_model(vocab: int, markov_rank: int = 4) -> MagicMock:
    """Draft model whose markov bias is identically zero."""
    model = MagicMock()

    def markov_embed(tokens):
        return torch.zeros(
            tokens.shape[0], tokens.shape[1], markov_rank, device=tokens.device
        )

    def markov_bias(emb):
        return torch.zeros(emb.shape[0], emb.shape[1], vocab, device=emb.device)

    model.markov_embed = MagicMock(side_effect=markov_embed)
    model.markov_bias = MagicMock(side_effect=markov_bias)
    return model


def _make_bigram_draft_model(M: torch.Tensor) -> MagicMock:
    """Draft model whose bias equals ``M[parent_token]`` (a next-token bigram)."""
    model = MagicMock()

    def markov_embed(tokens):
        return tokens.unsqueeze(-1)  # [R, batch, 1]

    def markov_bias(emb):
        return M[emb.squeeze(-1)]  # [R, batch, vocab]

    model.markov_embed = MagicMock(side_effect=markov_embed)
    model.markov_bias = MagicMock(side_effect=markov_bias)
    return model


def _build_beam_trees(
    logits: torch.Tensor,
    budget: int,
    topk: int,
    root_token_ids: torch.Tensor,
    draft_model=None,
) -> TreeLayout:
    out = empty_tree_layout(logits.shape[0], budget, device=logits.device)
    if draft_model is None:
        draft_model = _make_zero_bias_draft_model(logits.shape[-1])
    return build_beam_trees(logits, budget, topk, out, draft_model, root_token_ids)


def _assert_beam_tree_well_formed(layout: TreeLayout, req_idx: int, budget: int) -> None:
    node_count = int(layout.num_nodes[req_idx].item())
    assert 0 <= node_count <= budget
    parents = layout.parents[req_idx, :node_count].tolist()
    depths = layout.depths[req_idx, :node_count].tolist()
    for child_arr, parent_id in enumerate(parents):
        child_idx = child_arr + 1
        assert 0 <= parent_id < child_idx
        # Beam property: depth strictly increments by 1 along a parent chain.
        expected_depth = 1 if parent_id == 0 else depths[parent_id - 1] + 1
        assert depths[child_arr] == expected_depth
        assert int(layout.tokens[req_idx, child_arr].item()) >= 0

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


def test_build_beam_trees_zero_bias_branches() -> None:
    """
    Test that zero markov bias with topk > 1 produces a deterministic branching tree by base logits.
    """
    # [1, spec_num, vocab]: depth0 margin (40) exceeds the depth1/2 child band
    # width (20), so scores are cleanly ordered with no ties.
    logits = torch.tensor(
        [
            [
                [100.0, 60.0, 20.0, 0.0, 0.0, 0.0],
                [100.0, 90.0, 80.0, 0.0, 0.0, 0.0],
                [100.0, 90.0, 80.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    budget = 9
    layout = _build_beam_trees(logits, budget=budget, topk=3, root_token_ids=torch.tensor([0]))

    assert layout.num_nodes[0].item() == budget
    # Root children = top-3 of depth0; the best node then expands its top-3 each
    # depth (depth0's huge margin makes the beam stay on that single branch).
    assert layout.tokens[0, :9].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert layout.depths[0, :9].tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert layout.parents[0, :9].tolist() == [0, 0, 0, 1, 1, 1, 4, 4, 4]
    assert layout.first_child[0, :10].tolist() == [3, 6, -1, -1, 9, -1, -1, -1, -1, -1]
    _assert_beam_tree_well_formed(layout, 0, budget=budget)


def test_build_beam_trees_markov_conditions_on_root() -> None:
    """
    Test that markov bias is corrected applied.

    Given base logits all zeros and frontier tokens, candidate child nodes are determined by the 
    markov bias ``M``.
    """
    M = torch.tensor(
        [
            [-5.0, 10.0, -5.0],  # from token 0 -> prefer 1
            [-5.0, -5.0, 10.0],  # from token 1 -> prefer 2
            [10.0, -5.0, -5.0],  # from token 2 -> prefer 0
        ]
    )
    model = _make_bigram_draft_model(M)
    base_logits = torch.zeros(1, 3, 3)

    layout0 = _build_beam_trees(
        base_logits, budget=3, topk=1, root_token_ids=torch.tensor([0]), draft_model=model
    )
    layout1 = _build_beam_trees(
        base_logits, budget=3, topk=1, root_token_ids=torch.tensor([1]), draft_model=model
    )

    assert layout0.tokens[0, :3].tolist() == [1, 2, 0]
    assert layout0.parents[0, :3].tolist() == [0, 1, 2]
    assert layout1.tokens[0, :3].tolist() == [2, 0, 1]
    assert layout1.parents[0, :3].tolist() == [0, 1, 2]


def test_build_beam_trees_truncates_to_budget() -> None:
    """
    Test that beam trees are truncated to the target budget.
    """
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ]
    )
    layout = _build_beam_trees(logits, budget=2, topk=1, root_token_ids=torch.tensor([0]))

    assert layout.num_nodes[0].item() == 2
    assert layout.tokens[0, :2].tolist() == [0, 1]
    assert layout.parents[0, :2].tolist() == [0, 1]
    assert layout.depths[0, :2].tolist() == [1, 2]
    # Padding positions keep the initial -1 / 0 fill.
    assert layout.tokens[0, 2:].tolist() == [-1]
    assert layout.parents[0, 2:].tolist() == [-1]
    assert layout.depths[0, 2:].tolist() == [0]
    _assert_beam_tree_well_formed(layout, 0, budget=2)


def test_build_beam_trees_is_batched_and_returns_out() -> None:
    """
    Test that build_beam_trees is batched and returns the out parameter.
    """
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            [
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0],
            ],
        ]
    )
    out = empty_tree_layout(2, budget=3, device=logits.device)
    layout = build_beam_trees(logits, budget=3, topk=1, out=out,
                              draft_model=_make_zero_bias_draft_model(3),
                              root_token_ids=torch.tensor([0, 1]))

    assert layout is out
    assert layout.num_nodes.tolist() == [3, 3]
    assert layout.tokens[0, :3].tolist() == [0, 1, 2]
    assert layout.tokens[1, :3].tolist() == [1, 2, 0]
    _assert_beam_tree_well_formed(layout, 0, budget=3)
    _assert_beam_tree_well_formed(layout, 1, budget=3)


def test_build_beam_trees_random_are_well_formed() -> None:
    """
    Test that random beam trees are well-formed.
    """
    torch.manual_seed(1)
    logits = torch.randn(4, 5, 17)
    layout = _build_beam_trees(logits, budget=8, topk=4, root_token_ids=torch.arange(4))
    for req_idx in range(4):
        _assert_beam_tree_well_formed(layout, req_idx, budget=8)




def test_tree_kv_slot_layout_and_compact() -> None:
    """Siblings share RoPE positions but unique KV slots; compact packs the path."""
    num_computed = 10
    query_len = 4
    tree_depths = torch.zeros((1, 8), dtype=torch.int32)
    tree_depths[0, :3] = torch.tensor([1, 1, 2], dtype=torch.int32)
    idx_mapping = torch.zeros(1, dtype=torch.int32)
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    is_prefilling = torch.zeros(1, dtype=torch.int32)
    computed = torch.tensor([num_computed], dtype=torch.int32)
    pos = torch.zeros(query_len, dtype=torch.int64)
    slot_pos = torch.zeros(query_len, dtype=torch.int64)
    seq_lens = torch.zeros(2, dtype=torch.int32)

    prepare_tree_spec_pos_seq_lens(
        idx_mapping,
        query_start_loc,
        is_prefilling,
        computed,
        tree_depths,
        pos,
        slot_pos,
        seq_lens,
    )
    assert pos.tolist() == [10, 11, 11, 12]
    assert slot_pos.tolist() == [10, 11, 12, 13]
    assert seq_lens[0].tolist() == 14

    block_size = 16
    cache = torch.arange(2 * block_size, dtype=torch.float32).view(2, block_size, 1)
    before = cache.clone()
    block_table = torch.zeros((1, 4), dtype=torch.int32)
    block_table[0, 1] = 1
    path_node_ids = torch.tensor([[2, -1, -1]], dtype=torch.long)
    compact_tree_kv_along_path(
        [cache],
        block_table,
        block_size,
        idx_mapping,
        computed,
        path_node_ids,
    )
    # node 2 lives at logical 12; accepted path depth 1 dest is logical 11.
    assert cache[0, 11].tolist() == before[0, 12].tolist()
    assert cache[0, 12].tolist() == before[0, 12].tolist()

    # Overlapping src/dst: gather then scatter so 13→12 does not read the
    # already-written 12→11 result.
    cache = before.clone()
    compact_tree_kv_along_path(
        [cache],
        block_table,
        block_size,
        idx_mapping,
        computed,
        torch.tensor([[2, 3, -1]], dtype=torch.long),
    )
    assert cache[0, 11].tolist() == before[0, 12].tolist()
    assert cache[0, 12].tolist() == before[0, 13].tolist()


def test_tree_query_compact_along_non_prefix_path() -> None:
    """Packed siblings are not a prefix; compact gathers path rows and linear RoPE."""
    query_len = 4
    hidden = torch.arange(query_len * 2, dtype=torch.float32).view(query_len, 2)
    aux = hidden.clone() + 100
    pos = torch.tensor([10, 11, 11, 12], dtype=torch.int64)
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    before_h = hidden.clone()
    before_aux = aux.clone()

    compact_tree_query_along_path(
        [hidden, aux],
        query_start_loc,
        torch.tensor([[2, -1, -1]], dtype=torch.long),
        linearize_positions=pos,
    )
    assert torch.equal(hidden[0], before_h[0])
    assert torch.equal(hidden[1], before_h[2])
    assert torch.equal(hidden[2], before_h[2])
    assert torch.equal(hidden[3], before_h[3])
    assert torch.equal(aux[1], before_aux[2])
    assert torch.equal(pos, torch.tensor([10, 11, 11, 12]))

    hidden = before_h.clone()
    pos = torch.tensor([10, 11, 11, 12], dtype=torch.int64)
    compact_tree_query_along_path(
        [hidden],
        query_start_loc,
        torch.tensor([[2, 1, -1]], dtype=torch.long),
        linearize_positions=pos,
    )
    assert torch.equal(hidden[0], before_h[0])
    assert torch.equal(hidden[1], before_h[2])
    assert torch.equal(hidden[2], before_h[1])
    assert torch.equal(hidden[3], before_h[3])
    assert torch.equal(pos, torch.tensor([10, 11, 12, 12]))

    # path=[1] with a same-depth leftover sibling: compact leaves the sibling
    # on the rejected suffix sharing RoPE 11. DFlash must PAD that suffix so
    # it cannot clobber the accepted node's draft KV slot.
    hidden = before_h.clone()
    pos = torch.tensor([10, 11, 11, 12], dtype=torch.int64)
    compact_tree_query_along_path(
        [hidden],
        query_start_loc,
        torch.tensor([[1, -1, -1]], dtype=torch.long),
        linearize_positions=pos,
    )
    assert torch.equal(hidden[0], before_h[0])
    assert torch.equal(hidden[1], before_h[1])
    assert torch.equal(hidden[2], before_h[2])
    assert torch.equal(pos, torch.tensor([10, 11, 11, 12]))
    slots = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
    mask_rejected_dflash_context_slots(
        slots,
        query_start_loc,
        torch.tensor([2], dtype=torch.int32),
        pad_slot_id=-1,
    )
    assert torch.equal(slots, torch.tensor([100, 101, -1, -1]))

