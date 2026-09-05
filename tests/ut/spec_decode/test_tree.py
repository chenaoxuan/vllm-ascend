"""CPU unit tests for DFlash draft-tree construction."""

import torch

from vllm_ascend.worker.v2.input_batch import prepare_tree_spec_pos_seq_lens
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import (
    compact_tree_kv_along_path,
    compact_tree_query_along_path,
    mask_rejected_dflash_context_slots,
)
from vllm_ascend.worker.v2.spec_decode.tree.utils import (
    build_beam_trees,
    build_heap_trees,
    build_multi_order_trees,
    empty_tree_layout,
)


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
    layout = build_heap_trees(logits, budget=3, topk=3, out=out)

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


def test_build_beam_trees_zero_bias_branches() -> None:
    """Zero markov bias with topk > 1 branches by base logits, then fills budget."""
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
    out = empty_tree_layout(1, budget, device=logits.device)
    layout = build_beam_trees(
        logits,
        budget=budget,
        topk=3,
        out=out,
        root_token_ids=torch.tensor([0]),
    )

    assert layout is out
    assert layout.num_nodes[0].tolist() == budget
    assert layout.tokens[0, :9].tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert layout.depths[0, :9].tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert layout.parents[0, :9].tolist() == [0, 0, 0, 1, 1, 1, 4, 4, 4]
    assert layout.first_child[0, :10].tolist() == [3, 6, -1, -1, 9, -1, -1, -1, -1, -1]


def test_build_multi_order_trees_truncates_to_budget() -> None:
    """Uncorrected multi_order grows a spine then prunes to budget, batched."""
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
    out = empty_tree_layout(2, budget=2, device=logits.device)
    layout = build_multi_order_trees(logits, budget=2, topk=1, out=out)

    assert layout is out
    assert layout.num_nodes.tolist() == [2, 2]
    assert layout.tokens[0, :2].tolist() == [0, 1]
    assert layout.parents[0, :2].tolist() == [0, 1]
    assert layout.depths[0, :2].tolist() == [1, 2]
    assert layout.tokens[1, :2].tolist() == [1, 2]
    assert layout.parents[1, :2].tolist() == [0, 1]
    assert layout.depths[1, :2].tolist() == [1, 2]


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
