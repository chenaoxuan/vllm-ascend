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
    build_trees,
    empty_tree_layout,
)


def _build_trees(logits: torch.Tensor, budget: int, topk: int) -> TreeLayout:
    out = empty_tree_layout(logits.shape[0], budget, device=logits.device)
    return build_trees(logits, budget, topk, out)


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

