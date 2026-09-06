"""CPU UT: tree Triton wrappers fall back to Torch golden and stay consistent."""

import torch

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder, align_up
from vllm_ascend.worker.v2.spec_decode.tree.layout import (
    empty_tree_layout,
    finalize_tree_layout_torch,
)
from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import (
    greedy_tree_reject,
    greedy_tree_reject_torch,
)


def _chain_tree(num_reqs: int, budget: int, depth: int, device="cpu"):
    tree = empty_tree_layout(num_reqs, budget, device=device)
    for r in range(num_reqs):
        for i in range(depth):
            tree.tokens[r, i] = i
            tree.parents[r, i] = i
            tree.depths[r, i] = i + 1
            tree.first_child[r, i] = i + 1
        tree.first_child[r, depth] = -1
        tree.num_nodes[r] = depth
    return tree


def test_greedy_tree_reject_torch_golden_chain():
    budget, spec_len, vocab = 8, 3, 16
    tree = _chain_tree(2, budget, 3)
    target_ids = torch.zeros(2, budget + 1, dtype=torch.long)
    target_ids[0, :4] = torch.tensor([0, 1, 2, 9])
    target_ids[1, :2] = torch.tensor([0, 7])
    logits = torch.zeros(2, budget + 1, vocab)
    logits.scatter_(-1, target_ids.unsqueeze(-1), 5.0)
    out = greedy_tree_reject_torch(tree, logits, spec_len)
    assert out[0, :4].tolist() == [0, 1, 2, 9]
    assert out[1, :2].tolist() == [0, 7]
    # Dispatch on CPU must use torch golden.
    out2 = greedy_tree_reject(tree, logits, spec_len)
    assert torch.equal(out, out2)


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


def test_tree_attention_mask_torch_builder():
    device = torch.device("cpu")
    # Bypass singleton cache: exercise the torch fill path directly.
    class _B:
        device = device
        _tree_attn_mask = None
        _tree_mask_caps = (0, 0, 0)

        _get_tree_attention_mask_impl = AttentionMaskBuilder._get_tree_attention_mask_impl

    builder = _B()
    vis = torch.zeros(1, 2, 2, dtype=torch.bool)
    vis[0, 0, 0] = True
    vis[0, 1, 0] = True
    vis[0, 1, 1] = True
    seq_lens = torch.tensor([10], dtype=torch.int32)
    with torch.device(device):
        mask = AttentionMaskBuilder._get_tree_attention_mask_impl(
            builder, vis, seq_lens, 1
        )
    assert mask.shape == (1, 1, 3, align_up(10))
    prev = 10 - 3
    assert not bool(mask[0, 0, 1, prev + 1])
