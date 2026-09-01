"""CPU unit tests for greedy and block tree rejection."""

import torch

from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import (
    block_tree_reject,
    greedy_tree_reject,
)
from vllm_ascend.worker.v2.spec_decode.tree.utils import empty_tree_layout


def _logits_from_greedy_ids(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(
        token_ids.shape[0],
        token_ids.shape[1],
        vocab_size,
        dtype=torch.float32,
        device=token_ids.device,
    )
    logits.scatter_(-1, token_ids.unsqueeze(-1), 10.0)
    return logits


def test_greedy_tree_reject_batch_mixed_trees() -> None:
    budget = 8
    spec_len = 3
    vocab_size = 10
    tree = empty_tree_layout(3, budget, device="cpu")

    # req 0: spine 0 -> 1 -> 2, accept all drafts then bonus 7
    tree.tokens[0, :3] = torch.tensor([0, 1, 2])
    tree.parents[0, :3] = torch.tensor([0, 1, 2])
    tree.num_nodes[0] = 3
    tree.first_child[0, :4] = torch.tensor([1, 2, 3, -1])

    # req 1: root children {0, 1}; target picks token 1 then bonus 9
    tree.tokens[1, :3] = torch.tensor([0, 1, 5])
    tree.parents[1, :3] = torch.tensor([0, 0, 1])
    tree.num_nodes[1] = 3
    tree.first_child[1, 0] = 1
    tree.next_sibling[1, 1] = 2
    tree.first_child[1, 1] = 3

    # req 2: same spine as req 0, but root greedy token misses every child
    tree.tokens[2, :3] = torch.tensor([0, 1, 2])
    tree.parents[2, :3] = torch.tensor([0, 1, 2])
    tree.num_nodes[2] = 3
    tree.first_child[2, :4] = torch.tensor([1, 2, 3, -1])

    target_ids = torch.zeros(3, budget + 1, dtype=torch.long)
    target_ids[0, :4] = torch.tensor([0, 1, 2, 7])
    target_ids[1, :4] = torch.tensor([1, 0, 9, 0])
    target_ids[2, :4] = torch.tensor([9, 0, 1, 2])
    target_logits = _logits_from_greedy_ids(target_ids, vocab_size)

    sampled = greedy_tree_reject(tree, target_logits, spec_len)

    assert sampled.shape == (3, spec_len + 1)
    assert sampled.tolist() == [
        [0, 1, 2, 7],
        [1, 9, -1, -1],
        [9, -1, -1, -1],
    ]


def test_block_tree_reject_paper_path() -> None:
    """Reject X3 then X4 (locks p'≈0.091), then accept X2→X5 and recover Y."""
    mb = torch.tensor([0.3, 0.4, 0.3])
    ms = torch.tensor([0.6, 0.3, 0.1])
    target = torch.log(mb.clamp(min=1e-12)).view(1, 1, 3).expand(1, 6, 3).contiguous()
    draft = torch.log(ms.clamp(min=1e-12)).view(1, 1, 3).expand(1, 2, 3).contiguous()

    tree = empty_tree_layout(1, 5, device="cpu")
    # X1=a, X2=c, X3=b, X4=c, X5=a
    tree.tokens[0, :5] = torch.tensor([0, 2, 1, 2, 0])
    tree.parents[0, :5] = torch.tensor([0, 0, 1, 1, 2])
    tree.depths[0, :5] = torch.tensor([1, 1, 2, 2, 2])
    tree.num_nodes[0] = 5
    tree.first_child[0, 0] = 1
    tree.next_sibling[0, 1] = 2
    tree.first_child[0, 1] = 3
    tree.next_sibling[0, 3] = 4
    tree.first_child[0, 2] = 5

    sampled = block_tree_reject(
        tree,
        target,
        draft,
        2,
        # 0.7 ∈ (7/11, 1): correct p(X4)≈0.636 rejects; p(X4)=1 (wiped p') would accept.
        etas=torch.tensor([[0.9, 0.7, 0.0, 1.0, 1.0]]),
        recover_u=torch.zeros(1),
    )
    # X1 pruned after X4; accept X2=c, X5=a; Y from M_b at X5, u=0 -> a
    assert sampled.tolist() == [[2, 0, 0]]
