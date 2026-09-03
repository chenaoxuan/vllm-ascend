"""CPU unit tests for greedy and block tree rejection."""

from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import (
    TreeRejectionSampler,
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

    path = torch.full((3, spec_len), -1, dtype=torch.long)
    greedy_tree_reject(tree, target_logits, spec_len, path_node_ids=path)
    assert path.tolist() == [
        [1, 2, 3],
        [2, -1, -1],
        [-1, -1, -1],
    ]


def test_greedy_tree_reject_budget_one_empty_and_draft() -> None:
    """budget=1 used to IndexError when first_child is -1 (child-1 == -2)."""
    budget = 1
    spec_len = 1
    vocab_size = 10
    tree = empty_tree_layout(3, budget, device="cpu")

    tree.tokens[1, 0] = 3
    tree.parents[1, 0] = 0
    tree.num_nodes[1] = 1
    tree.first_child[1, 0] = 1

    tree.tokens[2, 0] = 3
    tree.parents[2, 0] = 0
    tree.num_nodes[2] = 1
    tree.first_child[2, 0] = 1

    target_ids = torch.zeros(3, budget + 1, dtype=torch.long)
    target_ids[0, 0] = 7
    target_ids[1, 0] = 3
    target_ids[1, 1] = 8
    target_ids[2, 0] = 9
    target_logits = _logits_from_greedy_ids(target_ids, vocab_size)

    sampled = greedy_tree_reject(tree, target_logits, spec_len)
    assert sampled.tolist() == [
        [7, -1],
        [3, 8],
        [9, -1],
    ]


def test_tree_rejection_sampler_call_uses_greedy_tree_reject() -> None:
    budget = 8
    spec_len = 3
    vocab_size = 10
    tree = empty_tree_layout(3, budget, device="cpu")
    tree.tokens[0, :3] = torch.tensor([0, 1, 2])
    tree.parents[0, :3] = torch.tensor([0, 1, 2])
    tree.num_nodes[0] = 3
    tree.first_child[0, :4] = torch.tensor([1, 2, 3, -1])
    tree.tokens[1, :3] = torch.tensor([0, 1, 5])
    tree.parents[1, :3] = torch.tensor([0, 0, 1])
    tree.num_nodes[1] = 3
    tree.first_child[1, 0] = 1
    tree.next_sibling[1, 1] = 2
    tree.first_child[1, 1] = 3
    tree.tokens[2, :3] = torch.tensor([0, 1, 2])
    tree.parents[2, :3] = torch.tensor([0, 1, 2])
    tree.num_nodes[2] = 3
    tree.first_child[2, :4] = torch.tensor([1, 2, 3, -1])

    target_ids = torch.zeros(3, budget + 1, dtype=torch.long)
    target_ids[0, :4] = torch.tensor([0, 1, 2, 7])
    target_ids[1, :4] = torch.tensor([1, 0, 9, 0])
    target_ids[2, :4] = torch.tensor([9, 0, 1, 2])
    target_logits = _logits_from_greedy_ids(target_ids, vocab_size)
    logits = target_logits.reshape(3 * (budget + 1), vocab_size)

    spec_config = SimpleNamespace(
        num_speculative_tokens=spec_len,
        rejection_sample_method="standard",
        synthetic_acceptance_rates=None,
    )
    sampler = SimpleNamespace(
        compute_nans=False,
        req_states=SimpleNamespace(
            prefill_len=SimpleNamespace(gpu=torch.zeros(3, dtype=torch.int32)),
        ),
    )
    rejection_sampler = TreeRejectionSampler(sampler, spec_config, torch.device("cpu"))
    cu = np.array([0, 9, 18, 27], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=3,
        cu_num_logits=torch.from_numpy(cu),
        cu_num_logits_np=cu,
        idx_mapping=torch.arange(3, dtype=torch.int32),
        seq_lens=torch.full((3,), 100, dtype=torch.int32),
        tree_tokens=tree.tokens,
        tree_depths=tree.depths,
        tree_parents=tree.parents,
        tree_num_nodes=tree.num_nodes,
        tree_visibility=tree.visibility,
        tree_first_child=tree.first_child,
        tree_next_sibling=tree.next_sibling,
    )

    output = rejection_sampler(logits, input_batch)
    assert output.sampled_token_ids.tolist() == [
        [0, 1, 2, 7],
        [1, 9, -1, -1],
        [9, -1, -1, -1],
    ]
    assert output.num_sampled.tolist() == [4, 2, 1]
    assert output.num_rejected.tolist() == [5, 7, 8]
    assert rejection_sampler.path_node_ids.tolist() == [
        [1, 2, 3],
        [2, -1, -1],
        [-1, -1, -1],
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
