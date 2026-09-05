from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import (
    TreeRejectionSampler,
    block_tree_reject,
)
from vllm_ascend.worker.v2.spec_decode.tree.layout import empty_tree_layout


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
    assert torch.equal(input_batch.path_node_ids, rejection_sampler.path_node_ids)


def test_tree_rejection_sampler_ragged_logit_counts_do_not_mix_requests() -> None:
    """Uneven cu_num_logits must not reshape as ``[R, total // R, V]``.

    Concatenating 3+1 logits and viewing as ``[2, 2, V]`` would steal req0's
    bonus column into req1 and drop req0's bonus, which is the bs>1 mixed
    prefill/decode packing path.
    """
    budget = 2
    spec_len = 2
    vocab_size = 8
    tree = empty_tree_layout(2, budget, device="cpu")
    for req in range(2):
        tree.tokens[req, :2] = torch.tensor([1, 2])
        tree.parents[req, :2] = torch.tensor([0, 1])
        tree.num_nodes[req] = 2
        tree.first_child[req, :3] = torch.tensor([1, 2, -1])

    logits0 = _logits_from_greedy_ids(torch.tensor([[1, 2, 7]]), vocab_size).squeeze(0)
    logits1 = _logits_from_greedy_ids(torch.tensor([[1]]), vocab_size).squeeze(0)
    logits = torch.cat([logits0, logits1], dim=0)

    spec_config = SimpleNamespace(
        num_speculative_tokens=spec_len,
        rejection_sample_method="standard",
        synthetic_acceptance_rates=None,
    )
    sampler = SimpleNamespace(
        compute_nans=False,
        req_states=SimpleNamespace(
            prefill_len=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int32)),
        ),
    )
    rejection_sampler = TreeRejectionSampler(sampler, spec_config, torch.device("cpu"))
    cu = np.array([0, 3, 4], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=2,
        cu_num_logits=torch.from_numpy(cu),
        cu_num_logits_np=cu,
        idx_mapping=torch.arange(2, dtype=torch.int32),
        seq_lens=torch.full((2,), 100, dtype=torch.int32),
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
        [1, 2, 7],
        [1, 0, -1],
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

