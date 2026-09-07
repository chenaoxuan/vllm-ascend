from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.tree.layout import empty_tree_layout
from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import (
    TreeRejectionSampler,
    block_tree_reject,
)


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


def _mock_tree_ascend_config(rejection_sampler: str = "greedy"):
    return SimpleNamespace(
        tree_spec_config=SimpleNamespace(rejection_sampler=rejection_sampler),
    )


def _paper_magicmtp_tree_and_logits():
    """Shared MagicMTP paper fixture (depth-indexed draft + node-indexed proposal)."""
    mb = torch.tensor([0.3, 0.4, 0.3])
    ms = torch.tensor([0.6, 0.3, 0.1])
    target = torch.log(mb.clamp(min=1e-12)).view(1, 1, 3).expand(1, 6, 3).contiguous()
    draft_depth = torch.log(ms.clamp(min=1e-12)).view(1, 1, 3).expand(1, 2, 3).contiguous()
    proposal = torch.log(ms.clamp(min=1e-12)).view(1, 1, 3).expand(1, 6, 3).contiguous()

    tree = empty_tree_layout(1, 5, device="cpu")
    tree.tokens[0, :5] = torch.tensor([0, 2, 1, 2, 0])
    tree.parents[0, :5] = torch.tensor([0, 0, 1, 1, 2])
    tree.depths[0, :5] = torch.tensor([1, 1, 2, 2, 2])
    tree.num_nodes[0] = 5
    tree.first_child[0, 0] = 1
    tree.next_sibling[0, 1] = 2
    tree.first_child[0, 1] = 3
    tree.next_sibling[0, 3] = 4
    tree.first_child[0, 2] = 5
    return tree, target, draft_depth, proposal


@patch(
    "vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler.get_ascend_config",
    return_value=_mock_tree_ascend_config("greedy"),
)
def test_tree_rejection_sampler_call_uses_greedy_tree_reject(_mock_cfg) -> None:
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


@patch(
    "vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler.get_ascend_config",
    return_value=_mock_tree_ascend_config("greedy"),
)
def test_tree_rejection_sampler_ragged_logit_counts_do_not_mix_requests(_mock_cfg) -> None:
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


def test_block_tree_reject_paper_path_depth_and_node_indexed() -> None:
    """Reject X3 then X4, accept X2→X5; depth- and node-indexed proposal agree."""
    tree, target, draft_depth, proposal = _paper_magicmtp_tree_and_logits()
    etas = torch.tensor([[0.9, 0.7, 0.0, 1.0, 1.0]])
    recover_u = torch.zeros(1)

    path_depth = torch.full((1, 2), -1, dtype=torch.long)
    sampled_depth = torch.full((1, 3), -1, dtype=torch.long)
    out_depth = block_tree_reject(
        tree,
        target,
        draft_depth,
        2,
        path_node_ids=path_depth,
        sampled_token_ids=sampled_depth,
        etas=etas,
        recover_u=recover_u,
    )

    path_node = torch.full((1, 2), -1, dtype=torch.long)
    sampled_node = torch.full((1, 3), -1, dtype=torch.long)
    out_node = block_tree_reject(
        tree,
        target,
        proposal,
        2,
        path_node_ids=path_node,
        sampled_token_ids=sampled_node,
        etas=etas,
        recover_u=recover_u,
    )
    assert out_depth.tolist() == [[2, 0, 0]]
    assert path_depth.tolist() == [[2, 5]]
    assert torch.equal(out_depth, out_node)
    assert torch.equal(path_depth, path_node)


@patch(
    "vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler.get_ascend_config",
    return_value=_mock_tree_ascend_config("magicmtp"),
)
def test_tree_rejection_sampler_call_uses_magicmtp(_mock_cfg) -> None:
    """TreeRejectionSampler routes to block_tree_reject when configured."""
    tree, target, _draft_depth, proposal = _paper_magicmtp_tree_and_logits()
    logits = target.reshape(6, 3)

    spec_config = SimpleNamespace(
        num_speculative_tokens=2,
        rejection_sample_method="standard",
        synthetic_acceptance_rates=None,
    )
    sampler = SimpleNamespace(
        compute_nans=False,
        req_states=SimpleNamespace(
            prefill_len=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int32)),
        ),
    )
    rejection_sampler = TreeRejectionSampler(sampler, spec_config, torch.device("cpu"))
    cu = np.array([0, 6], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=1,
        cu_num_logits=torch.from_numpy(cu),
        cu_num_logits_np=cu,
        idx_mapping=torch.arange(1, dtype=torch.int32),
        seq_lens=torch.full((1,), 100, dtype=torch.int32),
        tree_tokens=tree.tokens,
        tree_depths=tree.depths,
        tree_parents=tree.parents,
        tree_num_nodes=tree.num_nodes,
        tree_visibility=tree.visibility,
        tree_first_child=tree.first_child,
        tree_next_sibling=tree.next_sibling,
        tree_proposal_logits=proposal,
    )

    with patch(
        "vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler.block_tree_reject",
        wraps=block_tree_reject,
    ) as wrapped:
        def _fixed_block(*args, **kwargs):
            kwargs["etas"] = torch.tensor([[0.9, 0.7, 0.0, 1.0, 1.0]])
            kwargs["recover_u"] = torch.zeros(1)
            return block_tree_reject(*args, **kwargs)

        wrapped.side_effect = _fixed_block
        output = rejection_sampler(logits, input_batch, draft_logits=None)

    assert wrapped.called
    assert output.sampled_token_ids.tolist() == [[2, 0, 0]]
    assert rejection_sampler.path_node_ids.tolist() == [[2, 5]]
    assert torch.equal(input_batch.path_node_ids, rejection_sampler.path_node_ids)
