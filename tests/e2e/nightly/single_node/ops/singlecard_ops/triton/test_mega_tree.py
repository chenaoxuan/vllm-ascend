# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.mega_tree import from_sorted
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout
from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import greedy_tree_reject


def test_mega_tree_all_budgets_and_request_strides():
    """The last node inherits visibility from parent 127 across four mask words."""
    init_device_properties_triton()
    requests, depth, candidates = 2, 15, 64
    ids = torch.arange(requests * depth * candidates).reshape(requests, depth, candidates)
    rank = torch.arange(candidates)
    scores = torch.full((requests, depth, candidates, candidates), -100.0)
    slots = torch.empty_like(scores, dtype=torch.int32)
    expected_tokens = []
    for req in range(requests):
        order = (rank + req) % candidates
        slots[req] = order
        scores[req, 0, 0] = -0.0001 * rank
        scores[req, 1, order[0]] = -1.0 - 0.01 * rank
        scores[req, 2, order[62], 0] = -0.001
        expected_tokens.append(torch.cat((ids[req, 0, order], ids[req, 1, order[:63]], ids[req, 2, order[:1]])))
    expected_tokens = torch.stack(expected_tokens)
    ids, scores, slots = (t.to("npu") for t in (ids, scores, slots))
    parents = [0] * 64 + [1] * 63 + [127]
    depths = torch.tensor([1] * 64 + [2] * 63 + [3], dtype=torch.int32)
    visibility = torch.eye(128, dtype=torch.bool)
    for slot, parent in enumerate(parents):
        if parent:
            visibility[slot] |= visibility[parent - 1]

    for budget in range(1, 129):
        # Give each request a larger token stride and keep a guard on both sides.
        storage = torch.full((requests, budget + 2), -7, dtype=torch.long, device="npu")
        node_data = torch.empty((requests, budget), dtype=torch.int32, device="npu")
        links = torch.empty((requests, budget + 1), dtype=torch.int32, device="npu")
        out = TreeLayout(
            tokens=storage[:, 1:-1],
            depths=node_data,
            parents=torch.empty_like(node_data),
            num_nodes=node_data.new_empty(requests),
            visibility=torch.empty((requests, budget, budget), dtype=torch.bool, device="npu"),
            first_child=links,
            next_sibling=torch.empty_like(links),
        )
        from_sorted(ids, scores, slots, out)
        first_child, next_sibling = [-1] * (budget + 1), [-1] * (budget + 1)
        for node, parent in enumerate(parents[:budget], 1):
            next_sibling[node], first_child[parent] = first_child[parent], node
        assert out.tokens.cpu().equal(expected_tokens[:, :budget])
        assert out.num_nodes.cpu().tolist() == [budget] * requests
        assert storage[:, [0, -1]].cpu().tolist() == [[-7, -7]] * requests
        for actual, expected in (
            (out.depths, depths[:budget]),
            (out.parents, parents[:budget]),
            (out.visibility, visibility[:budget, :budget]),
            (out.first_child, first_child),
            (out.next_sibling, next_sibling),
        ):
            torch.testing.assert_close(actual.cpu(), torch.as_tensor(expected, dtype=actual.dtype).expand_as(actual))


@pytest.mark.parametrize("budget", [1, 15, 48, 128])
def test_verify_full_and_truncated_tree_in_one_batch(budget):
    depth = 15
    deepest = min(depth, budget)
    # The second request only has logits for a prefix of its stored tree.
    short = min(3, deepest // 2)
    tokens = torch.arange(1, budget + 1).repeat(2, 1)
    tokens[1] += 1000
    parents = torch.tensor(list(range(deepest)) + [0] * (budget - deepest)).repeat(2, 1)
    depths = torch.tensor(list(range(1, deepest + 1)) + [1] * (budget - deepest)).repeat(2, 1)
    tokens, parents, depths = (x.to("npu", torch.int32) for x in (tokens, parents, depths))
    counts = tokens.new_empty(2)
    links = tokens.new_empty((2, budget + 1))
    tree = TreeLayout(
        torch.empty_like(tokens),
        torch.empty_like(depths),
        torch.empty_like(parents),
        counts,
        tokens.new_empty((2, budget, budget), dtype=torch.bool),
        links,
        torch.empty_like(links),
    )
    finalize_tree_layout(tree, tokens, depths, parents, budget)
    logits_cu = tokens.new_tensor([0, budget + 1, budget + 2 + short])
    query_cu = tokens.new_tensor([0, budget + 1, 2 * (budget + 1)])
    targets = tokens.new_tensor(list(range(1, budget + 2)) + list(range(1001, 1002 + short)))
    acceptance = TreeAcceptance(
        path_node_ids=tokens.new_empty((2, depth), dtype=torch.long),
        num_sampled=torch.empty_like(counts),
        num_accepted=torch.empty_like(counts),
        endpoint_node=torch.empty_like(counts),
        active=torch.ones_like(counts, dtype=torch.bool),
    )
    sampled = tokens.new_empty((2, depth + 1), dtype=torch.long)
    greedy_tree_reject(tree, targets, logits_cu, tree.verified_node_counts(query_cu, logits_cu), sampled, acceptance)
    assert acceptance.num_accepted.cpu().tolist() == [deepest, short]
    assert acceptance.num_sampled.cpu().tolist() == [deepest + 1, short + 1]
    assert acceptance.endpoint_node.cpu().tolist() == [deepest, short]
    for req, count in enumerate((deepest, short)):
        assert acceptance.path_node_ids[req].cpu().tolist() == list(range(1, count + 1)) + [-1] * (depth - count)
        expected = [1000 * req + i for i in range(1, count + 2)] + [-1] * (depth - count)
        assert sampled[req].cpu().tolist() == expected
