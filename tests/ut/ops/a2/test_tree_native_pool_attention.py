# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C4 dual-source UT. Chain expansion and D2H below belong only to the oracle."""

import pytest
import torch

from tests.ut.ops.a2.test_tree_native_attention import make_case, native_ops  # noqa: F401


def make_pool_case(requests=1, budget=16, prefix=127, heads=8, dtype=torch.bfloat16):
    q, tree, linear, _ = make_case(requests=requests, budget=budget, prefix=prefix, heads=heads, dtype=dtype)
    n, _, dim = q.shape
    count, block = budget + 1, tree["cmp_kv"].shape[1]
    chains = []
    for req in range(requests):
        for local in range(count):
            chain = []
            while local >= 0:
                chain.append(req * count + local)
                local = (local - 1) // 2
            chains.append(chain[::-1])
    depth = max(map(len, chains))
    ancestors = torch.tensor([c + [-1] * (depth - len(c)) for c in chains], dtype=torch.int64)
    torch.manual_seed(915)
    nodes = torch.randn(n, dim, dtype=dtype)
    # Independent, branch-specific compressed caches are for the old-op oracle only.
    pages = max(1, ((prefix + depth) // 4 + block - 1) // block)
    expanded = torch.zeros(1 + n * pages, block, 1, dim, dtype=dtype)
    table = torch.arange(1, 1 + n * pages, dtype=torch.int32).view(n, pages)
    history, history_table, slots, original, topk, sinks, query = [
        value.cpu()
        for value in (
            tree["cmp_kv"],
            tree["cmp_block_table"],
            tree["ori_sparse_indices"],
            tree["ori_kv"],
            tree["cmp_sparse_indices"],
            tree["sinks"],
            q,
        )
    ]
    refs = []
    for row, chain in enumerate(chains):
        for k in range((prefix + len(chain)) // 4):
            value = (
                history[history_table[row, k // block], k % block, 0]
                if k < prefix // 4
                else nodes[chain[(k + 1) * 4 - 1 - prefix]]
            )
            expanded[table[row, k // block], k % block, 0] = value
        ori_slots = slots[row, 0]
        keys = original.view(-1, dim)[ori_slots[ori_slots >= 0].long()]
        selected = topk[row, 0]
        selected = selected[selected >= 0].long()
        if selected.numel():
            keys = torch.cat((keys, expanded[table[row, selected // block], selected % block, 0]))
        scores = query[row].float() @ keys.float().T * dim**-0.5
        weights = torch.softmax(torch.cat((scores, sinks[:, None]), -1), -1)[:, :-1]
        refs.append(weights @ keys.float())
    tree.update(
        cmp_tree_kv=nodes.npu(),
        tree_ancestors=ancestors.npu(),
        tree_prefix=torch.full((n,), prefix, dtype=torch.int32).npu(),
    )
    linear.update(cmp_kv=expanded.npu(), cmp_block_table=table.npu())
    return q, tree, linear, torch.stack(refs)


@pytest.mark.parametrize("requests", [1, 2, 4, 8])
@pytest.mark.parametrize("budget,prefix", [(16, 0), (16, 3), (16, 31), (42, 127), (42, 511)])
def test_compressed_pools_match_native(requests, budget, prefix):
    q, tree, linear, reference = make_pool_case(requests, budget, prefix)
    expected, expected_lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, return_softmax_lse=True, **linear)
    actual, lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, return_softmax_lse=True, **tree)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    torch.testing.assert_close(lse, expected_lse, atol=0.0001, rtol=0.0001)
    torch.testing.assert_close(actual.cpu().float(), reference, atol=0.006, rtol=0.02)


def test_compressed_sibling_isolation():
    # prefix=2 makes nodes 1 and 2 complete different versions of logical KV 0.
    q, tree, _, _ = make_pool_case(requests=2, prefix=2)
    old = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    tree["cmp_tree_kv"][2].fill_(10)
    new = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    torch.testing.assert_close(old[3], new[3], atol=0, rtol=0)  # [0,1,3]
    torch.testing.assert_close(old[17:], new[17:], atol=0, rtol=0)  # Other request.
    assert not torch.equal(old[5], new[5])  # [0,2,5]


def test_history_pool_ignores_uncommitted_tail():
    q, tree, _, _ = make_pool_case(prefix=31)
    old = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    # Only seven compressed rows belong to history. Poison the physical tail.
    pool, table = tree["cmp_kv"], tree["cmp_block_table"]
    logical = torch.arange(7, 9, device=q.device)
    pool[table[0, logical // 32].long(), logical % 32] = 100
    new = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    torch.testing.assert_close(old, new, atol=0, rtol=0)


@pytest.mark.parametrize("heads,dtype", [(4, torch.float16), (16, torch.bfloat16), (32, torch.bfloat16)])
def test_pool_heads_and_dtype(heads, dtype):
    q, tree, linear, _ = make_pool_case(requests=2, prefix=31, heads=heads, dtype=dtype)
    actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[0]
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)


def test_pool_graph_replay_reads_updated_buffers():
    q, tree, _, _ = make_pool_case(prefix=31)
    for _ in range(3):
        torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    for _ in range(3):
        tree["cmp_tree_kv"].mul_(0.9)
        expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("missing", ["cmp_tree_kv", "tree_ancestors", "tree_prefix"])
def test_pool_inputs_must_be_supplied_together(missing):
    q, tree, _, _ = make_pool_case(prefix=3)
    tree.pop(missing)
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)


def test_c4_pool_long_prefix_and_logical_topk_bounds():
    q, tree, linear, _ = make_pool_case(budget=42, prefix=4095)
    expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[0]
    actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    # Out-of-causal-range indices must be rejected before node-pool addressing.
    tree["cmp_sparse_indices"][:, :, -2:] = 999999
    linear["cmp_sparse_indices"] = tree["cmp_sparse_indices"]
    expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[0]
    actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
