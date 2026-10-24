# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full C128 visibility: compact original pages and untruncated compressed history."""

import pytest
import torch

from tests.ut.ops.a2.test_tree_native_attention import native_ops  # noqa: F401


def make_c128_case(requests=1, budget=16, prefix=127, heads=8, dtype=torch.bfloat16, inactive_last=False):
    """CPU-only fixture construction; H2D is explicit at the operator boundary."""
    torch.manual_seed(128)
    prefixes = [prefix] * requests if isinstance(prefix, int) else prefix
    assert len(prefixes) == requests
    count, block, dim, window, ratio = budget + 1, 32, 512, 128, 128
    chains = []
    for node in range(count):
        chain = []
        while node >= 0:
            chain.append(node)
            node = (node - 1) // 2
        chains.append(chain[::-1])
    depth, n = max(map(len, chains)), requests * count
    max_len = max(prefixes) + depth
    max_pages = (max(prefixes) + count + block - 1) // block
    # Only pages actually read by the sliding window need storage, even at 131K.
    ori_table = torch.zeros(requests, max_pages, dtype=torch.int32)
    ori_pages = [torch.zeros(block, 1, dim, dtype=dtype)]
    for req, start in enumerate(prefixes):
        for page in range(max(0, start + 1 - window) // block, (start + count + block - 1) // block):
            ori_table[req, page] = len(ori_pages)
            ori_pages.append(torch.randn(block, 1, dim, dtype=dtype))
    original = torch.stack(ori_pages)
    cmp_pages = max(1, (max_len // ratio + block - 1) // block)
    cmp_table = torch.arange(1, 1 + requests * cmp_pages, dtype=torch.int32).view(requests, cmp_pages)
    compressed = torch.randn(1 + requests * cmp_pages, block, 1, dim, dtype=dtype)
    nodes, query = torch.randn(n, dim, dtype=dtype), torch.randn(n, heads, dim, dtype=dtype)
    sinks = torch.randn(heads)
    ancestors = torch.full((n, depth), -1, dtype=torch.int64)
    slots = torch.full((n, 1, window), -1, dtype=torch.int32)
    linear_table = torch.zeros(n, max_pages, dtype=torch.int32)
    linear_pages = [torch.zeros(block, 1, dim, dtype=dtype)]
    linear_cmp_table = torch.arange(1, 1 + n * cmp_pages, dtype=torch.int32).view(n, cmp_pages)
    linear_cmp = torch.zeros(1 + n * cmp_pages, block, 1, dim, dtype=dtype)
    lengths, references = [], []
    for req, start in enumerate(prefixes):
        for local, chain in enumerate(chains):
            row, end = req * count + local, start + len(chain)
            lengths.append(end)
            ancestors[row, : len(chain)] = torch.tensor([req * count + v for v in chain])
            keys = []
            for index, pos in enumerate(range(max(0, end - window), end)):
                source = pos if pos < start else start + chain[pos - start]
                slot = ori_table[req, source // block] * block + source % block
                slots[row, 0, index] = slot
                value = original.view(-1, dim)[slot]
                keys.append(value)
                page = pos // block
                if linear_table[row, page] == 0:
                    linear_table[row, page] = len(linear_pages)
                    linear_pages.append(torch.zeros(block, 1, dim, dtype=dtype))
                linear_pages[linear_table[row, page]][pos % block, 0] = value
            for k in range(end // ratio):
                value = (
                    compressed[cmp_table[req, k // block], k % block, 0]
                    if k < start // ratio
                    else nodes[req * count + chain[(k + 1) * ratio - 1 - start]]
                )
                keys.append(value)
                linear_cmp[linear_cmp_table[row, k // block], k % block, 0] = value
            keys = torch.stack(keys).float()
            scores = query[row].float() @ keys.T * dim**-0.5
            weights = torch.softmax(torch.cat((scores, sinks[:, None]), -1), -1)[:, :-1]
            references.append(weights @ keys)
    if inactive_last:
        lengths[-1] = 0
        slots[-1].fill_(-1)
        references[-1].zero_()
    seq = torch.tensor(lengths, dtype=torch.int32)
    cu = torch.arange(n + 1, dtype=torch.int32)
    args = dict(
        cu_seqlens_q=cu.npu(),
        cu_seqlens_ori_kv=torch.cat((seq.new_zeros(1), seq.cumsum(0, dtype=torch.int32))).npu(),
        cu_seqlens_cmp_kv=torch.cat((seq.new_zeros(1), (seq // ratio).cumsum(0, dtype=torch.int32))).npu(),
        seqused_kv=seq.npu(),
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=window - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )
    metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=heads,
        num_heads_kv=1,
        head_dim=dim,
        seqused_q=cu.new_empty(0).npu(),
        batch_size=n,
        max_seqlen_q=1,
        max_seqlen_kv=max_len,
        ori_topk=0,
        cmp_topk=0,
        has_ori_kv=True,
        has_cmp_kv=True,
        device="npu",
        **args,
    )
    common = dict(metadata=metadata, sinks=sinks.npu(), softmax_scale=dim**-0.5, **args)
    tree = dict(
        ori_kv=original.npu(),
        ori_block_table=ori_table.repeat_interleave(count, 0).npu(),
        ori_sparse_indices=slots.npu(),
        cmp_kv=compressed.npu(),
        cmp_block_table=cmp_table.repeat_interleave(count, 0).npu(),
        cmp_tree_kv=nodes.npu(),
        tree_ancestors=ancestors.npu(),
        tree_prefix=torch.tensor(prefixes, dtype=torch.int32).repeat_interleave(count).npu(),
        **common,
    )
    linear = dict(
        ori_kv=torch.stack(linear_pages).npu(),
        ori_block_table=linear_table.npu(),
        cmp_kv=linear_cmp.npu(),
        cmp_block_table=linear_cmp_table.npu(),
        **common,
    )
    return query.npu(), tree, linear, torch.stack(references)


@pytest.mark.parametrize("requests", [1, 2, 4, 8])
@pytest.mark.parametrize("budget,prefix", [(16, 0), (16, 125), (16, 127), (42, 128), (42, 511)])
def test_c128_complete_visibility(requests, budget, prefix):
    q, tree, linear, reference = make_c128_case(requests, budget, prefix)
    expected, expected_lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, return_softmax_lse=True, **linear)
    actual, lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, return_softmax_lse=True, **tree)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    torch.testing.assert_close(lse, expected_lse, atol=0.0001, rtol=0.0001)
    torch.testing.assert_close(actual.cpu().float(), reference, atol=0.006, rtol=0.02)


@pytest.mark.parametrize("completed", [513, 1025])
def test_c128_history_exceeds_fixed_topk_capacity(completed):
    q, tree, linear, reference = make_c128_case(budget=2, prefix=completed * 128 - 1)
    assert "cmp_sparse_indices" not in tree
    actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[0]
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    torch.testing.assert_close(actual.cpu().float(), reference, atol=0.006, rtol=0.02)


def test_c128_siblings_and_different_prefixes():
    q, tree, linear, _ = make_c128_case(requests=4, prefix=[126, 127, 128, 255])
    expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[0]
    old = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    torch.testing.assert_close(old, expected, atol=0.001, rtol=0.01)
    tree["cmp_tree_kv"][2].fill_(10)
    new = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    torch.testing.assert_close(new[3], old[3], atol=0, rtol=0)
    torch.testing.assert_close(new[17:], old[17:], atol=0, rtol=0)
    assert not torch.equal(new[5], old[5])


@pytest.mark.parametrize("heads,dtype", [(4, torch.float16), (16, torch.bfloat16), (32, torch.bfloat16)])
def test_c128_heads_dtype_and_padding(heads, dtype):
    q, tree, linear, _ = make_c128_case(requests=2, prefix=127, heads=heads, dtype=dtype, inactive_last=True)
    actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[0]
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    torch.testing.assert_close(actual[-1], torch.zeros_like(actual[-1]), atol=0, rtol=0)


def test_c128_graph_replay_reads_updated_nodes():
    q, tree, _, _ = make_c128_case(prefix=127)
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
