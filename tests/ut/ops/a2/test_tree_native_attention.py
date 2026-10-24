# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native tree C4 prototype: compressed KV is fixed, original KV uses slots.

Fixtures and reference gathering run on CPU. Only indices/metadata are prepared
for the operator; its original KV pool contains each history/tree row once.
Load the experimental binding before pytest (see the operator README).
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env


@pytest.fixture(scope="module", autouse=True)
def native_ops():
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.npu.set_device(0)
    assert hasattr(torch.ops._C_ascend, "npu_sparse_attn_sharedkv_tree"), "Load the tree binding before pytest"


def make_case(requests=1, budget=16, prefix=127, heads=8, window=128, dtype=torch.bfloat16, inactive_last=False):
    """Binary tree, one logical query per node, shuffled physical KV pages."""
    torch.manual_seed(20261024)
    count, dim, block = budget + 1, 512, 32
    local_chains = []
    for node in range(count):
        chain, parent = [], node
        while parent >= 0:
            chain.append(parent)
            parent = (parent - 1) // 2
        local_chains.append(chain[::-1])
    depth = max(map(len, local_chains))
    n = requests * count
    pages = (prefix + count + block - 1) // block
    physical = torch.randperm(requests * pages).view(requests, pages).int() + 1
    pool = torch.randn(1 + requests * pages, block, 1, dim, dtype=dtype)
    cmp_pages = max(1, (prefix + depth + 4 * block - 1) // (4 * block))
    compressed = torch.randn(1 + requests * cmp_pages, block, 1, dim, dtype=dtype)
    cmp_table = torch.arange(1, 1 + requests * cmp_pages, dtype=torch.int32).view(requests, cmp_pages)
    linear_pages = (prefix + depth + block - 1) // block
    linear = torch.zeros(1 + n * linear_pages, block, 1, dim, dtype=dtype)
    linear_table = torch.arange(1, 1 + n * linear_pages, dtype=torch.int32).view(n, linear_pages)
    indices = torch.full((n, 1, 128), -1, dtype=torch.int32)
    topk = torch.full((n, 1, 512), -1, dtype=torch.int32)
    lengths, references = [], []
    q = torch.randn(n, heads, dim, dtype=dtype)
    sinks = torch.randn(heads)
    for req in range(requests):
        for node, chain in enumerate(local_chains):
            row = req * count + node
            end = prefix + len(chain)
            lengths.append(end)
            positions = list(range(max(0, end - window), end))
            addresses = [p if p < prefix else prefix + chain[p - prefix] for p in positions]
            slots = [int(physical[req, p // block]) * block + p % block for p in addresses]
            indices[row, 0, : len(slots)] = torch.tensor(slots, dtype=torch.int32)
            keys = pool.view(-1, dim)[slots]
            for pos, key in zip(positions, keys):
                linear[linear_table[row, pos // block], pos % block, 0] = key
            chosen = list(range(end // 4))[::-1][:512]
            topk[row, 0, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
            ck = [compressed[cmp_table[req, k // block], k % block, 0] for k in chosen]
            if ck:
                keys = torch.cat((keys, torch.stack(ck)))
            scores = q[row].float() @ keys.float().T * dim**-0.5
            prob = torch.softmax(torch.cat((scores, sinks[:, None]), dim=-1), dim=-1)[:, :-1]
            references.append(prob @ keys.float())
    if inactive_last:
        lengths[-1] = 0
        indices[-1].fill_(-1)
        topk[-1].fill_(-1)
        references[-1].zero_()
    seq = torch.tensor(lengths, dtype=torch.int32)
    cu = torch.arange(n + 1, dtype=torch.int32)
    args = dict(
        cu_seqlens_q=cu.npu(),
        cu_seqlens_ori_kv=torch.cat((seq.new_zeros(1), seq.cumsum(0, dtype=torch.int32))).npu(),
        cu_seqlens_cmp_kv=torch.cat((seq.new_zeros(1), (seq // 4).cumsum(0, dtype=torch.int32))).npu(),
        seqused_kv=seq.npu(),
        cmp_ratio=4,
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
        max_seqlen_kv=max(lengths),
        ori_topk=0,
        cmp_topk=512,
        has_ori_kv=True,
        has_cmp_kv=True,
        device="npu",
        **args,
    )
    common = dict(
        cmp_kv=compressed.npu(),
        cmp_block_table=cmp_table.repeat_interleave(count, 0).npu(),
        cmp_sparse_indices=topk.npu(),
        sinks=sinks.npu(),
        metadata=metadata,
        softmax_scale=dim**-0.5,
        **args,
    )
    return (
        q.npu(),
        dict(
            ori_kv=pool.npu(),
            ori_block_table=physical.repeat_interleave(count, 0).npu(),
            ori_sparse_indices=indices.npu(),
            **common,
        ),
        dict(ori_kv=linear.npu(), ori_block_table=linear_table.npu(), **common),
        torch.stack(references),
    )


def test_isolated_operator_matches_original():
    q, _, linear, _ = make_case(prefix=13)
    old = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[0]
    new = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **linear)[0]
    torch.testing.assert_close(new, old, atol=0, rtol=0)


@pytest.mark.parametrize("requests", [1, 2, 4, 8])
@pytest.mark.parametrize("budget,prefix", [(16, 0), (16, 3), (16, 31), (42, 127), (42, 511)])
def test_indexed_c4(requests, budget, prefix):
    q, tree, linear, reference = make_case(requests=requests, budget=budget, prefix=prefix)
    expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[0]
    actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    # Independent FP32 softmax also detects a shared address/mask error.
    torch.testing.assert_close(actual.cpu().float(), reference, atol=0.006, rtol=0.02)


def test_tree_operator_meta_shapes():
    query = torch.empty(17, 8, 512, device="meta", dtype=torch.bfloat16)
    for return_lse in (False, True):
        output, lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(
            query, layout_q="TND", return_softmax_lse=return_lse
        )
        assert output.shape == query.shape and output.dtype == query.dtype
        assert lse.shape == ((17, 8, 1) if return_lse else (0,))
        assert lse.dtype == torch.float32


@pytest.mark.parametrize(
    "heads,dtype,window", [(4, torch.float16, 1), (16, torch.bfloat16, 16), (32, torch.bfloat16, 128)]
)
def test_indexed_c4_heads_dtype_and_sink(heads, dtype, window):
    q, tree, linear, _ = make_case(requests=2, prefix=127, heads=heads, dtype=dtype, window=window)
    tree["ori_sparse_indices"].masked_fill_(tree["ori_sparse_indices"] < 0, -12345)
    expected, expected_lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, return_softmax_lse=True, **linear)
    actual, actual_lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, return_softmax_lse=True, **tree)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    torch.testing.assert_close(actual_lse, expected_lse, atol=0.0001, rtol=0.0001)


def test_c4_sibling_kv_is_invisible():
    q, tree, _, _ = make_case(prefix=31)
    first = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    # Node 2 is a sibling of node 1: query 3 has ancestors [0, 1, 3],
    # while query 5 has ancestors [0, 2, 5]. Only the latter may change.
    slot = tree["ori_sparse_indices"][2, 0, 32].long()
    tree["ori_kv"].view(-1, 512)[slot] = 100
    second = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    torch.testing.assert_close(first[3], second[3], atol=0, rtol=0)
    assert not torch.equal(first[5], second[5])


def test_c4_inactive_padding_row():
    q, tree, linear, reference = make_case(requests=2, prefix=3, inactive_last=True)
    actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[0]
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
    torch.testing.assert_close(actual[-1].cpu().float(), reference[-1], atol=0, rtol=0)


def test_c4_graph_replay_reads_live_slots_and_kv():
    q, tree, _, _ = make_case(requests=2, prefix=127)
    run = lambda: torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    with torch.inference_mode():
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
        for _ in range(3):
            tree["ori_kv"].mul_(0.9)
            tree["ori_sparse_indices"].copy_(tree["ori_sparse_indices"].roll(1, 0))
            expected = run()
            graph.replay()
            torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_c4_topk_at_long_prefix():
    q, tree, linear, _ = make_case(requests=1, budget=42, prefix=4095)
    actual = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[0]
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
