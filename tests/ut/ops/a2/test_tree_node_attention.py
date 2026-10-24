# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.node_attention import node_attention


@pytest.mark.parametrize("ratio,prefix", [(4, 0), (4, 13), (128, 125)])
@pytest.mark.parametrize("requests", [1, 2])
def test_shared_attention_matches_independent_branches(ratio, prefix, requests):
    torch.npu.set_device(0)
    torch.manual_seed(41)
    device = "npu"
    n, h, d, bs, window = 5 * requests, 4, 512, 8, 16
    chains = [[0], [0, 1], [0, 2], [0, 1, 3], [0, 2, 4]]
    chains = [[node + 5 * req for node in chain] for req in range(requests) for chain in chains]
    depths = [len(c) - 1 for c in chains]
    plan = dict(
        n=n,
        max_q=3,
        max_kv=prefix + 3,
        ancestors=torch.tensor([c + [0] * (3 - len(c)) for c in chains], device=device),
        start=torch.full((n,), prefix, dtype=torch.int32, device=device),
        depths=torch.tensor(depths, device=device),
        req_ids=torch.arange(n, device=device) // 5,
    )
    pages = (prefix + 5 + bs - 1) // bs
    table = torch.arange(1, requests * pages + 1, dtype=torch.int32, device=device).view(requests, pages)
    original = torch.randn(requests * pages + 1, bs, 1, d, device=device, dtype=torch.bfloat16) * 0.1
    compressed = torch.randn(requests * pages + 1, bs, 1, d, device=device, dtype=torch.bfloat16) * 0.1
    nodes = torch.randn(n, d, device=device, dtype=torch.bfloat16) * 0.1
    query = torch.randn(n, h, d, device=device, dtype=torch.bfloat16)
    sinks = torch.randn(h, device=device)
    topk = torch.tensor([[[2, 0, 3, -1]]] * n, dtype=torch.int32, device=device) if ratio == 4 else None
    scale = d**-0.5
    actual = node_attention(query, original, compressed, nodes, table, table, plan, topk, sinks, ratio, window, scale)
    oq, cq, nq, q, s = [x.cpu().float() for x in (original, compressed, nodes, query, sinks)]
    for node, chain in enumerate(chains):
        keys = []
        end = prefix + len(chain)
        for pos in range(max(0, end - window), end):
            address = pos if pos < prefix else prefix + chain[pos - prefix] - chain[0]
            keys.append(oq[1 + (node // 5) * pages + address // bs, address % bs, 0])
        chosen = [2, 0, 3, -1] if ratio == 4 else list(range(end // ratio))
        for k in chosen:
            if not 0 <= k < end // ratio:
                continue
            if k < prefix // ratio:
                keys.append(cq[1 + (node // 5) * pages + k // bs, k % bs, 0])
            else:
                ancestor = chain[(k + 1) * ratio - 1 - prefix]
                keys.append(nq[ancestor])
        kv = torch.stack(keys)
        score = q[node] @ kv.T * scale
        probabilities = torch.softmax(torch.cat((score, s[:, None]), 1), 1)[:, :-1]
        expected = probabilities @ kv
        torch.testing.assert_close(actual[node].cpu().float(), expected, atol=0.001, rtol=0.01)


def _native_attention(
    query, original, compressed, nodes, original_table, compressed_table, plan, topk, sinks, ratio, window, scale
):
    n, h, d = query.shape
    device = query.device
    length = plan["max_kv"]
    bs = 32
    ob = (length + bs - 1) // bs
    cb = max(1, (length // ratio + bs - 1) // bs)
    ori = torch.zeros((1 + n * ob, bs, 1, d), dtype=query.dtype, device=device)
    cmp = torch.zeros((1 + n * cb, bs, 1, d), dtype=query.dtype, device=device)
    ot = torch.arange(1, 1 + n * ob, dtype=torch.int32, device=device).view(n, ob)
    ct = torch.arange(1, 1 + n * cb, dtype=torch.int32, device=device).view(n, cb)
    starts, depths, ancestors, reqs = [plan[k].cpu().tolist() for k in ("start", "depths", "ancestors", "req_ids")]
    for node in range(n):
        prefix, depth, req = starts[node], depths[node], reqs[node]
        end = prefix + depth + 1
        pos = torch.arange(max(0, end - window), end, device=device)
        logical = pos.clone()
        branch = pos >= prefix
        selected = plan["ancestors"][node, (pos - prefix).clamp(0, plan["max_q"] - 1)]
        source = torch.where(branch, prefix + selected - ancestors[node][0], pos)
        page = original_table[req, source // original.shape[1]].long()
        ori[ot[node, logical // bs].long(), logical % bs] = original[page, source % original.shape[1]]
        for k in range(end // ratio):
            if k < prefix // ratio:
                page = compressed_table[req, k // compressed.shape[1]].long()
                value = compressed[page, k % compressed.shape[1]]
            else:
                ancestor = ancestors[node][(k + 1) * ratio - 1 - prefix]
                value = nodes[ancestor].reshape(1, d)
            cmp[ct[node, k // bs].long(), k % bs] = value
    seq = (plan["start"] + plan["depths"] + 1).int()
    cu = torch.arange(n + 1, dtype=torch.int32, device=device)
    ocu = torch.cat((seq.new_zeros(1), seq.cumsum(0, dtype=torch.int32)))
    ccu = torch.cat((seq.new_zeros(1), (seq // ratio).cumsum(0, dtype=torch.int32)))
    args = dict(
        cu_seqlens_q=cu,
        cu_seqlens_ori_kv=ocu,
        cu_seqlens_cmp_kv=ccu,
        seqused_kv=seq,
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=window - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )
    meta = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=h,
        num_heads_kv=1,
        head_dim=d,
        seqused_q=cu.new_empty(0),
        batch_size=n,
        max_seqlen_q=1,
        max_seqlen_kv=length,
        ori_topk=0,
        cmp_topk=topk.shape[-1] if topk is not None else 0,
        has_ori_kv=True,
        has_cmp_kv=True,
        device="npu",
        **args,
    )
    out, _ = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
        query,
        ori_kv=ori,
        cmp_kv=cmp,
        ori_block_table=ot,
        cmp_block_table=ct,
        cmp_sparse_indices=topk,
        sinks=sinks,
        metadata=meta,
        softmax_scale=scale,
        **args,
    )
    return out


@pytest.mark.parametrize("ratio,prefix", [(4, 0), (4, 85), (4, 511), (128, 125), (128, 270)])
def test_attention_preserves_native_bf16_probability_rounding(ratio, prefix):
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.npu.set_device(0)
    torch.manual_seed(41)
    n, h, d, bs, window = 5, 8, 512, 32, 128
    chains = [[0], [0, 1], [0, 2], [0, 1, 3], [0, 2, 4]]
    plan = dict(
        n=n,
        max_q=3,
        max_kv=prefix + 3,
        ancestors=torch.tensor([c + [0] * (3 - len(c)) for c in chains], device="npu"),
        start=torch.full((n,), prefix, dtype=torch.int32, device="npu"),
        depths=torch.tensor([len(c) - 1 for c in chains], device="npu"),
        req_ids=torch.zeros(n, dtype=torch.long, device="npu"),
    )
    pages = (prefix + n + bs - 1) // bs
    table = torch.arange(1, pages + 1, dtype=torch.int32, device="npu")[None]
    ori = torch.randn(pages + 1, bs, 1, d, device="npu", dtype=torch.bfloat16)
    cmp = torch.randn_like(ori)
    nodes = torch.randn(n, d, device="npu", dtype=torch.bfloat16)
    q = torch.randn(n, h, d, device="npu", dtype=torch.bfloat16)
    sinks = torch.randn(h, device="npu")
    topk = torch.arange(512, device="npu", dtype=torch.int32).expand(n, 512).contiguous()
    topk = torch.where(topk < (prefix + plan["depths"] + 1)[:, None] // ratio, topk, -1)[:, None]
    if ratio == 128:
        topk = None
    args = (q, ori, cmp, nodes, table, table, plan, topk, sinks, ratio, window, d**-0.5)
    expected = _native_attention(*args)
    actual = node_attention(*args)
    difference = actual.float() - expected.float()
    relative_l2 = difference.norm() / expected.float().norm()
    assert relative_l2.item() < 0.0003
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
