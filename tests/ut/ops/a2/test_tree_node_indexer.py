# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.node_indexer import node_indexer


@pytest.mark.parametrize("topk", [2, 16])
def test_sparse_indexer_excludes_siblings(topk):
    torch.npu.set_device(0)
    torch.manual_seed(7)
    device = "npu"
    n, h, d, bs, prefix = 5, 4, 128, 4, 13
    chains = [[0], [0, 1], [0, 2], [0, 1, 3], [0, 2, 4]]
    depth = [len(c) - 1 for c in chains]
    plan = dict(
        n=n,
        max_q=3,
        max_kv=prefix + 3,
        ancestors=torch.tensor([c + [0] * (3 - len(c)) for c in chains], device=device),
        start=torch.full((n,), prefix, dtype=torch.int32, device=device),
        depths=torch.tensor(depth, device=device),
        req_ids=torch.zeros(n, dtype=torch.long, device=device),
    )
    table = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
    hist = torch.randint(-127, 128, (3, bs, 1, d), dtype=torch.int8, device=device)
    nodes = torch.randint(-127, 128, (n, d), dtype=torch.int8, device=device)
    hs = torch.rand(3, bs, 1, 1, device=device).half()
    ns = torch.rand(n, device=device).half()
    query = torch.randint(-127, 128, (n, h, d), dtype=torch.int8, device=device)
    weights = torch.randn(n, h, device=device).half()
    qs = torch.rand(n, h, device=device).half()
    actual = node_indexer(query, weights, qs, hist, hs, nodes, ns, table, plan, topk).cpu()
    q, w, s, k, ks, nk, nks = [x.cpu() for x in (query, weights, qs, hist, hs, nodes, ns)]
    for node, chain in enumerate(chains):
        length = (prefix + len(chain)) // 4
        keys = []
        scales = []
        for pos in range(length):
            if pos < prefix // 4:
                keys.append(k[1 + pos // bs, pos % bs, 0])
                scales.append(ks[1 + pos // bs, pos % bs, 0, 0])
            else:
                ancestor = chain[(pos + 1) * 4 - 1 - prefix]
                keys.append(nk[ancestor])
                scales.append(nks[ancestor])
        dot = q[node].int() @ torch.stack(keys).int().T
        relu = (dot.float() / 1024).clamp_min(0).half().float()
        score = ((w[node] * s[node]).float()[:, None] * relu).sum(0) * torch.stack(scales).float()
        expected = (
            score.argsort(descending=True, stable=True)[:topk].int()
            if topk < length
            else torch.arange(length, dtype=torch.int32)
        )
        torch.testing.assert_close(actual[node, 0, : len(expected)], expected, rtol=0, atol=0)
        assert (actual[node, 0, len(expected) :] == -1).all()
