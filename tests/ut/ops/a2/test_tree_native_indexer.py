# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
"""Native tree indexer: branch isolation, native parity and live FULL inputs."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401


def make_case(bs=1, budget=16, prefix=127, topk=16):
    torch.npu.set_device(0)
    torch.manual_seed(123)
    chains = [[0]]
    for i in range(1, budget + 1):
        chains.append(chains[(i - 1) // 2] + [i])
    width = max(map(len, chains))
    n = bs * (budget + 1)
    ancestors, depths, requests, prefixes = [], [], [], []
    for b in range(bs):
        for chain in chains:
            ancestors.append([b * (budget + 1) + i for i in chain] + [-1] * (width - len(chain)))
            depths.append(len(chain) - 1)
            requests.append(b)
            prefixes.append(prefix + b % 3)
    plan = dict(
        n=n,
        max_q=width,
        max_kv=prefix + width + 2,
        ancestors=torch.tensor(ancestors, dtype=torch.int64, device="npu"),
        depths=torch.tensor(depths, dtype=torch.int64, device="npu"),
        req_ids=torch.tensor(requests, dtype=torch.int64, device="npu"),
        start=torch.tensor(prefixes, dtype=torch.int32, device="npu"),
        query_cu=torch.arange(n + 1, dtype=torch.int32, device="npu"),
    )
    block = 16
    pages = (plan["max_kv"] // 4 + block - 1) // block + 1
    storage = torch.randint(-32, 32, (1 + bs * pages, block + 3, 1, 128), dtype=torch.int8, device="npu")
    scales = (torch.rand(1 + bs * pages, block + 3, 1, 1, device="npu") * 0.5 + 0.1).half()
    table = torch.arange(1, 1 + bs * pages, dtype=torch.int32, device="npu").reshape(bs, pages).flip(1)
    return dict(
        plan=plan,
        query=torch.randint(-32, 32, (n, 64, 128), dtype=torch.int8, device="npu"),
        weights=(torch.randn(n, 64, device="npu") * 0.2).half(),
        query_scale=(torch.rand(n, 64, device="npu") * 0.5 + 0.1).half(),
        history=storage[:, :block],
        history_scale=scales[:, :block],
        nodes=torch.randint(-32, 32, (n, 128), dtype=torch.int8, device="npu"),
        node_scale=(torch.rand(n, device="npu") * 0.5 + 0.1).half(),
        table=table,
        topk=topk,
    )


def metadata(case):
    p = case["plan"]
    seq = (p["start"] + p["depths"] + 1).int()
    lengths, residual = seq // 4, seq % 4
    result = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=64,
        num_heads_k=1,
        head_dim=128,
        topk=case["topk"],
        quant_mode=2,
        cu_seqlens_q=p["query_cu"],
        seqused_k=lengths,
        cmp_residual_k=residual,
        batch_size=p["n"],
        max_seqlen_q=1,
        max_seqlen_k=max(p["max_kv"] // 4, 1),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        device="npu:0",
    )
    return lengths, residual, result


def native(case, meta=None):
    p = case["plan"]
    lengths, residual, schedule = metadata(case) if meta is None else meta
    return torch.ops._C_ascend.quant_lightning_indexer_tree(
        case["query"],
        case["history"],
        case["weights"],
        case["query_scale"],
        case["history_scale"],
        case["nodes"],
        case["node_scale"],
        p["ancestors"],
        p["start"],
        p["req_ids"],
        case["table"],
        p["query_cu"],
        lengths,
        residual,
        schedule,
        case["topk"],
    )


def expand(case):
    """Reference-only KV materialization; excluded from the new native operator."""
    p = case["plan"]
    block = case["history"].shape[1]
    capacity = case["table"].shape[1] * block
    positions = torch.arange(capacity, device="npu")[None, :].expand(p["n"], -1)
    history_ids = case["table"][p["req_ids"][:, None], positions // block].long()
    old = case["history"][history_ids, positions % block, 0]
    old_scale = case["history_scale"][history_ids, positions % block, 0, 0]
    steps = (positions + 1) * 4 - 1 - p["start"][:, None]
    ancestors = p["ancestors"].gather(1, steps.clamp(0, p["max_q"] - 1))
    tree = positions >= p["start"][:, None] // 4
    new = case["nodes"][ancestors.clamp_min(0)]
    new_scale = case["node_scale"][ancestors.clamp_min(0)]
    keys = torch.where(tree[..., None], new, old).reshape(-1, block, 1, 128)
    scales = torch.where(tree, new_scale, old_scale).reshape(-1, block, 1, 1)
    table = torch.arange(keys.shape[0], dtype=torch.int32, device="npu").reshape(p["n"], -1)
    return keys, scales, table


def linear(case, expanded=None, meta=None):
    keys, scales, table = expand(case) if expanded is None else expanded
    lengths, residual, schedule = metadata(case) if meta is None else meta
    return torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query=case["query"],
        key=keys,
        weights=case["weights"],
        query_dequant_scale=case["query_scale"],
        key_dequant_scale=scales.squeeze(-1),
        topk=case["topk"],
        quant_mode=2,
        cu_seqlens_q=case["plan"]["query_cu"],
        seqused_k=lengths,
        cmp_residual_k=residual,
        block_table=table,
        metadata=schedule,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        return_value=0,
    )[0]


def assert_reference(case, output, *, check_ranking=True):
    # D2H at the UT boundary: independently gather each visible branch on CPU.
    c = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in case.items()}
    p = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in case["plan"].items()}
    actual = output.cpu()
    max_score_gap = 0.0
    for row in range(p["n"]):
        prefix, depth, req = (int(p[k][row]) for k in ("start", "depths", "req_ids"))
        length = (prefix + depth + 1) // 4
        valid_positions, keys, scales = [], [], []
        for pos in range(length):
            if pos < prefix // 4:
                page = c["table"][req, pos // c["history"].shape[1]]
                keys.append(c["history"][page, pos % c["history"].shape[1], 0])
                scales.append(c["history_scale"][page, pos % c["history"].shape[1], 0, 0])
            else:
                node = int(p["ancestors"][row, (pos + 1) * 4 - 1 - prefix])
                if node < 0:
                    continue
                keys.append(c["nodes"][node])
                scales.append(c["node_scale"][node])
            valid_positions.append(pos)
        ids = actual[row, 0]
        selected = ids[ids >= 0].long()
        assert len(selected) == min(case["topk"], len(keys))
        assert len(selected.unique()) == len(selected)
        assert set(selected.tolist()).issubset(valid_positions)
        if not keys:
            assert (ids == -1).all()
            continue
        dot = c["query"][row].int() @ torch.stack(keys).int().T
        relu = (dot.float() / 1024).clamp_min(0).half().float()
        factors = (c["weights"][row] * c["query_scale"][row]).float()
        scores = (factors[:, None] * relu).sum(0) * torch.stack(scales).float()
        ranked = scores.sort(descending=True, stable=True).values[: len(selected)]
        chosen = scores[torch.tensor([valid_positions.index(int(i)) for i in selected])]
        # Native Cube reduction and CPU sum can differ by FP32 accumulation order.
        chosen = chosen.sort(descending=True).values
        max_score_gap = max(max_score_gap, float((chosen - ranked).abs().max()))
        if check_ranking:
            torch.testing.assert_close(chosen, ranked, atol=2e-5, rtol=2e-5)
    return max_score_gap


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("budget", [16, 42])
@pytest.mark.parametrize("prefix,topk", [(0, 16), (13, 2), (127, 16), (2303, 512)])
def test_tree_indexer_matches_native_and_reference(bs, budget, prefix, topk):
    case = make_case(bs=bs, budget=budget, prefix=prefix, topk=topk)
    before = [case[k].clone() for k in ("history", "history_scale", "nodes", "node_scale")]
    actual = native(case)
    expected = linear(case)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert_reference(case, actual)
    for key, saved in zip(("history", "history_scale", "nodes", "node_scale"), before):
        torch.testing.assert_close(case[key], saved, atol=0, rtol=0)


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
def test_tree_indexer_absent_ancestor(bs):
    case = make_case(bs=bs, prefix=13, topk=16)
    case["plan"]["ancestors"][:, 2] = -1
    assert_reference(case, native(case))


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("prefix", [127, 2303])
def test_tree_indexer_full_replay(bs, prefix):
    case = make_case(bs=bs, prefix=prefix, topk=16)
    native(case)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = native(case)
    for delta in (1, -1):
        case["plan"]["start"].add_(delta)
        case["nodes"].neg_()
        case["query"].neg_()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, linear(case), atol=0, rtol=0)
        assert_reference(case, captured)


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("prefix,topk", [(127, 512), (2303, 512)])
def test_tree_indexer_adapter_full_replay(bs, prefix, topk):
    from vllm_ascend.ops.triton.spec_decode.tree.native_indexer import native_indexer
    from vllm_ascend.ops.triton.spec_decode.tree.node_indexer import node_indexer

    case = make_case(bs=bs, prefix=prefix, topk=topk)
    op_args = tuple(
        case[k]
        for k in (
            "query",
            "weights",
            "query_scale",
            "history",
            "history_scale",
            "nodes",
            "node_scale",
            "table",
            "plan",
            "topk",
        )
    )

    def run():
        case["plan"]["query_metadata"] = {}
        first = native_indexer(*op_args)
        # A second layer shares device metadata without freezing it across replay.
        second = native_indexer(*op_args)
        return first, second

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        first, second = run()
    for delta in (1, -1):
        case["plan"]["start"].add_(delta)
        case["query"].neg_()
        graph.replay()
        torch.npu.synchronize()
        expected = node_indexer(*op_args) if prefix == 127 else linear(case)
        torch.testing.assert_close(first, expected, atol=0, rtol=0)
        torch.testing.assert_close(second, expected, atol=0, rtol=0)


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
def test_tree_indexer_model_dispatch(bs):
    from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerOps
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

    case = make_case(bs=bs, prefix=2303, topk=512)
    plan = case["plan"]
    plan["node_kv"] = {
        cp._key(case["history"]): dict(original=case["history"], rows=case["nodes"].unsqueeze(1), table=case["table"]),
        cp._key(case["history_scale"]): dict(original=case["history_scale"], rows=case["node_scale"].view(-1, 1, 1)),
    }
    config = SimpleNamespace(tree_spec_config=SimpleNamespace(indexer_backend="native"))
    ops = AscendIndexerOps(case["topk"])

    def run():
        plan["query_metadata"] = {}
        return ops.select_topk(
            case["query"], case["weights"], case["query_scale"], case["history"], case["history_scale"], None
        )

    with (
        patch.object(cp, "_LAYOUT", {}),
        patch.object(cp, "_PLAN", plan),
        patch("vllm_ascend.ascend_config.get_ascend_config", return_value=config),
    ):
        torch.testing.assert_close(run(), linear(case), atol=0, rtol=0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
        case["plan"]["start"].add_(1)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, linear(case), atol=0, rtol=0)
