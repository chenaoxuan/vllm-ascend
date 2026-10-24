# SPDX-License-Identifier: Apache-2.0
"""Tree-only aggregation: independent reference, branch masks, dynamic FULL replay."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.node_compressor import aggregate_nodes


def make_case(bs, ratio, dim, prefix, budget=16):
    torch.npu.set_device(0)
    torch.manual_seed(42)
    n_each = budget + 1
    chains = [[0]]
    for i in range(1, n_each):
        chains.append(chains[(i - 1) // 2] + [i])
    width = max(map(len, chains))
    n = bs * n_each
    ancestors, depths, requests, prefixes = [], [], [], []
    for b in range(bs):
        for chain in chains:
            ancestors.append([b * n_each + i for i in chain] + [-1] * (width - len(chain)))
            depths.append(len(chain) - 1)
            requests.append(b)
            prefixes.append(prefix + b % 2)
    plan = dict(
        n=n,
        max_q=width,
        ancestors=torch.tensor(ancestors, dtype=torch.int64, device="npu"),
        depths=torch.tensor(depths, dtype=torch.int64, device="npu"),
        req_ids=torch.tensor(requests, dtype=torch.int64, device="npu"),
        start=torch.tensor(prefixes, dtype=torch.int32, device="npu"),
    )
    coeff = 2 if ratio == 4 else 1
    block = 16
    pages = (prefix + width + block + 2) // block
    # Nontrivial page order and padded page stride catch linear-cache assumptions.
    storage = torch.randn(1 + bs * pages, block + 3, 2 * coeff * dim, device="npu") * 0.3
    state = storage[:, :block]
    table = torch.arange(1, 1 + bs * pages, dtype=torch.int32, device="npu").reshape(bs, pages).flip(1)
    nodes = torch.randn(n + 1, 2 * coeff * dim, device="npu") * 0.3
    norm = 1 + torch.randn(dim, device="npu") * 0.1
    angle = torch.randn(prefix + width + 8, 64, device="npu")
    cos, sin = angle.cos(), angle.sin()
    return dict(state=state, nodes=nodes, table=table, plan=plan, norm=norm, cos=cos, sin=sin, ratio=ratio, dim=dim)


def native(case):
    p = case["plan"]
    return torch.ops._C_ascend.compressor_tree(
        case["state"],
        case["nodes"],
        case["norm"],
        case["cos"],
        case["sin"],
        case["table"],
        p["start"],
        p["ancestors"],
        p["depths"],
        p["req_ids"],
        case["ratio"],
        64,
        1e-6,
    )


def triton(case):
    return aggregate_nodes(
        case["state"],
        case["nodes"],
        case["table"],
        case["plan"],
        case["norm"],
        case["cos"],
        case["sin"],
        case["ratio"],
        case["dim"],
        64,
        1e-6,
        torch.bfloat16,
    )


def projection(case, hidden_dim=4096, return_pair=False):
    """Construct native node projection once; return its graph-safe invocation."""
    n, d, r = case["plan"]["n"], case["dim"], case["ratio"]
    c = 2 if r == 4 else 1
    x = torch.randn(n, hidden_dim, dtype=torch.bfloat16, device="npu") * 0.1
    wkv = torch.randn(c * d, hidden_dim, dtype=torch.bfloat16, device="npu") * 0.1
    wgate = torch.randn_like(wkv) * 0.1
    ape = torch.randn(r, c * d, device="npu") * 0.1
    residue = ((case["plan"]["start"] + case["plan"]["depths"]) % r).int()
    table = torch.zeros(n, r, dtype=torch.int32, device="npu")
    table.scatter_(1, residue.long()[:, None], torch.arange(1, n + 1, dtype=torch.int32, device="npu")[:, None])
    cu = torch.arange(n + 1, dtype=torch.int32, device="npu")
    cos, sin = torch.ones(n, 64, device="npu"), torch.zeros(n, 64, device="npu")

    def run():
        case["nodes"].zero_()
        return torch.ops._C_ascend.compressor(
            x,
            wkv,
            wgate,
            case["nodes"].view(n + 1, 1, 2 * c * d),
            ape,
            case["norm"],
            sin,
            cos,
            state_block_table=table,
            cu_seqlens=cu,
            seqused=None,
            start_pos=residue,
            rope_head_dim=64,
            cmp_ratio=r,
            coff=c,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=1,
        )

    def run_tree():
        # Only the reserved sentinel needs clearing; every real node is written.
        case["nodes"][:1].zero_()
        torch.ops._C_ascend.compressor_tree_project(x, wkv, wgate, case["nodes"], ape, residue, cu)

    def update():
        x.mul_(0.9)
        residue.copy_((residue + 1) % r)
        table.zero_().scatter_(
            1, residue.long()[:, None], torch.arange(1, n + 1, dtype=torch.int32, device="npu")[:, None]
        )

    return (run, run_tree, update) if return_pair else run


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio,dim", [(4, 128), (4, 512), (128, 512)])
def test_native_projected_state(bs, ratio, dim):
    case = make_case(bs, ratio, dim, 4 * ratio - 3)
    project = projection(case)
    project()
    before = case["nodes"].clone()
    out = native(case)
    torch.testing.assert_close(out, triton(case), atol=0.001, rtol=0.016)
    torch.testing.assert_close(out.cpu(), reference(case), atol=0.001, rtol=0.016)
    assert torch.equal(before, case["nodes"])


def reference(case):
    # D2H: independent CPU oracle at the explicit UT boundary, never production.
    s, nd, table, norm, cos, sin = [case[k].cpu() for k in ("state", "nodes", "table", "norm", "cos", "sin")]
    p = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in case["plan"].items()}
    n, d, r = p["n"], case["dim"], case["ratio"]
    c = 2 if r == 4 else 1
    out = torch.zeros(n, d, dtype=torch.float64)
    for node in range(n):
        start, depth, req = int(p["start"][node]), int(p["depths"][node]), int(p["req_ids"][node])
        pos = start + depth
        if (pos + 1) % r:
            continue
        keys, scores = [], []
        for half in range(c):
            for position in range(pos + 1 - r * c + half * r, pos + 1 - r * c + (half + 1) * r):
                if position < 0:
                    continue
                if position < start:
                    row = s[table[req, position // s.shape[1]], position % s.shape[1]]
                else:
                    row = nd[p["ancestors"][node, position - start] + 1]
                keys.append(row[half * d : (half + 1) * d].double())
                scores.append(row[(c + half) * d : (c + half + 1) * d].double())
        x = (torch.stack(scores).softmax(0) * torch.stack(keys)).sum(0)
        x = x / (x.square().mean() + 1e-6).sqrt() * norm.double()
        rotary = x[-64:].clone()
        paired = rotary[torch.arange(64) ^ 1] * torch.where(torch.arange(64) % 2 == 0, -1, 1)
        x[-64:] = rotary * cos[pos + 1 - r] + paired * sin[pos + 1 - r]
        out[node] = x
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio,dim", [(4, 128), (4, 512), (128, 512)])
@pytest.mark.parametrize("prefix_kind", ["zero", "boundary", "history"])
def test_tree_compressor(bs, ratio, dim, prefix_kind):
    prefix = {"zero": 0, "boundary": ratio - 3, "history": 4 * ratio - 1}[prefix_kind]
    case = make_case(bs, ratio, dim, prefix)
    state_before, nodes_before = case["state"].clone(), case["nodes"].clone()
    out = native(case)
    torch.testing.assert_close(out, triton(case), atol=0.001, rtol=0.016)
    torch.testing.assert_close(out.cpu(), reference(case), atol=0.001, rtol=0.016)
    assert torch.equal(case["state"], state_before)
    assert torch.equal(case["nodes"], nodes_before)


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio,dim", [(4, 128), (4, 512), (128, 512)])
def test_tree_compressor_full_replay(bs, ratio, dim):
    case = make_case(bs, ratio, dim, 4 * ratio - 3)
    native(case)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = native(case)
    for shift in (1, -1):
        case["plan"]["start"].add_(shift)
        case["nodes"].mul_(0.9)
        graph.replay()
        torch.testing.assert_close(out, native(case), atol=0, rtol=0)
        torch.testing.assert_close(out.cpu(), reference(case), atol=0.001, rtol=0.016)


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("budget", [16, 42])
@pytest.mark.parametrize("ratio,dim", [(4, 128), (4, 512), (128, 512)])
def test_tree_projection_exact(bs, budget, ratio, dim):
    case = make_case(bs, ratio, dim, 4 * ratio - 3, budget)
    legacy, tree, update = projection(case, return_pair=True)
    for _ in range(2):
        legacy()
        expected = case["nodes"].clone()
        expected_out = native(case)
        case["nodes"].fill_(float("nan"))
        tree()
        torch.testing.assert_close(case["nodes"], expected, atol=0, rtol=0)
        torch.testing.assert_close(native(case), expected_out, atol=0, rtol=0)
        update()


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio,dim", [(4, 128), (4, 512), (128, 512)])
def test_tree_projection_full_replay(bs, ratio, dim):
    case = make_case(bs, ratio, dim, 4 * ratio - 3)
    legacy, tree, update = projection(case, return_pair=True)
    tree()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        tree()
        out = native(case)
    for _ in range(3):
        update()
        legacy()
        expected = case["nodes"].clone()
        expected_out = native(case)
        case["nodes"].fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(case["nodes"], expected, atol=0, rtol=0)
        torch.testing.assert_close(out, expected_out, atol=0, rtol=0)


@pytest.mark.parametrize("bs", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio,dim", [(4, 128), (4, 512), (128, 512)])
def test_tree_compressor_dispatch_replay(bs, ratio, dim):
    """Real tree adapter including metadata production; cache commit is unchanged."""
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp
    from vllm_ascend.worker.v2.spec_decode.tree.node_pool import compress_nodes

    case = make_case(bs, ratio, dim, 4 * ratio - 3)
    n, c = case["plan"]["n"], 2 if ratio == 4 else 1
    hidden = torch.randn(n, 4096, dtype=torch.bfloat16, device="npu") * 0.1
    module = SimpleNamespace(
        compress_ratio=ratio,
        head_dim=dim,
        coff=c,
        rope_head_dim=64,
        norm_eps=1e-6,
        wkv=SimpleNamespace(weight=torch.randn(c * dim, 4096, dtype=torch.bfloat16, device="npu") * 0.1),
        wgate=SimpleNamespace(weight=torch.randn(c * dim, 4096, dtype=torch.bfloat16, device="npu") * 0.1),
        ape=torch.randn(ratio, c * dim, device="npu") * 0.1,
        norm=SimpleNamespace(weight=case["norm"]),
    )

    def legacy(x, nodes, table, cu, positions, cos, sin):
        return torch.ops._C_ascend.compressor(
            x,
            module.wkv.weight,
            module.wgate.weight,
            nodes.squeeze(-2),
            module.ape,
            module.norm.weight,
            sin,
            cos,
            state_block_table=table,
            cu_seqlens=cu,
            seqused=None,
            start_pos=positions,
            rope_head_dim=64,
            cmp_ratio=ratio,
            coff=c,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=1,
        )

    module._compress = legacy
    metadata = SimpleNamespace(
        state=SimpleNamespace(req_metadata=SimpleNamespace(block_table=case["table"])),
        cache=SimpleNamespace(
            req_metadata=SimpleNamespace(
                full_compress_cos=case["cos"],
                full_compress_sin=case["sin"],
                storage_block_size=16,
                block_table=case["table"],
            )
        ),
    )
    config = SimpleNamespace(tree_spec_config=SimpleNamespace(compressor_backend="triton"))
    plan = dict(
        case["plan"],
        hidden_ids=torch.arange(n, device="npu").flip(0),
        token_index=torch.arange(n, device="npu"),
        query_cu=torch.arange(n + 1, dtype=torch.int32, device="npu"),
    )
    recorded = []
    compressed_cache = torch.zeros(2, 16, 1, dim, dtype=torch.bfloat16, device="npu")

    def record(plan, original, snapshot, metadata, ratio):
        if ratio == 1:
            recorded.append(snapshot)

    def run():
        # Same round-local lifecycle as the real plan producer. Device updates
        # from compress_nodes are captured and rerun on every FULL replay.
        plan["query_metadata"] = {}
        plan["node_workspace_owners"] = set()
        plan["cache_views"] = {}
        return compress_nodes(module, hidden, case["state"], metadata)

    with (
        patch.object(cp, "_build_plan", return_value=plan),
        patch.object(cp, "_KV_STACK", [[compressed_cache]]),
        patch.object(cp, "_LAYOUT", None),
        patch("vllm_ascend.ascend_config.get_ascend_config", return_value=config),
        patch("vllm_ascend.worker.v2.spec_decode.tree.node_pool.record_nodes", side_effect=record),
    ):
        expected, slots = run()
        expected, slots, nodes = expected.clone(), slots.clone(), recorded[-1].clone()
        config.tree_spec_config.compressor_backend = "native"
        actual, actual_slots = run()
        torch.testing.assert_close(recorded[-1], nodes, atol=0, rtol=0)
        torch.testing.assert_close(actual_slots, slots, atol=0, rtol=0)
        torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.016)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual, actual_slots = run()
        graph_nodes = recorded[-1]
        for _ in range(2):
            hidden.mul_(0.9)
            plan["start"].add_(1)
            graph.replay()
            # Preserve replay data before eager reuse of the workspace.
            replay, replay_slots, replay_nodes = actual.clone(), actual_slots.clone(), graph_nodes.clone()
            config.tree_spec_config.compressor_backend = "triton"
            expected, slots = run()
            torch.testing.assert_close(replay_nodes, recorded[-1], atol=0, rtol=0)
            torch.testing.assert_close(replay_slots, slots, atol=0, rtol=0)
            torch.testing.assert_close(replay, expected, atol=0.001, rtol=0.016)
            config.tree_spec_config.compressor_backend = "native"
