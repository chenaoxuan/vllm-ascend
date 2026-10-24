import json

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

from vllm_ascend.ops.triton.spec_decode.tree.node_compressor import aggregate_nodes  # noqa: E402


def test_node_compressor_matches_independent_chains():
    torch.npu.set_device(0)
    torch.manual_seed(42)
    device = "npu"
    chains = [[0], [0, 1], [0, 2], [0, 1, 3], [0, 2, 4], [0, 1, 3, 5], [0, 2, 4, 6], [0, 1, 3, 5, 7]]
    depths = [len(c) - 1 for c in chains]
    n = len(chains)
    results = []
    for ratio, head in [(4, 128), (4, 512), (128, 512)]:
        for prefix in [0, ratio - 3, ratio - 1, ratio * 2 - 3]:
            coeff = 2 if ratio == 4 else 1
            dim = 2 * coeff * head
            x = torch.randn(n, 1024, dtype=torch.bfloat16, device=device) * 0.1
            wkv = torch.randn(coeff * head, 1024, dtype=torch.bfloat16, device=device) * 0.1
            wgate = torch.randn_like(wkv) * 0.1
            ape = torch.randn(ratio, coeff * head, device=device) * 0.1
            norm = torch.randn(head, device=device) * 0.1 + 1
            rope_cos = torch.cos(torch.randn(512, 64, device=device))
            rope_sin = torch.sin(torch.randn(512, 64, device=device))
            history = torch.randn(prefix, dim, device=device) * 0.1
            state = torch.zeros(prefix + 8, 1, dim, device=device)
            state[1 : prefix + 1, 0].copy_(history)

            def compress(
                hidden, st, bt, cu, start, cos, sin, wkv=wkv, wgate=wgate, ape=ape, norm=norm, ratio=ratio, coeff=coeff
            ):
                return torch.ops._C_ascend.compressor(
                    hidden,
                    wkv,
                    wgate,
                    st,
                    ape,
                    norm,
                    sin,
                    cos,
                    state_block_table=bt,
                    cu_seqlens=cu,
                    seqused=None,
                    start_pos=start,
                    rope_head_dim=64,
                    cmp_ratio=ratio,
                    coff=coeff,
                    norm_eps=1e-6,
                    rotary_mode=2,
                    cache_mode=1,
                )

            nodes = torch.zeros(n + 1, 1, dim, device=device)
            residue = (torch.tensor(depths, dtype=torch.int32, device=device) + prefix) % ratio
            bt = torch.zeros(n, ratio, dtype=torch.int32, device=device)
            bt.scatter_(1, residue.long()[:, None], torch.arange(1, n + 1, dtype=torch.int32, device=device)[:, None])
            compress(
                x,
                nodes,
                bt,
                torch.arange(n + 1, dtype=torch.int32, device=device),
                residue,
                torch.ones(n, 64, device=device),
                torch.zeros(n, 64, device=device),
            )
            plan = dict(
                n=n,
                max_q=5,
                ancestors=torch.tensor([c + [0] * (5 - len(c)) for c in chains], dtype=torch.long, device=device),
                start=torch.full((n,), prefix, dtype=torch.int32, device=device),
                depths=torch.tensor(depths, dtype=torch.long, device=device),
                req_ids=torch.zeros(n, dtype=torch.long, device=device),
            )
            table = torch.arange(1, prefix + 8, dtype=torch.int32, device=device)[None]
            actual = aggregate_nodes(
                state, nodes, table, plan, norm, rope_cos, rope_sin, ratio, head, 64, 1e-6, x.dtype
            )
            max_diff = 0
            state_exact = True
            for endpoint, chain in enumerate(chains):
                ref = state.clone()
                count = min(len(chain), len(chain) // ratio + 1)
                positions = [p for p in range(prefix, prefix + len(chain)) if (p + 1) % ratio == 0]
                rc = torch.zeros(count, 64, device=device)
                rs = torch.zeros_like(rc)
                for i, p in enumerate(positions):
                    rc[i] = rope_cos[p + 1 - ratio]
                    rs[i] = rope_sin[p + 1 - ratio]
                out = compress(
                    x[chain],
                    ref,
                    table,
                    torch.tensor([0, len(chain)], dtype=torch.int32, device=device),
                    torch.tensor([prefix], dtype=torch.int32, device=device),
                    rc,
                    rs,
                )
                state_exact &= torch.equal(nodes[endpoint + 1], ref[prefix + len(chain)])
                if (prefix + len(chain)) % ratio == 0:
                    expected = out[len(positions) - 1]
                    max_diff = max(max_diff, (actual[endpoint].float() - expected.float()).abs().max().item())
                    torch.testing.assert_close(actual[endpoint], expected, atol=0.016, rtol=0.016)
            assert state_exact, (ratio, head, prefix, "state mismatch")
            results.append(dict(ratio=ratio, head=head, prefix=prefix, state_exact=state_exact, max_diff=max_diff))
    print(json.dumps(results, indent=2))
