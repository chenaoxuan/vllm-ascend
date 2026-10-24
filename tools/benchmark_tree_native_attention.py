# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare C4 prototype operator execution, excluding fixture/metadata setup.

This is not an end-to-end benchmark. The linear baseline has already expanded
its branch KV; costs of that expansion are deliberately reported as excluded.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.spec_decode.tree.node_attention import node_attention
from vllm_ascend.utils import bootstrap_custom_op_env


def measure(run, repeats, steps):
    for _ in range(3):
        run()
    torch.npu.synchronize()
    begin = time.perf_counter()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    torch.npu.synchronize()
    capture_ms = (time.perf_counter() - begin) * 1000
    graph.replay()
    torch.npu.synchronize()
    events, wall = [], []
    for _ in range(repeats):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            graph.replay()
        end.record()
        end.synchronize()
        events.append(start.elapsed_time(end) / steps)
        torch.npu.synchronize()
        begin = time.perf_counter()
        for _ in range(steps):
            run()
        torch.npu.synchronize()
        wall.append((time.perf_counter() - begin) * 1000 / steps)
    return dict(
        captured_replay_ms=statistics.median(events),
        captured_range_ms=[min(events), max(events)],
        eager_wall_ms=statistics.median(wall),
        eager_range_ms=[min(wall), max(wall)],
        capture_ms=capture_ms,
    ), output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--ratio", type=int, choices=(4, 128), default=4)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--pool", action="store_true", help="Benchmark dual-source compressed node pools")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--prefix", type=int, default=511)
    args = parser.parse_args()
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.ops.load_library(args.binding)
    torch.npu.set_device(0)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "tree_native_fixture", root / "tests/ut/ops/a2/test_tree_native_attention.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    if args.pool:
        from tests.ut.ops.a2.test_tree_native_pool_attention import make_pool_case
    make_fixture = make_pool_case if args.pool else fixture.make_case
    if args.ratio == 128:
        from tests.ut.ops.a2.test_tree_native_c128_attention import make_c128_case

        make_fixture = make_c128_case
    sources = sorted((root / "csrc/attention/sparse_attn_sharedkv_tree").rglob("*"))
    manifest = args.source_manifest or Path(args.binding).resolve().parent / "source_manifest.json"
    environment = dict(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        device=torch.npu.get_device_name(0),
        binding=str(Path(args.binding).resolve()),
        custom_opp_path=os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        binding_sha256=hashlib.sha256(Path(args.binding).read_bytes()).hexdigest(),
        build_source_sha256=json.loads(manifest.read_text()) if manifest.exists() else None,
        current_source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()
        },
    )
    results = []
    for requests in (1, 2, 4, 8):
        for budget in (16, 42):
            prefix = args.prefix
            q, tree, linear, _ = make_fixture(requests=requests, budget=budget, prefix=prefix)
            n, heads, dim = q.shape
            count = budget + 1
            chains = []
            for req in range(requests):
                for node in range(count):
                    chain, parent = [], node
                    while parent >= 0:
                        chain.append(req * count + parent)
                        parent = (parent - 1) // 2
                    chains.append(chain[::-1])
            depth = max(map(len, chains))
            ancestors = torch.tensor([c + [0] * (depth - len(c)) for c in chains], device=q.device)
            req_ids = torch.arange(n, device=q.device) // count
            plan = dict(
                n=n,
                max_q=depth,
                max_kv=prefix + depth,
                ancestors=ancestors,
                req_ids=req_ids,
                start=torch.full((n,), prefix, dtype=torch.int32, device=q.device),
                depths=torch.tensor([len(c) - 1 for c in chains], device=q.device),
            )
            ori_table = tree["ori_block_table"][::count].contiguous()
            cmp_table = tree["cmp_block_table"][::count].contiguous()
            k = ((prefix + plan["depths"] + 1) // args.ratio - 1).clamp_min(0)
            compressed = tree["cmp_kv"]
            block = compressed.shape[1]
            nodes = compressed[cmp_table[req_ids, k // block].long(), k % block, 0].contiguous()
            if args.pool or args.ratio == 128:
                nodes = tree["cmp_tree_kv"]
            calls = {
                "linear_native": lambda q=q, linear=linear: torch.ops._C_ascend.npu_sparse_attn_sharedkv(q, **linear)[
                    0
                ],
                "indexed_native": lambda q=q, tree=tree: torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[
                    0
                ],
                "triton": partial(
                    node_attention,
                    q,
                    tree["ori_kv"],
                    compressed,
                    nodes,
                    ori_table,
                    cmp_table,
                    plan,
                    tree.get("cmp_sparse_indices"),
                    tree["sinks"],
                    args.ratio,
                    128,
                    dim**-0.5,
                ),
            }
            expected = calls["linear_native"]()
            record = dict(
                ratio=args.ratio,
                requests=requests,
                budget=budget,
                queries=n,
                prefix=prefix,
                heads=heads,
                excluded=["KV expansion", "indices", "metadata"],
                operators={},
            )
            for name, run in calls.items():
                actual = run()
                torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.01)
                timing, output = measure(run, args.repeats, args.steps)
                torch.testing.assert_close(output, expected, atol=0.001, rtol=0.01)
                record["operators"][name] = timing
            results.append(record)
            print("TREE_NATIVE_BENCH", json.dumps(record), flush=True)
            args.output.write_text(json.dumps(results, indent=2))
            environment["loaded_op_libraries"] = sorted(
                {
                    line.split()[-1]
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if "libcust_op" in line or "tree_attention_binding.so" in line
                }
            )
            args.output.with_suffix(".environment.json").write_text(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
