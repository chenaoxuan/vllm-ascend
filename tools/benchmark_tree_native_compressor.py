# SPDX-License-Identifier: Apache-2.0
"""Measure tree reduction and native projection+reduction separately, under FULL replay."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from tools.benchmark_tree_native_attention import measure
from vllm_ascend.utils import bootstrap_custom_op_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.ops.load_library(args.binding)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "compressor_fixture", root / "tests/ut/ops/a2/test_tree_native_compressor.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    results = []
    for bs in (1, 2, 4, 8):
        for budget in (16, 42):
            for ratio, dim in ((4, 128), (4, 512), (128, 512)):
                case = fixture.make_case(bs, ratio, dim, 4 * ratio - 3, budget)
                project, tree_project, _ = fixture.projection(case, return_pair=True)
                project()
                torch.testing.assert_close(fixture.native(case), fixture.triton(case), atol=0.001, rtol=0.016)
                row = dict(bs=bs, budget=budget, ratio=ratio, dim=dim, prefix=4 * ratio - 3)
                for backend in ("triton", "native"):
                    reduce = getattr(fixture, backend)
                    row[backend + "_reduce"], _ = measure(lambda reduce=reduce, case=case: reduce(case), 3, 20)

                    def full(project=project, reduce=reduce, case=case):
                        project()
                        return reduce(case)

                    row[backend + "_projection_reduce"], _ = measure(full, 3, 20)
                expected = case["nodes"].clone()
                tree_project()
                torch.testing.assert_close(case["nodes"], expected, atol=0, rtol=0)
                row["legacy_projection"], _ = measure(project, 3, 20)
                row["tree_projection"], _ = measure(tree_project, 3, 20)

                def tree_full(tree_project=tree_project, case=case):
                    tree_project()
                    return fixture.native(case)

                row["tree_projection_reduce"], _ = measure(tree_full, 3, 20)
                results.append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps(dict(results=results), indent=2))
    evidence = dict(
        results=results,
        hidden_dim=4096,
        dtype="BF16",
        repeats=3,
        steps=20,
        note="Excludes metadata creation, cache-view allocation, state commit, and all other model work.",
        binding_sha256=hashlib.sha256(Path(args.binding).read_bytes()).hexdigest(),
        build_sources=json.loads((Path(args.binding).parent / "source_manifest.json").read_text()),
    )
    args.output.write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
