# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
"""Compare tree indexer paths under FULL replay, with KV expansion charged separately."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from tools.benchmark_tree_native_attention import measure
from vllm_ascend.utils import bootstrap_custom_op_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.ops.load_library(args.binding)
    from tests.ut.ops.a2.test_tree_native_indexer import assert_reference, expand, linear, make_case, metadata, native
    from vllm_ascend.ops.triton.spec_decode.tree.native_indexer import native_indexer
    from vllm_ascend.ops.triton.spec_decode.tree.node_indexer import node_indexer

    rows = []
    for bs in (1, 2, 4, 8):
        for budget in (16, 42):
            for prefix in (127, 2303):
                case = make_case(bs=bs, budget=budget, prefix=prefix, topk=512)
                meta, expanded = metadata(case), expand(case)
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

                def adapter(case=case, op_args=op_args):
                    case["plan"]["query_metadata"] = {}
                    return native_indexer(*op_args)

                actual = adapter()
                torch.testing.assert_close(native(case, meta), linear(case, expanded, meta), atol=0, rtol=0)
                assert_reference(case, actual)
                baseline = node_indexer(*op_args)
                # Keep structural validity checks for the unchanged baseline;
                # report its ranking error separately. Native still passes the
                # strict oracle and exact expanded-native parity checks above.
                baseline_gap = assert_reference(case, baseline, check_ranking=False)
                # D2H only at the benchmark verification boundary, outside timing.
                a, b = actual.cpu().flatten(1).tolist(), baseline.cpu().flatten(1).tolist()
                overlap = [len((set(x) & set(y)) - {-1}) / max(1, len(set(x) - {-1})) for x, y in zip(a, b)]
                row = dict(
                    bs=bs,
                    budget=budget,
                    prefix=prefix,
                    topk=512,
                    triton_reference_max_score_gap=baseline_gap,
                    min_selected_overlap=min(overlap),
                    mean_selected_overlap=sum(overlap) / len(overlap),
                )
                print("BEGIN", json.dumps(row), flush=True)
                for name, fn in (
                    ("triton_path", partial(node_indexer, *op_args)),
                    ("native_path_with_metadata", adapter),
                    ("native_operator", partial(native, case, meta)),
                    ("expanded_native_operator", partial(linear, case, expanded, meta)),
                    ("expand_and_native_operator", partial(linear, case, meta=meta)),
                ):
                    row[name], _ = measure(fn, 3, args.steps)
                rows.append(row)
                report = dict(
                    results=rows,
                    repeats=3,
                    steps=args.steps,
                    note="UT only; excludes unchanged Q/K projection, transforms, quantization and state commit. "
                    "Native path includes device metadata generation and preserves the existing select-all shortcut. "
                    "Expanded raw operator excludes expansion; expand_and_native_operator includes it.",
                )
                args.output.write_text(json.dumps(report, indent=2))
                print("RESULT", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
