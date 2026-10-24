# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run native tree UTs with an explicitly selected standalone binding."""

import argparse

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    args, pytest_args = parser.parse_known_args()
    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.ops.load_library(args.binding)
    return pytest.main(pytest_args or ["tests/ut/ops/a2/test_tree_native_attention.py", "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
