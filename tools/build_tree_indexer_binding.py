# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the experimental tree binding; leave the installed extension intact."""

import argparse
import hashlib
import json
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", required=True, type=Path)
    parser.add_argument("--cann-path", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    npu = Path(torch_npu.__file__).parent
    args.build_directory.mkdir(parents=True, exist_ok=True)
    source = args.build_directory / "tree_indexer_binding.cpp"
    source.write_text('#include "attention/quant_lightning_indexer_tree/torch_binding.h"\n')
    result = load(
        name="tree_indexer_binding",
        sources=[str(source)] + [str(p) for p in sorted((root / "csrc/aclnn_torch_adapter").glob("*.cpp"))],
        extra_include_paths=[str(root / "csrc"), str(npu / "include"), str(args.cann_path / "include")],
        extra_cflags=["-O2", "-DTREE_INDEXER_STANDALONE"],
        extra_ldflags=[f"-L{npu / 'lib'}", "-ltorch_npu", f"-L{args.cann_path / 'lib64'}", "-lascendcl"],
        build_directory=str(args.build_directory),
        is_python_module=False,
        verbose=True,
    )
    sources = sorted((root / "csrc/attention/quant_lightning_indexer_tree").rglob("*"))
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}
    (args.build_directory / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
