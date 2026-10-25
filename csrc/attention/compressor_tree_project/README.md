# Native tree node projection

`_C_ascend.compressor_tree_project` is the tree-only BF16 TH projection stage
for Ascend 910B/910C. It retains native Cube multiplication and FP32 APE addition,
and writes one unique FP32 node row directly. It never aggregates a linear
window or runs Norm/RoPE; `compressor_tree` performs the branch-aware epilogue.

The internal contract is one query per node: `cu = arange(n + 1)` (int32),
`positions` contains APE residues, and contiguous `nodes` contains n + 1 rows.
Row zero is reserved and left untouched. The caller clears that sentinel;
every real node channel is overwritten. Dynamic inputs remain on device.
C4 and C128 retain batched APE loads: an experimental per-node APE load
reduced bytes but regressed latency because of additional DMA synchronization.

The CANN ABI retains shape-only norm/RoPE/table/output placeholders to reuse
native tiling checks. These buffers are neither read nor written by this
projection kernel. They have no device fill/copy kernels. The tree path avoids
synthetic page-table construction, dummy cos/sin fills, and full node zeroing.
Unused linear aggregation workspace is removed. Non-tree operators are intact.

The source is derived from the adjacent native `compressor` implementation;
its CANN license headers are preserved. The 910B and 910C configurations use
the same arch32 kernel and device-specific tiling resources.
Normal A2/A3 source installation builds `compressor_tree_project`, `compressor_tree`,
`quant_lightning_indexer_tree`, and `sparse_attn_sharedkv_tree` into the bundled
`custom_transformer` vendor through `csrc/build_aclnn.sh`. Load the CANN
environment and rebuild from the updated checkout:

```bash
COMPILE_CUSTOM_KERNELS=1 python -m pip install -v -e . --no-deps --no-build-isolation
nm -D vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so \
  | grep -E 'aclnn(CompressorTreeProject|CompressorTree|QuantLightningIndexerTree|SparseAttnSharedkvTree)(GetWorkspaceSize)?$'
```

All four operators must export both their ACLNN execution function and their
`GetWorkspaceSize` function (eight symbols in total). Restart all workers after
rebuilding: the binding caches
custom library paths when it loads and ACLNN function addresses on first use.

For isolated development, build `ASCEND_OP_NAME=compressor_tree_project` into
a separate custom vendor, then use `tools.build_tree_compressor_binding` to
register projection together with tree aggregation. Add each vendor root
(the directory containing `op_api`) to `ASCEND_CUSTOM_OPP_PATH` before starting
Python. Both tree vendors must be available for tree verification; retain the
original vendor for non-tree model paths.

External UTs compare FP32 node rows exactly against the original linear
AscendC projection, then check tree reductions against an independent CPU
FP64 reference. FULL tests change device inputs/positions without recapture.
With the rebuilt production extension and vendor environment loaded, run:

```bash
TREE_VALIDATION_ROOT=/absolute/path/validation/native_only
python "$TREE_VALIDATION_ROOT/run_tests.py" \
  "$TREE_VALIDATION_ROOT/tests/ut/ops/a2/test_tree_native_compressor.py" -q
```

Tests are not stored in the production repository. Historical projection and
aggregation benchmark scripts remain under
`validation/archive/head-48c937639/tools/` for use with their matching checkout.
