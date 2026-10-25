# Native tree compressor aggregation

`_C_ascend.compressor_tree` is the Ascend 910B/910C BF16-output tree aggregation operator.
The tree-specific `compressor_tree_project` projects each node once into FP32
state. This operator performs tree-specific aggregation and the epilogue.
It does not change non-tree compression, indexer, or accepted-chain commitment.

The inputs are shared paged history, one projected state row per tree node
(plus reserved row zero), FP32 norm/RoPE tables, and device-resident ancestor,
request, depth and prefix indices. Historical state is read through its page
table; tree state is read through ancestors. Negative/unused ancestor slots and
positions before the sequence start are masked before softmax reduction. C4
retains its previous/current-window interleaving; C128 uses one window.
Non-emitting nodes produce zeros. State inputs are never modified.

The native AIV kernel fuses the per-channel window softmax/weighted reduction,
RMSNorm and interleaved RoPE. Output storage belongs to the operation and remains
stable during FULL graph replay. Tensor values never cross to the host.

Tree verification uses this operator when `tree_spec_config.enabled=True`.
AscendC is the only tree compressor implementation; there is no backend selector.
Current supported head dimensions are 128/512, compression ratios 4/128 and
RoPE dimension 64.

Normal A2/A3 source installation includes this operator and the other three
tree operators in the bundled `custom_transformer` vendor. See
`../compressor_tree_project/README.md` for the installation command.

For isolated development, build a separate `compressor_tree` vendor, following
the isolated CMake build in `sparse_attn_sharedkv_tree/README.md`, with
`ASCEND_OP_NAME=compressor_tree`.
Use `tools.build_tree_compressor_binding` for a standalone extension; normal
extension builds include this operator from `torch_binding.cpp`. Preserve the
existing custom vendor in `ASCEND_CUSTOM_OPP_PATH` for non-tree model paths. Also build and load the separate
`compressor_tree_project` vendor for the native tree projection stage.
For incremental builds, refresh the copied kernel under
`binary/ascend910b/src/compressor_tree/op_kernel/` and invalidate only this
operator's generated `.done` stamps. Verify the installed entry key is 1.

Tree-specific validation lives outside this repository. With the rebuilt
production extension and vendor environment loaded, run:

```bash
TREE_VALIDATION_ROOT=/absolute/path/validation/native_only
python "$TREE_VALIDATION_ROOT/run_tests.py" \
  "$TREE_VALIDATION_ROOT/tests/ut/ops/a2/test_tree_native_compressor.py" \
  "$TREE_VALIDATION_ROOT/tests/ut/models/a2/test_tree_node_compressor.py" -q
```

The tests cover multi-request branch masks, page strides, an independent CPU
FP64 oracle, projected states, and FULL replay. The original linear AscendC
compressor is a test reference. Historical Triton/native benchmark scripts
are preserved under `validation/archive/head-48c937639/tools/` and require
that historical checkout; they are not current native-only benchmarks.

Projection and tree reduction are separate native launches so all node states
are complete before ancestor aggregation. No inter-core dependency is assumed
between different nodes inside the projection launch.
