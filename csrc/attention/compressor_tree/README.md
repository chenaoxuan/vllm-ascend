# Native tree compressor aggregation

`_C_ascend.compressor_tree` is an isolated Ascend 910B BF16-output prototype.
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

Select the prototype with `tree_spec_config.compressor_backend="native"` and
`enabled=True, node_pool=True`. The default is `"triton"`. Native attention has
its own independent selector. Current supported head dimensions are 128/512,
compression ratios 4/128 and RoPE dimension 64.

Build a separate `compressor_tree` vendor, following the isolated CMake build
in `sparse_attn_sharedkv_tree/README.md`, with `ASCEND_OP_NAME=compressor_tree`.
Use `tools.build_tree_compressor_binding` for a standalone extension; normal
extension builds include this operator from `torch_binding.cpp`. Preserve the
existing custom vendor in `ASCEND_CUSTOM_OPP_PATH` for non-tree model paths. Also build and load the separate
`compressor_tree_project` vendor for the native tree projection stage.
For incremental builds, refresh the copied kernel under
`binary/ascend910b/src/compressor_tree/op_kernel/` and invalidate only this
operator's generated `.done` stamps. Verify the installed entry key is 1.

Validation sources:

- `tests/ut/ops/a2/test_tree_native_compressor.py`: multi-request branch masks,
  page strides, independent FP64 oracle, projected states and FULL replay.
- `tools/benchmark_tree_native_compressor.py`: reduction alone and projection
  plus reduction, with allocation/setup, model execution and commit excluded.

Projection and tree reduction are separate native launches so all node states
are complete before ancestor aggregation. No inter-core dependency is assumed
between different nodes inside the projection launch.
