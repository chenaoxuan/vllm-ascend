# Native tree node projection

`_C_ascend.compressor_tree_project` is the tree-only BF16 TH projection stage
for Ascend 910B. It retains native Cube multiplication and FP32 APE addition,
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
its CANN license headers are preserved. Only the 910B configuration is exposed.
Build `ASCEND_OP_NAME=compressor_tree_project` into a separate custom vendor,
then use `tools.build_tree_compressor_binding` to register projection together
with tree aggregation. Both vendors must be available when running the native
compressor backend; retain the original vendor for non-tree model paths.

UTs compare FP32 node rows exactly against the original projection, then check
tree reductions against independent FP64 and Triton references. FULL tests
change device inputs/positions without recapture. Performance is measured for
projection, aggregation, and their combined execution, excluding other model
work and accepted-state commitment.
