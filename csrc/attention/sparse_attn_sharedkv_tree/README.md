# Native DSV4 tree attention

This independent operator starts from `sparse_attn_sharedkv` at `29d7134cc`.
The DSV4 tree adapter uses it whenever
`additional_config.tree_spec_config.enabled=True`. AscendC is the only
compressed tree attention implementation; there is no backend selector.
Prefill, chain speculation and draft attention use their existing operators.
Host classes, ACLNN names and kernel names are distinct, so a package can
contain both operators.

## Tree contract

- One TND query record per tree node, including the root. A budget of 16 or 42
  non-root nodes means 17 or 43 query records per request.
- `seqused_kv` is the **logical** prefix length plus ancestor count.
- `ori_sparse_indices` contains physical original-KV slots, in chronological
  ancestor order, followed by `-1` padding. Its width is a multiple of 128.
- The number of valid slots must equal the logical sliding-window length. This
  operator reuses the original metadata task partition. Arbitrarily shortening
  a window or inserting holes requires matching metadata changes.
- Original KV rows are shared. No branch KV expansion is performed by the
  operator. The UT expands branches only for the original-operator oracle.
- C4 uses logical `cmp_sparse_indices` Top-K positions, filtered by the existing
  tree indexer before calling attention. C128 omits this tensor and retains full
  CFA traversal, including histories beyond 512/1024 compressed rows.
  C4 must supply `min(topk_width, seqused_kv // 4)` valid positions followed by
  padding, matching the original metadata contract. When prefixes change on
  graph replay, the indexer must refresh Top-K as well as the attention metadata.
- Optional dual-source inputs are supplied together: `cmp_tree_kv` is contiguous
  `[nodes,512]` FP16/BF16, `tree_ancestors` is int64 `[queries,max_depth]`, and
  `tree_prefix` is int32 `[queries]`. All are device tensors. Ancestors contain
  global node IDs, in root-to-query order; unused tail entries are never read.
- A logical compressed index `k < prefix // ratio` reads paged history. Otherwise
  it reads node `ancestors[(k+1)*ratio-1-prefix]`. Every completed tree window must
  have a valid ancestor. Logical positions never become physical cache indices.
- C4 collects both sources into the existing native merge workspace. C128 keeps
  contiguous paged-history transfers and reads only the completed tree tail from
  node rows. Neither path expands branch KV outside the operator.
- Omitting all three pool inputs retains the fixed compressed paged-layout path.
- QK and PV must consume identical original-KV slots. Joint softmax, sink and
  probability rounding are retained from the native kernel.

## Isolated development build

Normal A2/A3 source installation includes all four tree operators in the
bundled `custom_transformer` vendor. See `../compressor_tree_project/README.md`
for the installation command. The isolated build below is for development;
use `ASCEND_COMPUTE_UNIT=ascend910_93` and a separate build directory for A3.

Use a separate build and install directory. Do not use `build_aclnn.sh` with
only this operator: it clears the existing custom-op installation.

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -S csrc -B csrc/build_tree_native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DASCEND_OP_NAME=sparse_attn_sharedkv_tree \
  -DVENDOR_NAME=custom_tree -DENABLE_BUILD_PKG=ON \
  -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=ON
cmake --build csrc/build_tree_native \
  --target ops_transformer_kernel cust_opapi cust_opmaster cust_proto \
           optiling_compat modify_vendor gen_version_info -j8
cmake --install csrc/build_tree_native --prefix /absolute/path/tree_native_install
python -m tools.build_tree_attention_binding \
  --cann-path /usr/local/Ascend/cann-9.1.0 \
  --build-directory /absolute/path/tree_attention_binding
```

Add `/absolute/path/tree_native_install/packages/vendors/custom_tree_transformer`
to `ASCEND_CUSTOM_OPP_PATH` before starting the process, alongside the existing
vendor path. Set the corresponding `op_api/lib` path in `LD_LIBRARY_PATH` if
needed. The original metadata operator is still required from the existing
installation. The standalone binding only registers the new operator; it does
not overwrite `vllm_ascend_C`. Normal extension builds include the same binding
header from `torch_binding.cpp`.

Tree-specific tests and their launchers are maintained outside this repository.
The validation package uses the workspace layout
`<workspace>/{vllm,vllm-ascend,validation/native_only}`. After loading the CANN
and workspace environment, set its location and run the external launcher:

```bash
TREE_VALIDATION_ROOT=/absolute/path/validation/native_only
python "$TREE_VALIDATION_ROOT/tools/run_tree_attention_ut.py" \
  --binding /absolute/path/tree_attention_binding/tree_attention_binding.so \
  "$TREE_VALIDATION_ROOT/tests/ut/ops/a2/test_tree_native_attention.py" \
  "$TREE_VALIDATION_ROOT/tests/ut/ops/a2/test_tree_native_pool_attention.py" \
  "$TREE_VALIDATION_ROOT/tests/ut/ops/a2/test_tree_native_c128_attention.py" \
  "$TREE_VALIDATION_ROOT/tests/ut/ops/a2/test_tree_native_integration.py" \
  "$TREE_VALIDATION_ROOT/tests/ut/ops/a2/test_tree_attention_precision.py" -q
```

Never report compilation or warmup time as kernel time. Benchmark reports must
separate operator replay, metadata/index construction and full-path costs.

The generic `all` target also builds ONNX/protobuf dependencies that this
operator does not need. After kernel-header changes, invalidate only this
operator's generated `sparse_attn_sharedkv_tree_ascend910b_*.done` stamps under
`csrc/build_tree_native/binary/ascend910b/gen/` before rebuilding: the existing
CANN CMake rules use order-only dependencies for source copying. Verify that
OPC actually recompiles the kernel before installing or measuring it.

Historical three-backend benchmark scripts are archived outside the repository
under `validation/archive/head-48c937639/tools/`. They require the matching
historical checkout and are not runnable against the native-only code. Current
precision tests use independent CPU and original linear AscendC references;
they do not preserve a production Triton compute backend.

CPU-only registration/shape tests, without NPU initialization:

```bash
python "$TREE_VALIDATION_ROOT/tools/run_tree_attention_ut.py" --cpu-only \
  --binding /absolute/path/tree_attention_binding/tree_attention_binding.so \
  "$TREE_VALIDATION_ROOT/tests/ut/ops/test_tree_native_attention_meta.py" -q
```

The regression suites compare C4 and C128 outputs with independently expanded
inputs to the original AscendC operator and with CPU softmax references. They
cover sibling isolation, long compressed histories, BF16 probability rounding,
and live topology/prefix updates during FULL replay. Operator timings exclude
index/metadata construction and branch KV expansion; end-to-end performance
must be measured separately.
