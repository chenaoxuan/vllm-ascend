# Native DSV4 tree attention prototype

This independent operator starts from `sparse_attn_sharedkv` at `29d7134cc`.
It preserves the original operator and the default Triton node-pool backend.
The DSV4 node-pool adapter selects it with `attention_backend="native"` in
`additional_config.tree_spec_config`, alongside `enabled=True` and
`node_pool=True`. Only compressed tree verification layers use this selection;
prefill, chain speculation, draft attention and accepted-state commit retain
their existing paths. Host classes, ACLNN names and kernel names
are distinct, so a package can contain both operators.

## Tree prototype contract

- One TND query record per tree node, including the root. A budget of 16 or 42
  non-root nodes means 17 or 43 query records per request.
- `seqused_kv` is the **logical** prefix length plus ancestor count.
- `ori_sparse_indices` contains physical original-KV slots, in chronological
  ancestor order, followed by `-1` padding. Its width is a multiple of 128.
- The number of valid slots must equal the logical sliding-window length. This
  prototype reuses the original metadata task partition. Arbitrarily shortening
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

```bash
python -m tools.run_tree_attention_ut \
  --binding /absolute/path/tree_attention_binding/tree_attention_binding.so \
  tests/ut/ops/a2/test_tree_native_attention.py \
  tests/ut/ops/a2/test_tree_native_pool_attention.py \
  tests/ut/ops/a2/test_tree_native_c128_attention.py \
  tests/ut/ops/a2/test_tree_native_integration.py -q
```

Never report compilation or warmup time as kernel time. Benchmark reports must
separate operator replay, metadata/index construction and full-path costs.

The generic `all` target also builds ONNX/protobuf dependencies that this
operator does not need. After kernel-header changes, invalidate only this
operator's generated `sparse_attn_sharedkv_tree_ascend910b_*.done` stamps under
`csrc/build_tree_native/binary/ascend910b/gen/` before rebuilding: the existing
CANN CMake rules use order-only dependencies for source copying. Verify that
OPC actually recompiles the kernel before installing or measuring it.

```bash
python -m tools.benchmark_tree_native_attention \
  --binding /absolute/path/tree_attention_binding/tree_attention_binding.so \
  --output /absolute/path/tree_native_benchmark.json
```

The benchmark checks all three outputs before measuring graph-replay event time
and eager synchronized wall time. It excludes branch expansion, indices and
metadata, so these numbers are not full-path or end-to-end speedups.


For dual-source C4 benchmarks add `--pool`; for full C128 use `--ratio 128`.
`--source-manifest /path/to/manifest.json` records the exact kernel source
snapshot when it differs from the binding build. Build and current-source
hashes are separate fields; never identify an older binary with the current
working tree. The C4-only and pool ABI bindings must each match their installed
operator package. Run them in separate processes.

CPU-only registration/shape tests, without NPU initialization:

```bash
python -m tools.run_tree_attention_ut \
  --binding /absolute/path/tree_attention_binding/tree_attention_binding.so \
  tests/ut/ops/test_tree_native_attention_meta.py -q
```

Current validation: 16 Meta tests passed; C4 prototype passed 29 NPU tests,
C4 dual-source passed 59, and the final C128 vendor passed all 86 (overlapping
regression suites). Each stage completed eight BS1/2/4/8 × budget16/42 timing
cases at prefix511, BF16, heads8. Numerical tolerances were unchanged.
The TND host guard accounts for the leading cu_seqlens_q sentinel.
The production backend is unchanged; no end-to-end validation has been run.
Timings exclude index/metadata construction and branch KV expansion.
