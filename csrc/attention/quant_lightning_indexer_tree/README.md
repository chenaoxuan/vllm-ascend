# AscendC tree indexer

This independent A2/A3 operator derives from the adjacent native
`quant_lightning_indexer_v2`. Its INT8 Cube scoring, FP16 rounding boundaries,
head reduction and native Top-K merge are retained. The original operator and
Q/weight projection, RoPE, Hadamard and quantization paths are unchanged.

The tree ABI accepts shared paged key/scale history, unique node key/scale rows,
ancestor indices, per-node prefixes and request IDs. One query is one tree
endpoint (`cu = arange(nodes + 1)`). Each query can read historical positions
and its own ancestors. Absent ancestors get score negative infinity and index
`-1` before sorting. K/scale chains are never materialized by this operator.

The supported contract is A2/A3, INT8 Q/K, FP16 scales/weights, 64 query
heads, head dimension 128, compression ratio 4, TND queries and paged history.
History pages may have padded strides; rows within each page must be contiguous.
Node rows and metadata must be contiguous and colocated. This operator reads
all caches without modifying them; accepted-chain commitment is unchanged.

Normal A2/A3 source installation includes this operator and the other three
tree operators in the bundled `custom_transformer` vendor. See
`../compressor_tree_project/README.md` for the installation command.

For isolated development, build `ASCEND_OP_NAME=quant_lightning_indexer_tree`
into its own custom vendor.
`tools.build_tree_indexer_binding` builds an isolated PyTorch binding for UTs.
The original vendor remains necessary for metadata generation and the native
linear reference. Tests compare with expanded native inputs and an independent
CPU scoring oracle, and update device inputs during FULL graph replay.

The Python adapter preserves the existing select-all shortcut when the host
capacity proves that all compressed positions fit in Top-K. It does not move
this decision ahead of the original query preparation. Thus short-context
end-to-end tests do not exercise the native scoring kernel; long-context tests
are necessary to measure its benefit.

Tree verification uses this operator when
`additional_config.tree_spec_config.enabled=True`. AscendC is the only tree
indexer scoring implementation; the select-all shortcut uses device tensor
operations and does not invoke a second scoring backend.
The production extension must be rebuilt with this operator; the standalone
binding loader is only for isolated validation and benchmarking.

The external A2 regression suite covers exact expanded-linear index parity,
independent CPU scoring, absent-ancestor masking, padded history pages,
select-all ordering, and changing inputs during FULL replay. With the rebuilt
production extension and vendor environment loaded, run:

```bash
TREE_VALIDATION_ROOT=/absolute/path/validation/native_only
python "$TREE_VALIDATION_ROOT/run_tests.py" \
  "$TREE_VALIDATION_ROOT/tests/ut/ops/a2/test_tree_native_indexer.py" -q
```

Tests are maintained outside the production repository. The historical
three-backend performance utility is archived under
`validation/archive/head-48c937639/tools/` and requires the matching historical
checkout. Current precision references use CPU scoring and the original
linear AscendC operator.
