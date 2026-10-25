# AscendC tree indexer

This independent A2 operator derives from the adjacent native
`quant_lightning_indexer_v2`. Its INT8 Cube scoring, FP16 rounding boundaries,
head reduction and native Top-K merge are retained. The original operator and
Q/weight projection, RoPE, Hadamard and quantization paths are unchanged.

The tree ABI accepts shared paged key/scale history, unique node key/scale rows,
ancestor indices, per-node prefixes and request IDs. One query is one tree
endpoint (`cu = arange(nodes + 1)`). Each query can read historical positions
and its own ancestors. Absent ancestors get score negative infinity and index
`-1` before sorting. K/scale chains are never materialized by this operator.

The first supported contract is A2, INT8 Q/K, FP16 scales/weights, 64 query
heads, head dimension 128, compression ratio 4, TND queries and paged history.
History pages may have padded strides; rows within each page must be contiguous.
Node rows and metadata must be contiguous and colocated. This operator reads
all caches without modifying them; accepted-chain commitment is unchanged.

Build `ASCEND_OP_NAME=quant_lightning_indexer_tree` into its own custom vendor.
`tools.build_tree_indexer_binding` builds an isolated PyTorch binding for UTs.
The original vendor remains necessary for metadata generation and the native
linear reference. Tests compare with expanded native inputs and an independent
CPU scoring oracle, and update device inputs during FULL graph replay.

The Python adapter preserves the existing select-all shortcut when the host
capacity proves that all compressed positions fit in Top-K. It does not move
this decision ahead of the original query preparation. Thus short-context
end-to-end tests do not exercise the native scoring kernel; long-context tests
are necessary to measure its benefit.

Select this backend with `additional_config.tree_spec_config.indexer_backend`
set to `"native"`, alongside `enabled=True` and `node_pool=True`. The default
remains `"triton"`. Attention and compressor backend selection is independent.
The production extension must be rebuilt with this operator; the standalone
binding loader is only for isolated validation and benchmarking.

On A2 with CANN 9.1, the current implementation passes 56 multi-batch tests,
including exact expanded-native index parity, independent CPU scoring,
absent-ancestor masking, padded history pages and changing inputs during FULL
replay. The performance utility covers BS 1/2/4/8 and budgets 16/42; it reports
metadata-inclusive tree execution and KV expansion separately. These operator
timings exclude unchanged query preprocessing and accepted-chain commitment.
