# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
"""AscendC scoring over shared history and sparse tree ancestors, without KV expansion."""

import torch

from vllm_ascend.ops.triton.spec_decode.tree.node_indexer import node_indexer


def native_indexer(query, weights, query_scale, history, history_scale, nodes, node_scale, table, plan, topk):
    # Preserve the existing tree shortcut, at the same point after Q preparation.
    # It uses a host capacity bound, so FULL replay never branches on device data.
    if max(1, plan["max_kv"] // 4) <= topk:
        return node_indexer(query, weights, query_scale, history, history_scale, nodes, node_scale, table, plan, topk)
    key = ("native_indexer", topk)
    saved = plan["query_metadata"].get(key)
    if saved is None:
        seq = (plan["start"] + plan["depths"] + 1).int()
        lengths, residual = seq // 4, seq % 4
        schedule = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            num_heads_q=query.shape[1],
            num_heads_k=1,
            head_dim=query.shape[2],
            topk=topk,
            quant_mode=2,
            cu_seqlens_q=plan["query_cu"],
            seqused_k=lengths,
            cmp_residual_k=residual,
            batch_size=plan["n"],
            max_seqlen_q=1,
            max_seqlen_k=max(plan["max_kv"] // 4, 1),
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=4,
            device=str(query.device),
        )
        saved = lengths, residual, schedule
        plan["query_metadata"][key] = saved
    lengths, residual, schedule = saved
    return torch.ops._C_ascend.quant_lightning_indexer_tree(
        query,
        history,
        weights.reshape(query.shape[:2]).half(),
        query_scale.reshape(query.shape[:2]).half(),
        history_scale,
        nodes.reshape(query.shape[0], query.shape[2]),
        node_scale,
        plan["ancestors"],
        plan["start"],
        plan["req_ids"],
        table,
        plan["query_cu"],
        lengths,
        residual,
        schedule,
        topk,
    )
