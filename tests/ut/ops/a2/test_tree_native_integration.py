# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production node-plan adapter, including metadata producers in FULL replay."""

import pytest
import torch

from tests.ut.ops.a2.test_tree_native_attention import native_ops  # noqa: F401
from tests.ut.ops.a2.test_tree_native_c128_attention import make_c128_case
from tests.ut.ops.a2.test_tree_native_pool_attention import make_pool_case
from vllm_ascend.ops.triton.spec_decode.tree.native_attention import native_attention
from vllm_ascend.ops.triton.spec_decode.tree.node_attention import node_attention


@pytest.mark.parametrize("requests", [1, 2, 4, 8])
@pytest.mark.parametrize("ratio,prefix", [(4, 3), (4, 127), (128, 127), (128, 511)])
def test_node_adapter_and_live_full_replay(requests, ratio, prefix):
    fixture = make_pool_case if ratio == 4 else make_c128_case
    q, tree, _, _ = fixture(requests=requests, prefix=prefix)
    n = q.shape[0]
    ancestors = tree["tree_ancestors"]
    depths = (ancestors >= 0).sum(-1) - 1
    plan = dict(
        n=n,
        max_q=ancestors.shape[1],
        max_kv=prefix + ancestors.shape[1] + 2,
        ancestors=ancestors,
        req_ids=torch.arange(n, device=q.device) // 17,
        start=tree["tree_prefix"],
        depths=depths,
        query_cu=torch.arange(n + 1, dtype=torch.int32, device=q.device),
    )
    args = (
        q,
        tree["ori_kv"],
        tree["cmp_kv"],
        tree["cmp_tree_kv"],
        tree["ori_block_table"][::17].contiguous(),
        tree["cmp_block_table"][::17].contiguous(),
        plan,
        tree.get("cmp_sparse_indices"),
        tree["sinks"],
        ratio,
        128,
        512**-0.5,
    )

    def run():
        # These producers must be captured, not frozen in Python before capture.
        plan["query_metadata"] = {}
        plan["seqused"] = (plan["start"] + plan["depths"] + 1).int()
        if ratio == 4:
            # The real indexer also refreshes its logical Top-K after a prefix
            # change; keeping the old valid count violates the native contract.
            count = plan["seqused"].view(n, 1, 1) // ratio
            positions = torch.arange(512, device=q.device).view(1, 1, 512)
            tree["cmp_sparse_indices"].copy_(torch.where(positions < count, count - 1 - positions, -1))
        return native_attention(*args)

    actual = run()
    expected = torch.ops._C_ascend.npu_sparse_attn_sharedkv_tree(q, **tree)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, node_attention(*args), atol=0.001, rtol=0.01)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for delta in (1, -1):
        plan["start"].add_(delta)
        tree["cmp_tree_kv"].mul_(0.9)
        expected = run()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        torch.testing.assert_close(captured, node_attention(*args), atol=0.001, rtol=0.01)
