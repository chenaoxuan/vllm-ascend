# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native tree aggregation over device-resident history and immutable node rows."""

import torch


def aggregate_native_nodes(
    state, nodes, table, plan, norm, cos, sin, ratio, head_dim, rope_dim, eps, dtype, workspace=None
):
    """All tensor inputs stay on device; geometry and dtype are host metadata.

    The native operator owns its output (including during FULL capture). State
    is read-only and accepted-chain commitment retains the existing node format.
    """
    if dtype != torch.bfloat16:
        raise ValueError("Native tree compressor currently requires BF16 output")
    return torch.ops._C_ascend.compressor_tree(
        state,
        nodes,
        norm,
        cos,
        sin,
        table,
        plan["start"],
        plan["ancestors"],
        plan["depths"],
        plan["req_ids"],
        ratio,
        rope_dim,
        eps,
    )
