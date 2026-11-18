# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.worker.v2.spec_decode.tree.shared_metadata import SharedGDNTreeMetadata


@pytest.mark.parametrize("num_nodes", [0, 32])
@pytest.mark.parametrize("sampled", [0, 1, 4, 16])
def test_complete_gdn_tree_topology(num_nodes, sampled):
    def strided_row(values):
        return torch.tensor(values, device="npu", dtype=torch.int32).repeat_interleave(2)[None, ::2]

    parents = [node // 2 for node in range(32)]
    depths = [0]
    for parent in parents:
        depths.append(depths[parent] + 1)
    args = (
        strided_row(parents),
        strided_row([num_nodes])[0],
        strided_row(depths[1:]),
        strided_row([max(1, sampled)])[0],
        strided_row(list(range(1, 33))),
        strided_row([sampled])[0],
        4,
    )
    expected = SharedGDNTreeMetadata.create(*args)
    actual = SharedGDNTreeMetadata.create(*args, use_fused_topology=True)
    assert torch.equal(actual.init_state_columns, expected.init_state_columns)
    assert torch.equal(actual.conv_window_rows, expected.conv_window_rows)
    assert torch.equal(actual.conv_prefix_src, expected.conv_prefix_src)
