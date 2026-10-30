# SPDX-License-Identifier: Apache-2.0
"""Tree state page strides, storage offsets and live graph resume indices."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.gdn import _chunked_tree_verify
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    compute_tree_init_state_indices,
    compute_tree_resume_column,
    compute_tree_verify_plan,
)

pytestmark = pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")


def _inputs(generator):
    tokens, key_heads, value_heads, dimension = 17, 2, 6, 128
    q, k = [
        torch.nn.functional.normalize(torch.randn(tokens, key_heads, dimension, generator=generator), dim=-1)
        .bfloat16()
        .npu()
        .unsqueeze(0)
        for _ in range(2)
    ]
    v = torch.randn(tokens, value_heads, dimension, generator=generator).bfloat16().npu()[None]
    g = -(torch.rand(tokens, value_heads, generator=generator) * 0.2 + 0.1).npu()[None]
    beta = torch.rand(tokens, value_heads, generator=generator).bfloat16().npu()[None]
    return [q, k, v, g, beta]


def _copy_live_plan(destination, source):
    for old, new in zip(destination.chunks, source.chunks):
        for name in ("token_indices", "state_indices", "init_state_indices", "out_indices", "cu_seqlens"):
            getattr(old, name).copy_(getattr(new, name))


@pytest.mark.parametrize("graph", [False, True])
def test_tree_strided_offset_pages_and_two_round_resume_match_dense(graph):
    from vllm_ascend import vllm_ascend_C  # noqa: F401

    generator = torch.Generator().manual_seed(2030)
    pages, heads, dim, tokens = 24, 6, 128, 17
    elements, leading, trailing = heads * dim * dim, 64, 1024
    initial = torch.randn(pages, heads, dim, dim, generator=generator) * 0.03
    dense = initial.npu()
    backing = torch.full((pages, leading + elements + trailing), 123.5, device="npu")
    state = backing[:, leading : leading + elements].view_as(dense)
    assert state.storage_offset() == leading
    assert state.stride(0) > elements
    state.copy_(dense)
    indices = torch.randperm(pages, generator=generator)[:tokens].int().npu()[None]
    parents = torch.tensor([[(node - 1) // 2 for node in range(1, tokens)]], device="npu")
    counts = torch.tensor([tokens - 1], dtype=torch.int32, device="npu")
    accepted = torch.ones(1, dtype=torch.int32, device="npu")
    lengths = torch.tensor([tokens], dtype=torch.int32, device="npu")
    init = compute_tree_init_state_indices(indices, accepted, parents, counts)
    plan = compute_tree_verify_plan(lengths, indices, init, tokens)
    inputs = _inputs(generator)
    scale = dim**-0.5

    def run(pool):
        return _chunked_tree_verify(*inputs, pool, indices, accepted, scale, plan, tokens)

    if graph:
        run(state)
        state.copy_(dense)
        capture = torch.npu.NPUGraph()
        with torch.npu.graph(capture):
            actual = run(state)
        state.copy_(dense)

    for step in range(2):
        if step:
            # Accepted node16 is in chunk two and has non-prefix path 1,3,7,16.
            path = torch.tensor([[1, 3, 7, 16, -1, -1, -1]], dtype=torch.int32, device="npu")
            accepted.copy_(compute_tree_resume_column(path, torch.tensor([5], dtype=torch.int32, device="npu")))
            assert accepted.cpu().tolist() == [17]
            init = compute_tree_init_state_indices(indices, accepted, parents, counts)
            _copy_live_plan(plan, compute_tree_verify_plan(lengths, indices, init, tokens))
            for target, replacement in zip(inputs, _inputs(generator)):
                target.copy_(replacement)
        expected = run(dense)
        if graph:
            capture.replay()
        else:
            actual = run(state)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, dense, rtol=0, atol=0)
        assert torch.all(backing[:, :leading] == 123.5).cpu().item()
        assert torch.all(backing[:, leading + elements :] == 123.5).cpu().item()


@pytest.mark.parametrize("layout", ["transposed_inner", "padded_inner", "overlapping_pages"])
def test_tree_rejects_unsupported_inner_state_layout(layout):
    from vllm_ascend import vllm_ascend_C  # noqa: F401

    pages, heads, dim = 24, 6, 128
    if layout == "transposed_inner":
        state = torch.zeros(pages, heads, dim, dim, device="npu").transpose(-1, -2)
    elif layout == "padded_inner":
        state = torch.zeros(pages, heads, dim + 1, dim, device="npu")[:, :, :dim]
    else:
        state = torch.zeros(1, heads, dim, dim, device="npu").expand(pages, -1, -1, -1)
    args = dict(
        query=torch.zeros(1, 2, dim, dtype=torch.bfloat16, device="npu"),
        key=torch.zeros(1, 2, dim, dtype=torch.bfloat16, device="npu"),
        value=torch.zeros(1, heads, dim, dtype=torch.bfloat16, device="npu"),
        state=state,
        beta=torch.ones(1, heads, dtype=torch.bfloat16, device="npu"),
        g=torch.zeros(1, heads, device="npu"),
        scale=dim**-0.5,
        actual_seq_lengths=torch.tensor([0, 1], dtype=torch.int32, device="npu"),
        ssm_state_indices=torch.zeros(1, dtype=torch.int32, device="npu"),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device="npu"),
        init_state_indices=torch.zeros(1, dtype=torch.int32, device="npu"),
    )
    with pytest.raises(RuntimeError, match="dense within each non-overlapping page"):
        torch.ops._C_ascend.npu_recurrent_gated_delta_rule_tree(**args)
