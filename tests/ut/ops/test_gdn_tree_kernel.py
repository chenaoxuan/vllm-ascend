# SPDX-License-Identifier: Apache-2.0
"""Hardware regressions for the tree-only GDN state snapshot operator."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.gdn import _chunked_tree_verify
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    compute_tree_init_state_indices,
    compute_tree_verify_plan,
)


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
@pytest.mark.parametrize("tokens", [1, 2, 8, 16, 17])
@pytest.mark.parametrize("layout", ["chain", "siblings"])
def test_tree_kernel_outputs_and_all_state_pages(tokens, layout):
    from vllm_ascend import vllm_ascend_C  # noqa: F401

    # Compare every output and every snapshot to an independent CPU recurrence.
    # Non-contiguous pages detect accidental use of token/column IDs as page IDs.
    torch.manual_seed(1234)
    heads, dim = 2, 128
    q = torch.nn.functional.normalize(torch.randn(tokens, heads, dim), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(tokens, heads, dim), dim=-1).bfloat16()
    v = torch.randn(tokens, heads, dim).bfloat16()
    beta = torch.rand(tokens, heads).bfloat16()
    decay = -(torch.rand(tokens, heads) * 0.2 + 0.1)  # Operator input is log decay.
    slots = torch.randperm(tokens + 5)[:tokens].to(torch.int32)
    committed = tokens + 5
    initial = torch.randn(tokens + 6, heads, dim, dim) * 0.03
    parents = torch.tensor(
        [max(0, node - 1) if layout == "chain" else (node - 1) // 2 for node in range(1, tokens)], dtype=torch.int32
    ).reshape(1, -1)
    states = initial.clone()
    outputs = []
    for node in range(tokens):
        parent_page = committed if node == 0 else int(slots[parents[0, node - 1]])
        state = states[parent_page].clone() * decay[node, :, None, None].exp()
        residual = v[node].float() - torch.einsum("hvk,hk->hv", state, k[node].float())
        state = state + (residual * beta[node].float()[:, None])[:, :, None] * k[node].float()[:, None, :]
        states[int(slots[node])] = state
        outputs.append(torch.einsum("hvk,hk->hv", state, q[node].float()) * dim**-0.5)
    expected = torch.stack(outputs)
    device = "npu:0"
    state_device = initial.to(device)
    indices = slots.reshape(1, -1).to(device)
    # Initial page is deliberately separate from the output pages.
    init = torch.tensor(
        [committed] + [int(slots[p]) for p in parents.reshape(-1)], dtype=torch.int32, device=device
    ).reshape(1, -1)
    actual = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    accepted = torch.ones(1, dtype=torch.int32, device=device)
    plan = compute_tree_verify_plan(torch.tensor([tokens], device=device), indices, init, tokens)
    args = [t.to(device).unsqueeze(0) for t in (q, k, v, decay, beta)]
    if plan is None:
        result = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_tree(
            query=args[0].squeeze(0),
            key=args[1].squeeze(0),
            value=args[2].squeeze(0),
            state=state_device,
            g=args[3].squeeze(0),
            beta=args[4].squeeze(0),
            scale=dim**-0.5,
            actual_seq_lengths=actual,
            ssm_state_indices=indices.reshape(-1),
            num_accepted_tokens=accepted,
            init_state_indices=init.reshape(-1),
        )
    else:
        result = _chunked_tree_verify(*args, state_device, indices, accepted, dim**-0.5, plan, tokens)
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu().float(), expected.bfloat16().float(), rtol=0.02, atol=0.001)
    torch.testing.assert_close(state_device.cpu(), states, rtol=0.002, atol=0.0002)
    # Unused pages, including the committed prefix, must remain untouched.
    unused = sorted(set(range(tokens + 6)) - set(slots.tolist()))
    torch.testing.assert_close(state_device.cpu()[unused], initial[unused], rtol=0, atol=0)


def test_tree_init_pages_resume_from_selected_node():
    indices = torch.tensor([[19, 8, 14, 3, 21]], dtype=torch.int32)
    parents = torch.tensor([[0, 0, 1, 2]], dtype=torch.int32)
    actual = compute_tree_init_state_indices(
        indices,
        torch.tensor([4], dtype=torch.int32),
        parents,
        torch.tensor([4], dtype=torch.int32),
    )
    assert actual.tolist() == [[3, 19, 19, 8, 14]]


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
def test_tree_kernel_padded_state_pages_match_dense():
    from vllm_ascend import vllm_ascend_C  # noqa: F401

    torch.manual_seed(432)
    tokens, heads, dim = 3, 2, 128
    shape = (8, heads, dim, dim)
    dense = torch.randn(shape, device="npu", dtype=torch.float32)
    # Model Runner V2 places SSM rows in hybrid KV pages with a larger stride.
    backing = torch.empty((8, heads * dim * dim + 1024), device="npu")
    padded = backing[:, : heads * dim * dim].view(shape)
    padded.copy_(dense)
    q, k = [torch.nn.functional.normalize(torch.randn(tokens, heads, dim), dim=-1).bfloat16().npu() for _ in range(2)]
    v = torch.randn_like(q)
    beta = torch.rand(tokens, heads, dtype=torch.bfloat16, device="npu")
    g = -torch.rand(tokens, heads, device="npu")
    kwargs = dict(
        query=q,
        key=k,
        value=v,
        g=g,
        beta=beta,
        scale=dim**-0.5,
        actual_seq_lengths=torch.tensor([0, tokens], dtype=torch.int32, device="npu"),
        ssm_state_indices=torch.tensor([1, 4, 6], dtype=torch.int32, device="npu"),
        init_state_indices=torch.tensor([7, 1, 1], dtype=torch.int32, device="npu"),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32, device="npu"),
    )
    op = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_tree
    expected = op(state=dense, **kwargs)
    actual = op(state=padded, **kwargs)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(padded, dense, rtol=0, atol=0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
@pytest.mark.parametrize("accepted_node", [0, 2, 6, 16])
def test_two_round_verify_restores_the_accepted_branch(accepted_node):
    from vllm_ascend import vllm_ascend_C  # noqa: F401
    from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import compute_tree_resume_column

    torch.manual_seed(789)
    tokens, heads, dim = 17, 2, 128
    slots = torch.randperm(24)[:tokens].to(torch.int32)
    parent_ids = [(node - 1) // 2 for node in range(1, tokens)]
    parents = torch.tensor([parent_ids], dtype=torch.int32, device="npu")
    counts = torch.tensor([tokens - 1], dtype=torch.int32, device="npu")
    indices = slots.reshape(1, -1).npu()
    expected_states = torch.randn(24, heads, dim, dim) * 0.03
    backing = torch.empty((24, heads * dim * dim + 1024), device="npu")
    actual_states = backing[:, : heads * dim * dim].view(expected_states.shape)
    actual_states.copy_(expected_states)
    accepted = torch.ones(1, dtype=torch.int32, device="npu")
    root_page = int(slots[0])
    for step in range(2):
        q, k = [torch.nn.functional.normalize(torch.randn(tokens, heads, dim), dim=-1).bfloat16() for _ in range(2)]
        v = torch.randn(tokens, heads, dim).bfloat16()
        beta = torch.rand(tokens, heads).bfloat16()
        g = -torch.rand(tokens, heads)
        expected_outputs = []
        for node in range(tokens):
            page = root_page if node == 0 else int(slots[parent_ids[node - 1]])
            state = expected_states[page].clone() * g[node, :, None, None].exp()
            residual = v[node].float() - torch.einsum("hvk,hk->hv", state, k[node].float())
            state += (residual * beta[node].float()[:, None])[:, :, None] * k[node].float()[:, None, :]
            expected_states[int(slots[node])] = state
            expected_outputs.append(torch.einsum("hvk,hk->hv", state, q[node].float()) * dim**-0.5)
        init = compute_tree_init_state_indices(indices, accepted, parents, counts)
        plan = compute_tree_verify_plan(torch.tensor([tokens], device="npu"), indices, init, tokens)
        args = [tensor.npu().unsqueeze(0) for tensor in (q, k, v, g, beta)]
        result = _chunked_tree_verify(*args, actual_states, indices, accepted, dim**-0.5, plan, tokens)
        torch.testing.assert_close(
            result.cpu().float(), torch.stack(expected_outputs).bfloat16().float(), rtol=0.02, atol=0.001
        )
        torch.testing.assert_close(actual_states.cpu(), expected_states, rtol=0.002, atol=0.0002)
        if step == 0:
            # A deliberately non-prefix path; node 16 also lives in chunk two.
            path = []
            node = accepted_node
            while node:
                path.append(node)
                node = parent_ids[node - 1]
            path.reverse()
            table = torch.full((1, 7), -1, dtype=torch.int32, device="npu")
            if path:
                table[0, : len(path)] = torch.tensor(path, dtype=torch.int32, device="npu")
            accepted = compute_tree_resume_column(table, torch.tensor([len(path) + 1], dtype=torch.int32, device="npu"))
            assert accepted.cpu().tolist() == [accepted_node + 1]
            root_page = int(slots[accepted_node])


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
def test_tree_verify_graph_replays_live_parent_pages_and_padded_state():
    from vllm_ascend import vllm_ascend_C  # noqa: F401

    torch.manual_seed(127)
    tokens, heads, dim = 17, 2, 128
    indices = torch.arange(tokens, dtype=torch.int32, device="npu").reshape(1, -1)
    parents = torch.tensor([[(node - 1) // 2 for node in range(1, tokens)]], dtype=torch.int32, device="npu")
    init = compute_tree_init_state_indices(
        indices,
        torch.ones(1, dtype=torch.int32, device="npu"),
        parents,
        torch.tensor([tokens - 1], dtype=torch.int32, device="npu"),
    )
    plan = compute_tree_verify_plan(torch.tensor([tokens], device="npu"), indices, init, tokens)
    q, k = [torch.nn.functional.normalize(torch.randn(tokens, heads, dim), dim=-1).bfloat16().npu() for _ in range(2)]
    v = torch.randn_like(q)
    g = -torch.rand(tokens, heads, device="npu")
    beta = torch.rand(tokens, heads, dtype=torch.bfloat16, device="npu")
    args = [tensor.unsqueeze(0) for tensor in (q, k, v, g, beta)]
    initial = torch.randn(24, heads, dim, dim, device="npu") * 0.02
    backing = torch.empty((24, heads * dim * dim + 1024), device="npu")
    state = backing[:, : heads * dim * dim].view(initial.shape)
    accepted = torch.ones(1, dtype=torch.int32, device="npu")
    state.copy_(initial)
    _chunked_tree_verify(*args, state, indices, accepted, dim**-0.5, plan, tokens)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = _chunked_tree_verify(*args, state, indices, accepted, dim**-0.5, plan, tokens)
    for root_page in [0, 16]:
        # Update the captured input buffer rather than replacing its address.
        plan.chunks[0].init_state_indices[:1].copy_(indices[0, root_page : root_page + 1])
        reference = initial.clone()
        state.copy_(initial)
        expected = _chunked_tree_verify(*args, reference, indices, accepted, dim**-0.5, plan, tokens)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(state, reference, rtol=0, atol=0)
