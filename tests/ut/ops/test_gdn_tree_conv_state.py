# SPDX-License-Identifier: Apache-2.0
"""Independent tree-convolution and accepted-prefix state regressions."""

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.gdn import tree_causal_conv1d
from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    compute_tree_conv_rows,
    compute_tree_prefix_source,
    compute_tree_resume_column,
)


@pytest.fixture(autouse=True)
def cpu_tree_page_io(monkeypatch):
    # Mock only the NPU copy boundary. The tree convolution and history
    # bookkeeping below still execute their actual production code.
    import vllm_ascend.ops.gdn as gdn

    def gather(state, indices, valid, **kwargs):
        assert state.device.type == "cpu"
        selected = state.index_select(0, indices.long())
        mask = valid.reshape(-1, *([1] * (state.ndim - 1)))
        return torch.where(mask, selected, torch.zeros_like(selected))

    def scatter(state, indices, valid, source):
        assert state.device.type == "cpu"
        for row in range(indices.numel()):
            if bool(valid[row]):
                state[int(indices[row])].copy_(source[row])

    monkeypatch.setattr(gdn, "gather_ssm_states", gather)
    monkeypatch.setattr(gdn, "scatter_tree_conv_pages_", scatter)


@pytest.mark.parametrize("accepted_path", [[], [2], [2, 6], [1, 3, 7, 15]])
@pytest.mark.parametrize("dim_first", [False, True])
def test_conv_windows_and_resume_follow_only_accepted_ancestors(accepted_path, dim_first):
    torch.manual_seed(67)
    slots, width, channels = 17, 4, 32
    parents = [(node - 1) // 2 for node in range(1, slots)]
    parent_table = torch.tensor([parents], dtype=torch.int32)
    history = torch.randn(width - 1, channels).bfloat16()
    weight = torch.randn(channels, width).bfloat16()
    bias = torch.randn(channels).bfloat16()
    state = torch.randn(4, slots + width - 1, channels).bfloat16()
    state[2, : width - 1] = history
    if dim_first:
        state = state.transpose(1, 2).contiguous()
    cache = torch.full((1, slots), 2, dtype=torch.int32)
    starts = torch.tensor([0, slots], dtype=torch.int32)
    windows = compute_tree_conv_rows(parent_table, torch.tensor([slots - 1]), width, width - 1)
    prefix = torch.arange(width - 1, dtype=torch.int32).reshape(1, -1)
    first_x = None
    for step in range(2):
        x = torch.randn(slots, channels).bfloat16()
        expected = []
        for node in range(slots):
            ancestry = [node]
            while ancestry[-1] > 0:
                ancestry.append(parents[ancestry[-1] - 1])
            sequence = torch.cat((history, x[list(reversed(ancestry))]), dim=0)[-width:]
            out = (sequence.float() * weight.T.float()).sum(dim=0) + bias.float()
            expected.append(torch.nn.functional.silu(out).bfloat16())
        with patch("vllm_ascend.ops.gdn.is_conv_state_dim_first", return_value=dim_first):
            actual = tree_causal_conv1d(x, weight, state, cache, starts, windows, prefix, bias, True)
        torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
        if step == 0:
            first_x = x
            path = torch.full((1, slots - 1), -1, dtype=torch.int32)
            path[0, : len(accepted_path)] = torch.tensor(accepted_path, dtype=torch.int32)
            sampled = torch.tensor([len(accepted_path) + 1], dtype=torch.int32)
            prefix = compute_tree_prefix_source(path, sampled, width)
            resume = compute_tree_resume_column(path, sampled)
            assert resume.tolist() == [(accepted_path[-1] if accepted_path else 0) + 1]
            history = torch.cat((history, first_x[[0] + accepted_path]), dim=0)[-(width - 1) :]


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("accepted_path", [[], [2, 6], [1, 3, 7, 16]])
@pytest.mark.parametrize("leading,trailing", [(0, 0), (64, 37)])
def test_conv_two_round_strided_offset_pages_preserve_other_cache_views(dim_first, accepted_path, leading, trailing):
    generator = torch.Generator().manual_seed(2031)
    pages, slots, width, channels = 5, 17, 4, 32
    columns = slots + width - 1
    elements = columns * channels
    backing = torch.full((pages, leading + elements + trailing), -91.0, dtype=torch.bfloat16)
    shape = (pages, channels, columns) if dim_first else (pages, columns, channels)
    state = backing[:, leading : leading + elements].view(shape)
    state.copy_(torch.randn(shape, generator=generator).bfloat16())
    cache_page = 3
    history = torch.randn(width - 1, channels, generator=generator).bfloat16()
    if dim_first:
        state[cache_page, :, : width - 1] = history.T
    else:
        state[cache_page, : width - 1] = history
    parents = [(node - 1) // 2 for node in range(1, slots)]
    windows = compute_tree_conv_rows(
        torch.tensor([parents], dtype=torch.int32),
        torch.tensor([slots - 1]),
        width,
        width - 1,
    )
    prefix = torch.arange(width - 1, dtype=torch.int32)[None]
    weights = torch.randn(channels, width, generator=generator).bfloat16()
    bias = torch.randn(channels, generator=generator).bfloat16()
    cache = torch.full((1, slots), cache_page, dtype=torch.int32)
    starts = torch.tensor([0, slots], dtype=torch.int32)
    for step in range(2):
        x = torch.randn(slots, channels, generator=generator).bfloat16()
        expected_out = []
        for node in range(slots):
            ancestry = [node]
            while ancestry[-1]:
                ancestry.append(parents[ancestry[-1] - 1])
            window = torch.cat((history, x[list(reversed(ancestry))]), dim=0)[-width:]
            value = (window.float() * weights.T.float()).sum(0) + bias.float()
            expected_out.append(torch.nn.functional.silu(value).bfloat16())
        before = backing.clone()
        expected_state = state.clone()
        if dim_first:
            expected_state[cache_page, :, : width - 1] = history.T
            expected_state[cache_page, :, width - 1 :] = x.T
        else:
            expected_state[cache_page, : width - 1] = history
            expected_state[cache_page, width - 1 :] = x
        with patch("vllm_ascend.ops.gdn.is_conv_state_dim_first", return_value=dim_first):
            actual = tree_causal_conv1d(x, weights, state, cache, starts, windows, prefix, bias, True)
        torch.testing.assert_close(actual, torch.stack(expected_out), rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        torch.testing.assert_close(backing[:, :leading], before[:, :leading], rtol=0, atol=0)
        torch.testing.assert_close(backing[:, leading + elements :], before[:, leading + elements :], rtol=0, atol=0)
        if not step:
            path = torch.full((1, slots - 1), -1, dtype=torch.int32)
            path[0, : len(accepted_path)] = torch.tensor(accepted_path, dtype=torch.int32)
            prefix = compute_tree_prefix_source(path, torch.tensor([len(accepted_path) + 1], dtype=torch.int32), width)
            history = torch.cat((history, x[[0] + accepted_path]), dim=0)[-(width - 1) :]


@pytest.mark.parametrize("dim_first", [False, True])
def test_conv_inactive_row_alias_does_not_undo_active_page_updates(dim_first):
    generator = torch.Generator().manual_seed(2032)
    slots, width, channels = 17, 4, 32
    columns = slots + width - 1
    shape = (3, channels, columns) if dim_first else (3, columns, channels)
    state = torch.randn(shape, generator=generator).bfloat16()
    history = state[0, :, : width - 1].T.clone() if dim_first else state[0, : width - 1].clone()
    parents = [(node - 1) // 2 for node in range(1, slots)]
    # One real BS1 request plus one zero-length graph-padding row. Both
    # metadata rows carry harmless page0, but only the real row may store.
    windows = compute_tree_conv_rows(
        torch.tensor([parents], dtype=torch.int32),
        torch.tensor([slots - 1]),
        width,
        2 * (width - 1),
    )
    cache = torch.zeros((2, slots), dtype=torch.int32)
    starts = torch.tensor([0, slots, slots], dtype=torch.int32)
    prefix = torch.arange(width - 1, dtype=torch.int32)[None].expand(2, -1).clone()
    weight = torch.randn(channels, width, generator=generator).bfloat16()
    for step in range(2):
        x = torch.randn(slots, channels, generator=generator).bfloat16()
        expected = []
        for node in range(slots):
            ancestry = [node]
            while ancestry[-1]:
                ancestry.append(parents[ancestry[-1] - 1])
            taps = torch.cat((history, x[list(reversed(ancestry))]), dim=0)[-width:]
            expected.append((taps.float() * weight.T.float()).sum(0).bfloat16())
        untouched = state[1:].clone()
        with patch("vllm_ascend.ops.gdn.is_conv_state_dim_first", return_value=dim_first):
            actual = tree_causal_conv1d(x, weight, state, cache, starts, windows, prefix, None, False)
        torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
        actual_history = state[0, :, : width - 1].T if dim_first else state[0, : width - 1]
        torch.testing.assert_close(actual_history, history, rtol=0, atol=0)
        torch.testing.assert_close(state[1:], untouched, rtol=0, atol=0)
        if not step:
            path = torch.tensor([[1, 3, 7, 16, -1, -1, -1]], dtype=torch.int32)
            prefix[0] = compute_tree_prefix_source(path, torch.tensor([5], dtype=torch.int32), width)[0]
            history = x[[3, 7, 16]].clone()
