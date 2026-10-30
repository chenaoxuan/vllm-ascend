# SPDX-License-Identifier: Apache-2.0
"""Accepted tree rows must become the next DFlash2 draft's linear context.

These are BS1 contract tests with depth 7 and budget 16. CPU tests replace only
the NPU scatter primitive; the optional NPU test uses the production preparation
kernel. No model weights or LLM runner are loaded.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import prepare_dflash_inputs
from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.commit import TreeCommitCoordinator

_DEPTH = 7
_BUDGET = 16
_TARGET_ROWS = _BUDGET + 1
_QUERY_ROWS = _DEPTH + 1
_WIDTH = 2
_AUX_LAYERS = 5
_CONTEXT_CAPACITY = 32


def _cpu_scatter_nd_update(tensor, indices, updates):
    """CPU implementation of the generic scatter boundary, including PAD rows."""
    live = (indices >= 0).all(dim=1)
    coordinates = tuple(indices[live, axis].long() for axis in range(indices.shape[1]))
    tensor[coordinates] = updates[live]
    return tensor


def _coordinator():
    runner = SimpleNamespace(
        tree_kv_compact=object(),
        speculator=SimpleNamespace(packed_dsv4_verify=lambda: False),
    )
    return TreeCommitCoordinator(runner)


def _round_inputs(device, root_position, accepted_nodes, round_index):
    # last_hidden_states aliases aux[0]. A sequential gather/write per tensor
    # would compact that storage twice, corrupting the overlapping [2,3,16] path.
    storage = torch.arange(_TARGET_ROWS * _AUX_LAYERS * _WIDTH, device=device, dtype=torch.float32).reshape(
        _TARGET_ROWS, _AUX_LAYERS * _WIDTH
    )
    storage += round_index * 1000
    aux = [storage[:, i * _WIDTH : (i + 1) * _WIDTH] for i in range(_AUX_LAYERS)]
    last_hidden = aux[0]
    positions = root_position + torch.arange(_TARGET_ROWS, device=device) % 3
    positions[0] = root_position
    path = torch.full((1, _DEPTH), -1, dtype=torch.long, device=device)
    path[0, : len(accepted_nodes)] = torch.tensor(accepted_nodes, device=device)
    for depth, node in enumerate(accepted_nodes, 1):
        positions[node] = root_position + depth
    acceptance = TreeAcceptance.from_path(
        path, torch.tensor([len(accepted_nodes) + 1], dtype=torch.int32, device=device)
    )
    qsl = torch.tensor([0, _TARGET_ROWS], dtype=torch.int32, device=device)
    return last_hidden, aux, positions, acceptance, qsl


@pytest.mark.parametrize("precompute_slice", [(0, _TARGET_ROWS), (1, 4)])
def test_two_round_tree_commit_and_dflash_context_consumers(monkeypatch, precompute_slice):
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", _cpu_scatter_nd_update, raising=False)
    commit = _coordinator()
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.hidden_states = torch.full((_CONTEXT_CAPACITY, _WIDTH), -999.0)
    speculator.context_positions = torch.zeros(_CONTEXT_CAPACITY, dtype=torch.long)
    speculator._context_slot_mappings = torch.full((1, _CONTEXT_CAPACITY), PAD_SLOT_ID, dtype=torch.int32)
    speculator._layer_group_idx = None
    # The model boundary is deliberately recorded rather than implementing a
    # fake KV updater. Real DFlash methods decide the features/positions/slots.
    weights = torch.tensor([1.0, 3.0, 7.0, 11.0, 13.0]).reshape(1, _AUX_LAYERS, 1)

    def combine(features):
        return (features.reshape(-1, _AUX_LAYERS, _WIDTH) * weights).sum(dim=1)

    model = SimpleNamespace(
        combine_hidden_states=combine,
        precompute_and_store_context_kv=Mock(),
    )
    speculator.model = model
    for round_index, (root, nodes) in enumerate(((9, [2, 3, 16]), (13, [8, 12]))):
        last, aux, positions, acceptance, qsl = _round_inputs("cpu", root, nodes, round_index)
        originals = [tensor.clone() for tensor in (last, *aux)]
        original_positions = positions.clone()
        source_rows = torch.tensor([0, *nodes])
        accepted_rows = len(nodes) + 1
        expected_features = [torch.cat((tensor[source_rows], tensor[accepted_rows:]), dim=0) for tensor in originals]
        commit.compact_queries([last, *aux], qsl, acceptance, positions)
        for actual, expected in zip((last, *aux), expected_features):
            torch.testing.assert_close(actual, expected)
        expected_positions = original_positions.clone()
        expected_positions[:accepted_rows] = torch.arange(root, root + accepted_rows)
        torch.testing.assert_close(positions, expected_positions)

        DFlashSpeculator._store_hidden_states(speculator, _TARGET_ROWS, last, aux)
        expected_hidden = combine(torch.cat(expected_features[1:], dim=-1))
        torch.testing.assert_close(speculator.hidden_states[:_TARGET_ROWS], expected_hidden)
        assert torch.all(speculator.hidden_states[_TARGET_ROWS:] == -999)
        # The CPU consumer test supplies this producer boundary explicitly;
        # the NPU test below validates the actual preparation kernel's values.
        speculator.context_positions.zero_()
        speculator.context_positions[:accepted_rows] = expected_positions[:accepted_rows]
        speculator._context_slot_mappings.fill_(PAD_SLOT_ID)
        speculator._context_slot_mappings[0, :accepted_rows] = torch.arange(40, 40 + accepted_rows)
        start, end = precompute_slice
        DFlashSpeculator._precompute_context_kv(speculator, start, end)
        hidden_arg, position_arg, slot_arg = model.precompute_and_store_context_kv.call_args.args
        torch.testing.assert_close(hidden_arg, expected_hidden[start:end])
        torch.testing.assert_close(position_arg, speculator.context_positions[start:end])
        torch.testing.assert_close(slot_arg, speculator._context_slot_mappings[0, start:end])
        assert slot_arg.shape == (end - start,)

    assert model.precompute_and_store_context_kv.call_count == 2


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Production DFlash input preparation requires an NPU",
)
def test_npu_two_round_nonprefix_tree_context_and_query_indices():
    device = "npu"
    capacity, max_reqs, state_id, block_size = _CONTEXT_CAPACITY, 4, 2, 4
    commit = _coordinator()
    buffers = SimpleNamespace(
        input_ids=torch.full((capacity,), -999, dtype=torch.long, device=device),
        positions=torch.full((capacity,), -999, dtype=torch.long, device=device),
        query_start_loc=torch.full((max_reqs + 1,), -999, dtype=torch.int32, device=device),
        seq_lens=torch.full((max_reqs,), -999, dtype=torch.int32, device=device),
    )
    query_slots = torch.full((capacity,), -999, dtype=torch.int32, device=device)
    context_positions = torch.full((capacity,), -999, dtype=torch.long, device=device)
    context_slots = torch.full((capacity,), -999, dtype=torch.int32, device=device)
    sample_indices = torch.full((max_reqs * _DEPTH,), -999, dtype=torch.long, device=device)
    sample_pos = torch.full_like(sample_indices, -999)
    sample_mapping = torch.full_like(sample_indices, -999)
    output_temperature = torch.zeros(max_reqs, device=device)
    output_seeds = torch.zeros(max_reqs, dtype=torch.long, device=device)
    input_temperature = torch.tensor([0.0, 0.1, 0.2, 0.3], device=device)
    input_seeds = torch.tensor([11, 12, 13, 14], dtype=torch.long, device=device)
    last_sampled = torch.tensor([1000, 1001, 4242, 1003], dtype=torch.long, device=device)
    next_prefill = torch.full_like(last_sampled, 9999)
    table_host = [11, 3, 19, 5, 23, 7, 29, 9] + list(range(31, 55))
    block_table = torch.tensor([table_host], dtype=torch.int32, device=device)
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.hidden_states = torch.empty((capacity, _WIDTH), device=device)
    speculator.context_positions = context_positions
    speculator._context_slot_mappings = context_slots.unsqueeze(0)
    speculator._layer_group_idx = None
    speculator.model = SimpleNamespace(
        combine_hidden_states=lambda features: features.reshape(-1, _AUX_LAYERS, _WIDTH).sum(dim=1),
        precompute_and_store_context_kv=Mock(),
    )
    for round_index, (root, nodes) in enumerate(((9, [2, 3, 16]), (13, [8, 12]))):
        last, aux, positions, acceptance, qsl = _round_inputs(device, root, nodes, round_index)
        expected_accepted = torch.stack(
            [tensor[torch.tensor([0, *nodes], device=device)].clone() for tensor in aux]
        ).sum(dim=0)
        commit.compact_queries([last, *aux], qsl, acceptance, positions)
        batch = SimpleNamespace(
            num_reqs=1,
            num_scheduled_tokens=torch.tensor([_TARGET_ROWS], dtype=torch.int32),
            positions=positions,
            query_start_loc=qsl,
            idx_mapping=torch.tensor([state_id], dtype=torch.int32, device=device),
        )
        num_rejected = torch.tensor([_TARGET_ROWS - len(nodes) - 1], dtype=torch.int32, device=device)
        prepare_dflash_inputs(
            buffers,
            query_slots,
            context_positions,
            context_slots,
            sample_indices,
            sample_pos,
            sample_mapping,
            output_temperature,
            output_seeds,
            batch,
            acceptance.num_sampled,
            num_rejected,
            last_sampled,
            next_prefill,
            input_temperature,
            input_seeds,
            block_table,
            block_size,
            0,
            1,
            1,
            248070,
            _QUERY_ROWS,
            _DEPTH,
            max_reqs,
            capacity,
            128,
            sample_from_anchor=False,
            kv_cache_block_size=block_size,
        )
        DFlashSpeculator._store_hidden_states(speculator, _TARGET_ROWS, last, aux)
        DFlashSpeculator._precompute_context_kv(speculator, 0, _TARGET_ROWS)
        hidden_arg, position_arg, slot_arg = speculator.model.precompute_and_store_context_kv.call_args.args
        accepted_rows = len(nodes) + 1
        torch.testing.assert_close(hidden_arg[:accepted_rows], expected_accepted)
        torch.testing.assert_close(position_arg, context_positions[:_TARGET_ROWS])
        torch.testing.assert_close(slot_arg, context_slots[:_TARGET_ROWS])
        valid_positions = list(range(root, root + accepted_rows))
        expected_context_slots = [
            table_host[position // block_size] * block_size + position % block_size for position in valid_positions
        ]
        assert context_positions[:_TARGET_ROWS].cpu().tolist() == (
            valid_positions + [0] * (_TARGET_ROWS - accepted_rows)
        )
        assert context_slots[:_TARGET_ROWS].cpu().tolist() == (
            expected_context_slots + [PAD_SLOT_ID] * (_TARGET_ROWS - accepted_rows)
        )
        query_positions = list(range(root + accepted_rows, root + accepted_rows + _QUERY_ROWS))
        assert buffers.positions[:_QUERY_ROWS].cpu().tolist() == query_positions
        assert buffers.input_ids[:_QUERY_ROWS].cpu().tolist() == [4242] + [248070] * _DEPTH
        assert query_slots[:_QUERY_ROWS].cpu().tolist() == [
            table_host[position // block_size] * block_size + position % block_size for position in query_positions
        ]
        assert sample_indices[:_DEPTH].cpu().tolist() == list(range(1, _QUERY_ROWS))
        assert sample_pos[:_DEPTH].cpu().tolist() == query_positions[1:]
        assert sample_mapping[:_DEPTH].cpu().tolist() == [state_id] * _DEPTH
        assert buffers.query_start_loc.cpu().tolist() == [0] + [_QUERY_ROWS] * max_reqs
        assert buffers.seq_lens[0].cpu().item() == query_positions[-1] + 1
        assert torch.all(query_slots[_QUERY_ROWS:] == PAD_SLOT_ID).cpu().item()
        assert torch.all(sample_mapping[_DEPTH:] == -1).cpu().item()
