"""Byte-preserving accepted-row transport shared by KV and hidden states."""

from dataclasses import dataclass

import torch
import torch_npu


@dataclass
class TreeRowMove:
    """Device row indices and optional persistent gather storage.

    ``source`` and ``destination`` are [rows, index_dims]. A negative
    destination disables the write. Indices address the leading dimensions;
    each remaining row is transported as bytes, including quantized payloads.
    """

    tensor: torch.Tensor
    source: torch.Tensor
    destination: torch.Tensor
    scratch: torch.Tensor | None = None


def relocate_tree_rows(moves: list[TreeRowMove], *, deduplicate: bool) -> None:
    """Gather every source before writing any aliased destination.

    Paged caches can have several bindings to the same physical view. Their
    destination slots are combined and deduplicated before scatter. Query
    rows occupy disjoint request intervals and do not need that sorting step.
    All data and indices stay on device; only shapes and view identities are
    inspected by the host.
    """
    transfers = {}
    writes = []
    for move in moves:
        view = move.tensor.view(torch.int8)
        dimensions = move.source.shape[1]
        if dimensions == 1:
            gathered = view.index_select(0, move.source[:, 0])
        else:
            gathered = view[tuple(move.source[:, axis] for axis in range(dimensions))]
        if move.scratch is not None:
            scratch = move.scratch.view(torch.int8)
            scratch.copy_(gathered)
            gathered = scratch
        if not deduplicate:
            writes.append((view, move.destination, gathered))
            continue
        key = (view.data_ptr(), tuple(view.shape), tuple(view.stride()))
        transfer = transfers.setdefault(key, (view, [], [], dimensions))
        transfer[1].append(move.destination)
        transfer[2].append(gathered)

    for view, destinations, values, dimensions in transfers.values():
        destination = torch.cat(destinations)
        keep = (destination >= 0).all(dim=1)
        flat = destination[:, 0]
        capacity = view.shape[0]
        for axis in range(1, dimensions):
            flat = flat * view.shape[axis] + destination[:, axis]
            capacity *= view.shape[axis]
        # Invalid rows sort after real slots and never acquire a write target.
        flat = torch.where(keep, flat, capacity)
        flat, order = flat.sort()
        keep = keep[order]
        data = torch.cat(values).index_select(0, order)
        duplicate = torch.zeros_like(keep)
        duplicate[1:] = keep[1:] & keep[:-1] & (flat[1:] == flat[:-1])
        destination = destination.index_select(0, order)
        destination = torch.where((keep & ~duplicate).unsqueeze(1), destination, -1)
        writes.append((view, destination, data))

    for view, destination, data in writes:
        torch_npu.npu_scatter_nd_update_(view, destination, data)
