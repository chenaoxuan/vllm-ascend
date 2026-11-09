# SPDX-License-Identifier: Apache-2.0
"""One verification step's logical GDN metadata, shared across cache groups."""

from dataclasses import dataclass, field

import torch

from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    GDN_RECURRENT_MAX_VERIFY_TOKENS,
    TreeVerifyChunk,
    TreeVerifyPlan,
    compute_tree_conv_rows,
    compute_tree_prefix_source,
)


@dataclass(frozen=True)
class _VerifyGeometry:
    token_indices: torch.Tensor
    owned: torch.Tensor
    out_indices: torch.Tensor
    cu_seqlens: torch.Tensor


@dataclass(frozen=True)
class SharedGDNTreeMetadata:
    """Logical metadata for one complete BS1 tree, without physical state pages.

    A model state's prepare_attn owns this object for one step. Builders bind
    their own page tables, then retain their existing graph-stable copy targets.
    """

    slots: int
    conv_width: int
    init_state_columns: torch.Tensor
    conv_window_rows: torch.Tensor
    conv_prefix_src: torch.Tensor
    _verify_geometry: dict[int, tuple[_VerifyGeometry, ...]] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        parents: torch.Tensor,
        num_nodes: torch.Tensor,
        depths: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        prev_path_node_ids: torch.Tensor,
        prev_num_sampled: torch.Tensor,
        conv_width: int,
    ) -> "SharedGDNTreeMetadata":
        if parents.ndim != 2 or parents.shape[0] != 1 or parents.shape[1] < 1:
            raise ValueError("Shared GDN tree metadata requires BS1 rootless parents [1, budget].")
        if depths.shape != parents.shape:
            raise ValueError("TreeLayout depths must have the same rootless [1, budget] shape as parents.")
        if any(tensor.shape != (1,) for tensor in (num_nodes, num_accepted_tokens, prev_num_sampled)):
            raise ValueError("Shared GDN tree counts must have shape [1].")
        if prev_path_node_ids.ndim != 2 or prev_path_node_ids.shape[0] != 1:
            raise ValueError("The previous accepted path must have shape [1, path_capacity].")
        if conv_width < 1:
            raise ValueError("The GDN convolution width must be positive.")

        slots = parents.shape[1] + 1
        device = parents.device
        chain_parents = torch.arange(slots - 1, device=device, dtype=torch.long)
        parent_columns = torch.where(num_nodes[:, None] > 0, parents.to(torch.long), chain_parents[None, :])
        root_column = (num_accepted_tokens.to(torch.long) - 1).clamp(min=0).unsqueeze(1)
        init_state_columns = torch.cat((root_column, parent_columns), dim=1)
        return cls(
            slots=slots,
            conv_width=conv_width,
            init_state_columns=init_state_columns,
            conv_window_rows=compute_tree_conv_rows(
                parents,
                num_nodes,
                conv_width,
                conv_width - 1,
                tree_depths=depths,
            ),
            conv_prefix_src=compute_tree_prefix_source(prev_path_node_ids, prev_num_sampled, conv_width),
        )

    def _check_state_table(self, state_indices: torch.Tensor) -> None:
        if state_indices.shape != (1, self.slots):
            raise ValueError(
                f"Expected a complete BS1 state table [1, {self.slots}], got {tuple(state_indices.shape)}."
            )
        if state_indices.device != self.init_state_columns.device:
            raise ValueError("Shared GDN topology and group state pages must be on the same device.")

    def bind_init_state_indices(self, state_indices: torch.Tensor) -> torch.Tensor:
        """Resolve shared logical parents against this group's physical pages."""
        self._check_state_table(state_indices)
        return state_indices.gather(1, self.init_state_columns).to(torch.int32)

    def _get_verify_geometry(self, scratch_base: int) -> tuple[_VerifyGeometry, ...]:
        geometry = self._verify_geometry.get(scratch_base)
        if geometry is not None:
            return geometry
        device = self.init_state_columns.device
        width = self.slots
        cells = torch.arange(width, dtype=torch.long, device=device)
        scratch = cells + scratch_base
        chunks = []
        for start in range(0, self.slots, width):
            length = min(width, self.slots - start)
            owned = cells < length
            tokens = torch.where(owned, cells + start, torch.zeros_like(cells))
            lengths = torch.zeros(2, dtype=torch.int32, device=device)
            lengths[1:].fill_(length)
            chunks.append(
                _VerifyGeometry(
                    token_indices=tokens,
                    owned=owned,
                    out_indices=torch.where(owned, tokens, scratch),
                    cu_seqlens=lengths,
                )
            )
        geometry = tuple(chunks)
        self._verify_geometry[scratch_base] = geometry
        return geometry

    def bind_verify_plan(
        self,
        state_indices: torch.Tensor,
        init_state_indices: torch.Tensor,
        scratch_base: int,
    ) -> TreeVerifyPlan | None:
        """Share chunk geometry while binding each group's state and parent pages."""
        self._check_state_table(state_indices)
        if init_state_indices.shape != state_indices.shape or init_state_indices.device != state_indices.device:
            raise ValueError("Initial-state pages must match this group's complete state table.")
        width = GDN_RECURRENT_MAX_VERIFY_TOKENS
        if self.slots <= width:
            return None
        width = self.slots  # single whole-tree chunk (triton has no per-call limit)
        if scratch_base < 0:
            raise ValueError(f"The output scratch base must be non-negative, got {scratch_base}.")
        flat_state = state_indices.reshape(-1)
        flat_init = init_state_indices.reshape(-1)
        chunks = tuple(
            TreeVerifyChunk(
                token_indices=geometry.token_indices,
                state_indices=torch.where(geometry.owned, flat_state.index_select(0, geometry.token_indices), -1),
                init_state_indices=torch.where(geometry.owned, flat_init.index_select(0, geometry.token_indices), -1),
                out_indices=geometry.out_indices,
                cu_seqlens=geometry.cu_seqlens,
                width=width,
            )
            for geometry in self._get_verify_geometry(scratch_base)
        )
        return TreeVerifyPlan(rows=1, slots=self.slots, width=width, chunks=chunks)
