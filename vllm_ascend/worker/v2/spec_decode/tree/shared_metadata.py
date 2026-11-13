# SPDX-License-Identifier: Apache-2.0
"""One verification step's logical GDN metadata, shared across cache groups."""

from dataclasses import dataclass, field

import torch

from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    GDN_RECURRENT_MAX_VERIFY_TOKENS,
    TreeVerifyChunk,
    TreeVerifyPlan,
    TreeVerifyPlanBuffers,
    compute_tree_conv_rows,
    compute_tree_prefix_source,
)


@dataclass(frozen=True)
class _VerifyGeometry:
    num_chunks: int
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
    _verify_geometry: dict[int, _VerifyGeometry] = field(default_factory=dict, repr=False, compare=False)

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

    def _get_verify_geometry(self, scratch_base: int) -> _VerifyGeometry:
        geometry = self._verify_geometry.get(scratch_base)
        if geometry is not None:
            return geometry
        device = self.init_state_columns.device
        width = self.slots
        num_chunks = -(-self.slots // width)
        cells = torch.arange(width, dtype=torch.long, device=device).unsqueeze(0)
        starts = torch.arange(num_chunks, dtype=torch.long, device=device).unsqueeze(1) * width
        token_indices = cells + starts  # [C, W] global token column of each cell
        # A chunk verifies at most ``width`` tokens: all but the last chunk
        # are full, so the raw ``slots - start`` over-counts there and must be
        # clamped (the per-chunk loop used ``min(width, slots - start)``).
        lengths = (self.slots - starts.squeeze(1)).clamp(max=width)  # [C] per-chunk real length
        owned = cells < lengths.unsqueeze(1)
        owned_tokens = torch.where(owned, token_indices, torch.zeros_like(token_indices))
        scratch = cells.squeeze(0) + scratch_base
        out_indices = torch.where(owned, owned_tokens, scratch.unsqueeze(0).expand_as(owned_tokens))
        cu_seqlens = torch.zeros((num_chunks, 2), dtype=torch.int32, device=device)
        cu_seqlens[:, 1].copy_(lengths.to(torch.int32))
        geometry = _VerifyGeometry(
            num_chunks=num_chunks,
            token_indices=owned_tokens,
            owned=owned,
            out_indices=out_indices,
            cu_seqlens=cu_seqlens,
        )
        self._verify_geometry[scratch_base] = geometry
        return geometry

    def _bind_stacked(
        self,
        state_indices: torch.Tensor,
        init_state_indices: torch.Tensor,
        scratch_base: int,
    ) -> tuple[_VerifyGeometry, torch.Tensor, torch.Tensor] | None:
        """Validate and gather the group's state/init pages in one pass.

        Returns the stacked geometry plus the stacked per-chunk state/init
        tables (``[C, W]``), or ``None`` when the tree fits one recurrent-op
        call (no plan). Bit-identical to the per-chunk gather loop of
        :meth:`bind_verify_plan`: same ``index_select`` indices, same
        ``where(owned, ..., -1)`` masking, same element values.
        """
        self._check_state_table(state_indices)
        if init_state_indices.shape != state_indices.shape or init_state_indices.device != state_indices.device:
            raise ValueError("Initial-state pages must match this group's complete state table.")
        if self.slots <= GDN_RECURRENT_MAX_VERIFY_TOKENS:
            return None
        if scratch_base < 0:
            raise ValueError(f"The output scratch base must be non-negative, got {scratch_base}.")
        geometry = self._get_verify_geometry(scratch_base)
        tokens = geometry.token_indices.reshape(-1)
        flat_state = state_indices.reshape(-1)
        flat_init = init_state_indices.reshape(-1)
        shape = geometry.token_indices.shape
        state = torch.where(geometry.owned, flat_state.index_select(0, tokens).reshape(shape), -1)
        init = torch.where(geometry.owned, flat_init.index_select(0, tokens).reshape(shape), -1)
        return geometry, state, init

    def bind_plan(
        self,
        state_indices: torch.Tensor,
        init_state_indices: torch.Tensor,
        scratch_base: int,
    ) -> TreeVerifyPlan | None:
        """Stacked binding for the eager path (no stable buffers).

        Four device dispatches for the whole plan instead of ``4 * chunks``:
        the per-chunk tensors are views (``[i]``) onto the stacked results.
        """
        bound = self._bind_stacked(state_indices, init_state_indices, scratch_base)
        if bound is None:
            return None
        geometry, state, init = bound
        width = self.slots  # single whole-tree chunk (op has no per-call limit)
        return TreeVerifyPlan(
            rows=1,
            slots=self.slots,
            width=width,
            chunks=tuple(
                TreeVerifyChunk(
                    token_indices=geometry.token_indices[i],
                    state_indices=state[i],
                    init_state_indices=init[i],
                    out_indices=geometry.out_indices[i],
                    cu_seqlens=geometry.cu_seqlens[i],
                    width=width,
                )
                for i in range(geometry.num_chunks)
            ),
        )

    def bind_and_copy_into(
        self,
        state_indices: torch.Tensor,
        init_state_indices: torch.Tensor,
        scratch_base: int,
        buffers: "TreeVerifyPlanBuffers",
        plan_views: "TreeVerifyPlan | None" = None,
    ) -> TreeVerifyPlan | None:
        """Stacked binding into a group's stable graph buffers.

        Bit-identical to ``bind_verify_plan(...).copy_into(buffers)`` — the
        same five arrays land in the same buffer positions — but with nine
        dispatches (two gathers, two masks, five bulk copies) instead of
        ``4 * chunks + 5 * chunks``. The returned plan views the stable
        buffers, which a full graph reads by address.

        ``plan_views`` is a plan prebuilt from the same buffers (its views are
        fixed by buffer addresses and chunk shapes): passing it skips the
        per-step view construction on the hot path.
        """
        bound = self._bind_stacked(state_indices, init_state_indices, scratch_base)
        if bound is None:
            return None
        geometry, state, init = bound
        width = self.slots  # single whole-tree chunk (op has no per-call limit)
        num_chunks = geometry.num_chunks
        tokens = width
        buffers.token_indices[:num_chunks, :tokens].copy_(geometry.token_indices)
        buffers.state_indices[:num_chunks, :tokens].copy_(state)
        buffers.init_state_indices[:num_chunks, :tokens].copy_(init)
        buffers.out_indices[:num_chunks, :tokens].copy_(geometry.out_indices)
        # cu_seqlens buffers are sized for the graph's row capacity
        # (``rows + 1`` columns); this BS1 plan writes its two columns only,
        # exactly like ``copy_into``'s ``[: self.rows + 1]``.
        buffers.cu_seqlens[:num_chunks, : 1 + 1].copy_(geometry.cu_seqlens)
        if plan_views is None:
            return TreeVerifyPlan.views_from_buffers(buffers, 1, self.slots, width, num_chunks)
        if (plan_views.rows, plan_views.slots, plan_views.width, plan_views.num_chunks) != (
            1,
            self.slots,
            width,
            num_chunks,
        ):
            raise ValueError(
                f"cached verify-plan views ({plan_views.rows}/{plan_views.slots}/{plan_views.width}"
                f"/{plan_views.num_chunks}) do not match this tree ({1}/{self.slots}/{width}/{num_chunks})"
            )
        return plan_views

    def bind_verify_plan(
        self,
        state_indices: torch.Tensor,
        init_state_indices: torch.Tensor,
        scratch_base: int,
    ) -> TreeVerifyPlan | None:
        """Legacy per-chunk binding (kept for the diagnostic fallback path).

        Keeps the old per-chunk dispatch pattern (one ``index_select`` +
        ``where`` per tensor per chunk) sourced from the stacked geometry
        rows, so ``VLLM_ASCEND_GDN_VERIFY_PLAN_STACKED=0`` reproduces the
        pre-stacked device-op schedule exactly.
        """
        self._check_state_table(state_indices)
        if init_state_indices.shape != state_indices.shape or init_state_indices.device != state_indices.device:
            raise ValueError("Initial-state pages must match this group's complete state table.")
        width = self.slots  # single whole-tree chunk (op has no per-call limit)
        if self.slots <= GDN_RECURRENT_MAX_VERIFY_TOKENS:
            return None
        if scratch_base < 0:
            raise ValueError(f"The output scratch base must be non-negative, got {scratch_base}.")
        geometry = self._get_verify_geometry(scratch_base)
        flat_state = state_indices.reshape(-1)
        flat_init = init_state_indices.reshape(-1)
        chunks = tuple(
            TreeVerifyChunk(
                token_indices=geometry.token_indices[i],
                state_indices=torch.where(geometry.owned[i], flat_state.index_select(0, geometry.token_indices[i]), -1),
                init_state_indices=torch.where(
                    geometry.owned[i], flat_init.index_select(0, geometry.token_indices[i]), -1
                ),
                out_indices=geometry.out_indices[i],
                cu_seqlens=geometry.cu_seqlens[i],
                width=width,
            )
            for i in range(geometry.num_chunks)
        )
        return TreeVerifyPlan(rows=1, slots=self.slots, width=width, chunks=chunks)
