"""Tree-aware GDN RecoverSSM reference kernels.

This module is the missing adapter between the linear RecoverSSM operator from
PR #22045 and the draft-tree metadata used by ``tree261113-rebase``.  It keeps
the committed checkpoint read-only during verification, writes one compact
record per tree node, and reconstructs only the accepted root-to-leaf path at
commit time.

The code intentionally uses the ordinary Triton launcher (``kernel[grid]``),
so it is easy to validate and then switch to the ``ck.run`` launcher used by
the current triton-ascend runtime when that ABI is required.

Layouts
--------
* q/k: ``[T, HK, K]``; v: ``[T, HV, V]``; output: ``[T, HV, V]``
* checkpoint_state: ``[P, HV, V, K]`` (BF16 or FP32)
* replay_cache: ``[P, HV, S, V + K + 1]`` (FP32)
* query_start_loc: cumulative offsets ``[R + 1]``
* root_state_indices: checkpoint page for each request ``[R]``
* parent_slots: local parent slot table ``[R, S]``; root is ``-1`` and every
  non-root parent is smaller than its child slot

The tree builder may retain its existing ``[base, len_0, ...]`` convention;
convert it to cumulative offsets before calling this module. Tree node ids are
1-based (slot 0 is the root). ``num_sampled`` is the number of recurrent
records to compose: root-only acceptance is 1, and the sampler's trailing
bonus token is not an additional recurrent record. Pass that count as
``accepted_records = clamp(num_sampled, 1, S)``.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _tree_recover_verify_kernel(
    A_log_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    checkpoint_ptr,
    replay_ptr,
    query_start_ptr,
    root_index_ptr,
    parent_ptr,
    token_index_ptr,
    output_index_ptr,
    out_ptr,
    scale,
    softplus_beta,
    softplus_threshold,
    stride_a_token,
    stride_a_head,
    stride_b_token,
    stride_b_head,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_head,
    stride_v_token,
    stride_v_head,
    stride_checkpoint_page,
    stride_checkpoint_head,
    stride_checkpoint_v,
    stride_checkpoint_k,
    stride_replay_page,
    stride_replay_head,
    stride_replay_slot,
    stride_replay_dim,
    stride_out_token,
    stride_out_head,
    stride_parent_row,
    stride_token_index,
    stride_output_index,
    num_key_heads: tl.constexpr,
    num_value_heads: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    MAX_S: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    HAS_INDICES: tl.constexpr,
):
    """Verify one request and emit records indexed by local tree slot."""
    pid_v = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.program_id(2)
    start = tl.load(query_start_ptr + row)
    end = tl.load(query_start_ptr + row + 1)
    root_page = tl.load(root_index_ptr + row).to(tl.int64)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    page_base = (
        checkpoint_ptr
        + root_page * stride_checkpoint_page
        + head * stride_checkpoint_head
    )
    root_state = tl.load(
        page_base
        + offs_v[:, None] * stride_checkpoint_v
        + offs_k[None, :] * stride_checkpoint_k,
        mask=mask_state,
        other=0.0,
    ).to(tl.float32)
    state = root_state
    prev_slot = tl.full((), -1, tl.int32)
    key_head = head // (num_value_heads // num_key_heads)
    neg_a = -tl.exp(tl.load(A_log_ptr + head).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + head).to(tl.float32)

    for local_slot in range(0, MAX_S):
        live = local_slot < (end - start)
        token_cell = start + local_slot
        if HAS_INDICES:
            src = tl.load(token_index_ptr + token_cell * stride_token_index).to(tl.int64)
            dst = tl.load(output_index_ptr + token_cell * stride_output_index).to(tl.int64)
        else:
            src = token_cell.to(tl.int64)
            dst = token_cell.to(tl.int64)
        parent = tl.load(
            parent_ptr + row * stride_parent_row + local_slot
        ).to(tl.int32)

        # A continuation can keep the previous state.  A sibling or a deep
        # branch reconstructs from the immutable root checkpoint and the
        # records on its ancestor chain.
        if live & (parent != prev_slot):
            decay_product = tl.full([BK], 1.0, tl.float32)
            correction_sum = tl.zeros([BV, BK], tl.float32)
            cursor = parent
            for _ in range(0, MAX_S):
                has_ancestor = cursor >= 0
                safe_cursor = tl.maximum(cursor, 0)
                record = (
                    replay_ptr
                    + row * stride_replay_page
                    + head * stride_replay_head
                    + safe_cursor.to(tl.int64) * stride_replay_slot
                )
                correction = tl.load(
                    record + offs_v * stride_replay_dim,
                    mask=mask_v & has_ancestor,
                    other=0.0,
                ).to(tl.float32)
                ancestor_key = tl.load(
                    record + (V + offs_k) * stride_replay_dim,
                    mask=mask_k & has_ancestor,
                    other=0.0,
                ).to(tl.float32)
                ancestor_decay = tl.load(
                    record + (V + K) * stride_replay_dim,
                    mask=has_ancestor,
                    other=1.0,
                ).to(tl.float32)
                correction_sum += correction[:, None] * (
                    ancestor_key * decay_product
                )[None, :]
                decay_product *= tl.where(has_ancestor, ancestor_decay, 1.0)
                next_cursor = tl.load(
                    parent_ptr + row * stride_parent_row + safe_cursor,
                    mask=has_ancestor,
                    other=-1,
                )
                cursor = tl.where(has_ancestor, next_cursor, -1)
            state = root_state * decay_product[None, :] + correction_sum

        q = tl.load(
            q_ptr
            + src * stride_q_token
            + key_head * stride_q_head
            + offs_k,
            mask=mask_k & live,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            k_ptr
            + src * stride_k_token
            + key_head * stride_k_head
            + offs_k,
            mask=mask_k & live,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + src * stride_v_token + head * stride_v_head + offs_v,
            mask=mask_v & live,
            other=0.0,
        ).to(tl.float32)
        if USE_QK_L2NORM:
            q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6)
            k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        x = tl.load(a_ptr + src * stride_a_token + head * stride_a_head).to(tl.float32) + dt_bias
        softplus_x = tl.where(
            softplus_beta * x <= softplus_threshold,
            tl.log(1.0 + tl.exp(softplus_beta * x)) / softplus_beta,
            x,
        )
        decay = tl.exp(neg_a * softplus_x)
        beta = tl.sigmoid(
            tl.load(b_ptr + src * stride_b_token + head * stride_b_head).to(tl.float32)
        )
        decayed_state = state * decay
        correction = (v - tl.sum(decayed_state * k[None, :], axis=1)) * beta
        state = decayed_state + correction[:, None] * k[None, :]
        output = tl.sum(state * (q * scale)[None, :], axis=1)
        tl.store(
            out_ptr + dst * stride_out_token + head * stride_out_head + offs_v,
            output,
            mask=mask_v & live,
        )

        record = (
            replay_ptr
            + row * stride_replay_page
            + head * stride_replay_head
            + local_slot * stride_replay_slot
        )
        tl.store(record + offs_v * stride_replay_dim, correction, mask=mask_v & live)
        if pid_v == 0:
            tl.store(
                record + (V + offs_k) * stride_replay_dim,
                k,
                mask=mask_k & live,
            )
            tl.store(record + (V + K) * stride_replay_dim, decay, mask=live)
        prev_slot = tl.where(live, local_slot, prev_slot)


@triton.jit
def _tree_recover_commit_kernel(
    checkpoint_ptr,
    replay_ptr,
    state_ptr,
    state_page_ptr,
    path_ptr,
    accepted_ptr,
    tree_row_ptr,
    stride_checkpoint_page,
    stride_checkpoint_head,
    stride_checkpoint_v,
    stride_checkpoint_k,
    stride_replay_page,
    stride_replay_head,
    stride_replay_slot,
    stride_replay_dim,
    stride_state_page,
    stride_state_head,
    stride_state_v,
    stride_state_k,
    stride_state_page_table,
    stride_path_row,
    scale_unused,
    num_value_heads: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    MAX_S: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Compose records on the accepted path and write one checkpoint page."""
    pid_v = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.program_id(2)
    if tl.load(tree_row_ptr + row) == 0:
        return
    accepted = tl.minimum(tl.maximum(tl.load(accepted_ptr + row), 1), MAX_S)
    root_page = tl.load(state_page_ptr + row * stride_state_page_table).to(tl.int64)
    root_base = checkpoint_ptr + root_page * stride_checkpoint_page + head * stride_checkpoint_head
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    initial = tl.load(
        root_base + offs_v[:, None] * stride_checkpoint_v + offs_k[None, :] * stride_checkpoint_k,
        mask=mask_state,
        other=0.0,
    ).to(tl.float32)
    decay_product = tl.full([BK], 1.0, tl.float32)
    correction_sum = tl.zeros([BV, BK], tl.float32)

    # ``accepted`` includes the root record. path_ptr contains 1-based child
    # node ids, so path[0] is the first non-root slot. Walk all accepted
    # records backwards, including slot zero for the root token.
    for reverse_offset in range(0, MAX_S):
        live = reverse_offset < accepted
        record_index = accepted - 1 - reverse_offset
        path_pos = record_index - 1
        safe_path_pos = tl.maximum(path_pos, 0)
        child_slot = tl.load(
            path_ptr + row * stride_path_row + safe_path_pos
        ).to(tl.int64)
        slot = tl.where(record_index == 0, 0, child_slot)
        record = (
            replay_ptr
            + row * stride_replay_page
            + head * stride_replay_head
            + slot * stride_replay_slot
        )
        correction = tl.load(
            record + offs_v * stride_replay_dim,
            mask=mask_v & live,
            other=0.0,
        ).to(tl.float32)
        key = tl.load(
            record + (V + offs_k) * stride_replay_dim,
            mask=mask_k & live,
            other=0.0,
        ).to(tl.float32)
        decay = tl.load(record + (V + K) * stride_replay_dim, mask=live, other=1.0).to(tl.float32)
        correction_sum += correction[:, None] * (key * decay_product)[None, :]
        decay_product *= tl.where(live, decay, 1.0)

    safe_last = tl.maximum(accepted - 2, 0)
    final_slot = tl.where(
        accepted > 1,
        tl.load(path_ptr + row * stride_path_row + safe_last),
        0,
    )
    dst_page = tl.load(
        state_page_ptr + row * stride_state_page_table + final_slot
    ).to(tl.int64)
    dst = state_ptr + dst_page * stride_state_page + head * stride_state_head
    state = initial * decay_product[None, :] + correction_sum
    tl.store(
        dst + offs_v[:, None] * stride_state_v + offs_k[None, :] * stride_state_k,
        state,
        mask=mask_state,
    )


def tree_recoverssm_verify(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    replay_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    root_state_indices: torch.Tensor,
    parent_slots: torch.Tensor,
    scale: float | None = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True,
    token_indices: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tree verification; checkpoint pages are read-only and records are per slot."""
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
        raise ValueError("q/k/v must be [T,HK,K], [T,HK,K], [T,HV,V]")
    tokens, hk, dk = q.shape
    _, hv, dv = v.shape
    rows, max_s = parent_slots.shape
    if query_start_loc.shape != (rows + 1,) or root_state_indices.numel() != rows:
        raise ValueError("tree row metadata has incompatible shape")
    if replay_cache.shape != (checkpoint_state.shape[0], hv, max_s, dv + dk + 1):
        raise ValueError("replay_cache must be [P,HV,S,V+K+1]")
    if checkpoint_state.shape[1:] != (hv, dv, dk):
        raise ValueError("checkpoint_state must be [P,HV,V,K]")
    if scale is None:
        scale = dk**-0.5
    if out is None:
        out = torch.empty((tokens, hv, dv), dtype=v.dtype, device=v.device)
    has_indices = token_indices is not None or out_indices is not None
    if has_indices and (token_indices is None or out_indices is None):
        raise ValueError("token_indices and out_indices must be provided together")
    block_v = min(triton.next_power_of_2(dv), 64)
    grid = (triton.cdiv(dv, block_v), hv, rows)
    dummy = query_start_loc
    _tree_recover_verify_kernel[grid](
        A_log, a.reshape(tokens, hv), b.reshape(tokens, hv), dt_bias,
        q, k, v, checkpoint_state, replay_cache, query_start_loc,
        root_state_indices, parent_slots,
        token_indices if has_indices else dummy,
        out_indices if has_indices else dummy,
        out, float(scale), softplus_beta, softplus_threshold,
        a.stride(-2), a.stride(-1), b.stride(-2), b.stride(-1),
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        checkpoint_state.stride(0), checkpoint_state.stride(1),
        checkpoint_state.stride(2), checkpoint_state.stride(3),
        replay_cache.stride(0), replay_cache.stride(1), replay_cache.stride(2), replay_cache.stride(3),
        out.stride(0), out.stride(1), parent_slots.stride(0),
        token_indices.stride(0) if has_indices else 1,
        out_indices.stride(0) if has_indices else 1,
        num_key_heads=hk, num_value_heads=hv, K=dk, V=dv, MAX_S=max_s,
        BK=triton.next_power_of_2(dk), BV=block_v,
        USE_QK_L2NORM=use_qk_l2norm_in_kernel, HAS_INDICES=has_indices,
        num_warps=4,
    )
    return out


def tree_recoverssm_commit(
    checkpoint_state: torch.Tensor,
    replay_cache: torch.Tensor,
    state_pages: torch.Tensor,
    path_node_ids: torch.Tensor,
    accepted_records: torch.Tensor,
    tree_rows: torch.Tensor,
) -> None:
    """Commit an accepted path into the destination page table.

    ``path_node_ids`` contains 1-based child node ids; the state table is
    ``[R,S]`` and therefore destination slot is the node id itself.
    ``accepted_records`` includes the root record (root-only is 1) and excludes
    the sampler's trailing bonus record. The root page is ``state_pages[:, 0]``.
    """
    rows = state_pages.shape[0]
    _, hv, dv, dk = checkpoint_state.shape
    max_s = replay_cache.shape[2]
    block_v = min(triton.next_power_of_2(dv), 64)
    grid = (triton.cdiv(dv, block_v), hv, rows)
    _tree_recover_commit_kernel[grid](
        checkpoint_state, replay_cache, checkpoint_state, state_pages, path_node_ids,
        accepted_records, tree_rows,
        checkpoint_state.stride(0), checkpoint_state.stride(1),
        checkpoint_state.stride(2), checkpoint_state.stride(3),
        replay_cache.stride(0), replay_cache.stride(1), replay_cache.stride(2), replay_cache.stride(3),
        checkpoint_state.stride(0), checkpoint_state.stride(1),
        checkpoint_state.stride(2), checkpoint_state.stride(3), state_pages.stride(1),
        path_node_ids.stride(0), 0.0,
        num_value_heads=hv, K=dk, V=dv, MAX_S=max_s, BK=triton.next_power_of_2(dk), BV=block_v,
        num_warps=4,
    )
