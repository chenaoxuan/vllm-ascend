# SPDX-License-Identifier: Apache-2.0
"""Triton GDN recurrence with one state page per draft-tree node.

For each token, decay the parent's state, compute
``delta = beta * (value - state @ key)``, update the state by
``delta[:, None] * key[None, :]``, and project it onto the scaled query.
The computation uses fp32; state pages support fp32 or bf16 storage.

``init_state_indices`` selects the parent page and ``ssm_state_indices``
selects the output page. Nodes must follow their parents. A continuation can
keep the state on chip; sibling groups can share their parent's tile.

``actual_seq_lengths`` stores a base followed by per-row token counts, not
cumulative endpoints. Active rows precede zero-length graph padding. Each
row can verify the whole tree in one call; there is no fixed token limit.
Optional token/output index tables map a padded plan to packed operands
without materializing gathers and scatters. Padding entries are not read;
all entries inside active prefixes must contain valid token and page IDs.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
)

# ---- Pre-bound fast dispatch cache ----
_CK_CACHE = {}


def _validate_inputs(
    query,
    key,
    value,
    state,
    beta,
    g,
    gk,
    actual_seq_lengths,
    ssm_state_indices,
    init_state_indices,
    scale,
    token_indices=None,
    out_indices=None,
):
    """Full input validation (fast path runs this once per specialization)."""
    assert scale is not None, "scale cannot be empty"
    assert beta is not None, "beta is required"
    assert actual_seq_lengths is not None, "actual_seq_lengths (cu_seqlens) is required"
    assert ssm_state_indices is not None, "ssm_state_indices is required"
    assert init_state_indices is not None, "tree mode requires init_state_indices"
    assert gk is None, "gk not supported (unused in vllm call path)"
    assert (
        query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and value.dtype == torch.bfloat16
        and beta.dtype == torch.bfloat16
    )
    assert actual_seq_lengths.dtype == torch.int32
    assert ssm_state_indices.dtype == torch.int32
    assert init_state_indices.dtype == torch.int32
    assert state.dtype in (torch.float32, torch.bfloat16)
    if g is not None:
        assert g.dtype == torch.float32
    # Require dense, non-overlapping state pages.
    assert state.dim() == 4
    assert (
        state.stride(3) == 1
        and state.stride(2) == state.size(3)
        and state.stride(1) == state.size(2) * state.size(3)
        and state.stride(0) >= state.size(1) * state.size(2) * state.size(3)
    ), "state must be dense within each non-overlapping page"
    T, NK, DK = query.shape
    NV, DV = value.shape[1], value.shape[2]
    assert state.shape[1] == NV and state.shape[2] == DV and state.shape[3] == DK
    assert beta.shape[0] >= T and (g is None or g.shape[0] >= T)
    assert NV % NK == 0, "GQA requires NV to be an integer multiple of NK"
    assert ssm_state_indices.numel() >= T and init_state_indices.numel() >= T
    if token_indices is not None:
        assert out_indices is not None, "token indirection needs the output table too"
        assert token_indices.dtype in (torch.int32, torch.int64), "the token table must be integer"
        assert out_indices.dtype in (torch.int32, torch.int64), "the output table must be integer"
        assert token_indices.numel() >= T and out_indices.numel() >= T, (
            "the indirection tables must cover every token position of the call"
        )


_raw_stream_fn = None


def _raw_stream(device):
    global _raw_stream_fn
    if _raw_stream_fn is None:
        try:
            from torch_npu._C import _npu_getCurrentRawStreamNoWait as fn
        except ImportError:
            from torch_npu._C import _npu_getCurrentRawStream as fn
        _raw_stream_fn = fn
    return _raw_stream_fn(device)


@triton.jit
def _load_tree_token(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    ssm_ptr,
    init_ptr,
    tok_ptr,
    out_tok_ptr,
    seq_start,
    token,
    length,
    scale,
    NK,
    DK,
    NV,
    DV,
    head,
    khead,
    k_offs,
    v_offs,
    HAS_G: tl.constexpr,
    HAS_IDX: tl.constexpr,
):
    live = token < length
    cell = seq_start + token
    page = tl.load(ssm_ptr + cell, mask=live, other=-1)
    parent = tl.load(init_ptr + cell, mask=live, other=-1)
    if HAS_IDX:
        src = tl.load(tok_ptr + cell, mask=live, other=0)
        dst = tl.load(out_tok_ptr + cell, mask=live, other=0)
    else:
        src = cell
        dst = cell
    src64 = src.to(tl.int64)
    head64 = head.to(tl.int64)
    qk_base = src64 * (NK * DK) + khead * DK
    query = tl.load(q_ptr + qk_base + k_offs, mask=live & (k_offs < DK), other=0.0).to(tl.float32) * scale
    key = tl.load(k_ptr + qk_base + k_offs, mask=live & (k_offs < DK), other=0.0).to(tl.float32)
    value = tl.load(
        v_ptr + src64 * (NV * DV) + head64 * DV + v_offs,
        mask=live & (v_offs < DV),
        other=0.0,
    ).to(tl.float32)
    beta = tl.load(beta_ptr + src * NV + head, mask=live, other=0.0).to(tl.float32)
    if HAS_G:
        log_decay = tl.load(g_ptr + src * NV + head, mask=live, other=0.0)
    else:
        log_decay = tl.full((), 0.0, tl.float32)
    return page, parent, dst.to(tl.int64), query, key, value, beta, log_decay


@triton.jit
def _run_tree_tile(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    out_ptr,
    cu_ptr,
    ssm_ptr,
    init_ptr,
    tok_ptr,
    out_tok_ptr,
    scale,
    NK,
    DK,
    NV,
    DV,
    PAGE_STRIDE,
    GQA,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    HAS_G: tl.constexpr,
    SYNC_RELOAD: tl.constexpr,
    HAS_IDX: tl.constexpr,
    REUSE_PARENT: tl.constexpr,
    METADATA_ROW_STRIDE: tl.constexpr,
    B,
    pid_v,
    head,
    batch,
):
    head64 = head.to(tl.int64)
    khead = head // GQA

    # cu = [base, len_0, len_1, ...] (fork convention, kernel accumulates row starts)
    if METADATA_ROW_STRIDE > 0:
        seq_start = tl.load(cu_ptr) + batch * METADATA_ROW_STRIDE
    else:
        lens = tl.load(
            cu_ptr + 1 + tl.arange(0, BLOCK_B),
            mask=tl.arange(0, BLOCK_B) < B,
            other=0,
        )
        seq_start = tl.load(cu_ptr) + tl.sum(tl.where(tl.arange(0, BLOCK_B) < batch, lens, 0))
    T = tl.load(cu_ptr + batch + 1)

    # ---- Resident address components (int64 base) ----
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_offs = tl.arange(0, BLOCK_K)
    m_v = v_offs < DV
    m_k = k_offs < DK
    v_offs64 = v_offs.to(tl.int64)
    k_offs64 = k_offs.to(tl.int64)
    page_in = head64 * (DV * DK) + v_offs64[:, None] * DK + k_offs64[None, :]

    # Root token's initial state page
    if T > 0:
        init0 = tl.load(init_ptr + seq_start)
        state = tl.load(
            state_ptr + init0.to(tl.int64) * PAGE_STRIDE + page_in,
            mask=(m_v[:, None] & m_k[None, :]),
            other=0.0,
        ).to(tl.float32)
        m_vk = m_v[:, None] & m_k[None, :]

        if REUSE_PARENT:
            ssm_prev = tl.full((), -1, tl.int32)
            t = 0
            ssm_t, init_t, dst_t64, q_t, k_t, v_t, beta_t, log_g_t = _load_tree_token(
                q_ptr,
                k_ptr,
                v_ptr,
                g_ptr,
                beta_ptr,
                ssm_ptr,
                init_ptr,
                tok_ptr,
                out_tok_ptr,
                seq_start,
                0,
                T,
                scale,
                NK,
                DK,
                NV,
                DV,
                head,
                khead,
                k_offs64,
                v_offs64,
                HAS_G,
                HAS_IDX,
            )
            while t < T:
                group_init = init_t
                if (t > 0) & (group_init != ssm_prev):
                    if SYNC_RELOAD:
                        tl.debug_barrier()
                    state = tl.load(
                        state_ptr + group_init.to(tl.int64) * PAGE_STRIDE + page_in,
                        mask=m_vk,
                        other=0.0,
                    ).to(tl.float32)
                parent_state = state
                group_start = t
                same_parent = tl.full((), True, tl.int1)
                while (t < T) & same_parent:
                    # Keep the parent loop-invariant across its children,
                    # instead of copying a second matrix on every iteration.
                    # Subsequent children observe stored precision, like a reload.
                    state = parent_state
                    if t > group_start:
                        state = parent_state.to(state_ptr.dtype.element_ty).to(tl.float32)
                    # Next-token operands are independent of the current
                    # state update; keep one small vector set in flight.
                    (
                        next_ssm,
                        next_init,
                        next_dst,
                        next_q,
                        next_k,
                        next_v,
                        next_beta,
                        next_log_g,
                    ) = _load_tree_token(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        g_ptr,
                        beta_ptr,
                        ssm_ptr,
                        init_ptr,
                        tok_ptr,
                        out_tok_ptr,
                        seq_start,
                        t + 1,
                        T,
                        scale,
                        NK,
                        DK,
                        NV,
                        DV,
                        head,
                        khead,
                        k_offs64,
                        v_offs64,
                        HAS_G,
                        HAS_IDX,
                    )

                    # ---- GDN recurrence (all fp32) ----
                    if HAS_G:
                        g_t = tl.exp(log_g_t)
                        state = state * g_t
                    kS = tl.sum(state * k_t[None, :], axis=1)
                    delta = (v_t - kS) * beta_t
                    state = state + delta[:, None] * k_t[None, :]
                    o_t = tl.sum(state * q_t[None, :], axis=1)

                    # Write out
                    tl.store(
                        out_ptr + dst_t64 * (NV * DV) + head64 * DV + v_offs64,
                        o_t.to(out_ptr.dtype.element_ty),
                        mask=m_v,
                    )
                    tl.store(
                        state_ptr + ssm_t.to(tl.int64) * PAGE_STRIDE + page_in,
                        state.to(state_ptr.dtype.element_ty),
                        mask=m_vk,
                    )
                    ssm_prev = ssm_t
                    t += 1
                    if REUSE_PARENT:
                        # An in-place output changes the parent and is a continuation.
                        same_parent = (next_init == group_init) & (ssm_t != group_init)
                    else:
                        same_parent = tl.full((), False, tl.int1)
                    ssm_t = next_ssm
                    init_t = next_init
                    dst_t64 = next_dst
                    q_t = next_q
                    k_t = next_k
                    v_t = next_v
                    beta_t = next_beta
                    log_g_t = next_log_g
        else:
            ssm_prev = tl.full((), -1, tl.int32)

            for t in range(0, T):
                # Direct scalar reads (independent of the recurrence chain, pipelinable)
                tok_t = seq_start + t
                ssm_t = tl.load(ssm_ptr + tok_t)
                init_t = tl.load(init_ptr + tok_t)
                if HAS_IDX:
                    # ``tok_t`` is a cell of the caller's padded grid. The tables
                    # carry the packed token this cell reads (q/k/v/g/beta stay
                    # ungathered) and the output row it writes (no scatter). Cells
                    # outside a row's prefix are never reached: the loop stops at
                    # the row's ``cu_seqlens`` length.
                    src_t = tl.load(tok_ptr + tok_t)
                    dst_t = tl.load(out_tok_ptr + tok_t)
                else:
                    src_t = tok_t
                    dst_t = tok_t
                src_t64 = src_t.to(tl.int64)
                dst_t64 = dst_t.to(tl.int64)
                beta_t = tl.load(beta_ptr + src_t * NV + head).to(tl.float32)

                # Tree branch: parent page != previous token's write page -> reload from parent
                if (t > 0) & (init_t != ssm_prev):
                    if SYNC_RELOAD:
                        tl.debug_barrier()
                    state = tl.load(
                        state_ptr + init_t.to(tl.int64) * PAGE_STRIDE + page_in,
                        mask=m_vk,
                        other=0.0,
                    ).to(tl.float32)
                ssm_prev = ssm_t

                # Current token's q/k/v vectors
                qk_base = src_t64 * (NK * DK) + khead * DK
                q_t = tl.load(q_ptr + qk_base + k_offs64, mask=m_k, other=0.0).to(tl.float32) * scale
                k_t = tl.load(k_ptr + qk_base + k_offs64, mask=m_k, other=0.0).to(tl.float32)
                v_t = tl.load(
                    v_ptr + src_t64 * (NV * DV) + head64 * DV + v_offs64,
                    mask=m_v,
                    other=0.0,
                ).to(tl.float32)

                # ---- GDN recurrence (all fp32) ----
                if HAS_G:
                    g_t = tl.exp(tl.load(g_ptr + src_t * NV + head))
                    state = state * g_t
                kS = tl.sum(state * k_t[None, :], axis=1)
                delta = (v_t - kS) * beta_t
                state = state + delta[:, None] * k_t[None, :]
                o_t = tl.sum(state * q_t[None, :], axis=1)

                # Write out
                tl.store(
                    out_ptr + dst_t64 * (NV * DV) + head64 * DV + v_offs64,
                    o_t.to(out_ptr.dtype.element_ty),
                    mask=m_v,
                )
                tl.store(
                    state_ptr + ssm_t.to(tl.int64) * PAGE_STRIDE + page_in,
                    state.to(state_ptr.dtype.element_ty),
                    mask=m_vk,
                )


@triton.jit(do_not_specialize=["B"])
def recurrent_gated_delta_rule_tree_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    state_ptr,
    out_ptr,
    cu_ptr,
    ssm_ptr,
    init_ptr,
    tok_ptr,
    out_tok_ptr,
    scale,
    NK,
    DK,
    NV,
    DV,
    PAGE_STRIDE,
    GQA,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    HAS_G: tl.constexpr,
    SYNC_RELOAD: tl.constexpr,
    HAS_IDX: tl.constexpr,
    REUSE_PARENT: tl.constexpr,
    METADATA_ROW_STRIDE: tl.constexpr,
    B,
    BALANCED_WORKERS: tl.constexpr,
):
    if BALANCED_WORKERS:
        v_tiles = tl.cdiv(DV, BLOCK_V)
        tiles_per_request = NV * v_tiles
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        live_rows = B
        if tl.load(cu_ptr + B) == 0:
            # FULL graph replay may have fewer live rows than the capture.
            # Trim only the zero suffix before assigning work to the cores.
            rows = tl.arange(0, BLOCK_B)
            lengths = tl.load(cu_ptr + 1 + rows, rows < B, other=0)
            live_rows = tl.max(tl.where(lengths > 0, rows + 1, 0))
        total = live_rows * tiles_per_request
        per_worker = total // workers
        extra = total % workers
        # Give every vector core a contiguous slice with at most one extra
        # tile. As the batch grows, each worker covers more requests.
        first = worker * per_worker + tl.minimum(worker, extra)
        count = per_worker + (worker < extra)
    else:
        first = 0
        count = 1
    for tile in range(first, first + count):
        if BALANCED_WORKERS:
            batch = tile // tiles_per_request
            head = (tile // v_tiles) % NV
            pid_v = tile % v_tiles
        else:
            pid_v = tl.program_id(0)
            head = tl.program_id(1)
            batch = tl.program_id(2)
        _run_tree_tile(
            q_ptr,
            k_ptr,
            v_ptr,
            g_ptr,
            beta_ptr,
            state_ptr,
            out_ptr,
            cu_ptr,
            ssm_ptr,
            init_ptr,
            tok_ptr,
            out_tok_ptr,
            scale,
            NK,
            DK,
            NV,
            DV,
            PAGE_STRIDE,
            GQA,
            BLOCK_V,
            BLOCK_K,
            BLOCK_B,
            HAS_G,
            SYNC_RELOAD,
            HAS_IDX,
            REUSE_PARENT,
            METADATA_ROW_STRIDE,
            B,
            pid_v,
            head,
            batch,
        )


def npu_recurrent_gated_delta_rule_tree_triton(
    query,
    key,
    value,
    state,
    *,
    beta=None,
    scale=None,
    actual_seq_lengths=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    g=None,
    gk=None,
    init_state_indices=None,
    token_indices=None,
    out_indices=None,
    out=None,
    BLOCK_V=64,
    BLOCK_B=64,
    SYNC_RELOAD=True,
    REUSE_PARENT=False,
    metadata_row_stride=0,
    fast=True,
    strict=False,
):
    """Triton replacement for torch.ops._C_ascend.npu_recurrent_gated_delta_rule_tree.

    cu_seqlens = [base, len_0, len_1, ...] (per-row token counts, kernel
    accumulates row starts; uses per-row token counts; zero-width
    padding rows only allowed at the end).
    Updates state in-place; returns [query.rows, NV, DV] bf16.

    ``token_indices``/``out_indices`` (optional, both or neither) let the kernel
    address q/k/v/g/beta and the output through integer tables instead of
    pre-gathered copies: position ``p`` of this call reads ``token_indices[p]``
    of the caller's packed q/k/v/g/beta and writes output row ``out_indices[p]``,
    while ``state``/``ssm_state_indices``/``init_state_indices`` stay in this
    call's own token order. They let a caller run the op directly on a padded
    token grid -- the GDN tree verify plan's ``token_indices``/``out_indices`` --
    with no gather, no scatter and no intermediate buffer. ``out`` must then
    span every row ``out_indices`` names.

    ``metadata_row_stride`` is the padded width of the metadata tables (and
    direct operands when no token table is supplied). Zero uses packed rows
    whose starts are the prefix sum of ``actual_seq_lengths``. The stride is
    a host-side shape, so variable live lengths need no host synchronization.

    ``REUSE_PARENT`` retains the input state across consecutive siblings. A
    branch rounds that tile to the state storage dtype, just like a reload;
    a continuation still uses the previous token's unrounded fp32 result.
    The caller enables it for sibling-grouped trees. Leave it disabled for
    chains or interleaved layouts to retain the original serial traversal.
    Its operand prefetch requires all inputs except state to stay read-only
    during the call; ``out`` must not overlap those inputs.
    """
    # Layout fix-up (every call; is_contiguous is cheap)
    if not query.is_contiguous():
        query = query.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    if not beta.is_contiguous():
        beta = beta.contiguous()
    if g is not None and not g.is_contiguous():
        g = g.contiguous()

    T, NK, DK = query.shape
    NV, DV = value.shape[1], value.shape[2]
    B = actual_seq_lengths.numel() - 1
    BLOCK_B = max(BLOCK_B, triton.next_power_of_2(B))
    if not isinstance(metadata_row_stride, int) or metadata_row_stride < 0:
        raise ValueError("metadata_row_stride must be a nonnegative integer")
    if metadata_row_stride:
        cells = B * metadata_row_stride
        if ssm_state_indices.numel() < cells or init_state_indices.numel() < cells:
            raise ValueError("state index tables must cover every fixed-width metadata row")

    has_idx = token_indices is not None or out_indices is not None
    if has_idx:
        if token_indices is None or out_indices is None:
            raise ValueError("token_indices and out_indices must be given together")
        if out is None:
            raise ValueError("token indirection writes caller-owned output rows, so out is required")
        if metadata_row_stride and (
            token_indices.numel() < B * metadata_row_stride or out_indices.numel() < B * metadata_row_stride
        ):
            raise ValueError("token index tables must cover every fixed-width metadata row")
        # The plan slices its tables off stable buffers; keep them dense for the
        # kernel's flat indexing without copying in the common case.
        if not token_indices.is_contiguous():
            token_indices = token_indices.contiguous()
        if not out_indices.is_contiguous():
            out_indices = out_indices.contiguous()
    elif out is None:
        out = torch.empty((T, NV, DV), dtype=torch.bfloat16, device=value.device)

    if B == 0:
        return out

    # Host shapes only: packed ragged rows are bounded by the whole table,
    # not by total/B. The scan itself masks every lane outside the live row.
    BLOCK_K = triton.next_power_of_2(DK)
    v_tiles = triton.cdiv(DV, BLOCK_V)
    grid = (v_tiles, NV, B)
    balanced_workers = False
    if REUSE_PARENT and metadata_row_stride > 0:
        init_device_properties_triton()
        cores = get_vectorcore_num()
        if v_tiles * NV * B > cores:
            balanced_workers = True
            grid = (cores, 1, 1)
    has_g = g is not None
    args = (
        query,
        key,
        value,
        g if has_g else beta,  # never dereferenced when HAS_G=0
        beta,
        state,
        out,
        actual_seq_lengths,
        ssm_state_indices,
        init_state_indices,
        token_indices if has_idx else beta,  # never dereferenced when HAS_IDX=0
        out_indices if has_idx else beta,  # ditto
        float(scale),
        NK,
        DK,
        NV,
        DV,
        state.stride(0),
        NV // NK,
    )
    spec_key = (
        BLOCK_V,
        BLOCK_K,
        BLOCK_B,
        has_g,
        SYNC_RELOAD,
        has_idx,
        REUSE_PARENT,
        metadata_row_stride,
        balanced_workers,
        NK,
        DK,
        NV,
        DV,
        state.stride(0),
        NV // NK,
        query.dtype,
        state.dtype,
        beta.dtype,
        actual_seq_lengths.dtype,
        ssm_state_indices.dtype,
        init_state_indices.dtype,
        token_indices.dtype if has_idx else None,
        out_indices.dtype if has_idx else None,
        out.dtype,
    )

    if strict or not fast:
        _validate_inputs(
            query,
            key,
            value,
            state,
            beta,
            g,
            gk,
            actual_seq_lengths,
            ssm_state_indices,
            init_state_indices,
            scale,
            token_indices,
            out_indices,
        )
        if not fast:
            recurrent_gated_delta_rule_tree_kernel[grid](
                *args,
                BLOCK_V=BLOCK_V,
                BLOCK_K=BLOCK_K,
                BLOCK_B=BLOCK_B,
                HAS_G=has_g,
                SYNC_RELOAD=SYNC_RELOAD,
                HAS_IDX=has_idx,
                REUSE_PARENT=REUSE_PARENT,
                METADATA_ROW_STRIDE=metadata_row_stride,
                B=B,
                BALANCED_WORKERS=balanced_workers,
            )
            return out

    ck = _CK_CACHE.get(spec_key)
    if ck is None:
        _validate_inputs(
            query,
            key,
            value,
            state,
            beta,
            g,
            gk,
            actual_seq_lengths,
            ssm_state_indices,
            init_state_indices,
            scale,
            token_indices,
            out_indices,
        )
        ck = recurrent_gated_delta_rule_tree_kernel.warmup(
            *args,
            grid=grid,
            BLOCK_V=BLOCK_V,
            BLOCK_K=BLOCK_K,
            BLOCK_B=BLOCK_B,
            HAS_G=has_g,
            SYNC_RELOAD=SYNC_RELOAD,
            HAS_IDX=has_idx,
            REUSE_PARENT=REUSE_PARENT,
            METADATA_ROW_STRIDE=metadata_row_stride,
            B=B,
            BALANCED_WORKERS=balanced_workers,
        )
        _CK_CACHE[spec_key] = ck

    # triton-ascend's launcher stub uses the full signature (constexprs are placeholders)
    device = torch.npu.current_device()
    stream = _raw_stream(device)
    ck.run(
        grid[0],
        grid[1],
        grid[2],
        stream,
        ck.function,
        ck.packed_metadata,
        None,
        None,
        None,
        *args,
        BLOCK_V,
        BLOCK_K,
        BLOCK_B,
        has_g,
        SYNC_RELOAD,
        has_idx,
        REUSE_PARENT,
        metadata_row_stride,
        B,
        balanced_workers,
    )
    return out


class ModelNew(torch.nn.Module):
    """Triton implementation wrapper for npu_recurrent_gated_delta_rule_tree."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        query,
        key,
        value,
        state,
        beta=None,
        scale=None,
        actual_seq_lengths=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        g=None,
        gk=None,
        init_state_indices=None,
    ):
        return npu_recurrent_gated_delta_rule_tree_triton(
            query,
            key,
            value,
            state,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            g=g,
            gk=gk,
            init_state_indices=init_state_indices,
        )
