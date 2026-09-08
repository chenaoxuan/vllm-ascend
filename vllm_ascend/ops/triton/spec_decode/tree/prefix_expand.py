from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(
    do_not_specialize=["num_reqs", "live", "num_nodes", "start", "child_depth"]
)
def expand_prefix_layer_kernel(
        top_scores_ptr,
        cand_ids_ptr,
        path_scores_ptr,
        frontier_ptr,
        tokens_ptr,
        depths_ptr,
        parents_ptr,
        parent_pos_ptr,
        num_reqs,
        live,
        num_nodes,
        start,
        child_depth,
        stride_ts_r,
        stride_ts_w,
        stride_cid_r,
        stride_cid_w,
        stride_ps_r,
        stride_fr_r,
        stride_tok_r,
        stride_dep_r,
        stride_par_r,
        stride_pp_r,
        WIDTH: tl.constexpr,
        N_FLAT: tl.constexpr,
        N_FLAT_PAD: tl.constexpr,
):
    # One request per program; grid-stride when R > vectorcores.
    #
    # Greedy top-WIDTH pick over the flattened WIDTH*WIDTH candidate matrix
    # (score = path_scores[frontier[j]] + top_scores[j][k], rows j >= live are
    # -inf). Vectorized: build the full score vector once, then WIDTH argmax
    # passes that mask out already-taken indices. Smallest flat index wins on
    # ties, matching the torch topk reference. This drops the per-request work
    # from O(WIDTH^4) scalar ops (nested argmax + taken scan) to O(WIDTH^3)
    # vectorized block ops and reads each score exactly once.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    neg_inf = -float("inf")
    big = tl.full((), N_FLAT_PAD, tl.int64)
    req = pid
    while req < num_reqs:
        # --- Precompute the N_FLAT candidate scores once -------------------
        offs = tl.arange(0, N_FLAT_PAD)
        valid = offs < N_FLAT
        i64 = offs.to(tl.int64)
        j = i64 // WIDTH  # parent row in [0, WIDTH)
        k = i64 - j * WIDTH  # col in [0, WIDTH)
        j_safe = tl.where(valid, j, 0)
        k_safe = tl.where(valid, k, 0)
        parent_id = tl.load(frontier_ptr + req * stride_fr_r + j_safe)
        ps = tl.load(path_scores_ptr + req * stride_ps_r + parent_id).to(tl.float32)
        ts = tl.load(
            top_scores_ptr + req * stride_ts_r + j_safe * stride_ts_w + k_safe
        ).to(tl.float32)
        ts = tl.where(valid & (j < live), ts, neg_inf)
        scores = tl.where(valid, ps + ts, neg_inf)

        # --- Greedy top-WIDTH selection (smallest index wins on ties) ------
        taken = ~valid  # padding permanently excluded
        for out_j in tl.range(0, WIDTH):
            vals = tl.where(taken, neg_inf, scores)
            best_val = tl.max(vals, axis=0)
            is_best = (vals == best_val) & (~taken)
            best_i = tl.min(tl.where(is_best, i64, big), axis=0)
            tl.store(parent_pos_ptr + req * stride_pp_r + out_j, best_i)
            taken = taken | (i64 == best_i)

        for out_j in tl.range(0, WIDTH):
            m_sel_i = tl.load(parent_pos_ptr + req * stride_pp_r + out_j)
            m_j = m_sel_i // WIDTH
            m_k = m_sel_i - m_j * WIDTH
            m_parent_id = tl.load(frontier_ptr + req * stride_fr_r + m_j)
            m_tok = tl.load(
                cand_ids_ptr + req * stride_cid_r + m_j * stride_cid_w + m_k
            )
            m_ps = tl.load(
                path_scores_ptr + req * stride_ps_r + m_parent_id
            ).to(tl.float32)
            m_ts = tl.load(
                top_scores_ptr + req * stride_ts_r + m_j * stride_ts_w + m_k
            ).to(tl.float32)
            m_ts = tl.where(m_j < live, m_ts, neg_inf)
            m_val = m_ps + m_ts
            tl.store(tokens_ptr + req * stride_tok_r + num_nodes + out_j, m_tok)
            tl.store(
                depths_ptr + req * stride_dep_r + num_nodes + out_j, child_depth
            )
            tl.store(parents_ptr + req * stride_par_r + start + out_j, m_parent_id)
            tl.store(path_scores_ptr + req * stride_ps_r + start + out_j, m_val)
            tl.store(parent_pos_ptr + req * stride_pp_r + out_j, m_j)
        req += nprog


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def expand_prefix_layer_triton(
        top_scores,
        cand_ids,
        path_scores,
        frontier,
        tokens,
        depths,
        parents,
        parent_pos,
        live: int,
        num_nodes: int,
        child_depth: int,
) -> None:
    # Inputs may be views; outputs share scratch storage (do not contiguous).
    ts = top_scores.contiguous()
    cids = cand_ids.contiguous()
    fr = frontier.contiguous()
    num_reqs, width, _ = ts.shape
    start = num_nodes + 1
    vec = get_vectorcore_num()
    grid = min(max(num_reqs, 1), max(vec, 1))
    n_flat = width * width
    expand_prefix_layer_kernel[(grid,)](
        ts,
        cids,
        path_scores,
        fr,
        tokens,
        depths,
        parents,
        parent_pos,
        num_reqs,
        live,
        num_nodes,
        start,
        child_depth,
        ts.stride(0),
        ts.stride(1),
        cids.stride(0),
        cids.stride(1),
        path_scores.stride(0),
        fr.stride(0),
        tokens.stride(0),
        depths.stride(0),
        parents.stride(0),
        parent_pos.stride(0),
        WIDTH=width,
        N_FLAT=n_flat,
        N_FLAT_PAD=_next_pow2(n_flat),
    )
