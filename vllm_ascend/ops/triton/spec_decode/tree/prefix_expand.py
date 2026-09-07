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
):
    # One request per program; grid-stride when R > vectorcores.
    # Small top-k: scan WIDTH^2, first index wins on ties.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    neg_inf = -float("inf")
    req = pid
    while req < num_reqs:
        for out_j in tl.range(0, WIDTH):
            best_val = tl.full((), neg_inf, tl.float32)
            best_i = tl.full((), 0, tl.int64)
            for i in tl.range(0, N_FLAT):
                i64 = i.to(tl.int64)
                j = i64 // WIDTH
                k = i64 - j * WIDTH
                taken = tl.full((), 0, tl.int32)
                for prev in tl.range(0, WIDTH):
                    prev_i = tl.load(parent_pos_ptr + req * stride_pp_r + prev)
                    taken = tl.where(
                        (prev < out_j) & (prev_i == i64), 1, taken
                    )
                parent_id = tl.load(frontier_ptr + req * stride_fr_r + j)
                ps = tl.load(
                    path_scores_ptr + req * stride_ps_r + parent_id
                ).to(tl.float32)
                ts = tl.load(
                    top_scores_ptr
                    + req * stride_ts_r
                    + j * stride_ts_w
                    + k
                ).to(tl.float32)
                ts = tl.where(j < live, ts, neg_inf)
                score = tl.where(taken == 0, ps + ts, neg_inf)
                better = (score > best_val) | (
                    (score == best_val) & (i64 < best_i)
                )
                best_val = tl.where(better, score, best_val)
                best_i = tl.where(better, i64, best_i)
            tl.store(parent_pos_ptr + req * stride_pp_r + out_j, best_i)

        for out_j in tl.range(0, WIDTH):
            i64 = tl.load(parent_pos_ptr + req * stride_pp_r + out_j)
            j = i64 // WIDTH
            k = i64 - j * WIDTH
            parent_id = tl.load(frontier_ptr + req * stride_fr_r + j)
            tok = tl.load(
                cand_ids_ptr + req * stride_cid_r + j * stride_cid_w + k
            )
            ps = tl.load(
                path_scores_ptr + req * stride_ps_r + parent_id
            ).to(tl.float32)
            ts = tl.load(
                top_scores_ptr + req * stride_ts_r + j * stride_ts_w + k
            ).to(tl.float32)
            ts = tl.where(j < live, ts, neg_inf)
            val = ps + ts
            tl.store(
                tokens_ptr + req * stride_tok_r + num_nodes + out_j, tok
            )
            tl.store(
                depths_ptr + req * stride_dep_r + num_nodes + out_j,
                child_depth,
            )
            tl.store(
                parents_ptr + req * stride_par_r + start + out_j, parent_id
            )
            tl.store(
                path_scores_ptr + req * stride_ps_r + start + out_j, val
            )
            tl.store(parent_pos_ptr + req * stride_pp_r + out_j, j)
        req += nprog


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
        N_FLAT=width * width,
    )
