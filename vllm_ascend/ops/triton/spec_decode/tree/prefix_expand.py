from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(
    do_not_specialize=["num_reqs", "frontier_len", "take", "child_depth", "num_nodes"]
)
def prefix_expand_depth_kernel(
    top_scores_ptr,  # [R, WIDTH, K] fp32
    cand_ids_ptr,  # [R, WIDTH, K] i64
    frontier_ptr,  # [R, WIDTH] i64
    path_scores_ptr,  # [R, max_nodes] fp32
    score_scratch_ptr,  # [R, WIDTH*K] fp32 temporary flat scores
    tokens_ptr,  # [R, super] i64
    depths_ptr,  # [R, super] i64
    parents_ptr,  # [R, max_nodes] i64
    num_reqs,
    frontier_len,
    take,
    child_depth,
    num_nodes,
    stride_ts_r,
    stride_ts_w,
    stride_ci_r,
    stride_ci_w,
    stride_fr_r,
    stride_ps_r,
    stride_sc_r,
    stride_tok_r,
    stride_dep_r,
    stride_par_r,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
):
    """One depth: path-adjust scores, top-take select, write nodes, update frontier.

    Cube matmuls (Domino / GRU) stay on torch; this fuses the small index/select
    writes that dominate launch count in the expand loop.
    """
    req = tl.program_id(0)
    if req >= num_reqs:
        return

    neg_inf = -1.0e30
    flat_n = WIDTH * K

    for w in tl.static_range(WIDTH):
        valid_w = w < frontier_len
        fnode = tl.load(frontier_ptr + req * stride_fr_r + w)
        ps = tl.load(path_scores_ptr + req * stride_ps_r + fnode)
        for j in tl.static_range(K):
            edge = tl.load(
                top_scores_ptr + req * stride_ts_r + w * stride_ts_w + j
            )
            flat = w * K + j
            sc = tl.where(valid_w, ps + edge, neg_inf)
            tl.store(score_scratch_ptr + req * stride_sc_r + flat, sc)

    for t_i in tl.range(0, take):
        best_val = neg_inf
        best_flat = 0
        for flat in tl.range(0, flat_n):
            sc = tl.load(score_scratch_ptr + req * stride_sc_r + flat)
            better = sc > best_val
            best_val = tl.where(better, sc, best_val)
            best_flat = tl.where(better, flat, best_flat)

        # Do not reuse static_range names (w/j): Ascend Triton treats them
        # as constexpr and rejects reassignment inside tl.range.
        parent_pos = best_flat // K
        kid = best_flat - parent_pos * K
        tok = tl.load(
            cand_ids_ptr + req * stride_ci_r + parent_pos * stride_ci_w + kid
        )
        parent_node = tl.load(frontier_ptr + req * stride_fr_r + parent_pos)
        slot = num_nodes + t_i
        node_id = slot + 1
        tl.store(tokens_ptr + req * stride_tok_r + slot, tok)
        tl.store(depths_ptr + req * stride_dep_r + slot, child_depth)
        tl.store(parents_ptr + req * stride_par_r + node_id, parent_node)
        tl.store(path_scores_ptr + req * stride_ps_r + node_id, best_val)
        tl.store(score_scratch_ptr + req * stride_sc_r + best_flat, neg_inf)

    for fw in tl.static_range(WIDTH):
        val = tl.where(fw < take, num_nodes + 1 + fw, 0)
        tl.store(frontier_ptr + req * stride_fr_r + fw, val)


def prefix_expand_depth_torch(
    top_scores,
    cand_ids,
    frontier,
    path_scores,
    tokens,
    depths,
    parents,
    frontier_len: int,
    take: int,
    child_depth: int,
    num_nodes: int,
    width: int,
    k: int,
) -> None:
    """Torch golden for ``prefix_expand_depth_triton`` (CPU / UT)."""
    import torch

    num_reqs = top_scores.shape[0]
    device = top_scores.device
    width_arange = torch.arange(width, device=device)
    valid = width_arange[None, :] < frontier_len
    parent_scores = torch.gather(path_scores, 1, frontier)
    cand_scores = parent_scores.unsqueeze(-1) + top_scores
    cand_scores = torch.where(
        valid[:, :, None], cand_scores, torch.full_like(cand_scores, float("-inf"))
    )
    flat = cand_scores.reshape(num_reqs, -1)
    vals, sel = torch.topk(flat, k=take, dim=-1)
    parent_pos = sel // k
    sel_tokens = torch.gather(cand_ids.reshape(num_reqs, -1), 1, sel)
    sel_parents = torch.gather(frontier, 1, parent_pos)
    start = num_nodes + 1
    tokens[:, num_nodes : num_nodes + take] = sel_tokens
    depths[:, num_nodes : num_nodes + take] = child_depth
    parents[:, start : start + take] = sel_parents
    path_scores[:, start : start + take] = vals
    new_ids = (
        torch.arange(start, start + take, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(num_reqs, -1)
    )
    if take < width:
        pad = torch.zeros(num_reqs, width - take, dtype=torch.long, device=device)
        new_ids = torch.cat([new_ids, pad], dim=-1)
    frontier.copy_(new_ids)


def prefix_expand_depth_triton(
    top_scores,
    cand_ids,
    frontier,
    path_scores,
    score_scratch,
    tokens,
    depths,
    parents,
    frontier_len: int,
    take: int,
    child_depth: int,
    num_nodes: int,
    width: int,
    k: int,
) -> None:
    num_reqs = top_scores.shape[0]
    vec = get_vectorcore_num()
    grid = min(max(num_reqs, 1), max(vec, 1))
    ts = top_scores.contiguous()
    ci = cand_ids.contiguous()
    sc = score_scratch.contiguous()
    prefix_expand_depth_kernel[(grid,)](
        ts,
        ci,
        frontier,
        path_scores,
        sc,
        tokens,
        depths,
        parents,
        num_reqs,
        frontier_len,
        take,
        child_depth,
        num_nodes,
        ts.stride(0),
        ts.stride(1),
        ci.stride(0),
        ci.stride(1),
        frontier.stride(0),
        path_scores.stride(0),
        sc.stride(0),
        tokens.stride(0),
        depths.stride(0),
        parents.stride(0),
        WIDTH=width,
        K=k,
    )


def prefix_domino_score_torch(
    s_proj,
    z,
    cand_vals,
    cand_ids,
    cand_w,
    cand_bias,
    valid_parent,
    k: int,
    use_silu: bool,
):
    """Cube Domino score after ``F.linear(w_s)`` (SiLU + einsum + topk/lse/mask).

    Ascend Triton scalar rematerialization of mid·w_c is slower than Cube torch
    at typical ``M=256,C=64``; keep this as the production path.
    """
    import torch
    import torch.nn.functional as F

    mid_in = z.unsqueeze(1) + s_proj
    mid = F.silu(mid_in) if use_silu else mid_in
    bias = torch.einsum("rwm,rcm->rwc", mid, cand_w)
    if cand_bias is not None:
        bias = bias + cand_bias.unsqueeze(1)
    logits = cand_vals.unsqueeze(1).to(bias.dtype) + bias
    logits = logits.float()
    top_vals, top_ids = torch.topk(logits, k=k, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_scores = top_vals - log_z
    width = s_proj.size(1)
    expanded = cand_ids.unsqueeze(1).expand(-1, width, -1)
    sel_ids = torch.gather(expanded, 2, top_ids)
    top_scores = torch.where(
        valid_parent[:, :, None],
        top_scores,
        torch.full_like(top_scores, float("-inf")),
    )
    return top_scores, sel_ids


def prefix_domino_score_triton(
    s_proj,
    z,
    cand_vals,
    cand_ids,
    cand_w,
    cand_bias,
    valid_parent,
    top_scores_out,
    cand_ids_out,
    k: int,
    use_silu: bool,
) -> None:
    """Dispatch to Cube torch (see ``prefix_domino_score_torch``)."""
    top_scores, sel_ids = prefix_domino_score_torch(
        s_proj,
        z,
        cand_vals,
        cand_ids,
        cand_w,
        cand_bias,
        valid_parent,
        k,
        use_silu,
    )
    top_scores_out.copy_(top_scores)
    cand_ids_out.copy_(sel_ids)


def prefix_gru_mix_torch(tokens, parent_h, gh, gru_table, out_h) -> None:
    """Torch GRU gate mix after Cube ``F.linear(W_hh)`` (same as update_hidden)."""
    import torch

    num_reqs, take, gru_h = parent_h.shape
    flat_tok = tokens.reshape(-1)
    gi = gru_table.index_select(0, flat_tok).view(num_reqs, take, 3 * gru_h)
    i_r, i_z, i_n = gi.split(gru_h, dim=-1)
    h_r, h_z, h_n = gh.split(gru_h, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    out_h.copy_((1.0 - z) * n + z * parent_h)


def prefix_gru_mix_triton(tokens, parent_h, gh, gru_table, out_h) -> None:
    """Dispatch to torch; Ascend scalar Triton over H≈1024 is slower than Cube."""
    prefix_gru_mix_torch(tokens, parent_h, gh, gru_table, out_h)
