import torch

from vllm_ascend.worker.v2.spec_decode.tree.utils import TreeLayout

_PAD_TOKEN_ID = -1


def greedy_tree_reject(
    tree: TreeLayout,
    target_logits: torch.Tensor,
    num_speculative_tokens: int,
) -> torch.Tensor:
    """Greedy-verify a draft tree by token-id comparison on device.

    ``target_logits`` is ``[batch_size, budget + 1, vocab_size]``, indexed by
    node id (column 0 is the already-accepted root).

    Returns ``[num_reqs, spec_len + 1]`` accepted token ids including bonus.
    Unused slots are ``-1``.
    """
    tokens = tree.tokens
    first_child = tree.first_child
    next_sibling = tree.next_sibling
    device = tokens.device
    num_reqs, budget = tokens.shape
    spec_len = num_speculative_tokens
    target_token_ids = target_logits.argmax(dim=-1)
    sampled_token_ids = torch.full(
        (num_reqs, spec_len + 1),
        _PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    neg_one = torch.tensor(-1, dtype=torch.long, device=device)

    for req_idx in range(num_reqs):  # tl.program_id(0)
        current = torch.zeros((), dtype=torch.long, device=device)
        alive = torch.ones((), dtype=torch.bool, device=device)
        for n_out in range(spec_len):
            t = target_token_ids[req_idx, current]
            sampled_token_ids[req_idx, n_out] = torch.where(
                alive, t, sampled_token_ids[req_idx, n_out]
            )
            child = first_child[req_idx, current].to(torch.long)
            found_child = neg_one
            for _ in range(budget):
                valid = alive & (found_child < 0) & (child >= 0)
                found_child = torch.where(
                    valid & (tokens[req_idx, child - 1] == t),
                    child,
                    found_child,
                )
                child = torch.where(
                    valid & (found_child < 0),
                    next_sibling[req_idx, child].to(torch.long),
                    child,
                )
            found = alive & (found_child >= 0)
            current = torch.where(found, found_child, current)
            alive = found
        t = target_token_ids[req_idx, current]
        sampled_token_ids[req_idx, spec_len] = torch.where(
            alive, t, sampled_token_ids[req_idx, spec_len]
        )

    return sampled_token_ids


def _update_parent_residual(
    p: torch.Tensor,
    mb: torch.Tensor,
    ms: torch.Tensor,
    token: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update parent residual ``M_b`` / ``M_s`` / ``p`` after rejecting ``token``.

    ``p`` is a 0-dim tensor. ``mb`` / ``ms`` are the parent's full-vocab
    ``M_b`` and ``M_s``. ``token`` is a Python int or a 0-dim long tensor.
    Returns the updated ``(p, M_b, M_s)``; does not sample.
    """
    residual = (p * mb - ms).clamp(min=0)
    z = residual.sum()
    denom = z + (1 - p)
    p_new = torch.where(denom > 0, z / denom, torch.zeros((), dtype=mb.dtype, device=mb.device))
    mb_new = torch.where(z > 0, residual / z, torch.zeros_like(mb))
    idx = torch.arange(ms.shape[-1], device=ms.device)
    ms_new = torch.where(idx == token, torch.zeros_like(ms), ms)
    mass = ms_new.sum()
    ms_new = torch.where(mass > 0, ms_new / mass, ms_new)
    return p_new, mb_new, ms_new


def _sample_token(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    cdf = torch.cumsum(probs, dim=-1)
    lo = torch.tensor(1e-12, dtype=probs.dtype, device=probs.device)
    u = torch.clamp(u, min=lo, max=cdf[-1])
    idx = torch.searchsorted(cdf, u, right=False)
    last = torch.tensor(probs.shape[0] - 1, dtype=torch.long, device=probs.device)
    return torch.minimum(idx, last)


def _first_leaf(first_child: torch.Tensor) -> torch.Tensor:
    cur = torch.zeros((), dtype=torch.long, device=first_child.device)
    for _ in range(first_child.numel()):
        child = first_child[cur].to(torch.long)
        cur = torch.where(child >= 0, child, cur)
    return cur


def _unlink(
    first_child: torch.Tensor,
    next_sibling: torch.Tensor,
    parent: torch.Tensor,
    node: torch.Tensor,
    active: torch.Tensor,
) -> None:
    neg_one = torch.tensor(-1, dtype=torch.long, device=first_child.device)
    child = first_child[parent].to(torch.long)
    is_first = active & (child == node)
    first_child[parent] = torch.where(
        is_first, next_sibling[node].to(torch.long), first_child[parent].to(torch.long)
    )
    prev = child
    for _ in range(first_child.numel()):
        safe_prev = prev.clamp(min=0)
        nxt = torch.where(prev >= 0, next_sibling[safe_prev].to(torch.long), neg_one)
        match = active & ~is_first & (prev >= 0) & (nxt == node)
        next_sibling[safe_prev] = torch.where(
            match, next_sibling[node].to(torch.long), next_sibling[safe_prev].to(torch.long)
        )
        step = active & ~is_first & (prev >= 0) & (nxt >= 0) & (nxt != node)
        prev = torch.where(step, nxt, prev)
    next_sibling[node] = torch.where(active, neg_one, next_sibling[node].to(torch.long))


def _prune_subtree(
    first_child: torch.Tensor,
    next_sibling: torch.Tensor,
    parents: torch.Tensor,
    node: torch.Tensor,
    active: torch.Tensor,
) -> None:
    neg_one = torch.tensor(-1, dtype=torch.long, device=first_child.device)
    zero = torch.zeros((), dtype=torch.long, device=first_child.device)
    is_root = node == 0
    first_child[0] = torch.where(active & is_root, neg_one, first_child[0].to(torch.long))
    safe = (node - 1).clamp(min=0)
    grand = torch.where(node > 0, parents[safe].to(torch.long), zero)
    _unlink(first_child, next_sibling, grand, node, active & ~is_root & (node > 0))


def _recompute_descendant_p(
    p: torch.Tensor,
    mb: torch.Tensor,
    ms: torch.Tensor,
    parents: torch.Tensor,
    tokens: torch.Tensor,
    has_draft: bool,
    active: torch.Tensor,
    start: torch.Tensor,
) -> None:
    one = p.new_ones(())
    zero = p.new_zeros(())
    for node in range(1, p.shape[0]):
        parent = parents[node - 1].to(torch.long)
        token = tokens[node - 1]
        valid = active & (token >= 0) & (start != node)
        safe_parent = parent.clamp(min=0)
        safe_token = token.clamp(min=0)
        if has_draft:
            ms_t = ms[safe_parent, safe_token]
            ratio = torch.where(ms_t > 0, mb[safe_parent, safe_token] / ms_t, zero)
        else:
            ratio = mb[safe_parent, safe_token]
        p[node] = torch.where(valid, torch.minimum(p[safe_parent] * ratio, one), p[node])


def _write_path(
    sampled: torch.Tensor,
    req_idx: int,
    spec_len: int,
    tau: torch.Tensor,
    parents: torch.Tensor,
    tokens: torch.Tensor,
    y: torch.Tensor,
) -> None:
    device = sampled.device
    zero = torch.zeros((), dtype=torch.long, device=device)
    pad = torch.tensor(_PAD_TOKEN_ID, dtype=torch.long, device=device)
    spec_cap = torch.tensor(spec_len, dtype=torch.long, device=device)
    cur = tau
    depth_tau = zero
    for _ in range(spec_len):
        is_node = cur > 0
        depth_tau = depth_tau + is_node.to(torch.long)
        safe = (cur - 1).clamp(min=0)
        cur = torch.where(is_node, parents[safe].to(torch.long), cur)
    sampled[req_idx, torch.minimum(depth_tau, spec_cap)] = y
    cur = tau
    d = depth_tau
    for _ in range(spec_len):
        is_node = cur > 0
        slot = (d - 1).clamp(min=0)
        write = is_node & (d > 0) & (d <= spec_cap)
        safe = (cur - 1).clamp(min=0)
        tok = torch.where(is_node, tokens[safe], pad)
        sampled[req_idx, slot] = torch.where(write, tok, sampled[req_idx, slot])
        d = torch.where(is_node, d - 1, d)
        cur = torch.where(is_node, parents[safe].to(torch.long), cur)


def block_tree_reject(
    tree: TreeLayout,
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor | None,
    num_speculative_tokens: int,
    etas: torch.Tensor | None = None,
    recover_u: torch.Tensor | None = None,
) -> torch.Tensor:
    """MagicMTP Block Verify on a draft tree, from leaf to root.

    ``target_logits`` is ``[num_reqs, budget + 1, vocab]``, node 0 = root.
    ``draft_logits`` is ``[num_reqs, spec_num, vocab]`` (DFlash depth rows) or
    ``None`` (q=1, one-hot residual). No temperature / top-k / top-p.

    Accepts the whole root→leaf block, or rejects the leaf and updates residual
    ``M_b`` / ``M_s`` / ``p`` at the parent. Recovers a single token ``Y`` from
    ``M_b`` at the accepted prefix.

    ``etas`` is ``[num_reqs, max_visits]`` accept/reject draws. ``recover_u``
    is ``[num_reqs]`` for the final ``Y``. Unused outputs are ``-1``.
    """
    tokens = tree.tokens
    device = tokens.device
    num_reqs, budget = tokens.shape
    spec_len = num_speculative_tokens
    vocab = target_logits.shape[-1]
    target_probs = torch.softmax(target_logits.float(), dim=-1)
    has_draft = draft_logits is not None
    draft_probs = torch.softmax(draft_logits.float(), dim=-1) if has_draft else None
    spec_num = draft_probs.shape[1] if draft_probs is not None else 0
    if etas is None:
        etas = torch.rand(num_reqs, budget, device=device)
    if recover_u is None:
        recover_u = torch.rand(num_reqs, device=device)
    sampled = torch.full(
        (num_reqs, spec_len + 1),
        _PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    zero_long = torch.zeros((), dtype=torch.long, device=device)
    vocab_idx = torch.arange(vocab, device=device)

    for req_idx in range(num_reqs):
        tok = tokens[req_idx]
        par = tree.parents[req_idx]
        fc = tree.first_child[req_idx].clone().to(torch.long)
        ns = tree.next_sibling[req_idx].clone().to(torch.long)
        mb = target_probs[req_idx].clone()
        ms = target_probs.new_zeros(budget + 1, vocab)
        if has_draft:
            ms[0] = draft_probs[req_idx, 0]
            if spec_num > 0:
                depth = tree.depths[req_idx].to(torch.long)
                step = depth.clamp(min=0, max=spec_num - 1)
                gathered = draft_probs[req_idx].index_select(0, step)
                valid_ms = (depth > 0) & (depth < spec_num) & (tok >= 0)
                ms[1:] = torch.where(valid_ms.unsqueeze(-1), gathered, ms[1:])
        p = target_probs.new_zeros(budget + 1)
        p[0] = 1
        true = torch.ones((), dtype=torch.bool, device=device)
        _recompute_descendant_p(p, mb, ms, par, tok, has_draft, true, zero_long)

        done = torch.zeros((), dtype=torch.bool, device=device)
        tau = zero_long
        max_visits = etas.shape[1]
        one = p.new_ones(())
        for visit in range(budget):
            leaf = _first_leaf(fc)
            still = (leaf > 0) & ~done
            eta = etas[req_idx, visit] if visit < max_visits else one
            accept = still & (eta < p[leaf])
            tau = torch.where(accept, leaf, tau)
            done = done | accept | (leaf == 0)
            reject = still & ~accept
            safe_leaf = (leaf - 1).clamp(min=0)
            parent = torch.where(leaf > 0, par[safe_leaf].to(torch.long), zero_long)
            token = torch.where(leaf > 0, tok[safe_leaf], zero_long)
            safe_parent = parent.clamp(min=0)
            safe_token = token.clamp(min=0)
            ms_parent = ms[safe_parent]
            if not has_draft:
                ms_parent = torch.where(
                    vocab_idx == safe_token, mb.new_ones(vocab), mb.new_zeros(vocab)
                )
            p_new, mb_new, ms_new = _update_parent_residual(
                p[safe_parent], mb[safe_parent], ms_parent, safe_token
            )
            p[safe_parent] = torch.where(reject, p_new, p[safe_parent])
            mb[safe_parent] = torch.where(reject, mb_new, mb[safe_parent])
            if has_draft:
                ms[safe_parent] = torch.where(reject, ms_new, ms[safe_parent])
            _unlink(fc, ns, parent, leaf, reject)
            do_prune = reject & (p_new <= 0)
            _prune_subtree(fc, ns, par, parent, do_prune)
            _recompute_descendant_p(
                p, mb, ms, par, tok, has_draft, reject & ~do_prune, safe_parent
            )

        y = _sample_token(mb[tau], recover_u[req_idx])
        _write_path(sampled, req_idx, spec_len, tau, par, tok, y)

    return sampled
