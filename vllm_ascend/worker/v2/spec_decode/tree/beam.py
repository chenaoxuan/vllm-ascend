import torch

from vllm_ascend.worker.v2.spec_decode.tree.builder import TreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout, finalize_tree_layout

# One request's latest beam expansion. Rejection reads this for the tree it verifies.
LAST_LIFE: dict = {}
# Globally kept token ids at each depth, after the topk^2 -> topk cut. Req 0.
LEVEL_SELECTED: list[list[int]] = []
_COVER_HITS: list[int] = []
_COVER_DENOM: list[int] = []
_COVER_TOPK = 0
_COVER_STEPS = 0


def _agent_dbg(location, message, data, hypothesis_id, limit=400):
    try:
        import importlib.util
        import sys

        mod = sys.modules.get("_agent_debug_trace")
        if mod is None:
            spec = importlib.util.spec_from_file_location(
                "_agent_debug_trace",
                "/home/specdec/spec260922/debug_trace.py",
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_agent_debug_trace"] = mod
            spec.loader.exec_module(mod)
        mod.dbg(location, message, data, hypothesis_id, limit=limit)
    except Exception:
        pass


def note_life_ctx(ctx: dict) -> None:
    LAST_LIFE["ctx"] = ctx


def _cover_rank_ok() -> bool:
    try:
        import torch

        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()) == int(dist.get_world_size()) - 1
    except Exception:
        return True
    return True


def publish_level_selected(rows: list[list[int]], topk: int) -> None:
    """Store this step's globally kept tokens. One row per draft depth."""
    global LEVEL_SELECTED, _COVER_TOPK
    LEVEL_SELECTED = rows
    _COVER_TOPK = int(topk)


def record_position_cover(want: int, fail_depth: int, selected: list[list[int]]) -> None:
    """Coverage of the target token inside the level-wide kept top-k.

    Position ``d`` is counted when the first ``d`` drafts were accepted.
    It hits when ``d`` was accepted, or when the target token at the first
    miss is one of the ``topk`` tokens kept from that depth's candidates.
    """
    global _COVER_STEPS
    if not _cover_rank_ok() or not selected:
        return
    n = len(selected)
    while len(_COVER_HITS) < n:
        _COVER_HITS.append(0)
        _COVER_DENOM.append(0)
    for depth in range(n):
        if depth > fail_depth:
            break
        _COVER_DENOM[depth] += 1
        if depth < fail_depth or int(want) in selected[depth]:
            _COVER_HITS[depth] += 1
    _COVER_STEPS += 1
    _write_cover()


def _write_cover() -> None:
    import json
    import os

    topk = _COVER_TOPK
    path = f"/home/specdec/spec260922/results/topk_cover_k{topk}.json"
    coverage = [
        round(hit / denom, 4) if denom else None
        for hit, denom in zip(_COVER_HITS, _COVER_DENOM)
    ]
    payload = {
        "topk": topk,
        "positions": list(range(len(_COVER_HITS))),
        "hits": list(_COVER_HITS),
        "denom": list(_COVER_DENOM),
        "coverage": coverage,
        "steps": _COVER_STEPS,
        "definition": (
            "position d counts only after the first d drafts are accepted; "
            "hit when that draft was accepted, or when the target token is "
            "among the topk ids kept from this depth's candidates"
        ),
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp, path)
        if _COVER_STEPS == 1 or _COVER_STEPS % 50 == 0:
            import logging

            logging.getLogger("vllm.topk_cover").info(
                "topk=%s steps=%s coverage=%s denom=%s",
                topk,
                _COVER_STEPS,
                coverage,
                list(_COVER_DENOM),
            )
    except Exception:
        pass


def _shadow_greedy(draft_model, draft_logits, root_token_ids):
    """Greedy chain on the same depth logits and Markov bias the beam uses."""
    spec = draft_logits.shape[1]
    prev = root_token_ids[:1].reshape(1, 1)
    shadow = []
    raw = []
    for depth in range(spec):
        raw.append(int(draft_logits[0, depth].argmax().item()))
        step = _markov_correct_logits(
            draft_model, draft_logits[:1, depth], prev
        )
        tid = step[0, 0].argmax().view(1)
        if draft_model is not None:
            tid = draft_model.map_draft_to_target(tid).reshape(-1)[:1]
        shadow.append(int(tid[0].item()))
        prev = tid.view(1, 1)
    return shadow, raw


def _trace_beam_life(
    draft_model,
    draft_logits,
    root_token_ids,
    level_recs,
    remap,
    tokens,
    depths,
    parent_ids,
    pool_scores,
    packed,
    num_nodes,
) -> None:
    """Record req 0: shadow chain, per-parent top-k, and who survived."""
    try:
        shadow, raw = _shadow_greedy(draft_model, draft_logits, root_token_ids)
        by_parent = {}
        for rec in level_recs:
            for j, pool in enumerate(rec["par_pool"]):
                if pool < 0:
                    fid = 0
                else:
                    fid = int(remap[0, pool].item())
                    if fid <= 0:
                        continue
                kept = [
                    tok
                    for tok, par in zip(rec["sel_tok"], rec["sel_par"])
                    if par == pool
                ]
                by_parent[fid] = {
                    "tok": rec["par_tok"][j],
                    "score": rec["par_score"][j],
                    "top": rec["top"][j],
                    "level_kept": kept,
                }
        n = int(num_nodes)
        kept_nodes = []
        scores = torch.gather(pool_scores[0], 0, packed[0, :n])
        for i in range(n):
            kept_nodes.append(
                {
                    "i": i + 1,
                    "t": int(tokens[0, i].item()),
                    "p": int(parent_ids[0, i].item()),
                    "d": int(depths[0, i].item()),
                    "s": round(float(scores[i].item()), 3),
                }
            )
        ctx = LAST_LIFE.get("ctx")
        LAST_LIFE.clear()
        LAST_LIFE.update(
            {
                "ctx": ctx,
                "root": int(root_token_ids[0].item()),
                "shadow": shadow,
                "raw": raw,
                "by_parent": by_parent,
                "kept": kept_nodes,
            }
        )
        _agent_dbg(
            "tree/beam.py:build",
            "life_build",
            {
                "root": LAST_LIFE["root"],
                "shadow": shadow,
                "raw": raw,
                "kept": kept_nodes,
                "by_parent": by_parent,
                "ctx": ctx,
            },
            "H6",
        )
    except Exception:
        pass


def _markov_correct_logits(
    draft_model,
    depth_logits: torch.Tensor,
    frontier_tokens: torch.Tensor,
) -> torch.Tensor:
    """Beam-only one-depth logit correction (optional DSpark Markov bias).

    When ``draft_model`` is set, mirrors ``DSparkSpeculator._sample_sequential``:
    bias from the previous **target-vocab** token via ``markov_embed`` +
    ``markov_bias``. When ``draft_model`` is None, expand the shared-depth base
    logits across the frontier.
    """
    base = depth_logits.unsqueeze(1)
    if draft_model is None:
        return base.expand(-1, frontier_tokens.size(1), -1)
    markov_emb = draft_model.markov_embed(frontier_tokens)
    # NPU unquantized gemm is 2D-only; flatten then restore the frontier layout.
    bias = draft_model.markov_bias(markov_emb.reshape(-1, markov_emb.shape[-1]))
    return base + bias.view(*markov_emb.shape[:-1], -1)


class BeamTreeBuilder(TreeBuilder):
    """Level-wise beam (PCTree) expansion; optional DSpark Markov bias."""

    required_backend = "dspark"
    method = "beam"

    def __init__(self, budget: int, topk: int, draft_model=None):
        super().__init__(budget, topk)
        self.draft_model = draft_model

    def build(
        self,
        draft_logits: torch.Tensor,
        out: TreeLayout,
        *,
        root_token_ids: torch.Tensor | None = None,
        draft_hidden: torch.Tensor | None = None,
        proposal_logits: torch.Tensor | None = None,
    ) -> TreeLayout:
        budget = self.budget
        topk = self.topk
        draft_model = self.draft_model
        num_reqs, spec_num, vocab = draft_logits.shape
        k = min(topk, vocab)
        device = draft_logits.device
        frontier_tokens = root_token_ids.unsqueeze(1)
        frontier_scores = torch.zeros(num_reqs, 1, dtype=torch.float32, device=device)
        frontier_pools = torch.full((num_reqs, 1), -1, dtype=torch.long, device=device)

        pool_tokens = torch.empty(num_reqs, 0, dtype=torch.long, device=device)
        pool_scores = torch.empty(num_reqs, 0, dtype=torch.float32, device=device)
        pool_parents = torch.empty(num_reqs, 0, dtype=torch.long, device=device)
        pool_depth = torch.empty(num_reqs, 0, dtype=torch.long, device=device)

        # Temp proposal indexed by root=0 and pool slot i -> i+1. Remapped later.
        # FP32 to match tree_proposal_logits; NPU IndexPut rejects BF16.
        max_pool = spec_num * k
        prop_temp = None
        if proposal_logits is not None:
            prop_temp = torch.full(
                (num_reqs, max_pool + 1, vocab),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )

        # Keep the full level-wise beam pool, then prune it once by the
        # configured global budget.
        level_recs = []
        for depth in range(spec_num):
            batch = frontier_tokens.size(1)
            step_logits = _markov_correct_logits(
                draft_model, draft_logits[:, depth], frontier_tokens
            )
            if prop_temp is not None:
                req_idx = torch.arange(num_reqs, device=device)
                step_f = step_logits.to(dtype=prop_temp.dtype)
                for j in range(batch):
                    parent_pool = frontier_pools[:, j]
                    temp_id = torch.where(
                        parent_pool < 0,
                        torch.zeros((), dtype=torch.long, device=device),
                        parent_pool + 1,
                    ).clamp(max=prop_temp.shape[1] - 1)
                    prop_temp[req_idx, temp_id] = step_f[:, j]
            log_probs = torch.log_softmax(step_logits.float(), dim=-1)
            top_vals, top_ids = log_probs.topk(k, dim=-1)
            if draft_model is not None:
                top_ids = draft_model.map_draft_to_target(top_ids)
            candidate_scores = frontier_scores.unsqueeze(-1) + top_vals
            num_candidates = batch * k
            cand_flat = candidate_scores.reshape(num_reqs, -1)
            top_ids_flat = top_ids.reshape(num_reqs, -1)
            parent_flat = frontier_pools.repeat_interleave(k, dim=-1)
            selected = cand_flat.topk(min(k, num_candidates), dim=-1).indices
            sel_tokens = torch.gather(top_ids_flat, 1, selected)
            sel_scores = torch.gather(cand_flat, 1, selected)
            sel_parents = torch.gather(parent_flat, 1, selected)
            n_add = selected.shape[1]

            pool_tokens = torch.cat([pool_tokens, sel_tokens], dim=-1)
            pool_scores = torch.cat([pool_scores, sel_scores], dim=-1)
            pool_parents = torch.cat([pool_parents, sel_parents], dim=-1)
            pool_depth = torch.cat(
                [
                    pool_depth,
                    torch.full(
                        (num_reqs, n_add),
                        depth,
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=-1,
            )
            # Previous per-parent trace. Paused: it syncs every depth.
            if False:
                try:
                    level_recs.append(
                        {
                            "d": depth,
                            "par_pool": frontier_pools[0].tolist(),
                            "par_tok": frontier_tokens[0].tolist(),
                            "par_score": [
                                round(float(x), 3)
                                for x in frontier_scores[0].tolist()
                            ],
                            "top": top_ids[0].tolist(),
                            "sel_tok": sel_tokens[0].tolist(),
                            "sel_par": sel_parents[0].tolist(),
                        }
                    )
                except Exception:
                    pass
            frontier_tokens = sel_tokens
            frontier_scores = sel_scores
            base = pool_tokens.size(-1) - n_add
            frontier_pools = base + torch.arange(
                n_add, device=device, dtype=torch.long
            ).unsqueeze(0).expand(num_reqs, -1)

        num_pool = pool_tokens.size(-1)
        num_nodes = min(int(budget), num_pool)
        # Cumulative log-probability never increases down an edge. Stable
        # global ordering therefore keeps every ancestor before its children,
        # including exact-score ties, so the budget prefix is ancestor-closed.
        packed = pool_scores.argsort(dim=-1, descending=True, stable=True)[
            :, :num_nodes
        ]
        packed_depth = torch.gather(pool_depth, 1, packed)
        packed_rank = packed_depth * num_pool + packed
        packed = torch.gather(
            packed,
            1,
            packed_rank.argsort(dim=-1),
        )

        remap = torch.zeros(num_reqs, num_pool, dtype=torch.long, device=device)
        node_ids = torch.arange(
            1, num_nodes + 1, dtype=torch.long, device=device
        ).unsqueeze(0)
        remap.scatter_(1, packed, node_ids.expand(num_reqs, num_nodes))
        pool_parent_packed = torch.gather(pool_parents, 1, packed)
        non_root = pool_parent_packed >= 0
        raw_parent = torch.gather(remap, 1, pool_parent_packed.clamp(min=0))
        parent_ids = torch.where(
            non_root, raw_parent, torch.zeros_like(raw_parent)
        )

        tokens = torch.gather(pool_tokens, 1, packed)
        depths = (torch.gather(pool_depth, 1, packed) + 1).to(torch.int32)
        finalize_tree_layout(out, tokens, depths, parent_ids, num_nodes)
        # Previous full lifecycle dump, including the shadow chain. Paused.
        if False:
            _trace_beam_life(
                draft_model,
                draft_logits,
                root_token_ids,
                level_recs,
                remap,
                tokens,
                depths,
                parent_ids,
                pool_scores,
                packed,
                num_nodes,
            )

        if proposal_logits is not None and prop_temp is not None:
            proposal_logits.fill_(float("-inf"))
            proposal_logits[:, 0] = prop_temp[:, 0]
            req_idx = torch.arange(num_reqs, device=device)
            for pool_idx in range(num_pool):
                final_id = remap[:, pool_idx]
                has = final_id > 0
                proposal_logits[req_idx, final_id.clamp(min=0)] = torch.where(
                    has.unsqueeze(-1),
                    prop_temp[:, pool_idx + 1],
                    proposal_logits[req_idx, final_id.clamp(min=0)],
                )
        return out
