"""One compressor call and one attention call for every tree chain.

The outer boundary groups real requests. Operators never see it.
They see one flat ``cu_seqlens`` whose segment ``k`` is tree chain ``k``,
so each chain is its own request, and the kernel is launched once.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

_LAYOUT: dict | None = None
_PLAN: dict | None = None
_KV_STACK: list[list[torch.Tensor]] = []


def set_tree_chain_layout(parents, num_nodes, depths, query_start_loc, start_pos) -> None:
    global _LAYOUT, _PLAN
    _LAYOUT = {
        "parents": parents,
        "num_nodes": num_nodes,
        "depths": depths,
        "qsl": query_start_loc,
        "start": start_pos,
    }
    _PLAN = None


def clear_tree_chain_layout() -> None:
    global _LAYOUT, _PLAN
    _LAYOUT = None
    _PLAN = None


def chain_layout_active() -> bool:
    return _LAYOUT is not None


@contextmanager
def chain_kv_scope(caches):
    """Caches that share the block table the next compressor call writes."""
    live = [c for c in caches if c is not None]
    _KV_STACK.append(live)
    try:
        yield
    finally:
        _KV_STACK.pop()


def _log(data: dict) -> None:
    try:
        from debug_trace import cmp_log
    except Exception:
        return
    cmp_log(data)


def _build_plan(num_tokens: int, device) -> dict | None:
    """Flatten (request, chain) into the batch the operators see.

    ``cu`` / ``attn_cu`` is the inner chain-token boundary the operators see.
    Segment ``k`` has length equal to chain ``k`` (root through that node).
    ``req_cu`` is the outer request boundary and is not passed to kernels.
    """
    global _PLAN
    if _PLAN is not None:
        return _PLAN
    layout = _LAYOUT
    if layout is None:
        return None
    num_nodes_t = layout["num_nodes"]
    if num_nodes_t is None or num_nodes_t.numel() == 0 or num_tokens <= 1:
        return None
    n_req = int(num_nodes_t.shape[0])
    qsl = [int(x) for x in layout["qsl"][: n_req + 1].detach().to("cpu").tolist()]
    prefixes_req = [int(x) for x in layout["start"][:n_req].detach().to("cpu").tolist()]
    if len(qsl) < 2 or qsl[0] != 0:
        return None
    if qsl[-1] < num_tokens:
        qsl[-1] = num_tokens

    rows: list[int] = []
    lens: list[int] = []
    cmp_rows: list[int] = []
    cmp_lens: list[int] = []
    seqused: list[int] = []
    chains: list[list[int]] = []
    req_index: list[int] = []
    prefix_of: list[int] = []
    token_index: list[int] = []
    req_counts = [0] * n_req
    mismatch = 0
    for req in range(n_req):
        tok0 = qsl[req]
        tok1 = min(qsl[req + 1], num_tokens)
        n_nodes = int(num_nodes_t[req].item())
        parents = layout["parents"][req, :n_nodes].tolist() if n_nodes > 0 else []
        depths = layout["depths"][req, :n_nodes].tolist() if n_nodes > 0 else []
        prefix = prefixes_req[req]
        for token in range(tok0, tok1):
            local = token - tok0
            req_index.append(req)
            prefix_of.append(prefix)
            token_index.append(token)
            req_counts[req] += 1
            depth = int(depths[local - 1]) if 0 < local <= n_nodes else 0
            if local == 0 or local > n_nodes or depth <= 0:
                rows.append(tok0)
                lens.append(1)
                # Root chain has no draft. Keep one token so the segment is non-empty.
                cmp_rows.append(tok0)
                cmp_lens.append(1)
                seqused.append(prefix + 1)
                chains.append([])
                continue
            chain: list[int] = []
            cur = local
            seen: set[int] = set()
            while cur > 0 and cur not in seen and len(chain) <= n_nodes:
                seen.add(cur)
                chain.append(cur)
                parent = int(parents[cur - 1])
                cur = 0 if parent == cur else parent
            chain.reverse()
            if len(chain) != depth:
                mismatch += 1
            seq = [tok0] + [tok0 + n if tok0 + n < num_tokens else tok0 for n in chain]
            rows.extend(seq)
            lens.append(len(seq))
            # Drafts land at prefix+0 onward after dst_from_zero. Drop the root.
            cmp_rows.extend(seq[1:])
            cmp_lens.append(len(seq) - 1)
            chains.append(chain)
            seqused.append(prefix + depth + 1)
    if not lens:
        return None
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)
    cmp_cu = [0]
    for length in cmp_lens:
        cmp_cu.append(cmp_cu[-1] + length)
    req_cu = [0]
    for count in req_counts:
        req_cu.append(req_cu[-1] + count)
    n = len(lens)
    _PLAN = {
        "prefix": prefix_of[0],
        "prefixes": prefix_of,
        "n": n,
        "n_req": n_req,
        "req_index": req_index,
        "req_cu": req_cu,
        "rows": torch.tensor(rows, dtype=torch.long, device=device),
        "cu": torch.tensor(cu, dtype=torch.int32, device=device),
        "start": torch.tensor(prefix_of, dtype=torch.int32, device=device),
        "lens": lens,
        "cmp_rows": torch.tensor(cmp_rows, dtype=torch.long, device=device),
        "cmp_cu": torch.tensor(cmp_cu, dtype=torch.int32, device=device),
        "cmp_lens": cmp_lens,
        "seqused": torch.tensor(seqused, dtype=torch.int32, device=device),
        "attn_cu": torch.tensor(cu, dtype=torch.int32, device=device),
        "token_index": torch.tensor(token_index, dtype=torch.long, device=device),
        "max_q": max(lens),
        "chains": chains,
        "kv_tables": {},
        "ori_views": {},
        "scratch_used": {},
        "records": [[] for _ in range(n)],
        "logged": False,
        "chain_mismatch": mismatch,
    }
    return _PLAN


def _used_blocks(block_table: torch.Tensor, rows: list[int] | None = None) -> set[int]:
    if rows is None:
        rows = [0]
    used: set[int] = set()
    width = int(block_table.shape[0])
    for row in rows:
        if row < 0 or row >= width:
            continue
        ids = block_table[row].detach().to("cpu").tolist()
        used.update(int(b) for b in ids if int(b) >= 0)
    return used


def _reserve_scratch(plan, cache: torch.Tensor, need: int, used: set[int]) -> list[int]:
    """Scratch block ids for one cache stay unique for the whole verify step."""
    taken = plan["scratch_used"].setdefault(cache.data_ptr(), set())
    found = _alloc_scratch(int(cache.shape[0]), need, used | taken)
    taken.update(found)
    return found


def _alloc_scratch(nblocks: int, need: int, used: set[int]) -> list[int]:
    found: list[int] = []
    block = nblocks - 1
    while block > 0 and len(found) < need:
        if block not in used:
            found.append(block)
        block -= 1
    if len(found) < need:
        raise RuntimeError(
            f"tree chain scratch short: need {need}, free {len(found)}, blocks {nblocks}"
        )
    found.reverse()
    return found


def _copy_blocks(cache: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
    if src.numel() == 0:
        return
    cache.index_copy_(0, dst, cache.index_select(0, src))


def _emit_count(start: int, length: int, ratio: int) -> int:
    if length <= 0 or start < 0 or ratio <= 1:
        return 0
    return (start + length) // ratio - start // ratio


def chain_compressor_batch(
    hidden: torch.Tensor,
    state_cache: torch.Tensor,
    kv_block_table: torch.Tensor,
    state_block_table: torch.Tensor,
    storage_block_size: int,
    ratio: int,
    req_metadata,
    slot_format: int,
):
    """Pack every chain and point the single compressor launch at scratch pages.

    Returns ``None`` when this forward is not a tree verify.
    """
    if _LAYOUT is None or not _KV_STACK:
        return None
    caches = _KV_STACK[-1]
    if not caches:
        return None
    plan = _build_plan(int(hidden.shape[0]), hidden.device)
    if plan is None:
        return None
    if int(kv_block_table.shape[0]) < 1:
        return None
    n = int(plan["n"])
    prefixes = plan["prefixes"]
    # Compressor sees the drafts that remain after accept, not the placeholder root.
    lens = plan["cmp_lens"]
    emits = [_emit_count(prefixes[i], lens[i], ratio) for i in range(n)]
    n_out = sum(emits)
    req_index = plan["req_index"]
    used = _used_blocks(kv_block_table, req_index)
    req_t = torch.tensor(req_index, dtype=torch.long, device=hidden.device)
    kv_rows = kv_block_table.index_select(0, req_t).to(dtype=torch.int32).clone()
    bs = int(storage_block_size)
    chain_cols: list[list[int]] = []
    for i, n_emit in enumerate(emits):
        if n_emit <= 0:
            chain_cols.append([])
            continue
        base = prefixes[i] // ratio
        chain_cols.append(sorted({(base + step) // bs for step in range(n_emit)}))
    if any(chain_cols):
        if int(caches[0].shape[1]) != bs:
            raise RuntimeError(
                f"chain page {tuple(caches[0].shape)} != block {bs}"
            )
        need = sum(len(cols) for cols in chain_cols)
        scratch = _reserve_scratch(plan, caches[0], need, used)
        src_ids: list[int] = []
        dst_ids: list[int] = []
        real_rows = [
            kv_block_table[req].tolist() if req < kv_block_table.shape[0] else []
            for req in range(int(kv_block_table.shape[0]))
        ]
        slot = 0
        for chain, cols in enumerate(chain_cols):
            real_row = real_rows[req_index[chain]]
            for col in cols:
                if col >= len(real_row):
                    continue
                src = int(real_row[col])
                dst = scratch[slot]
                slot += 1
                if src < 0:
                    continue
                src_ids.append(src)
                dst_ids.append(dst)
                kv_rows[chain, col] = dst
                plan["records"][chain].append((caches[0], src, dst, "kv"))
                for extra in caches[1:]:
                    plan["records"][chain].append((extra, src, dst, "kv"))
        if src_ids:
            src_t = torch.tensor(src_ids, dtype=torch.long, device=hidden.device)
            dst_t = torch.tensor(dst_ids, dtype=torch.long, device=hidden.device)
            for cache in caches:
                _copy_blocks(cache, src_t, dst_t)
        for cache in caches:
            plan["kv_tables"][cache.data_ptr()] = kv_rows
    else:
        for cache in caches:
            plan["kv_tables"][cache.data_ptr()] = kv_rows

    state_rows = _clone_state_rows(
        state_cache, state_block_table, plan, hidden.device, int(ratio), int(plan["prefix"])
    )
    full_cos = req_metadata.full_compress_cos
    full_sin = req_metadata.full_compress_sin
    cos = full_cos.view(full_cos.shape[0], full_cos.shape[-1])
    sin = full_sin.view(full_sin.shape[0], full_sin.shape[-1])
    token_size = sum(lens)
    # Compressor checks rope rows against this, not the emitted window count.
    n_rows = min(token_size, token_size // int(ratio) + n)
    n_rows = max(n_rows, 1)
    compress_cos, compress_sin, slot_mapping = torch.ops._C_ascend.compressor_metadata(
        cos,
        sin,
        plan["cmp_cu"],
        plan["start"],
        kv_rows,
        int(storage_block_size),
        int(slot_format),
        ratio,
        n_rows,
        n,
    )
    if not plan["logged"]:
        plan["logged"] = True
        slot_info = {"ndim": int(slot_mapping.ndim), "rows": int(slot_mapping.shape[0])}
        valid = slot_mapping >= 0
        if slot_mapping.ndim == 2:
            valid = slot_mapping[:, 0] >= 0
            offs = slot_mapping[:, 1]
        else:
            offs = torch.remainder(slot_mapping, int(storage_block_size))
        if int(valid.sum().item()) > 0:
            used_off = offs[valid]
            slot_info["valid"] = int(valid.sum().item())
            slot_info["off_min"] = int(used_off.min().item())
            slot_info["off_max"] = int(used_off.max().item())
        else:
            slot_info["valid"] = 0
        _log(
            {
                "pack": 1,
                "calls": 1,
                "op_cu": "chain",
                "n_req": int(plan["n_req"]),
                "req_cu": plan["req_cu"],
                "chain_cu": [int(x) for x in plan["cmp_cu"][:9].detach().to("cpu").tolist()],
                "q_cu_last": n,
                "prefix": int(plan["prefix"]),
                "n": n,
                "emit": n_out,
                "rope_rows": n_rows,
                "residual": int(plan["prefix"]) % ratio,
                "mismatch": int(plan["chain_mismatch"]),
                "lens": lens[:8],
                "cols": (chain_cols[0][:4] if chain_cols else []),
                "slots": slot_info,
            }
        )
    if int(plan["cmp_rows"].numel()) == 0:
        packed = hidden.new_empty((0, *hidden.shape[1:]))
    else:
        packed = hidden.index_select(0, plan["cmp_rows"])
    return {
        "hidden": packed,
        "cu": plan["cmp_cu"],
        "start": plan["start"],
        "kv_bt": kv_rows,
        "state_bt": state_rows,
        "state": state_cache,
        "cos": compress_cos,
        "sin": compress_sin,
        "slot": slot_mapping,
    }


def _state_block_size(state_cache: torch.Tensor) -> int:
    """Match the compressor kernel: block size is dim 1 after squeezing a unit axis."""
    view = state_cache
    if view.ndim >= 3 and int(view.shape[-2]) == 1:
        view = view.squeeze(-2)
    if view.ndim < 2:
        return 1
    return max(int(view.shape[1]), 1)


def _clone_state_rows(state_cache, state_block_table, plan, device, ratio: int, prefix: int):
    """One private state-page set per chain, taken from that chain's request.

    Chains of different requests do not read each other's state blocks.
    The winner's pages are copied back in full after accept.
    """
    del prefix
    n = int(plan["n"])
    width = int(state_block_table.shape[1])
    rows = torch.zeros((n, width), dtype=torch.int32, device=state_block_table.device)
    block_size = _state_block_size(state_cache)
    window = (2 if ratio == 4 else 1) * ratio
    by_req: dict[int, list[int]] = {}
    for chain, req in enumerate(plan["req_index"]):
        by_req.setdefault(int(req), []).append(chain)
    dbg = plan.setdefault("state_dbg", [])
    src_ids: list[int] = []
    dst_ids: list[int] = []
    for req, chains in by_req.items():
        if req < 0 or req >= int(state_block_table.shape[0]):
            continue
        real = [int(x) for x in state_block_table[req].detach().to("cpu").tolist()]
        for chain in chains:
            for col, block in enumerate(real):
                rows[chain, col] = block
        req_prefix = int(plan["prefixes"][chains[0]])
        max_len = max(int(plan["lens"][c]) for c in chains)
        lo = max(0, req_prefix - window)
        hi = req_prefix + max_len
        write_col = req_prefix // block_size
        col_blocks: list[tuple[int, int]] = []
        seen_cols: set[int] = set()
        for pos in range(lo, hi + 1):
            col = pos // block_size
            if col in seen_cols or col >= len(real):
                continue
            seen_cols.add(col)
            block = int(real[col])
            if block > 0:
                col_blocks.append((col, block))
        write_ids = {block for col, block in col_blocks if col >= write_col}
        write_blocks = [b for b in dict.fromkeys(b for _, b in col_blocks if b in write_ids)]
        read_blocks = [
            b
            for b in dict.fromkeys(b for col, b in col_blocks if col < write_col)
            if b not in write_ids
        ]
        live = read_blocks + write_blocks
        cursor = write_col
        cursor_id = int(real[cursor]) if cursor < len(real) else -1
        if len(dbg) < 6:
            dbg.append(
                {
                    "req": req,
                    "ratio": ratio,
                    "bs": block_size,
                    "shape": list(state_cache.shape),
                    "width": len(real),
                    "write_col": write_col,
                    "cols": [c for c, _ in col_blocks][:8],
                    "n_ro": len(read_blocks),
                    "n_write": len(write_blocks),
                    "live": live[:8],
                    "cursor": cursor,
                    "cursor_id": cursor_id,
                    "cursor_in_live": cursor_id in write_ids,
                }
            )
        if not live:
            continue
        scratch = _reserve_scratch(plan, state_cache, len(chains) * len(live), set(live))
        index_of = {block: i for i, block in enumerate(live)}
        for ci, chain in enumerate(chains):
            base = ci * len(live)
            for src in live:
                dst = scratch[base + index_of[src]]
                src_ids.append(src)
                dst_ids.append(dst)
                plan["records"][chain].append((state_cache, src, dst, "state"))
            for col, block in enumerate(real):
                if block in index_of:
                    rows[chain, col] = scratch[base + index_of[block]]
    if src_ids:
        _copy_blocks(
            state_cache,
            torch.tensor(src_ids, dtype=torch.long, device=device),
            torch.tensor(dst_ids, dtype=torch.long, device=device),
        )
    return rows


def chain_length_cus(seqused: torch.Tensor, cmp_ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner cu_seqlens for one operator launch.

    Segment ``k`` is tree chain ``k``. Ori length is ``seqused[k]``.
    Compressed length is ``seqused[k] // cmp_ratio``. The outer request
    boundary is not included.
    """
    n = int(seqused.shape[0])
    ori = torch.empty(n + 1, dtype=torch.int32, device=seqused.device)
    cmp = torch.empty(n + 1, dtype=torch.int32, device=seqused.device)
    ori[0] = 0
    cmp[0] = 0
    ori[1:] = torch.cumsum(seqused.to(torch.int64), dim=0).to(torch.int32)
    cmp_len = torch.div(seqused, int(cmp_ratio), rounding_mode="floor")
    cmp[1:] = torch.cumsum(cmp_len.to(torch.int64), dim=0).to(torch.int32)
    return ori, cmp


def pack_chain_tokens(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Repeat chain tokens into the one packed batch. Lengths may differ."""
    plan = _PLAN
    if tensor is None or plan is None or tensor.ndim == 0:
        return tensor
    if int(tensor.shape[0]) != int(plan["n"]):
        return tensor
    return tensor.index_select(0, plan["rows"])


def unpack_chain_tokens(packed: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Keep each chain's last token, the node that chain exists to score."""
    plan = _PLAN
    if plan is None:
        return packed
    last = (plan["cu"][1:] - 1).to(dtype=torch.long)
    ends = packed.index_select(0, last)
    out = packed.new_zeros((n_tokens, *ends.shape[1:]))
    out.index_copy_(0, plan["token_index"], ends)
    return out


def lookup_kv_table(cache: torch.Tensor):
    if _PLAN is None or cache is None:
        return None
    return _PLAN["kv_tables"].get(cache.data_ptr())


def chain_attention_view(ori_cache: torch.Tensor, ori_block_table: torch.Tensor, block_size: int):
    """Per-token sequences: one query each, ori tail patched to that chain."""
    if _PLAN is None:
        return None
    plan = _PLAN
    cached = plan["ori_views"].get(ori_cache.data_ptr())
    if cached is not None:
        return cached
    n = int(plan["n"])
    prefixes = plan["prefixes"]
    req_index = plan["req_index"]
    if ori_cache.ndim < 2 or int(ori_cache.shape[1]) != int(block_size):
        raise RuntimeError(
            f"ori page {tuple(ori_cache.shape)} != block {block_size}"
        )
    bs = int(block_size)
    cols: set[int] = set()
    for chain, nodes in enumerate(plan["chains"]):
        prefix = int(prefixes[chain])
        for depth in range(1, len(nodes) + 1):
            cols.add((prefix + depth) // bs)
    cols_l = sorted(cols)
    req_t = torch.tensor(req_index, dtype=torch.long, device=ori_block_table.device)
    rows = ori_block_table.index_select(0, req_t).to(dtype=torch.int32).clone()
    if not cols_l:
        view = {
            "cu": plan["attn_cu"],
            "seqused": plan["seqused"],
            "ori_bt": rows,
        }
        plan["ori_views"][ori_cache.data_ptr()] = view
        return view
    used = _used_blocks(ori_block_table, req_index)
    need_chains = [i for i, chain in enumerate(plan["chains"]) if chain]
    scratch = _reserve_scratch(plan, ori_cache, len(need_chains) * len(cols_l), used)
    real_rows = [ori_block_table[req].tolist() for req in range(int(ori_block_table.shape[0]))]
    src_ids: list[int] = []
    dst_ids: list[int] = []
    slot = 0
    chain_page: dict[int, dict[int, int]] = {}
    for chain in need_chains:
        chain_page[chain] = {}
        real_row = real_rows[req_index[chain]]
        for col in cols_l:
            if col >= len(real_row):
                continue
            src = int(real_row[col])
            dst = scratch[slot]
            slot += 1
            chain_page[chain][col] = dst
            rows[chain, col] = dst
            if src >= 0:
                src_ids.append(src)
                dst_ids.append(dst)
    if src_ids:
        _copy_blocks(
            ori_cache,
            torch.tensor(src_ids, dtype=torch.long, device=ori_cache.device),
            torch.tensor(dst_ids, dtype=torch.long, device=ori_cache.device),
        )
    src_b: list[int] = []
    src_o: list[int] = []
    dst_b: list[int] = []
    dst_o: list[int] = []
    bs = int(block_size)
    for chain in need_chains:
        prefix = int(prefixes[chain])
        real_row = real_rows[req_index[chain]]
        for depth, node in enumerate(plan["chains"][chain], start=1):
            src_pos = prefix + int(node)
            dst_pos = prefix + depth
            if src_pos == dst_pos:
                continue
            src_col = src_pos // bs
            dst_col = dst_pos // bs
            if src_col >= len(real_row) or dst_col not in chain_page[chain]:
                continue
            src_b.append(int(real_row[src_col]))
            src_o.append(src_pos % bs)
            dst_b.append(chain_page[chain][dst_col])
            dst_o.append(dst_pos % bs)
    if src_b:
        sb = torch.tensor(src_b, dtype=torch.long, device=ori_cache.device)
        so = torch.tensor(src_o, dtype=torch.long, device=ori_cache.device)
        db = torch.tensor(dst_b, dtype=torch.long, device=ori_cache.device)
        do = torch.tensor(dst_o, dtype=torch.long, device=ori_cache.device)
        ori_cache[db, do] = ori_cache[sb, so]
    view = {
        "cu": plan["attn_cu"],
        "seqused": plan["seqused"],
        "ori_bt": rows,
    }
    plan["ori_views"][ori_cache.data_ptr()] = view
    return view


def note_read(kind: str, **fields) -> None:
    """One record per kind per verify step."""
    plan = _PLAN
    if plan is None:
        return
    done = plan.setdefault("read_kinds", set())
    if kind in done:
        return
    done.add(kind)
    fields["read"] = kind
    _log(fields)


def indexer_batch(key_cache: torch.Tensor):
    if _PLAN is None:
        return None
    table = _PLAN["kv_tables"].get(key_cache.data_ptr())
    if table is None:
        note_read("indexer_miss", ptr=int(key_cache.data_ptr()))
        return None
    seqused = _PLAN["seqused"]
    return {
        "cu": _PLAN["attn_cu"],
        "seqused_k": torch.div(seqused, 4, rounding_mode="floor"),
        "residual": torch.remainder(seqused, 4),
        "block_table": table,
        "max_q": int(_PLAN["max_q"]),
        "max_k": int(seqused.max().item()) // 4,
        "batch": int(_PLAN["n"]),
    }


def _window_cases(plan, chain: list[int]) -> dict:
    """Residual, just-closed, and close-at-branch for the accepted chain."""
    prefix = int(plan["prefix"])
    depth_nodes: dict[int, list[int]] = {}
    for node, nodes in enumerate(plan["chains"]):
        if nodes:
            depth_nodes.setdefault(len(nodes), []).append(node)
    cases = {}
    for ratio in (4, 128):
        closed = []
        branch = []
        for depth, node in enumerate(chain, start=1):
            pos = prefix + depth
            if (pos + 1) % ratio != 0:
                continue
            siblings = [n for n in depth_nodes.get(depth, []) if n != node]
            closed.append(int(node))
            if siblings:
                branch.append({"node": int(node), "sib": len(siblings)})
        cases["r" + str(ratio)] = {
            "residual": prefix % ratio,
            "partial": prefix % ratio != 0,
            "closed": closed[:8],
            "at_branch": branch[:4],
        }
    return cases


def commit_winner(path_node_ids) -> dict | None:
    """Copy the accepted chain's scratch pages back onto the linear cache."""
    if _PLAN is None or path_node_ids is None:
        return None
    plan = _PLAN
    if path_node_ids.numel() == 0:
        return None
    row = [int(n) for n in path_node_ids[0].tolist()]
    winner = 0
    accepted = [n for n in row if n > 0]
    if accepted:
        winner = accepted[-1]
    if winner >= len(plan["records"]):
        return None
    chain = list(plan["chains"][winner]) if winner < len(plan["chains"]) else []
    copied = 0
    n_state = 0
    n_kv = 0
    seen: set[tuple[int, int, int]] = set()
    by_src: dict[tuple[int, int], list[int]] = {}
    by_dst: dict[tuple[int, int], list[int]] = {}
    src_sample: list[int] = []
    for item in plan["records"][winner]:
        cache, src, dst, kind = item
        src_i, dst_i = int(src), int(dst)
        key = (cache.data_ptr(), src_i, dst_i)
        by_src.setdefault((cache.data_ptr(), src_i), []).append(dst_i)
        by_dst.setdefault((cache.data_ptr(), dst_i), []).append(src_i)
        if len(src_sample) < 8:
            src_sample.append(src_i)
        if key in seen:
            continue
        seen.add(key)
        cache[src_i].copy_(cache[dst_i])
        copied += 1
        if kind == "state":
            n_state += 1
        else:
            n_kv += 1
    src_stomp = sum(1 for dsts in by_src.values() if len(set(dsts)) > 1)
    dst_stomp = sum(1 for srcs in by_dst.values() if len(set(srcs)) > 1)
    info = {
        "commit": 1,
        "prefix": plan["prefix"],
        "winner": winner,
        "path": accepted[:8],
        "chain": chain[:8],
        "path_eq_chain": accepted == chain,
        "pages": copied,
        "state": n_state,
        "kv": n_kv,
        "src_stomp": src_stomp,
        "dst_stomp": dst_stomp,
        "src": src_sample,
        "mismatch": int(plan.get("chain_mismatch", -1)),
        "win": _window_cases(plan, chain),
        "state_dbg": plan.get("state_dbg", []),
    }
    _log(info)
    return info
