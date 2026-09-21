# NPU checks for tree-chain compressor, attention and indexer state.
# Operators see one cu_seqlens. Segment k is chain k, and those lengths differ.
# Run inside the spec260922 container on a free NPU.

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp  # noqa: E402

KV_BLOCK = 32
ROPE_DIM = 64
ROPE_ROWS = 2048
SLOT_BLOCK_OFFSET = 2
HEADS = 4
HEAD_DIM = 512


def _segments(cu):
    vals = [int(x) for x in cu.detach().to("cpu").tolist()]
    return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]


def _emit(start, length, ratio):
    if length <= 0 or start < 0 or ratio <= 1:
        return 0
    return (start + length) // ratio - start // ratio


def _install(parents, depths, num_nodes, qsl, start):
    cp.clear_tree_chain_layout()
    cp.set_tree_chain_layout(parents, num_nodes, depths, qsl, start)


def _branch(prefix):
    # root, two depth-1 siblings, one of them continues one more step
    return (
        torch.tensor([[0, 0, 1]], dtype=torch.int32),
        torch.tensor([[1, 1, 2]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32),
        torch.tensor([prefix], dtype=torch.int32),
        [1, 2, 2, 3],
    )


def _short_siblings(prefix):
    return (
        torch.tensor([[0, 0]], dtype=torch.int32),
        torch.tensor([[1, 1]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([prefix], dtype=torch.int32),
        [1, 2, 2],
    )


def _closed_chain(prefix):
    # lengths 1,2,3,4 and a depth-1 sibling. The length-4 chain can close a window.
    return (
        torch.tensor([[0, 1, 2, 0]], dtype=torch.int32),
        torch.tensor([[1, 2, 3, 1]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([0, 5], dtype=torch.int32),
        torch.tensor([prefix], dtype=torch.int32),
        [1, 2, 3, 4, 2],
    )


def _straight(depth, prefix):
    parents = [0] + list(range(1, depth))
    depths = list(range(1, depth + 1))
    return (
        torch.tensor([parents], dtype=torch.int32),
        torch.tensor([depths], dtype=torch.int32),
        torch.tensor([depth], dtype=torch.int32),
        torch.tensor([0, depth + 1], dtype=torch.int32),
        torch.tensor([prefix], dtype=torch.int32),
        [1] + [i + 1 for i in range(1, depth + 1)],
    )


def _two_requests():
    # Request 0 is the branch tree at prefix 10.
    # Request 1 is a depth-3 chain at prefix 40, long enough to emit one C4 slot.
    parents = torch.tensor([[0, 0, 1, 0], [0, 1, 2, 0]], dtype=torch.int32)
    depths = torch.tensor([[1, 1, 2, 0], [1, 2, 3, 0]], dtype=torch.int32)
    num_nodes = torch.tensor([3, 3], dtype=torch.int32)
    qsl = torch.tensor([0, 4, 8], dtype=torch.int32)
    start = torch.tensor([10, 40], dtype=torch.int32)
    return parents, depths, num_nodes, qsl, start, [1, 2, 2, 3, 1, 2, 3, 4]


CASES = [
    ("c4_branch_residual_and_close", 4, "branch", 10),
    ("c4_window_not_full", 4, "short", 9),
    ("c4_just_closed", 4, "closed", 12),
    ("c128_window_not_full", 128, "straight20", 100),
    ("c128_just_closed", 128, "straight7", 120),
    ("c128_crosses_window", 128, "straight29", 100),
    ("c4_two_requests", 4, "two", 0),
]


def _tree(kind, prefix):
    if kind == "branch":
        return _branch(prefix)
    if kind == "short":
        return _short_siblings(prefix)
    if kind == "closed":
        return _closed_chain(prefix)
    if kind == "straight20":
        return _straight(19, prefix)
    if kind == "straight7":
        return _straight(7, prefix)
    if kind == "straight29":
        return _straight(29, prefix)
    if kind == "two":
        return _two_requests()
    raise AssertionError(kind)


def _reference_slots(cu, start, block_table, ratio, num_rows):
    start = start.detach().to("cpu")
    block_table = block_table.detach().to("cpu")
    lens = _segments(cu)
    prefix = [0]
    total = 0
    for idx, length in enumerate(lens):
        total += _emit(int(start[idx]), length, ratio)
        prefix.append(total)
    ref_slot = torch.empty((num_rows, 2), dtype=torch.int32)
    req = 0
    for row in range(num_rows):
        valid = row < total
        block_id = -1
        compressed = 0
        if valid:
            while req < len(lens) and prefix[req + 1] <= row:
                req += 1
            compressed = int(start[req]) // ratio + row - prefix[req]
            col = compressed // KV_BLOCK
            rope_pos = compressed * ratio
            if col >= block_table.shape[1] or rope_pos >= ROPE_ROWS:
                valid = False
            else:
                block_id = int(block_table[req, col])
                valid = block_id >= 0
        if valid:
            ref_slot[row, 0] = block_id
            ref_slot[row, 1] = compressed % KV_BLOCK
        else:
            ref_slot[row, 0] = -1
            ref_slot[row, 1] = KV_BLOCK - 1
    return ref_slot, total


def _fill_rope(ref_cos, ref_sin, rope_cos, rope_sin, slot, start, cu, ratio):
    """Rebuild cos/sin from the same compressed positions the slot rows use."""
    start = [int(x) for x in start.detach().to("cpu").tolist()]
    lens = _segments(cu)
    prefix = [0]
    total = 0
    for idx, length in enumerate(lens):
        total += _emit(start[idx], length, ratio)
        prefix.append(total)
    req = 0
    for row in range(ref_cos.shape[0]):
        if int(slot[row, 0]) < 0:
            continue
        while req < len(lens) and prefix[req + 1] <= row:
            req += 1
        compressed = start[req] // ratio + row - prefix[req]
        rope_pos = compressed * ratio
        ref_cos[row, 0, 0].copy_(rope_cos[rope_pos])
        ref_sin[row, 0, 0].copy_(rope_sin[rope_pos])


def _rope():
    values = torch.arange(ROPE_ROWS * ROPE_DIM, dtype=torch.float32).reshape(ROPE_ROWS, ROPE_DIM)
    return values.to(torch.bfloat16), (values * 0.25).to(torch.bfloat16)


def _run_compressor(kind, prefix, ratio, hidden_dim=4, state_dim=4):
    parents, depths, num_nodes, qsl, start, expect_lens = _tree(kind, prefix)
    _install(parents, depths, num_nodes, qsl, start)
    n_tok = int(qsl[-1])
    n_req = int(num_nodes.shape[0])
    device = torch.device("npu:0")
    hidden_dtype = torch.bfloat16 if hidden_dim > 4 else torch.float32
    hidden = torch.randn(n_tok, hidden_dim, dtype=hidden_dtype, device=device)
    kv_cache = torch.zeros(96, KV_BLOCK, 1, 8, device=device)
    state = torch.zeros(4096, 2, 1, state_dim, device=device)
    kv_table = torch.arange(1, 1 + n_req * 8, dtype=torch.int32, device=device).view(n_req, 8)
    state_table = torch.arange(1, 1 + n_req * 32, dtype=torch.int32, device=device).view(n_req, 32)
    if n_req > 1:
        state[int(state_table[0, 5])].fill_(1)
        state[int(state_table[1, 20])].fill_(2)
    rope_cos, rope_sin = _rope()
    meta = type("M", (), {})()
    meta.full_compress_cos = rope_cos.npu()
    meta.full_compress_sin = rope_sin.npu()
    with cp.chain_kv_scope([kv_cache]):
        packed = cp.chain_compressor_batch(
            hidden,
            state,
            kv_table,
            state_table,
            KV_BLOCK,
            ratio,
            meta,
            SLOT_BLOCK_OFFSET,
        )
    assert packed is not None
    plan = cp._PLAN
    assert _segments(plan["cu"]) == expect_lens
    assert len(set(expect_lens)) > 1 or kind.startswith("straight")
    assert max(expect_lens) == plan["max_q"]
    assert _segments(plan["attn_cu"]) == expect_lens
    assert torch.equal(plan["cu"], plan["attn_cu"])
    # Compressor cu drops the placeholder root. Attention cu still has it.
    assert plan["cmp_lens"] == [1 if x == 1 else x - 1 for x in expect_lens]
    slot = packed["slot"].detach().to("cpu")
    token_size = sum(plan["cmp_lens"])
    n_chain = len(plan["cmp_lens"])
    n_rows = max(min(token_size, token_size // ratio + n_chain), 1)
    assert slot.shape[0] == n_rows
    ref_slot, n_valid = _reference_slots(packed["cu"], plan["start"], packed["kv_bt"], ratio, n_rows)
    ref_cos = torch.ones((n_rows, 1, 1, ROPE_DIM), dtype=torch.bfloat16)
    ref_sin = torch.zeros_like(ref_cos)
    _fill_rope(ref_cos, ref_sin, rope_cos, rope_sin, ref_slot, plan["start"], packed["cu"], ratio)
    assert torch.equal(slot, ref_slot)
    assert torch.equal(packed["cos"].detach().to("cpu"), ref_cos)
    assert torch.equal(packed["sin"].detach().to("cpu"), ref_sin)
    valid = slot[:, 0] >= 0
    assert int(valid.sum()) == n_valid
    if n_valid:
        blocks = slot[valid, 0].tolist()
        assert len(blocks) == len(set(blocks))
        real_ids = set(int(x) for x in kv_table.detach().to("cpu").reshape(-1).tolist())
        assert set(blocks).isdisjoint(real_ids)
    return plan, packed, state, state_table, kv_table, kv_cache


def _attn_metadata(plan, ratio):
    seqused = plan["seqused"]
    ori_cu, cmp_cu = cp.chain_length_cus(seqused, ratio)
    segs = plan["attn_cu"][1:] - plan["attn_cu"][:-1]
    assert _segments(ori_cu) == [int(x) for x in seqused.tolist()]
    assert _segments(cmp_cu) == [int(x) // ratio for x in seqused.tolist()]
    assert int(segs.max()) == int(plan["max_q"])
    assert int(plan["max_q"]) == max(plan["lens"])
    assert plan["lens"] != [1] * len(plan["lens"])
    meta = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=HEADS,
        num_heads_kv=1,
        head_dim=HEAD_DIM,
        cu_seqlens_q=plan["attn_cu"],
        cu_seqlens_ori_kv=ori_cu,
        cu_seqlens_cmp_kv=cmp_cu,
        seqused_q=plan["attn_cu"].new_empty(0),
        seqused_kv=seqused,
        batch_size=int(seqused.shape[0]),
        max_seqlen_q=int(plan["max_q"]),
        max_seqlen_kv=int(seqused.max().item()),
        ori_topk=0,
        cmp_topk=512 if ratio == 4 else 0,
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=183,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=True,
        device="npu",
    )
    torch.npu.synchronize()
    assert tuple(meta.shape) == (1024,)
    assert meta.dtype == torch.int32
    assert meta.device.type == "npu"


def _indexer_metadata(cache, ratio):
    batch = cp.indexer_batch(cache)
    assert batch is not None
    assert _segments(batch["cu"]) == cp._PLAN["lens"]
    assert batch["max_q"] == cp._PLAN["max_q"]
    assert batch["batch"] == len(cp._PLAN["lens"])
    assert batch["seqused_k"].tolist() == [x // 4 for x in cp._PLAN["seqused"].tolist()]
    meta = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=64,
        num_heads_k=1,
        head_dim=128,
        topk=512,
        quant_mode=2,
        cu_seqlens_q=batch["cu"],
        seqused_k=batch["seqused_k"],
        cmp_residual_k=batch["residual"],
        batch_size=batch["batch"],
        max_seqlen_q=batch["max_q"],
        max_seqlen_k=max(batch["max_k"], 1),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        device="npu",
    )
    torch.npu.synchronize()
    assert tuple(meta.shape) == (1024,)
    assert meta.device.type == "npu"
    del ratio


@pytest.fixture(autouse=True)
def _device():
    torch.npu.set_device(0)
    cp.clear_tree_chain_layout()
    yield
    cp.clear_tree_chain_layout()


@pytest.mark.parametrize("name,ratio,kind,prefix", CASES, ids=[c[0] for c in CASES])
def test_compressor_metadata_matches_each_chain(name, ratio, kind, prefix):
    plan, packed, state, state_table, kv_table, kv_cache = _run_compressor(kind, prefix, ratio)
    lens = plan["cmp_lens"]
    starts = [int(x) for x in plan["start"].tolist()]
    emits = [_emit(starts[i], lens[i], ratio) for i in range(len(lens))]
    # Window facts the slots must follow. Lengths are the drafts that remain.
    if name == "c4_branch_residual_and_close":
        assert starts[0] % 4 == 2
        assert emits == [0, 0, 0, 1]
        valid = packed["slot"][:, 0] >= 0
        assert int(valid.sum()) == 1
        assert int(packed["slot"][valid][0, 1]) == 2
    if name == "c4_window_not_full":
        assert all(e == 0 for e in emits)
        assert int((packed["slot"][:, 0] >= 0).sum()) == 0
    if name == "c4_just_closed":
        assert emits == [0, 0, 0, 0, 0]
        assert int((packed["slot"][:, 0] >= 0).sum()) == 0
    if name == "c128_window_not_full":
        assert all(e == 0 for e in emits)
    if name == "c128_just_closed":
        # Seven drafts at prefix 120 end at 126, so the ratio-128 window stays open.
        assert sum(emits) == 0
    if name == "c128_crosses_window":
        assert sum(emits) >= 1
        assert len(set(packed["slot"][: sum(emits), 0].tolist())) == sum(emits)
    if name == "c4_two_requests":
        assert plan["req_cu"] == [0, 4, 8]
        assert plan["prefixes"][:4] == [10, 10, 10, 10]
        assert plan["prefixes"][4:] == [40, 40, 40, 40]
        # Column 1 is outside the emitted compressed page, so it stays on the request row.
        bt = packed["kv_bt"].detach().to("cpu")
        real = kv_table.detach().to("cpu")
        assert int(bt[0, 1]) == int(real[0, 1])
        assert int(bt[7, 1]) == int(real[1, 1])
        assert int(bt[0, 1]) != int(bt[7, 1])
        # State pages come from the request that owns the chain.
        sbt = packed["state_bt"].detach().to("cpu")
        for chain in range(4):
            page = int(sbt[chain, 5])
            assert float(state[page].reshape(-1)[0].item()) == 1.0
        for chain in range(4, 8):
            page = int(sbt[chain, 20])
            assert float(state[page].reshape(-1)[0].item()) == 2.0
        assert len({int(sbt[c, 5]) for c in range(4)}) == 4
    _attn_metadata(plan, ratio)
    if ratio == 4:
        _indexer_metadata(kv_cache, ratio)


def test_attention_reads_the_chain_slot_and_not_its_sibling():
    _install(*_branch(10)[:5])
    plan = cp._build_plan(4, torch.device("npu:0"))
    cache = torch.zeros(64, KV_BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    table = torch.arange(1, 9, dtype=torch.int32, device="npu").view(1, 8)
    # Node 3 lives at prefix+3. The length-3 chain must read it at prefix+depth.
    marker = torch.arange(HEAD_DIM, dtype=torch.float32).to(torch.bfloat16)
    cache[1, 13, 0].copy_(marker.npu())
    view = cp.chain_attention_view(cache, table, KV_BLOCK)
    assert _segments(view["cu"]) == [1, 2, 2, 3]
    assert view["seqused"].tolist() == [11, 12, 12, 13]
    col = 0
    long_page = int(view["ori_bt"][3, col])
    sib_page = int(view["ori_bt"][2, col])
    assert long_page != sib_page
    assert long_page != 1 and sib_page != 1
    got = cache[long_page, 12, 0].detach().to("cpu")
    sib = cache[sib_page, 12, 0].detach().to("cpu")
    assert torch.equal(got, marker)
    assert torch.equal(sib, torch.zeros_like(marker))

    q = torch.randn(int(view["cu"][-1]), HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    seqused = view["seqused"]
    ori_cu, cmp_cu = cp.chain_length_cus(seqused, 4)
    meta = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=HEADS,
        num_heads_kv=1,
        head_dim=HEAD_DIM,
        cu_seqlens_q=view["cu"],
        cu_seqlens_ori_kv=ori_cu,
        cu_seqlens_cmp_kv=cmp_cu,
        seqused_q=view["cu"].new_empty(0),
        seqused_kv=seqused,
        batch_size=int(seqused.shape[0]),
        max_seqlen_q=3,
        max_seqlen_kv=int(seqused.max().item()),
        ori_topk=0,
        cmp_topk=0,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=183,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device="npu",
    )
    sinks = torch.zeros(HEADS, dtype=torch.float32, device="npu")

    def _run():
        out, _lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q,
            ori_kv=cache,
            ori_block_table=view["ori_bt"],
            cu_seqlens_q=view["cu"],
            cu_seqlens_ori_kv=ori_cu,
            seqused_kv=seqused,
            sinks=sinks,
            metadata=meta,
            softmax_scale=HEAD_DIM**-0.5,
            cmp_ratio=4,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=183,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )
        torch.npu.synchronize()
        return out

    first = _run()
    assert tuple(first.shape) == tuple(q.shape)
    assert torch.isfinite(first.float()).all()
    cache[long_page, 12, 0].fill_(4)
    second = _run()
    # The last query of the long chain is the token that must see this slot.
    assert not torch.equal(first[-1].detach().to("cpu"), second[-1].detach().to("cpu"))
    restored = cp.unpack_chain_tokens(first, 4)
    assert restored.shape[0] == 4


def test_commit_moves_only_the_winner():
    plan, packed, state, state_table, kv_table, _kv_cache = _run_compressor("branch", 10, 4)
    del packed, state_table, kv_table
    records = plan["records"][3]
    kv_items = [item for item in records if item[3] == "kv"]
    state_items = [item for item in records if item[3] == "state"]
    assert kv_items and state_items
    cache, src, dst, _kind = kv_items[0]
    cache[dst].fill_(5)
    cache[src].fill_(0)
    _sib_cache, _sib_src, sib_dst, _sib_kind = [item for item in plan["records"][2] if item[3] == "kv"][0]
    assert int(sib_dst) != int(dst)
    cache[sib_dst].fill_(3)
    path = torch.tensor([[1, 3]], dtype=torch.int32)
    info = cp.commit_winner(path)
    assert info is not None
    assert info["path_eq_chain"] is True
    assert info["src_stomp"] == 0
    assert info["dst_stomp"] == 0
    assert float(cache[src].reshape(-1)[0].item()) == 5.0
    # Sibling scratch stays put. Both chains read the same linear source page.
    assert float(cache[sib_dst].reshape(-1)[0].item()) == 3.0
    assert info["win"]["r4"]["partial"] is True
    assert info["win"]["r4"]["at_branch"]


CMP_HEAD = 128
CMP_HIDDEN = 1024
INDEX_HEADS = 64
INDEX_DIM = 128


def _compressor_weights(ratio):
    coff = 2 if ratio == 4 else 1
    return {
        "coff": coff,
        "wkv": torch.randn(coff * CMP_HEAD, CMP_HIDDEN, dtype=torch.bfloat16, device="npu") * 0.02,
        "wgate": torch.randn(coff * CMP_HEAD, CMP_HIDDEN, dtype=torch.bfloat16, device="npu") * 0.02,
        "ape": torch.randn(ratio, coff * CMP_HEAD, dtype=torch.float32, device="npu"),
        "norm": torch.ones(CMP_HEAD, dtype=torch.float32, device="npu"),
    }


def _compressor_op(packed, hidden, state, ratio, weights):
    n_rows = int(packed["cos"].shape[0])
    rope = torch.ones(n_rows, ROPE_DIM, dtype=torch.float32, device="npu")
    out = torch.ops._C_ascend.compressor(
        hidden,
        weights["wkv"],
        weights["wgate"],
        state.squeeze(-2),
        weights["ape"],
        weights["norm"],
        rope,
        rope,
        state_block_table=packed["state_bt"],
        cu_seqlens=packed["cu"],
        seqused=None,
        start_pos=packed["start"],
        rope_head_dim=ROPE_DIM,
        cmp_ratio=ratio,
        coff=weights["coff"],
        norm_eps=1e-6,
        rotary_mode=2,
        cache_mode=1,
    )
    torch.npu.synchronize()
    return out


def _valid_rows(plan, ratio):
    starts = [int(x) for x in plan["start"].tolist()]
    return [_emit(starts[i], plan["lens"][i], ratio) for i in range(len(plan["lens"]))]


def test_compressor_op_keeps_chain_outputs_apart():
    ratio = 4
    state_dim = 2 * 2 * CMP_HEAD
    plan, packed, state, _state_table, _kv_table, _kv_cache = _run_compressor(
        "branch", 10, ratio, hidden_dim=CMP_HIDDEN, state_dim=state_dim
    )
    emits = _valid_rows(plan, ratio)
    assert emits == [0, 1, 1, 1]
    hidden = torch.randn(int(plan["cu"][-1]), CMP_HIDDEN, dtype=torch.bfloat16, device="npu") * 0.1
    weights = _compressor_weights(ratio)
    base = state.detach().clone()
    first = _compressor_op(packed, hidden, state, ratio, weights)
    assert tuple(first.shape) == (int(packed["cos"].shape[0]), CMP_HEAD)
    assert torch.isfinite(first.float()).all()
    state.copy_(base)
    changed = hidden.clone()
    # Chain 3 is packed at [5, 8). Raw positions 10 and 11 close the ratio-4
    # window; position 12 is only the residual of the next window.
    changed[6].fill_(4)
    second = _compressor_op(packed, changed, state, ratio, weights)
    # Rows 0 and 1 are the two depth-1 siblings. Row 2 is the longer chain.
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[2], second[2])
    bt = packed["state_bt"]
    wrote = []
    diverged = []
    for col in range(int(bt.shape[1])):
        page_a = int(bt[1, col])
        page_b = int(bt[2, col])
        if page_a == page_b:
            continue
        if not torch.equal(state[page_a], base[page_a]) or not torch.equal(state[page_b], base[page_b]):
            wrote.append(col)
        if not torch.equal(state[page_a], state[page_b]):
            diverged.append(col)
    assert wrote and diverged, f"wrote={wrote} diverged={diverged}"


def test_compressor_op_keeps_requests_apart():
    ratio = 4
    state_dim = 2 * 2 * CMP_HEAD
    plan, packed, state, _state_table, _kv_table, _kv_cache = _run_compressor(
        "two", 0, ratio, hidden_dim=CMP_HIDDEN, state_dim=state_dim
    )
    emits = _valid_rows(plan, ratio)
    assert sum(emits[:4]) == 3
    assert sum(emits[4:]) == 1
    hidden = torch.randn(int(plan["cu"][-1]), CMP_HIDDEN, dtype=torch.bfloat16, device="npu") * 0.1
    weights = _compressor_weights(ratio)
    base = state.detach().clone()
    first = _compressor_op(packed, hidden, state, ratio, weights)
    state.copy_(base)
    changed = hidden.clone()
    req1 = int(plan["cu"][4])
    changed[req1:].fill_(4)
    second = _compressor_op(packed, changed, state, ratio, weights)
    assert torch.equal(first[:3], second[:3])
    assert not torch.equal(first[3], second[3])


def test_compressor_op_ratio_128_writes_the_closed_chain():
    ratio = 128
    state_dim = 2 * CMP_HEAD
    plan, packed, state, _state_table, _kv_table, _kv_cache = _run_compressor(
        "straight7", 120, ratio, hidden_dim=CMP_HIDDEN, state_dim=state_dim
    )
    emits = _valid_rows(plan, ratio)
    assert sum(emits) == 1
    assert emits[-1] == 1
    hidden = torch.randn(int(plan["cu"][-1]), CMP_HIDDEN, dtype=torch.bfloat16, device="npu") * 0.1
    weights = _compressor_weights(ratio)
    base = state.detach().clone()
    first = _compressor_op(packed, hidden, state, ratio, weights)
    assert torch.isfinite(first.float()).all()
    state.copy_(base)
    changed = hidden.clone()
    changed[int(plan["cu"][-2]) :].fill_(4)
    second = _compressor_op(packed, changed, state, ratio, weights)
    assert not torch.equal(first[0], second[0])


def test_indexer_op_reads_each_chain_page():
    plan, packed, _state, _state_table, _kv_table, kv_cache = _run_compressor("branch", 10, 4)
    batch = cp.indexer_batch(kv_cache)
    assert _segments(batch["cu"]) == [1, 2, 2, 3]
    n_tok = int(batch["cu"][-1])
    key = torch.zeros(96, KV_BLOCK, 1, INDEX_DIM, dtype=torch.int8, device="npu")
    key_scale = torch.ones(96, KV_BLOCK, 1, dtype=torch.float16, device="npu")
    query = torch.zeros(n_tok, INDEX_HEADS, INDEX_DIM, dtype=torch.int8, device="npu")
    query_scale = torch.ones(n_tok, INDEX_HEADS, dtype=torch.float16, device="npu")
    weights = torch.ones(n_tok, INDEX_HEADS, dtype=torch.float16, device="npu")
    pattern = torch.full((INDEX_DIM,), 40, dtype=torch.int8, device="npu")
    long_page = int(packed["kv_bt"][3, 0])
    sib_page = int(packed["kv_bt"][2, 0])
    assert long_page != sib_page
    key[long_page, 2, 0] = pattern
    key[sib_page, 0, 0] = pattern
    # Packed positions: chain 2 ends at 4, chain 3 ends at 7.
    query[4] = pattern
    query[7] = pattern
    meta = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=INDEX_HEADS,
        num_heads_k=1,
        head_dim=INDEX_DIM,
        topk=1,
        quant_mode=2,
        cu_seqlens_q=batch["cu"],
        seqused_k=batch["seqused_k"],
        cmp_residual_k=batch["residual"],
        batch_size=batch["batch"],
        max_seqlen_q=batch["max_q"],
        max_seqlen_k=max(batch["max_k"], 1),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        device="npu",
    )
    indices, _values = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        topk=1,
        quant_mode=2,
        cu_seqlens_q=batch["cu"],
        seqused_k=batch["seqused_k"],
        cmp_residual_k=batch["residual"],
        block_table=batch["block_table"],
        metadata=meta,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        return_value=0,
    )
    torch.npu.synchronize()
    assert indices.shape[0] == n_tok
    assert int(indices[7].reshape(-1)[0]) == 2
    assert int(indices[4].reshape(-1)[0]) == 0


def test_attention_op_reads_each_chain_compressed_slot():
    plan, packed, _state, _state_table, _kv_table, _kv_cache = _run_compressor("branch", 10, 4)
    ori = torch.zeros(64, KV_BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    ori_table = torch.arange(1, 9, dtype=torch.int32, device="npu").view(1, 8)
    view = cp.chain_attention_view(ori, ori_table, KV_BLOCK)
    cmp = torch.zeros(96, KV_BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    long_page = int(packed["kv_bt"][3, 0])
    sib_page = int(packed["kv_bt"][2, 0])
    assert long_page != sib_page
    cmp[long_page, 2].fill_(1)
    n_tok = int(view["cu"][-1])
    q = torch.randn(n_tok, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    seqused = view["seqused"]
    ori_cu, cmp_cu = cp.chain_length_cus(seqused, 4)
    # Ratio-4 attention selects compressed tokens. Point every query at the new slot
    # when that chain's compressed length includes it.
    indices = torch.zeros(n_tok, 1, 512, dtype=torch.int32, device="npu")
    indices[3:] = 2
    meta = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=HEADS,
        num_heads_kv=1,
        head_dim=HEAD_DIM,
        cu_seqlens_q=view["cu"],
        cu_seqlens_ori_kv=ori_cu,
        cu_seqlens_cmp_kv=cmp_cu,
        seqused_q=view["cu"].new_empty(0),
        seqused_kv=seqused,
        batch_size=int(seqused.shape[0]),
        max_seqlen_q=3,
        max_seqlen_kv=int(seqused.max().item()),
        ori_topk=0,
        cmp_topk=512,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=183,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=True,
        device="npu",
    )
    sinks = torch.zeros(HEADS, dtype=torch.float32, device="npu")

    def _run():
        out, _lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q,
            ori_kv=ori,
            cmp_kv=cmp,
            cmp_sparse_indices=indices,
            ori_block_table=view["ori_bt"],
            cmp_block_table=packed["kv_bt"],
            cu_seqlens_q=view["cu"],
            cu_seqlens_ori_kv=ori_cu,
            cu_seqlens_cmp_kv=cmp_cu,
            seqused_kv=seqused,
            sinks=sinks,
            metadata=meta,
            softmax_scale=HEAD_DIM**-0.5,
            cmp_ratio=4,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=183,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )
        torch.npu.synchronize()
        return out

    first = _run()
    assert tuple(first.shape) == tuple(q.shape)
    assert torch.isfinite(first.float()).all()
    cmp[long_page, 2].fill_(4)
    second = _run()
    # Chain 2 occupies packed rows 3:5. Chain 3 occupies 5:8.
    assert torch.equal(first[3:5], second[3:5])
    assert not torch.equal(first[5:8], second[5:8])


def _scatter_block_slots(cache, values, slots):
    """Scatter ``values`` into paged ``cache`` with compressor block/offset slots.

    ``cache`` is ``[blocks, block, *tail]``. Invalid slots (block id < 0) are skipped.
    """
    valid = slots[:, 0] >= 0
    picked = slots[valid].to(torch.int64)
    updates = values[valid].reshape(int(valid.sum()), -1).contiguous()
    flat = (picked[:, 0] * cache.shape[1] + picked[:, 1]).view(-1, 1)
    flat_cache = cache.reshape(cache.shape[0] * cache.shape[1], -1)
    torch.ops._C_ascend.npu_scatter_nd_update_sk(flat_cache, flat, updates)
    torch.npu.synchronize()
    return valid


def test_compressor_output_scatters_onto_each_chain_page():
    ratio = 4
    state_dim = 2 * 2 * CMP_HEAD
    plan, packed, state, _state_table, _kv_table, _kv_cache = _run_compressor(
        "branch", 10, ratio, hidden_dim=CMP_HIDDEN, state_dim=state_dim
    )
    hidden = torch.randn(int(plan["cu"][-1]), CMP_HIDDEN, dtype=torch.bfloat16, device="npu") * 0.1
    weights = _compressor_weights(ratio)
    cmp_kv = _compressor_op(packed, hidden, state, ratio, weights)
    slots = packed["slot"]
    cache = torch.full((96, KV_BLOCK, CMP_HEAD), -7, dtype=torch.bfloat16, device="npu")
    valid = _scatter_block_slots(cache, cmp_kv, slots)
    assert int(valid.sum()) == 3
    slot_cpu = slots.detach().to("cpu")
    for row in range(3):
        block = int(slot_cpu[row, 0])
        offset = int(slot_cpu[row, 1])
        assert torch.equal(cache[block, offset], cmp_kv[row])
    blocks = [int(slot_cpu[row, 0]) for row in range(3)]
    assert len(set(blocks)) == 3
    # A page the slots do not name keeps the sentinel.
    used = set(blocks)
    untouched = next(i for i in range(1, 96) if i not in used)
    assert float(cache[untouched, 0, 0]) == -7.0


def test_pipeline_feeds_indexer_then_attention_without_repacking():
    ratio = 4
    state_dim = 2 * 2 * CMP_HEAD
    plan, packed, state, _state_table, _kv_table, kv_cache = _run_compressor(
        "branch", 10, ratio, hidden_dim=CMP_HIDDEN, state_dim=state_dim
    )
    hidden = torch.randn(int(plan["cu"][-1]), CMP_HIDDEN, dtype=torch.bfloat16, device="npu") * 0.1
    cmp_kv = _compressor_op(packed, hidden, state, ratio, _compressor_weights(ratio))
    slots = packed["slot"]
    attn_cache = torch.zeros(96, KV_BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    # Attention head is 512. Repeat the 128-d compressor row so the same slot is readable.
    wide = cmp_kv.repeat(1, HEAD_DIM // CMP_HEAD).reshape(cmp_kv.shape[0], 1, HEAD_DIM)
    _scatter_block_slots(attn_cache, wide, slots)
    key = torch.zeros(96, KV_BLOCK, 1, INDEX_DIM, dtype=torch.int8, device="npu")
    key_scale = torch.ones(96, KV_BLOCK, 1, dtype=torch.float16, device="npu")
    quant, _scale = torch_npu.npu_dynamic_quant(cmp_kv, dst_type=torch.int8)
    _scatter_block_slots(key, quant.reshape(quant.shape[0], 1, INDEX_DIM), slots)
    long_page = int(packed["kv_bt"][3, 0])
    sib_page = int(packed["kv_bt"][2, 0])
    assert long_page != sib_page
    pattern = torch.full((INDEX_DIM,), 40, dtype=torch.int8, device="npu")
    key[long_page, 2, 0] = pattern
    key[sib_page, 0, 0] = pattern
    n_tok = int(plan["cu"][-1])
    query = torch.zeros(n_tok, INDEX_HEADS, INDEX_DIM, dtype=torch.int8, device="npu")
    query[4] = pattern
    query[7] = pattern
    batch = cp.indexer_batch(kv_cache)
    assert batch["cu"].shape[0] == plan["cu"].shape[0]
    assert torch.equal(batch["cu"], plan["cu"])
    meta = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=INDEX_HEADS,
        num_heads_k=1,
        head_dim=INDEX_DIM,
        topk=1,
        quant_mode=2,
        cu_seqlens_q=batch["cu"],
        seqused_k=batch["seqused_k"],
        cmp_residual_k=batch["residual"],
        batch_size=batch["batch"],
        max_seqlen_q=batch["max_q"],
        max_seqlen_k=max(batch["max_k"], 1),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        device="npu",
    )
    indices, _values = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
        query=query,
        key=key,
        weights=torch.ones(n_tok, INDEX_HEADS, dtype=torch.float16, device="npu"),
        query_dequant_scale=torch.ones(n_tok, INDEX_HEADS, dtype=torch.float16, device="npu"),
        key_dequant_scale=key_scale,
        topk=1,
        quant_mode=2,
        cu_seqlens_q=batch["cu"],
        seqused_k=batch["seqused_k"],
        cmp_residual_k=batch["residual"],
        block_table=batch["block_table"],
        metadata=meta,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        return_value=0,
    )
    torch.npu.synchronize()
    assert indices.shape[0] == n_tok
    assert int(indices[7].reshape(-1)[0]) == 2
    assert int(indices[4].reshape(-1)[0]) == 0
    # Packed chain order already matches attention Q. Do not pack the indices again.
    chosen = indices.reshape(n_tok, -1)[:, :1]
    sparse = chosen.expand(n_tok, 512).contiguous().view(n_tok, 1, 512)
    ori = torch.zeros(64, KV_BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    ori_table = torch.arange(1, 9, dtype=torch.int32, device="npu").view(1, 8)
    view = cp.chain_attention_view(ori, ori_table, KV_BLOCK)
    attn_cache[long_page, 2].fill_(1)
    q = torch.randn(n_tok, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    seqused = view["seqused"]
    ori_cu, cmp_cu = cp.chain_length_cus(seqused, 4)
    sas = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=HEADS,
        num_heads_kv=1,
        head_dim=HEAD_DIM,
        cu_seqlens_q=view["cu"],
        cu_seqlens_ori_kv=ori_cu,
        cu_seqlens_cmp_kv=cmp_cu,
        seqused_q=view["cu"].new_empty(0),
        seqused_kv=seqused,
        batch_size=int(seqused.shape[0]),
        max_seqlen_q=int(plan["max_q"]),
        max_seqlen_kv=int(seqused.max().item()),
        ori_topk=0,
        cmp_topk=512,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=183,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=True,
        device="npu",
    )
    sinks = torch.zeros(HEADS, dtype=torch.float32, device="npu")

    def _attn():
        out, _lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q,
            ori_kv=ori,
            cmp_kv=attn_cache,
            cmp_sparse_indices=sparse,
            ori_block_table=view["ori_bt"],
            cmp_block_table=packed["kv_bt"],
            cu_seqlens_q=view["cu"],
            cu_seqlens_ori_kv=ori_cu,
            cu_seqlens_cmp_kv=cmp_cu,
            seqused_kv=seqused,
            sinks=sinks,
            metadata=sas,
            softmax_scale=HEAD_DIM**-0.5,
            cmp_ratio=4,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=183,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )
        torch.npu.synchronize()
        return out

    first = _attn()
    restored = cp.unpack_chain_tokens(first, 4)
    cu = plan["cu"].tolist()
    tokens = plan["token_index"].tolist()
    for chain in range(4):
        last = cu[chain + 1] - 1
        assert torch.equal(restored[tokens[chain]], first[last])
    attn_cache[long_page, 2].fill_(4)
    second = _attn()
    assert torch.equal(first[4], second[4])
    assert not torch.equal(first[7], second[7])


def test_c4_and_c128_caches_keep_separate_scratch():
    parents, depths, num_nodes, qsl, start, _lens = _straight(7, 120)
    _install(parents, depths, num_nodes, qsl, start)
    c4 = torch.zeros(96, KV_BLOCK, 1, 8, device="npu")
    c128 = torch.zeros(96, KV_BLOCK, 1, 8, device="npu")
    state = torch.zeros(4096, 2, 1, 4, device="npu")
    table = torch.arange(1, 9, dtype=torch.int32, device="npu").view(1, 8)
    state_table = torch.arange(1, 33, dtype=torch.int32, device="npu").view(1, 32)
    rope_cos, rope_sin = _rope()
    meta = type("M", (), {})()
    meta.full_compress_cos = rope_cos.npu()
    meta.full_compress_sin = rope_sin.npu()
    hidden = torch.zeros(int(qsl[-1]), 4, device="npu")
    with cp.chain_kv_scope([c4]):
        cp.chain_compressor_batch(hidden, state, table, state_table, KV_BLOCK, 4, meta, SLOT_BLOCK_OFFSET)
    with cp.chain_kv_scope([c128]):
        cp.chain_compressor_batch(hidden, state, table, state_table, KV_BLOCK, 128, meta, SLOT_BLOCK_OFFSET)

    def _dsts(cache):
        found = []
        for recs in cp._PLAN["records"]:
            for item in recs:
                if item[0] is cache and item[3] == "kv":
                    found.append(int(item[2]))
        return found

    d4 = _dsts(c4)
    d128 = _dsts(c128)
    assert d4 and d128
    assert set(d4).isdisjoint(set(range(1, 9)))
    assert set(d128).isdisjoint(set(range(1, 9)))
    c4[d4[0]].fill_(1)
    assert float(c128[d4[0]].reshape(-1)[0]) == 0.0
    view = cp.chain_attention_view(c4, table, KV_BLOCK)
    attn_ids = {int(x) for x in view["ori_bt"].reshape(-1).tolist()}
    assert set(d4).isdisjoint(attn_ids)


def _move_ori_from_zero(cache, block_ids, prefix, nodes):
    """Same addressing as TreeKvCompact with dst_from_zero: draft i lands at prefix+i."""
    gathered = []
    dsts = []
    for i, node in enumerate(nodes):
        src = prefix + int(node)
        dst = prefix + i
        src_block = block_ids[src // KV_BLOCK]
        gathered.append(cache[src_block, src % KV_BLOCK].clone())
        dsts.append((block_ids[dst // KV_BLOCK], dst % KV_BLOCK))
    for value, (block, offset) in zip(gathered, dsts):
        cache[block, offset] = value


def test_accept_moves_winner_only_and_advances_prefix():
    plan, packed, _state, _state_table, _kv_table, _kv_cache = _run_compressor("branch", 10, 4)
    records = plan["records"][3]
    kv_items = [item for item in records if item[3] == "kv"]
    cache, src, dst, _kind = kv_items[0]
    cache[dst].fill_(5)
    cache[src].zero_()
    sib_dst = [item[2] for item in plan["records"][2] if item[3] == "kv"][0]
    cache[sib_dst].fill_(3)
    info = cp.commit_winner(torch.tensor([[1, 3]], dtype=torch.int32))
    assert info["path_eq_chain"] is True
    assert float(cache[src].reshape(-1)[0]) == 5.0
    assert float(cache[sib_dst].reshape(-1)[0]) == 3.0

    ori = torch.zeros(8, KV_BLOCK, 1, 4, device="npu")
    block_ids = [1] * 8
    prefix = 10
    ori[1, prefix + 1].fill_(11)
    ori[1, prefix + 2].fill_(22)
    ori[1, prefix + 3].fill_(33)
    before_residual = ori[1, prefix + 2].clone()
    _move_ori_from_zero(ori, block_ids, prefix, [1, 3])
    assert torch.equal(ori[1, prefix + 0], torch.full_like(ori[1, 0], 11))
    assert torch.equal(ori[1, prefix + 1], torch.full_like(ori[1, 0], 33))
    assert torch.equal(ori[1, prefix + 2], before_residual)

    accepted = 2
    parents, depths, num_nodes, qsl, _start, lens = _branch(prefix + accepted)
    _install(parents, depths, num_nodes, qsl, torch.tensor([prefix + accepted], dtype=torch.int32))
    nxt = cp._build_plan(4, torch.device("cpu"))
    assert nxt["prefixes"][0] == prefix + accepted
    assert int(nxt["seqused"][0]) == prefix + accepted + 1
    assert (prefix + accepted) % 4 == 0
    assert nxt["lens"] == lens


def test_draft_hidden_follows_accepted_target_path():
    from vllm_ascend.models.deepseek_v4.dspark import DeepseekV4DSparkModel
    from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import (
        compact_tree_query_along_path,
        mask_rejected_dflash_context_slots,
    )

    hidden = torch.tensor([[10.0], [20.0], [30.0], [40.0]], device="npu")
    target_kv = torch.arange(8, dtype=torch.float32, device="npu")
    target_snapshot = target_kv.clone()
    positions = torch.tensor([80, 81, 81, 82], dtype=torch.int64, device="npu")
    qsl = torch.tensor([0, 4], dtype=torch.int32, device="npu")
    path = torch.tensor([[1, 3]], dtype=torch.int64, device="npu")
    compact_tree_query_along_path([hidden], qsl, path, positions)
    assert hidden[:3, 0].tolist() == [10.0, 20.0, 40.0]
    assert 30.0 not in hidden[:3, 0].tolist()
    assert positions[:3].tolist() == [80, 81, 82]
    assert torch.equal(target_kv, target_snapshot)

    slots = torch.tensor([5, 6, 7, 8], dtype=torch.int64, device="npu")
    mask_rejected_dflash_context_slots(
        slots,
        qsl,
        torch.tensor([1], dtype=torch.int32, device="npu"),
        pad_slot_id=-1,
    )
    assert slots.tolist() == [5, 6, 7, -1]

    state = torch.ones(4, 2, 1, 4, device="npu")
    state_before = state.clone()

    class _StateCache:
        prefix = "model.layers.0.state_cache"
        kv_cache = state

    model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
    wrote = DeepseekV4DSparkModel._store_paged_kv(
        model,
        hidden,
        slots,
        _StateCache(),
    )
    assert wrote is False
    assert torch.equal(state, state_before)


def test_draft_swa_kv_is_projected_per_layer():
    import vllm_ascend.models.deepseek_v4.dspark as dspark
    from vllm_ascend.models.deepseek_v4.dspark import DeepseekV4DSparkModel

    hidden_size = 4
    nope = 4
    rope = 4
    head = nope + rope
    block = 32

    class _Linear(torch.nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.weight = torch.full((head, hidden_size), scale, device="npu")

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight)

    class _Identity(torch.nn.Module):
        def forward(self, x):
            return x

    def _layer(prefix, scale, cache):
        attn = type("Attn", (), {})()
        attn.wkv = _Linear(scale)
        attn.kv_norm = _Identity()
        attn.nope_head_dim = nope
        attn.rope_head_dim = rope
        attn.head_dim = head
        attn.rotary_emb = None
        swa = type("Swa", (), {})()
        swa.prefix = prefix
        swa.kv_cache = cache
        swa.block_size = block
        attn.dsa_attn = type("Dsa", (), {"swa_cache_layer": swa})()
        return type("Layer", (), {"self_attn": attn})()

    cache0 = torch.full((4, block, 1, head), -3, dtype=torch.float32, device="npu")
    cache1 = torch.full((4, block, 1, head), -3, dtype=torch.float32, device="npu")
    state = torch.ones(4, 2, 1, head, device="npu")
    target_kv = torch.arange(8, dtype=torch.float32, device="npu")
    state_before = state.clone()
    target_before = target_kv.clone()
    layer0 = _layer("draft.swa.0", 1.0, cache0)
    layer1 = _layer("draft.swa.1", 2.0, cache1)
    # Same storage as layer 0, different projection. The second write must be skipped.
    layer_dup = _layer("draft.swa.0b", 9.0, cache0)
    model = DeepseekV4DSparkModel.__new__(DeepseekV4DSparkModel)
    model.layers = {"0": layer0, "1": layer1, "2": layer_dup}
    model.vllm_config = None
    # Flat slots: block 2, offsets 10 and 11, then a rejected PAD.
    slots = torch.tensor([2 * block + 10, 2 * block + 11, -1], dtype=torch.int64, device="npu")
    model._draft_kv_slots_by_name = {
        "draft.swa.0": slots,
        "draft.swa.1": slots,
        "draft.swa.0b": slots,
    }
    hidden = torch.ones(3, hidden_size, device="npu")
    positions = torch.tensor([10, 11, 12], dtype=torch.int64, device="npu")
    original_rope = dspark._apply_dsv4_rope
    dspark._apply_dsv4_rope = lambda rotary_emb, positions, x, inverse=False: x
    try:
        DeepseekV4DSparkModel.precompute_and_store_context_kv(
            model,
            hidden,
            positions,
            [],
        )
    finally:
        dspark._apply_dsv4_rope = original_rope
    torch.npu.synchronize()

    def _row(scale):
        return torch.full((head,), hidden_size * scale, device="npu")

    assert torch.equal(cache0[2, 10, 0], _row(1.0))
    assert torch.equal(cache0[2, 11, 0], _row(1.0))
    assert torch.equal(cache1[2, 10, 0], _row(2.0))
    assert torch.equal(cache1[2, 11, 0], _row(2.0))
    # Rejected suffix and every other page stay at the sentinel.
    assert float(cache0[2, 12, 0, 0]) == -3.0
    assert float(cache1[2, 12, 0, 0]) == -3.0
    assert float(cache0[0, 0, 0, 0]) == -3.0
    # The duplicate layer did not overwrite layer 0 with its own projection.
    assert float(cache0[2, 10, 0, 0]) != hidden_size * 9.0
    assert torch.equal(state, state_before)
    assert torch.equal(target_kv, target_before)


def test_ratio0_sparse_slots_skip_the_sibling():
    """Compress ratio 0 keeps one sequence and gathers slots, not a chain cu.

    The second sibling's query must not read the other sibling's slot, even
    though that slot sits inside the widened dense window.
    """
    from vllm_ascend.attention.tree_spec import build_tree_ori_sparse_indices

    window = 8
    prefix = 16
    budget = 2
    n_tok = 1 + budget
    block = 1
    vis = torch.zeros(1, budget, budget, dtype=torch.bool, device="npu")
    vis[0, 0, 0] = True
    vis[0, 1, 1] = True
    width = 128
    buffer = torch.full((8, 1, width), -1, dtype=torch.int32, device="npu")
    block_table = torch.tensor([[block, 2, 3, 4]], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([prefix + n_tok], dtype=torch.int32, device="npu")
    qsl = torch.tensor([0, n_tok], dtype=torch.int32, device="npu")
    indices = build_tree_ori_sparse_indices(
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        tree_visibility=vis,
        num_decodes=1,
        num_tokens=n_tok,
        window_size=window,
        storage_block_size=KV_BLOCK,
        buffer=buffer,
    )
    torch.npu.synchronize()

    def _slot(pos):
        return block * KV_BLOCK + pos

    sib_a = _slot(prefix + 1)
    sib_b = _slot(prefix + 2)
    row_b = indices[2, 0].detach().to("cpu")
    row_a = indices[1, 0].detach().to("cpu")
    assert int(sib_b) in row_b.tolist()
    assert int(sib_a) not in row_b.tolist()
    assert int(sib_a) in row_a.tolist()
    assert int(sib_b) not in row_a.tolist()

    ori = torch.zeros(8, KV_BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    for pos in range(prefix - window + 1, prefix + 1):
        ori[block, pos, 0, 0] = 0.1 * pos
    q = torch.randn(n_tok, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    cu = torch.tensor([0, n_tok], dtype=torch.int32, device="npu")
    seqused = seq_lens
    meta = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=HEADS,
        num_heads_kv=1,
        head_dim=HEAD_DIM,
        cu_seqlens_q=cu,
        seqused_q=cu.new_empty(0),
        seqused_kv=seqused,
        batch_size=1,
        max_seqlen_q=n_tok,
        max_seqlen_kv=int(seqused.max().item()),
        ori_topk=width,
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=window + budget - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device="npu",
    )
    sinks = torch.zeros(HEADS, dtype=torch.float32, device="npu")

    def _run():
        # No cmp_kv: a present cmp tensor selects CFA, which rejects ratio 0.
        out, _lse = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q,
            ori_kv=ori,
            ori_sparse_indices=indices,
            ori_block_table=block_table,
            cu_seqlens_q=cu,
            seqused_kv=seqused,
            sinks=sinks,
            metadata=meta,
            softmax_scale=HEAD_DIM**-0.5,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=window + budget - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )
        torch.npu.synchronize()
        return out

    base = _run()
    ori[block, prefix + 1, 0].fill_(40)
    skipped = _run()
    assert torch.equal(base[2], skipped[2])
    ori[block, prefix + 2, 0].fill_(40)
    touched = _run()
    assert not torch.equal(base[2], touched[2])


def test_accept_move_follows_compress_ratio():
    """Ratio 0 only relocates ori KV. Ratio 4 and 128 copy the winner's pages."""
    ratio0 = torch.full((8, KV_BLOCK, 1, 4), -3.0, device="npu")
    prefix = 10
    ratio0[1, prefix + 1].fill_(11)
    ratio0[1, prefix + 2].fill_(22)
    kept = ratio0.clone()
    _move_ori_from_zero(ratio0, [1] * 8, prefix, [1])
    assert torch.equal(ratio0[1, prefix + 0], torch.full_like(ratio0[1, 0], 11))
    assert float(ratio0[1, prefix + 2, 0, 0]) == 22.0

    for ratio, kind, start, winner, path in (
        (4, "branch", 10, 3, [1, 3]),
        (128, "straight7", 122, 7, [1, 2, 3, 4, 5, 6, 7]),
    ):
        plan, _packed, _state, _state_table, _kv_table, cache = _run_compressor(kind, start, ratio)
        for recs in plan["records"]:
            for item in recs:
                assert item[0].data_ptr() != ratio0.data_ptr()
                assert item[3] in ("kv", "state")
        kv_items = [item for item in plan["records"][winner] if item[3] == "kv"]
        assert kv_items, ratio
        page, src, dst, _kind = kv_items[0]
        page[dst].fill_(5)
        page[src].zero_()
        before_ratio0 = ratio0.clone()
        info = cp.commit_winner(torch.tensor([path], dtype=torch.int32))
        assert info["winner"] == winner
        assert float(page[src].reshape(-1)[0]) == 5.0
        assert torch.equal(ratio0, before_ratio0)
        assert torch.equal(ratio0[1, prefix + 2], kept[1, prefix + 2])
