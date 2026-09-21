# Build compressor, attention and indexer batch state from a tree.
# Operators see one cu_seqlens. Segment k is chain k, and those lengths differ.
# No NPU kernel is launched.

import importlib.util
from pathlib import Path

import torch

_CHAIN_PACK = (
    Path(__file__).resolve().parents[3]
    / "vllm_ascend"
    / "worker"
    / "v2"
    / "spec_decode"
    / "tree"
    / "chain_pack.py"
)


def _load_chain_pack():
    spec = importlib.util.spec_from_file_location("tree_chain_pack_ut", _CHAIN_PACK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cp = _load_chain_pack()


def _segments(cu):
    vals = [int(x) for x in cu.detach().to("cpu").tolist()]
    return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]


def _install(parents, depths, num_nodes, qsl, start):
    cp.clear_tree_chain_layout()
    cp.set_tree_chain_layout(parents, num_nodes, depths, qsl, start)


def _one_request():
    # token 0 root
    # token 1 and 2 are different depth-1 children
    # token 3 continues token 1, so its chain is longer
    parents = torch.tensor([[0, 0, 1]], dtype=torch.int32)
    depths = torch.tensor([[1, 1, 2]], dtype=torch.int32)
    num_nodes = torch.tensor([3], dtype=torch.int32)
    qsl = torch.tensor([0, 4], dtype=torch.int32)
    start = torch.tensor([10], dtype=torch.int32)
    return parents, depths, num_nodes, qsl, start


def _two_requests():
    parents = torch.tensor(
        [
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [
            [1, 1, 2],
            [1, 0, 0],
        ],
        dtype=torch.int32,
    )
    num_nodes = torch.tensor([3, 1], dtype=torch.int32)
    qsl = torch.tensor([0, 4, 6], dtype=torch.int32)
    start = torch.tensor([10, 40], dtype=torch.int32)
    return parents, depths, num_nodes, qsl, start


def test_inner_cu_uses_each_chain_length():
    _install(*_one_request())
    plan = cp._build_plan(4, torch.device("cpu"))
    assert plan is not None
    segments = _segments(plan["cu"])
    assert segments == [1, 2, 2, 3]
    assert len(set(segments)) > 1
    assert _segments(plan["attn_cu"]) == segments
    assert plan["max_q"] == 3
    assert plan["req_cu"] == [0, 4]
    assert torch.equal(plan["cu"], plan["attn_cu"])
    # KV length is the request prefix plus that chain, still one value per chain.
    assert plan["seqused"].tolist() == [11, 12, 12, 13]
    assert plan["chains"] == [[], [1], [2], [1, 3]]
    # Compressor drops the placeholder root. Drafts start at the same prefix
    # that dst_from_zero writes them to.
    assert plan["cmp_lens"] == [1, 1, 1, 2]
    assert _segments(plan["cmp_cu"]) == [1, 1, 1, 2]
    assert plan["cmp_rows"].tolist() == [0, 1, 2, 1, 3]


def test_requests_do_not_share_chains():
    _install(*_two_requests())
    plan = cp._build_plan(6, torch.device("cpu"))
    assert plan is not None
    assert plan["req_cu"] == [0, 4, 6]
    assert _segments(plan["cu"]) == [1, 2, 2, 3, 1, 2]
    assert plan["req_index"] == [0, 0, 0, 0, 1, 1]
    assert plan["prefixes"] == [10, 10, 10, 10, 40, 40]
    assert plan["start"].tolist() == [10, 10, 10, 10, 40, 40]
    # Request 1 chain must not point at request 0 tokens.
    packed = plan["rows"].tolist()
    cu = plan["cu"].tolist()
    req1 = packed[cu[4] : cu[6]]
    assert req1 == [4, 4, 5]
    assert plan["seqused"].tolist()[-2:] == [41, 42]


def test_pack_and_unpack_follow_variable_chains():
    _install(*_one_request())
    hidden = torch.tensor(
        [
            [10.0],
            [20.0],
            [30.0],
            [40.0],
        ]
    )
    cp._build_plan(4, hidden.device)
    packed = cp.pack_chain_tokens(hidden)
    assert packed[:, 0].tolist() == [10, 10, 20, 10, 30, 10, 20, 40]
    restored = cp.unpack_chain_tokens(packed, 4)
    assert restored[:, 0].tolist() == [10, 20, 30, 40]


def test_kv_cu_segments_match_each_chain_seqused():
    seqused = torch.tensor([11, 12, 12, 13], dtype=torch.int32)
    ori_cu, cmp_cu = cp.chain_length_cus(seqused, 4)
    assert _segments(ori_cu) == [11, 12, 12, 13]
    assert _segments(cmp_cu) == [2, 3, 3, 3]


def test_indexer_batch_uses_variable_chain_cu():
    _install(*_one_request())
    cp._build_plan(4, torch.device("cpu"))
    cache = torch.zeros(8, 4, 2)
    cp._PLAN["kv_tables"][cache.data_ptr()] = torch.zeros(4, 3, dtype=torch.int32)
    batch = cp.indexer_batch(cache)
    assert batch is not None
    assert _segments(batch["cu"]) == [1, 2, 2, 3]
    assert batch["max_q"] == 3
    assert batch["batch"] == 4
    assert batch["seqused_k"].tolist() == [2, 3, 3, 3]
    assert batch["residual"].tolist() == [3, 0, 0, 1]


def test_attention_view_keeps_distinct_chain_lengths():
    _install(*_one_request())
    cp._build_plan(4, torch.device("cpu"))
    block = 4
    cache = torch.arange(16 * block * 2, dtype=torch.float32).reshape(16, block, 2)
    table = torch.arange(8, dtype=torch.int32).view(1, 8)
    view = cp.chain_attention_view(cache, table, block)
    assert _segments(view["cu"]) == [1, 2, 2, 3]
    assert view["seqused"].tolist() == [11, 12, 12, 13]
    # Depth-1 siblings both rewrite the page that holds prefix+1, on private scratch.
    col = (10 + 1) // block
    left = int(view["ori_bt"][1, col].item())
    right = int(view["ori_bt"][2, col].item())
    real = int(table[0, col].item())
    assert left != right
    assert left != real
    assert right != real


def test_state_pages_follow_the_request_that_owns_the_chain():
    _install(*_two_requests())
    plan = cp._build_plan(6, torch.device("cpu"))
    state = torch.zeros(256, 2, 1, 4)
    table = torch.arange(1, 65, dtype=torch.int32).view(2, 32)
    state[int(table[0, 5])].fill_(1)
    state[int(table[1, 20])].fill_(2)
    rows = cp._clone_state_rows(state, table, plan, state.device, 4, 10)
    # Request 0 prefix 10 writes column 5. Siblings must not share that page.
    pages = [int(rows[chain, 5].item()) for chain in range(4)]
    assert len(set(pages)) == 4
    assert int(table[0, 5]) not in pages
    for page in pages:
        assert float(state[page].reshape(-1)[0]) == 1.0
    # Request 1 prefix 40 writes column 20, from its own table row.
    for chain in (4, 5):
        page = int(rows[chain, 20].item())
        assert page != int(table[1, 20].item())
        assert float(state[page].reshape(-1)[0]) == 2.0
    assert int(rows[4, 20]) != int(rows[5, 20])


def test_compressor_launches_once_with_chain_cu():
    _install(*_one_request())
    seen = {}

    def _metadata(cos, sin, cu, start, kv_rows, block, slot_format, ratio, n_rows, n):
        seen["cu"] = cu.detach().cpu().clone()
        seen["start"] = start.detach().cpu().clone()
        seen["n"] = int(n)
        seen["rows"] = int(kv_rows.shape[0])
        return (
            torch.zeros(n_rows, 4),
            torch.zeros(n_rows, 4),
            torch.full((n_rows, 2), -1, dtype=torch.int32),
        )

    class _Ops:
        class _C_ascend:
            compressor_metadata = staticmethod(_metadata)

    cache = torch.zeros(32, 4, 2)
    state = torch.zeros(32, 2, 1, 8)
    kv_table = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32)
    state_table = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32)
    meta = type("M", (), {})()
    meta.full_compress_cos = torch.zeros(64, 1, 4)
    meta.full_compress_sin = torch.zeros(64, 1, 4)
    original = cp.torch.ops
    cp.torch.ops = _Ops()
    table = None
    try:
        with cp.chain_kv_scope([cache]):
            packed = cp.chain_compressor_batch(
                torch.zeros(4, 4),
                state,
                kv_table,
                state_table,
                4,
                4,
                meta,
                2,
            )
        table = cp._PLAN["kv_tables"][cache.data_ptr()]
    finally:
        cp.torch.ops = original
        cp.clear_tree_chain_layout()
    assert packed is not None
    assert seen["n"] == 4
    assert seen["rows"] == 4
    assert _segments(seen["cu"]) == [1, 1, 1, 2]
    assert seen["start"].tolist() == [10, 10, 10, 10]
    assert _segments(packed["cu"]) == [1, 1, 1, 2]
    col = (10 // 4) // 4
    pages = [int(table[i, col].item()) for i in range(4)]
    # Only the two-draft chain closes a window, so only it leaves the real page.
    assert pages[1] == pages[0]
    assert pages[2] == pages[0]
    assert pages[3] != pages[0]


def _main():
    tests = [
        test_inner_cu_uses_each_chain_length,
        test_requests_do_not_share_chains,
        test_pack_and_unpack_follow_variable_chains,
        test_kv_cu_segments_match_each_chain_seqused,
        test_indexer_batch_uses_variable_chain_cu,
        test_attention_view_keeps_distinct_chain_lengths,
        test_state_pages_follow_the_request_that_owns_the_chain,
        test_compressor_launches_once_with_chain_cu,
    ]
    for test in tests:
        cp.clear_tree_chain_layout()
        test()
        cp.clear_tree_chain_layout()
        print("ok", test.__name__)


if __name__ == "__main__":
    _main()
