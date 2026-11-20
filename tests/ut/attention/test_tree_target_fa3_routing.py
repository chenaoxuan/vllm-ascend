"""CPU behavior checks for target FIA routing and retained draft FA3 metadata.

These tests call the production selectors and runner methods. They do not
instantiate a model or launch NPU kernels. Run in the normal Ascend Python
environment so imports resolve to the frozen source under test.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.compilation import backends
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.selector import AttentionSelectorConfig

from vllm_ascend import platform
from vllm_ascend.device.hardware_profile import AttentionBackendFamily

FIA = "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
FA3 = "vllm_ascend.attention.flash_attention_v3.AscendFlashAttentionBackend"


def make_config(tree=True, **tree_options):
    target = SimpleNamespace(
        architecture="Qwen3_5ForConditionalGeneration",
        hf_config=SimpleNamespace(model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"]),
        hf_text_config=SimpleNamespace(model_type="qwen3_5_text"),
    )
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=target,
        speculative_config=SimpleNamespace(
            method="dflash",
            parallel_drafting=True,
            target_model_config=target,
            draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"]),
            enable_adaptive_verification=False,
        ),
        additional_config={"tree_spec_config": {"enabled": tree, **tree_options}},
        kv_transfer_config=None,
    )


@pytest.fixture
def route(monkeypatch):
    # Fix the hardware/RL environment, retaining the real target/draft and
    # ordinary-backend validators. A2 does not take the A5 CANN draft path.
    profile = SimpleNamespace(
        supports=lambda _: False,
        attention_backend_family=AttentionBackendFamily.STANDARD,
    )
    monkeypatch.setattr(platform, "get_current_hardware_profile", lambda: profile)
    monkeypatch.setattr(platform, "_validate_fa3_backend", lambda *_: False)

    def select(config, tag="backbone", **selector_options):
        monkeypatch.setattr("vllm.config.get_current_vllm_config_or_none", lambda: config)
        monkeypatch.setattr(backends, "model_tag", tag)
        options = dict(head_size=256, block_size=128, dtype=torch.bfloat16, kv_cache_dtype="auto")
        options.update(selector_options)
        selector = AttentionSelectorConfig(**options)
        return platform.NPUPlatform._select_attn_backend_cls(selector)

    return select


@pytest.mark.parametrize("topk", [None, 1, 8])
@pytest.mark.parametrize("use_mega_tree", [False, True])
@pytest.mark.parametrize("budget", [16, 32, 48])
def test_tree_target_uses_mask_capable_backend(route, topk, use_mega_tree, budget):
    config = make_config(topk=topk, use_mega_tree=use_mega_tree, budget=budget)
    assert route(config) == FIA


@pytest.mark.parametrize("tree_config", [None, {}, {"enabled": False}])
def test_non_tree_target_also_uses_original_fia(route, tree_config):
    config = make_config(False)
    config.additional_config = {} if tree_config is None else {"tree_spec_config": tree_config}
    assert route(config) == FIA


@pytest.mark.parametrize("use_mega_tree", [False, True])
@pytest.mark.parametrize("non_causal", [False, True])
def test_tree_draft_keeps_fa3(route, use_mega_tree, non_causal):
    config = make_config(use_mega_tree=use_mega_tree, budget=32, topk=8)
    assert route(config, "dflash_head", use_non_causal=non_causal) == FA3


@pytest.mark.parametrize("tree", [False, True])
def test_encoder_target_is_not_routed_to_fa3(route, tree):
    assert route(make_config(tree), attn_type=AttentionType.ENCODER) == FIA


@pytest.mark.parametrize(
    "options",
    [
        {"dtype": torch.float32},
        {"kv_cache_dtype": "fp8"},
        {"has_sink": True},
        {"head_size": 512},
        {"use_pcp": True},
        {"dtype": torch.bfloat16, "kv_cache_dtype": "float16"},
        {"use_batch_invariant": True},
    ],
)
def test_non_tree_unsupported_target_retains_fia(route, options):
    assert route(make_config(False), **options) == FIA


def test_unrelated_model_target_is_not_enabled_by_guard(route):
    config = make_config(False)
    config.speculative_config.target_model_config.architecture = "OtherModel"
    assert route(config) == FIA


def make_runner(target_fa3):
    # __new__ avoids allocations and model loading; methods under test remain
    # exactly the production implementations.
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
    from vllm_ascend.attention.flash_attention_v3 import AscendFlashAttentionBackend
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNFusedAttentionBackend
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    config = make_config(not target_fa3)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.model_config = config.model_config
    runner.speculative_config = config.speculative_config
    runner.vllm_config = config
    runner.sync_spec_pp_cpu_counts = False
    runner.model_state = MambaHybridModelState.__new__(MambaHybridModelState)
    runner.speculator = SimpleNamespace(attn_backends={"draft.0": AscendFlashAttentionBackend})
    target_backend = AscendFlashAttentionBackend if target_fa3 else AscendAttentionBackend
    runner.attn_groups = [
        [
            SimpleNamespace(backend=target_backend, layer_names=["target.full.0"]),
            SimpleNamespace(backend=AscendGDNFusedAttentionBackend, layer_names=["target.gdn.0"]),
            SimpleNamespace(backend=AscendFlashAttentionBackend, layer_names=["draft.0"]),
        ]
    ]
    runner.parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    runner.ascend_config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=1),
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
    )
    return runner


@pytest.mark.parametrize("target_fa3", [False, True])
def test_qwen_target_always_requires_exact_length_sync(target_fa3):
    runner = make_runner(target_fa3)
    assert runner._can_skip_spec_cpu_length_sync() is False


@pytest.mark.parametrize("target_fa3", [False, True])
def test_fia_target_copies_and_consumes_post_rejection_lengths(monkeypatch, target_fa3):
    runner = make_runner(target_fa3)
    runner.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([98, 195], dtype=torch.int32)),
        num_computed_tokens_cpu=torch.tensor([100, 200], dtype=torch.int32),
        req_id_to_index={"a": 0, "b": 1},
    )
    runner.input_buffers = SimpleNamespace(seq_lens_cpu=torch.zeros(2, dtype=torch.int32))
    runner.num_computed_tokens_cpu = torch.zeros(2, dtype=torch.int32)
    runner.num_computed_tokens_event = Mock()
    runner.num_computed_tokens_stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "default-stream")
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=["a", "b"]),
        num_scheduled_tokens={"a": 2, "b": 3},
    )
    runner._copy_num_computed_tokens_to_cpu()
    runner._update_seq_lens_cpu(scheduler_output, ["a", "b"])
    runner.num_computed_tokens_stream.wait_stream.assert_called_once_with("default-stream")
    runner.num_computed_tokens_event.record.assert_called_once_with()
    runner.num_computed_tokens_event.synchronize.assert_called_once_with()
    assert runner.req_states.num_computed_tokens_cpu.tolist() == [98, 195]
    assert runner.input_buffers.seq_lens_cpu.tolist() == [100, 198]


def make_draft_builder(monkeypatch, mode, enforce_eager=False):
    from vllm_ascend.attention.flash_attention_v3 import AscendFlashAttentionMetadataBuilder

    impl = SimpleNamespace(
        num_heads=6,
        num_kv_heads=1,
        head_size=256,
        scale=1 / 16,
        logits_soft_cap=0.0,
        sliding_window=None,
    )
    config = make_config()
    config.speculative_config.enforce_eager = enforce_eager
    config.speculative_config.num_speculative_tokens = 15
    config.scheduler_config = SimpleNamespace(max_num_seqs=4)
    config.compilation_config = SimpleNamespace(
        cudagraph_mode=mode,
        # The actual DFlash query bucket below is deliberately absent: draft
        # graph metadata must survive independently of target capture gears.
        cudagraph_capture_sizes=[33],
        static_forward_context={
            "draft.0": SimpleNamespace(impl=impl),
            "draft.1": SimpleNamespace(impl=impl),
        },
    )
    builder = AscendFlashAttentionMetadataBuilder(
        SimpleNamespace(block_size=128, dtype=torch.bfloat16),
        ["draft.0", "draft.1"],
        config,
        torch.device("cpu"),
    )
    calls = []

    def cpu_tiling(spec, batch_size, query_start_loc, seq_lens, block_tables, max_query_len, causal):
        # Replace only AICPU tiling. The builder, cache ownership, copies,
        # graph keys, active-request handling and refresh method stay native.
        calls.append(batch_size)
        return torch.tensor(
            [batch_size, int(seq_lens.sum()), int(query_start_loc[-1]), max_query_len, int(block_tables[0, 0])],
            dtype=torch.int32,
        )

    monkeypatch.setattr(builder, "_compute_tiling", cpu_tiling)
    return builder, calls


def draft_common(seq_len=127):
    return SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=16,
        num_input_tokens=32,
        max_query_len=16,
        query_start_loc_cpu=torch.tensor([0, 16], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 16], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        block_table_tensor=torch.tensor([[3, 7]], dtype=torch.int32),
        slot_mapping=torch.arange(16, dtype=torch.int64),
        causal=False,
    )


@pytest.mark.parametrize("mode_name", ["FULL", "FULL_DECODE_ONLY"])
def test_draft_graph_tiling_refresh_preserves_captured_buffer(monkeypatch, mode_name):
    from vllm.config.compilation import CUDAGraphMode

    builder, calls = make_draft_builder(monkeypatch, getattr(CUDAGraphMode, mode_name))
    common = draft_common()
    metadata = builder.build_for_graph_capture(common)
    assert metadata.max_query_len == 16
    assert calls == [1]  # Two identical layers share one tiling computation.
    captured = next(iter(metadata.scheduler_metadata.values()))
    address = captured.data_ptr()
    torch.testing.assert_close(captured, torch.tensor([1, 127, 16, 16, 3], dtype=torch.int32))

    common.seq_lens.fill_(119)
    common.query_start_loc[-1] = 12
    common.block_table_tensor[0, 0] = 9
    builder.update_draft_decode_metadata(metadata)
    assert captured.data_ptr() == address
    torch.testing.assert_close(captured, torch.tensor([1, 119, 12, 16, 9], dtype=torch.int32))

    rebuilt = builder.build(0, common)
    assert next(iter(rebuilt.scheduler_metadata.values())).data_ptr() == address
    assert metadata.seq_lens.data_ptr() == common.seq_lens.data_ptr()


def test_draft_graph_contexts_keep_independent_tiling_storage(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode

    builder, _ = make_draft_builder(monkeypatch, CUDAGraphMode.FULL_DECODE_ONLY)
    first, second = draft_common(127), draft_common(255)
    first_meta = builder.build(0, first)
    second_meta = builder.build(0, second)
    first_buffer = next(iter(first_meta.scheduler_metadata.values()))
    second_buffer = next(iter(second_meta.scheduler_metadata.values()))
    assert first_buffer.data_ptr() != second_buffer.data_ptr()
    first.seq_lens.fill_(111)
    second.seq_lens.fill_(222)
    builder.update_draft_decode_metadata()
    assert first_buffer[1] == 111
    assert second_buffer[1] == 222


@pytest.mark.parametrize("mode_name,enforce_eager", [("FULL_DECODE_ONLY", True), ("NONE", False)])
def test_eager_draft_does_not_reuse_or_refresh_graph_tiling(monkeypatch, mode_name, enforce_eager):
    from vllm.config.compilation import CUDAGraphMode

    builder, calls = make_draft_builder(monkeypatch, getattr(CUDAGraphMode, mode_name), enforce_eager)
    common = draft_common()
    first = next(iter(builder.build(0, common).scheduler_metadata.values()))
    common.seq_lens.fill_(119)
    builder.update_draft_decode_metadata()
    assert calls == [1]
    assert first[1] == 127
    second = next(iter(builder.build(0, common).scheduler_metadata.values()))
    assert first.data_ptr() != second.data_ptr()
    assert second[1] == 119


def test_draft_padding_requests_do_not_enter_scheduler_tiling(monkeypatch):
    from vllm.config.compilation import CUDAGraphMode

    builder, calls = make_draft_builder(monkeypatch, CUDAGraphMode.FULL_DECODE_ONLY)
    common = draft_common()
    common.num_reqs = 2
    common.query_start_loc_cpu = torch.tensor([0, 16, 16], dtype=torch.int32)
    common.query_start_loc = common.query_start_loc_cpu.clone()
    common.seq_lens = torch.tensor([127, 999], dtype=torch.int32)
    common.block_table_tensor = torch.tensor([[3, 7], [19, 20]], dtype=torch.int32)
    metadata = builder.build(0, common)
    assert calls == [1]
    assert metadata.seq_lens.tolist() == [127]
    assert metadata.query_start_loc.tolist() == [0, 16]
    torch.testing.assert_close(
        next(iter(metadata.scheduler_metadata.values())), torch.tensor([1, 127, 16, 16, 3], dtype=torch.int32)
    )
