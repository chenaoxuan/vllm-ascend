# SPDX-License-Identifier: Apache-2.0
"""BS1 regressions for Qwen GDN tree target FULL capture after model loading."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState
from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import TreeRejectionSampler
from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape
from vllm_ascend.worker.v2.spec_decode.tree.speculator import AscendTreeSpeculator


def _config(tree=True):
    return SimpleNamespace(
        additional_config={"tree_spec_config": {"enabled": tree, "budget": 16, "topk": 8}},
        speculative_config=SimpleNamespace(
            num_speculative_tokens=7,
            use_dspark=lambda: False,
            uses_dynamic_speculative_decoding=lambda: False,
        ),
        cache_config=SimpleNamespace(use_kda_recoverssm=False),
    )


def _manager(query_len, *, tree=True):
    manager = ModelAclGraphManager.__new__(ModelAclGraphManager)
    manager.vllm_config = _config(tree)
    manager.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[17 if tree else 8],
        max_cudagraph_capture_size=17 if tree else 8,
    )
    # GDN supports uniform batches, so requested FULL resolves to this mode.
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.max_num_reqs = 1
    manager.decode_query_len = query_len
    manager.varlen_decode = False
    manager.ubatch_runner = None
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    manager._capture_descs = {}
    manager._candidates = {}
    manager.graphs = {}
    manager._graphs_captured = False
    manager.tree_mixed_batch = False
    manager.tree_full_graph_manager = None
    manager.tree_batch_shape = None
    manager._init_candidates()
    return manager


def test_chain_width_drops_budget17_capture_gear():
    manager = _manager(8)
    assert manager._capture_descs == {}
    assert not manager.needs_capture()
    manager._graphs_captured = True
    desc = manager.dispatch(1, 17, 17, 0, max_query_len=17)
    assert desc.cg_mode == CUDAGraphMode.NONE


def test_tree_width_retains_actual_full_capture_and_replay_descriptor():
    manager = _manager(17)
    assert manager.needs_capture()
    descriptors = manager._capture_descs[CUDAGraphMode.FULL]
    assert len(descriptors) == 1
    captured = descriptors[0]
    assert (captured.num_tokens, captured.num_reqs, captured.uniform_token_count) == (17, 1, 17)
    assert captured.max_query_len == 17
    manager._graphs_captured = True
    runtime = manager.dispatch(1, 17, 17, 0, max_query_len=17)
    assert runtime == captured
    assert runtime.cg_mode == CUDAGraphMode.FULL


def test_non_tree_chain_capture_and_dispatch_remain_eight_wide():
    manager = _manager(8, tree=False)
    manager._graphs_captured = True
    desc = manager.dispatch(1, 8, 8, 0, max_query_len=8)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert (desc.num_tokens, desc.num_reqs, desc.uniform_token_count) == (8, 1, 8)


@pytest.mark.parametrize(
    ("tree", "state_type", "dsv4", "expected_width"),
    [
        (True, AscendMambaHybridModelState, False, 17),
        (False, AscendMambaHybridModelState, False, 8),
        (True, AscendModelState, False, 8),
        (True, AscendModelState, True, 8),
    ],
    ids=["qwen-gdn-tree", "qwen-gdn-chain", "qwen-gqa-tree", "dsv4-tree"],
)
def test_load_model_restores_only_gdn_tree_target_width(monkeypatch, tree, state_type, dsv4, expected_width):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = _config(tree)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.device = torch.device("cpu")
    runner.max_num_reqs = 1
    runner.num_speculative_steps = 7
    runner.prompt_logprobs_worker = object()
    runner.model_state = state_type.__new__(state_type)
    runner.tree_shape = TreeShape(7, 16, 8, dsv4)
    speculator = AscendTreeSpeculator.__new__(AscendTreeSpeculator)
    speculator._dsv4_dspark_draft = dsv4
    speculator.topk = 8
    speculator.tree_shape = runner.tree_shape
    runner.speculator = speculator if tree else SimpleNamespace()

    sampler = TreeRejectionSampler.__new__(TreeRejectionSampler)
    sampler.device = torch.device("cpu")
    sampler.num_speculative_steps = 7
    sampler._result_buf = None
    sampler._buf_cap_reqs = 0
    sampler._sampled_buf = None
    sampler._ensure_bufs(1)
    sampled_buffer = sampler._sampled_buf
    runner.rejection_sampler = sampler
    runner.decode_query_len = 17 if tree else 8
    load_calls = []

    def core_load(self, load_dummy_weights=False, *args, **kwargs):
        load_calls.append((load_dummy_weights, args, kwargs))
        # Reproduce the rebase's real post-model-load assignment.
        self.decode_query_len = self.num_speculative_steps + 1

    monkeypatch.setattr(GPUModelRunner, "load_model", core_load)
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.model_runner.get_pd_dspark_aux_layer_ids",
        lambda config: (),
    )
    for module, names in (
        ("nz_warmup", ("start_nz_warm_thread", "join_nz_warm_thread")),
        ("early_kernel_warmup", ("start_early_kernel_warmup", "join_early_kernel_warmup")),
    ):
        for name in names:
            monkeypatch.setattr(
                f"vllm_ascend.model_executor.warmup.{module}.{name}",
                lambda *args, **kwargs: None,
            )

    runner.load_model(False, "load-boundary", load_option=True)
    assert load_calls == [(False, ("load-boundary",), {"load_option": True})]
    assert runner.decode_query_len == expected_width
    assert runner.rejection_sampler._sampled_buf is sampled_buffer
    assert sampled_buffer.shape == (1, 8)
    assert runner.rejection_sampler.num_speculative_steps == 7
    if tree and state_type is AscendMambaHybridModelState:
        manager = _manager(runner.decode_query_len)
        manager._graphs_captured = True
        assert manager.dispatch(1, 17, 17, 0, max_query_len=17).cg_mode == CUDAGraphMode.FULL
