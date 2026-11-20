# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.ascend_config import TreeSpecConfig
from vllm_ascend.patch.platform import patch_spec_decode_stats
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.model_states import mamba_hybrid
from vllm_ascend.worker.v2.spec_decode.tree import state_rollback


@pytest.mark.parametrize("budget", [1, 15, 48, 128])
@pytest.mark.parametrize("mega,topk", [(True, 1), (False, 1), (False, 8)])
def test_async_tree_reserves_verifier_width_and_restores_draft_depth(budget, mega, topk):
    depth = 15
    tree = TreeSpecConfig(enabled=True, budget=budget, topk=topk, use_mega_tree=mega)
    width = budget if mega or topk > 1 else min(budget, depth)
    config = SimpleNamespace(
        additional_config={"tree_spec_config": {"enabled": True}},
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )

    class ProbeScheduler(AsyncScheduler):
        def schedule(self):
            # Run upstream's own publication and in-flight accounting on two
            # successive steps, as well as our scheduler-to-worker adapter.
            output = SimpleNamespace(
                scheduled_spec_decode_tokens={"r": self.requests["r"].spec_token_ids},
                num_scheduled_tokens={"r": 1 + len(self.requests["r"].spec_token_ids)},
                num_spec_tokens_to_schedule=self.num_spec_tokens,
                pending_structured_output_tokens=False,
            )
            self._update_after_schedule(output)
            return output

    scheduler = ProbeScheduler.__new__(ProbeScheduler)
    scheduler.num_spec_tokens = depth
    scheduler.vllm_config = config
    scheduler.use_v2_model_runner = False
    scheduler.num_sampled_tokens_per_step = 1
    scheduler._spec_token_placeholders = [-1] * depth
    req = SimpleNamespace(
        is_prefill_chunk=False, use_structured_output=True, num_output_placeholders=0, spec_token_ids=[]
    )
    scheduler.requests = {"r": req}
    patch_spec_decode_stats._patch_tree_verify_width(ProbeScheduler)
    with (
        patch.object(patch_spec_decode_stats, "get_ascend_config", return_value=SimpleNamespace(tree_spec_config=tree)),
        patch.object(Scheduler, "_update_after_schedule"),
    ):
        prefill = scheduler.schedule()
        verify = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"r": 1}
    assert verify.num_scheduled_tokens == {"r": 1 + width}
    assert verify.num_spec_tokens_to_schedule == scheduler.num_spec_tokens == depth
    assert req.num_output_placeholders == width + 2
    assert req.spec_token_ids is scheduler._spec_token_placeholders and len(req.spec_token_ids) == width
    assert verify.pending_structured_output_tokens


@pytest.mark.parametrize(
    "budget,limit,expected",
    [
        (1, None, [2, 16, 33]),
        (15, None, [16, 33]),
        (48, None, [16, 33, 49]),
        (128, None, [16, 33, 129]),
        (48, 33, [16, 33]),
    ],
)
def test_mega_tree_capture_sizes_from_budget(budget, limit, expected):
    config = SimpleNamespace(
        additional_config={"tree_spec_config": {"enabled": True, "budget": budget, "topk": 8, "use_mega_tree": True}},
        speculative_config=SimpleNamespace(num_speculative_tokens=15),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512, max_num_seqs=1),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[16, 33], max_cudagraph_capture_size=limit, cudagraph_mode=CUDAGraphMode.FULL
        ),
    )
    NPUPlatform.apply_config_platform_defaults(config)
    assert config.compilation_config.cudagraph_capture_sizes == expected


@pytest.mark.parametrize("budget", [1, 15, 48])
def test_gdn_padded_batch_retains_the_previous_accepted_path(budget):
    depth = 15
    state = mamba_hybrid.AscendTreeMambaHybridModelState.__new__(mamba_hybrid.AscendTreeMambaHybridModelState)
    state.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=2, async_scheduling=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=depth),
    )
    state.model_config = SimpleNamespace(max_model_len=512)
    state.num_accepted_tokens_gpu = torch.ones(2, dtype=torch.int32)
    batch = Mock(
        num_reqs=1,
        num_reqs_after_padding=2,
        num_tokens=budget + 1,
        num_tokens_after_padding=2 * (budget + 1),
        is_prefilling_np=np.array([False]),
        prefill_runs_as_decode_np=None,
        idx_mapping=torch.tensor([1]),
        query_start_loc_np=np.array([0, budget + 1, 2 * (budget + 1)], dtype=np.int32),
        max_query_len=budget + 1,
    )
    with (
        patch.object(state_rollback, "tree_state_slot_count", return_value=budget + 1),
        patch.object(mamba_hybrid, "build_attn_metadata", side_effect=lambda **kw: kw["model_specific_attn_metadata"]),
    ):
        path, count = state._tree_buffers(torch.device("cpu"))
        path[1, 0], count[1] = budget, 2
        metadata = state._build_attn_metadata_for_batch(
            batch, CUDAGraphMode.FULL, (), None, [], None, for_capture=True, use_pcp_padding=False
        )
    assert metadata.tree_prev_path.shape == (2, max(depth, budget))
    assert metadata.tree_prev_path[:, 0].tolist() == [budget, 0]
    assert metadata.tree_prev_num_sampled.tolist() == [2, 0]


@pytest.mark.parametrize("budget", [1, 15, 128])
@pytest.mark.parametrize("page_floor", [None, 4200])
def test_tree_state_page_reuses_padding_and_covers_every_node(budget, page_floor):
    depth, conv_width, conv_dim = 15, 4, 64
    spec = MambaSpec(
        block_size=896,
        shapes=((conv_width - 1 + depth, conv_dim), (15, 16)),
        dtypes=(torch.bfloat16, torch.float32),
        num_speculative_blocks=depth,
        page_size_padded=page_floor,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=depth), cache_config=SimpleNamespace(block_size=896)
    )
    with (
        patch.object(state_rollback, "tree_state_slot_count", return_value=budget + 1),
        patch.object(attn_utils, "is_conv_state_dim_first", return_value=False),
    ):
        updated = attn_utils._widen_mamba_spec_for_tree(spec, config)
    if budget + 1 <= depth:
        assert updated is spec
        return
    conv_columns = conv_width - 1 + budget + 1
    contents = conv_columns * conv_dim * 2 + 15 * 16 * 4
    # Seven 128-token attention blocks share this page. Their BF16 strides
    # and the FP32 state each need integral offsets.
    expected = 28 * ((max(page_floor or 0, contents) + 27) // 28)
    assert updated.page_size_bytes == expected
    assert updated.shapes == ((conv_columns, conv_dim), (15, 16))
    assert updated.num_speculative_blocks == max(budget, depth)


@pytest.mark.parametrize("mega", [True, False])
def test_small_budget_is_valid_unless_the_entire_origin_chain_is_required(mega):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=15, use_dspark=lambda: False, use_dflash=lambda: True)
    )
    TreeSpecConfig(enabled=True, budget=1, topk=8, use_mega_tree=mega)._validate_backend(config)
    with pytest.raises(ValueError, match="budget >= num_speculative_tokens"):
        TreeSpecConfig(enabled=True, budget=1, topk=8, use_mega_tree=mega, keep_origin_chain=True)._validate_backend(
            config
        )
