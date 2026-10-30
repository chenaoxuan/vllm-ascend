# SPDX-License-Identifier: Apache-2.0
"""Tree FULL FIA accepts padded-page cache views without changing ordinary FIA."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm_ascend.attention.attention_v1 as attn_module
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState


def _case(contiguous):
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 8, 1, 256
    impl.scale = 256**-0.5
    impl.use_bnsd_kv_cache = False
    impl.enable_c8_quant = False
    impl.sliding_window = None
    impl._use_max_workspace_for_fia_graph = False
    impl._layer_name = "qwen.full_attn"
    raw = torch.arange(2 * 2 * 4 * 256, dtype=torch.float32).view(2, 2, 4, 1, 256)
    impl.key_cache, impl.value_cache = raw[:, 0], raw[:, 1]
    if contiguous:
        impl.key_cache = impl.key_cache.contiguous()
        impl.value_cache = impl.value_cache.contiguous()
    impl._tree_fia_bsnd_shape = Mock(return_value=(1, 17))
    metadata = SimpleNamespace(
        actual_seq_lengths_q=[17],
        num_decodes=1,
        num_actual_tokens=17,
        attn_state=AscendAttentionState.DecodeOnly,
        block_tables=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_lens_list=[8],
        causal=True,
        attn_mask=torch.zeros(1, 1, 17, 8, dtype=torch.bool),
    )
    query = torch.zeros(17, 8, 256)
    return impl, metadata, query, torch.empty_like(query)


@pytest.mark.parametrize("contiguous", [False, True])
def test_tree_full_fia_materializes_only_strided_cache_at_op_boundary(contiguous):
    impl, metadata, query, output = _case(contiguous)
    with (
        patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False)),
        patch.object(
            attn_module.torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", create=True
        ) as workspace,
        patch.object(attn_module.torch_npu, "npu_fused_infer_attention_score", create=True),
        patch.object(attn_module, "get_capture_resource", side_effect=lambda _, factory, *args: factory()),
        patch.object(attn_module, "register_task") as register,
    ):
        actual, tokens = impl.full_graph_tree_fia(query, query, query, metadata, output)
    assert tokens == 17
    assert actual.data_ptr() == output.data_ptr()
    workspace.assert_called_once()
    register.assert_called_once()
    op_params = register.call_args.args[1]
    for params in (workspace.call_args.kwargs, op_params):
        assert params["input_layout"] == "BSND"
        assert params["sparse_mode"] == 0
        for name, cache in (("key", impl.key_cache), ("value", impl.value_cache)):
            expected = cache.view(2, 4, 256)
            assert params[name].is_contiguous()
            torch.testing.assert_close(params[name], expected)
            assert (params[name].data_ptr() == cache.data_ptr()) == contiguous
    assert op_params["key"] is workspace.call_args.kwargs["key"]
    assert op_params["value"] is workspace.call_args.kwargs["value"]
    # Runtime updates replace metadata only; they must retain the dense recorded K/V.
    provider = register.call_args.args[2]
    runtime = SimpleNamespace(seq_lens_list=[7], attn_mask=None, block_tables=metadata.block_tables)
    updates = provider.resolve({"qwen.full_attn": runtime})
    assert "key" not in updates and "value" not in updates
    assert {**op_params, **updates}["key"] is op_params["key"]


@pytest.mark.parametrize("contiguous", [False, True])
def test_ordinary_full_fia_retains_existing_paged_cache_boundary(contiguous):
    impl, metadata, query, output = _case(contiguous)
    metadata.attn_mask = torch.zeros(2048, 2048, dtype=torch.int8)
    with (
        patch.object(attn_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False)),
        patch.object(
            attn_module.torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", create=True
        ) as workspace,
        patch.object(attn_module.torch_npu, "npu_fused_infer_attention_score", create=True),
        patch.object(attn_module, "get_capture_resource", side_effect=lambda _, factory, *args: factory()),
        patch.object(attn_module, "register_task") as register,
    ):
        impl.full_graph_fia(query, query, query, metadata, output)
    for params in (workspace.call_args.kwargs, register.call_args.args[1]):
        assert params["input_layout"] == "TND"
        for name, cache in (("key", impl.key_cache), ("value", impl.value_cache)):
            assert params[name].data_ptr() == cache.data_ptr()
            assert params[name].is_contiguous() == contiguous
