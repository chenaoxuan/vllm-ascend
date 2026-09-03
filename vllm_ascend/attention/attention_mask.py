#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from vllm_ascend.platform import ModelConfig
from vllm_ascend.utils import singleton
from vllm_ascend.worker.v2.spec_decode import dflash_tree_spec_enabled
from vllm_ascend.ascend_config import get_ascend_config


def _generate_attn_mask(max_seq_len, dtype):
    # Construct lower triangle matrix.
    mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    # TODO: Eliminate this part in the future.
    mask_value = float("-inf") if dtype == torch.float16 else 1
    attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype).masked_fill_(mask_flag, mask_value)
    return attn_mask

def align_up(value, alignment=128):
    return ((value + alignment - 1) // alignment) * alignment

@singleton
class AttentionMaskBuilder:
    def __init__(self, device: torch.device):
        self.attn_mask_cache = None
        self._seq_len_cached = 0
        self.device = device
        self.chunked_prefill_attn_mask = None

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype):
        if self.attn_mask_cache is None or max_seq_len > self._seq_len_cached:
            self.attn_mask_cache = _generate_attn_mask(max_seq_len, dtype)
            self._seq_len_cached = max_seq_len
        assert self.attn_mask_cache is not None, "Something is wrong in generate_attn_mask."
        if self.attn_mask_cache.dtype != dtype:
            self.attn_mask_cache = self.attn_mask_cache.to(dtype)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous().to(self.device, non_blocking=True)

    def get_splitfuse_attn_mask(self) -> torch.Tensor:
        if self.chunked_prefill_attn_mask is None:
            self.chunked_prefill_attn_mask = (
                torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(self.device)
            )
        return self.chunked_prefill_attn_mask

    def get_attention_mask(self, causal: bool,
                           model_config: ModelConfig,
                           tree_num_nodes: torch.Tensor | None = None,
                           tree_visibility: torch.Tensor | None = None,
                           seq_lens: torch.Tensor | None = None,
                           num_decode: int = 0
                           ):
        if not causal:
            # FIA applies any provided mask as defaultMask (sparse_mode=0),
            # which would wrongly mask out the upper triangle for
            # bidirectional attention, so non-causal attention must not
            # carry a mask here. The 310P mask builder overrides this
            # because its attention operators require an explicit
            # non-masking mask instead.
            return None

        if dflash_tree_spec_enabled() and num_decode > 0:
            return self.get_tree_attention_mask(tree_num_nodes, tree_visibility, seq_lens, num_decode)

        if model_config.runner_type == "pooling":
            return self.get_attn_mask(2048, torch.bool)

        return self.get_splitfuse_attn_mask()
    
    def get_tree_attention_mask(self, tree_num_nodes: torch.Tensor,
                                tree_visibility: torch.Tensor,
                                seq_lens: torch.Tensor,
                                num_decode):
        num_reqs = tree_visibility.shape[0]
        max_nodes = get_ascend_config().tree_spec_config.budget
        # TODO: align query_len with actual scheduled query length
        # (e.g. 1 + num_speculative_steps / num_scheduled_tokens). Using
        # 1 + budget breaks prev_kv_len and FIA S1 when budget differs.
        query_len = 1 + max_nodes

        attn_mask = torch.ones(num_decode, query_len,
            align_up(seq_lens.max(), 128), dtype=torch.bool, device=self.device)
        for i in range(num_decode):
            req_mask = attn_mask[i, :, :]
            seq_len = seq_lens[i].item()
            prev_kv_len = seq_len - query_len

            req_mask[:, :prev_kv_len + 1] = False
            # tree_visibility: True = can attend; FIA bool mask: True = masked out.
            # Draft columns are slot-indexed (contiguous j). KV slot_mapping uses
            # unique token-index coordinates while RoPE positions stay depth-based.
            req_mask[1:, prev_kv_len + 1 : prev_kv_len + 1 + max_nodes] = ~tree_visibility[i]
        return attn_mask