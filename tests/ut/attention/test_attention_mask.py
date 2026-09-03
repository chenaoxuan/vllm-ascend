#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder


class TestAttentionMaskBuilder(TestBase):
    def test_get_attn_mask(self):
        # if the len is less than max_seq_len, the attn_mask_cache will not be updated
        attention_mask_builder = AttentionMaskBuilder(torch.device("cpu"))
        attn_mask = attention_mask_builder.get_attn_mask(max_seq_len=512, dtype=torch.float16)
        self.assertEqual(attn_mask.shape, (512, 512))
        self.assertEqual(attn_mask[0][-1], torch.tensor(float("-inf"), dtype=torch.float16))
        self.assertEqual(attention_mask_builder._seq_len_cached, 512)
        self.assertEqual(attention_mask_builder.attn_mask_cache.shape, (512, 512))
        self.assertEqual(
            attention_mask_builder.attn_mask_cache[0][-1], torch.tensor(float("-inf"), dtype=torch.float16)
        )

        # if the len is greater than max_seq_len, the attn_mask_cache will be updated
        attn_mask = attention_mask_builder.get_attn_mask(max_seq_len=2048, dtype=torch.float16)
        self.assertEqual(attn_mask.shape, (2048, 2048))
        self.assertEqual(attn_mask[0][-1], torch.tensor(float("-inf"), dtype=torch.float16))
        self.assertEqual(attention_mask_builder._seq_len_cached, 2048)
        self.assertEqual(attention_mask_builder.attn_mask_cache.shape, (2048, 2048))
        self.assertEqual(
            attention_mask_builder.attn_mask_cache[0][-1], torch.tensor(float("-inf"), dtype=torch.float16)
        )

    def test_get_splitfuse_attn_mask(self):
        attention_mask_builder = AttentionMaskBuilder(torch.device("cpu"))
        attn_mask = attention_mask_builder.get_splitfuse_attn_mask()
        self.assertEqual(attn_mask.shape, (2048, 2048))

    def test_tree_attention_mask_per_request_prefix(self):
        builder = AttentionMaskBuilder(torch.device("cpu"))
        visibility = torch.zeros(2, 2, 2, dtype=torch.bool)
        visibility[0, 0, 0] = True
        visibility[1] = torch.eye(2, dtype=torch.bool)
        seq_lens = torch.tensor([10, 20], dtype=torch.int32)
        mask = builder.get_tree_attention_mask(None, visibility, seq_lens, num_decode=2)
        self.assertEqual(mask.shape, (2, 1, 3, 128))
        # query_len=3 → prev_kv is 7 and 17; prefix+root columns are visible.
        self.assertFalse(mask[0, 0, :, :8].any())
        self.assertFalse(mask[1, 0, :, :18].any())
        self.assertTrue(mask[0, 0, 0, 8:].all())
        self.assertTrue(mask[1, 0, 0, 18:].all())
        # req0 node1 sees slot 0; req1 is identity visibility.
        self.assertFalse(bool(mask[0, 0, 1, 8]))
        self.assertTrue(bool(mask[0, 0, 1, 9]))
        self.assertFalse(bool(mask[1, 0, 1, 18]))
        self.assertTrue(bool(mask[1, 0, 1, 19]))
        self.assertTrue(bool(mask[1, 0, 2, 18]))
        self.assertFalse(bool(mask[1, 0, 2, 19]))

    @patch("vllm_ascend.attention.attention_mask.dflash_tree_spec_enabled", return_value=True)
    def test_tree_attention_mask_requires_decode_classification(self, _mock):
        """num_decode=0 (budget query misclassified as prefill) drops the tree mask."""
        builder = AttentionMaskBuilder(torch.device("cpu"))
        visibility = torch.eye(2, dtype=torch.bool).unsqueeze(0)
        seq_lens = torch.tensor([10], dtype=torch.int32)
        model_config = SimpleNamespace(runner_type="generate")
        prefill = builder.get_attention_mask(
            True, model_config, None, visibility, seq_lens, num_decode=0
        )
        decode = builder.get_attention_mask(
            True, model_config, None, visibility, seq_lens, num_decode=1
        )
        self.assertEqual(prefill.shape, (2048, 2048))
        self.assertEqual(decode.shape, (1, 1, 3, 128))
