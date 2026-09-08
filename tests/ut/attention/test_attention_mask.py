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
from vllm_ascend.attention.attention_mask import (
    AttentionMaskBuilder,
    dummy_tree_mask_for_capture,
    tree_fia_bsnd_shape,
)


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
        mask = builder.get_tree_attention_mask(visibility, seq_lens, num_decode=2)
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

    def test_tree_fia_bsnd_shape_only_budget_aligned(self):
        with patch(
            "vllm_ascend.attention.attention_mask._tree_query_len",
            return_value=49,
        ):
            self.assertEqual(tree_fia_bsnd_shape(49), (1, 49))
            self.assertEqual(tree_fia_bsnd_shape(98), (2, 49))
            self.assertIsNone(tree_fia_bsnd_shape(64))
            self.assertIsNone(tree_fia_bsnd_shape(17))
            self.assertTrue(dummy_tree_mask_for_capture(49, 1))
            self.assertTrue(dummy_tree_mask_for_capture(98, 2))
            self.assertFalse(dummy_tree_mask_for_capture(98, 8))
            self.assertFalse(dummy_tree_mask_for_capture(64, 8))

    def test_capture_dummy_non_tree_batch_uses_splitfuse_mask(self):
        """bs>1 mixed capture gears are not 1+budget; 4D tree mask + sparse3 tiles fail."""
        builder = AttentionMaskBuilder(torch.device("cpu"))
        model_config = SimpleNamespace(runner_type="generate")
        with (
            patch(
                "vllm_ascend.attention.attention_mask._need_dummy_tree_mask_for_capture",
                return_value=True,
            ),
            patch(
                "vllm_ascend.attention.attention_mask._tree_query_len",
                return_value=49,
            ),
        ):
            mixed = builder.get_attention_mask(
                True,
                model_config,
                seq_lens=torch.tensor([8, 8], dtype=torch.int32),
                num_decode=2,
            )
            self.assertEqual(mixed.shape, (2048, 2048))

            tree = builder.get_attention_mask(
                True,
                model_config,
                seq_lens=torch.tensor([49, 49], dtype=torch.int32),
                num_decode=2,
            )
            self.assertEqual(tree.ndim, 4)
            self.assertEqual(tree.shape[0], 2)
            self.assertEqual(tree.shape[2], 49)

            vis = torch.eye(48, dtype=torch.bool).unsqueeze(0).expand(8, -1, -1)
            leaked = builder.get_attention_mask(
                True,
                model_config,
                tree_visibility=vis,
                seq_lens=torch.tensor([8] * 8, dtype=torch.int32),
                num_decode=8,
                num_tokens=64,
                for_capture=True,
            )
            self.assertEqual(leaked.shape, (2048, 2048))

            captured_tree = builder.get_attention_mask(
                True,
                model_config,
                tree_visibility=vis[:2],
                seq_lens=torch.tensor([49, 49], dtype=torch.int32),
                num_decode=2,
                num_tokens=98,
                for_capture=True,
            )
            self.assertEqual(captured_tree.ndim, 4)
            self.assertEqual(captured_tree.shape[2], 49)
