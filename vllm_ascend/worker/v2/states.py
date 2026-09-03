# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/states.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# This file is a part of the vllm-ascend project.
#

import torch
from vllm.v1.worker.gpu.states import RequestState
from vllm_ascend.worker.v2.spec_decode import dflash_tree_spec_enabled, tree_spec_budget


class AscendRequestState(RequestState):
    """Request state for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
    ):
        super().__init__(
            max_num_reqs,
            max_model_len,
            max_num_batched_tokens,
            num_speculative_steps,
            vocab_size,
            device,
        )
        # Ascend attention needs a torch CPU view of the upstream NumPy state.
        # Sharing storage keeps both APIs coherent without duplicate writes.
        self.num_computed_tokens_cpu: torch.Tensor = torch.from_numpy(self.num_computed_tokens_np)

        if dflash_tree_spec_enabled():
            max_nodes = tree_spec_budget(num_speculative_steps)
            if max_nodes is None:
                max_nodes = num_speculative_steps
            self.draft_tokens: torch.Tensor = torch.zeros(
                self.max_num_reqs,
                max_nodes,
                dtype=self.draft_tokens.dtype,
                device=device,
            )

            self.tree_depths: torch.Tensor = torch.zeros(
                self.max_num_reqs,
                max_nodes,
                dtype=torch.int32,
                device=device,
            )

            self.tree_parents: torch.Tensor = torch.zeros(
                self.max_num_reqs,
                max_nodes,
                dtype=torch.int32,
                device=device,
            )

            self.tree_num_nodes: torch.Tensor = torch.zeros(
                self.max_num_reqs,
                dtype=torch.int32,
                device=device,
            )

            self.tree_visibility: torch.Tensor = torch.zeros(
                self.max_num_reqs,
                max_nodes,
                max_nodes,
                dtype=torch.bool,
                device=device,
            )
            self.tree_first_child: torch.Tensor = torch.full(
                (self.max_num_reqs, max_nodes + 1),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self.tree_next_sibling: torch.Tensor = torch.full(
                (self.max_num_reqs, max_nodes + 1),
                -1,
                dtype=torch.int32,
                device=device,
            )
