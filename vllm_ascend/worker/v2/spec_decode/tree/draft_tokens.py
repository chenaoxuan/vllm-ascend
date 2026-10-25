# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publish synchronous tree proposals independently of draft-chain depth."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch

    from vllm_ascend.worker.v2.spec_decode.tree.speculator import AscendTreeSpeculator
    from vllm_ascend.worker.v2.states import AscendRequestState


class TreeDraftTokensHandler(DraftTokensHandler):
    """Reuse upstream publication and its single structured-output copy.

    Request-state tokens remain on device. The builder's selected-node count
    is uniform across requests and comes from host shapes, including during
    graph capture; it never reads the device ``tree.num_nodes`` buffer.
    """

    def __init__(
        self,
        device: torch.device,
        req_states: AscendRequestState,
        speculator: AscendTreeSpeculator,
    ) -> None:
        super().__init__(device)
        self.req_states = req_states
        self.speculator = speculator

    def set_draft_tokens(self, input_batch: InputBatch, draft_tokens: torch.Tensor) -> None:
        # MRV2 passes only K columns after #57053. The completed tree proposal
        # is already in request state, where storage retains its B-column width.
        num_nodes = self.speculator.tree_shape.validate_selected_nodes(self.speculator.tree_builder.num_selected_nodes)
        tokens = self.req_states.draft_tokens[input_batch.idx_mapping, :num_nodes]
        super().set_draft_tokens(input_batch, tokens)
