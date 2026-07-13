#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""DSpark confidence head and request-level dynamic verify-length scheduling.

The confidence head is a single linear projection over the draft hidden state
(optionally concatenated with the Markov transition embedding) that predicts,
per draft position, the probability the draft token is accepted by the target.

The confidence scores drive a request-level "dynamic speculation" schedule:
``survival = cumprod(confidence)`` gives the ordered survival probability of
each draft position; a global top-k over all (request, position) survival
values, bounded by a caller-supplied ``budget``, distributes extra verify slots
across requests. Unlike sglang, the budget is passed in as a parameter instead
of being derived from an SPS cost table.

Ported/adapted from sglang's ``DSparkConfidenceHead``
(``models/dspark.py``) and ``schedule_verify_lens_topk``
(``speculative/dspark_components/kernels/dspark_schedule.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)


class DSparkConfidenceHead(nn.Module):
    """Per-position confidence head: ``Linear(hidden[+markov_rank] -> 1)``.

    Mirrors sglang's ``DSparkConfidenceHead``. ``forward`` returns the raw
    logit; ``apply_sts`` maps it to a calibrated probability via a temperature
    (Simplified Temperature Scaling). The temperature defaults to ``1.0`` (no
    calibration) when no STS table is provided.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        markov_rank: int,
        with_markov: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.with_markov = bool(with_markov)
        input_dim = int(hidden_size) + (int(markov_rank) if self.with_markov else 0)
        self.proj = nn.Linear(input_dim, 1, bias=bias, dtype=dtype)
        self.register_buffer(
            "sts_temperatures", torch.ones((), dtype=torch.float32), persistent=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embed_stack: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.with_markov:
            if markov_embed_stack is None:
                raise ValueError(
                    "DSparkConfidenceHead(with_markov=True) requires markov_embed_stack."
                )
            features = torch.cat(
                [hidden_states, markov_embed_stack.to(dtype=hidden_states.dtype)],
                dim=-1,
            )
        else:
            features = hidden_states
        features = features.to(dtype=self.proj.weight.dtype)
        return self.proj(features).squeeze(-1)

    def apply_sts(self, confidence_raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(confidence_raw.float() / self.sts_temperatures)


def build_confidence_head(
    config,
    *,
    dtype: torch.dtype = torch.float32,
) -> Optional[DSparkConfidenceHead]:
    """Build a confidence head from a DSpark draft ``hf_config``.

    Returns ``None`` when the head is disabled via ``enable_confidence_head``.
    """
    if not bool(getattr(config, "enable_confidence_head", True)):
        return None
    hidden_size = int(config.hidden_size)
    markov_rank = int(getattr(config, "markov_rank", 0))
    with_markov = bool(getattr(config, "confidence_head_with_markov", markov_rank > 0))
    if with_markov and markov_rank <= 0:
        raise ValueError(
            "DSpark confidence_head_with_markov requires markov_rank > 0, "
            f"got markov_rank={markov_rank}."
        )
    return DSparkConfidenceHead(
        hidden_size=hidden_size,
        markov_rank=markov_rank,
        with_markov=with_markov,
        dtype=dtype,
    )


@dataclass
class DSparkScheduleConfig:
    """Bounds for the request-level dynamic verify-length schedule.

    ``max_verify_len == 0`` resolves to ``gamma`` (the number of draft tokens).
    """

    min_verify_len: int = 1
    max_verify_len: int = 0
    survival_eps: float = 1e-6

    def resolved_max_verify_len(self, gamma: int) -> int:
        return self.max_verify_len if self.max_verify_len > 0 else int(gamma)


def compute_survival(confidence: torch.Tensor) -> torch.Tensor:
    """Ordered survival probability: ``cumprod(confidence, dim=1)``."""
    return torch.cumprod(confidence.to(torch.float32), dim=1)


def schedule_verify_lens_topk(
    *,
    confidence: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    """Per-request verify length from confidence survival and a token budget.

    A global top-``budget`` selection over the ``[num_requests, max_len]``
    survival window (tie-break: smaller position first, then smaller request)
    distributes ``budget`` extra verify slots; each request's verify length is
    ``clamp(min_verify_len + selected, min_verify_len, max_len)``.

    Adapted from sglang's ``schedule_verify_lens_topk_from_survival``; ``budget``
    is a caller-supplied parameter (no SPS cost table).
    """
    survival_probs = compute_survival(confidence)
    num_requests, gamma = survival_probs.shape
    max_len = cfg.resolved_max_verify_len(gamma)
    device = survival_probs.device

    selected_extra = torch.zeros(num_requests, dtype=torch.int64, device=device)
    if budget > 0:
        candidate_window = survival_probs[:, :max_len]
        num_candidates = candidate_window.numel()
        if num_candidates > 0:
            request_index = (
                torch.arange(num_requests, device=device)
                .view(num_requests, 1)
                .expand_as(candidate_window)
            )
            position_index = (
                torch.arange(candidate_window.shape[1], device=device)
                .view(1, candidate_window.shape[1])
                .expand_as(candidate_window)
            )
            valid = candidate_window >= cfg.survival_eps

            flat_prob = candidate_window.reshape(-1).to(torch.float64)
            flat_request = request_index.reshape(-1)
            flat_position = position_index.reshape(-1)
            flat_valid = valid.reshape(-1)

            order = _value_independent_descending_order(
                probs=flat_prob,
                positions=flat_position,
                requests=flat_request,
                valid=flat_valid,
            )

            take = min(int(budget), num_candidates)
            chosen = order[:take]
            chosen_requests = flat_request[chosen]
            chosen_valid = flat_valid[chosen].to(torch.int64)
            selected_extra.scatter_add_(0, chosen_requests, chosen_valid)

    min_len = torch.full(
        (num_requests,), cfg.min_verify_len, dtype=torch.int64, device=device
    )
    verify_lens = min_len + selected_extra
    lower_bound = max(cfg.min_verify_len, 1)
    verify_lens = torch.clamp(verify_lens, min=lower_bound, max=max_len)
    return verify_lens.to(torch.int32)


def _value_independent_descending_order(
    *,
    probs: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    masked_prob = torch.where(valid, probs, torch.full_like(probs, float("-inf")))
    num_candidates = masked_prob.numel()
    order = torch.arange(num_candidates, device=probs.device)
    order = order[torch.argsort(requests[order], stable=True)]
    order = order[torch.argsort(positions[order], stable=True)]
    order = order[torch.argsort(-masked_prob[order], stable=True)]
    return order
