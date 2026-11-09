# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from Mterzzl/vllm-ascend-asd at 3d47910. Uses the V1 flat draft layout.

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _select_asd_prefix(
    draft_ids,
    target_ids,
    target_logits,
    bonus_ids,
    cu_draft,
    request_meta,
    spent_budget,
    output,
    LOGIT_STRIDE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    BUDGET: tl.constexpr,
    GATE: tl.constexpr,
    CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(cu_draft + row - 1, row > 0, other=0)
    end = tl.load(cu_draft + row)
    length = end - start
    slot = tl.load(request_meta + row * 3)
    fresh = tl.load(request_meta + row * 3 + 1) != 0
    commit = tl.load(request_meta + row * 3 + 2) != 0
    spent = tl.load(spent_budget + slot, ~fresh, other=0.0)
    remaining = tl.maximum(BUDGET - spent, 0.0)
    cumulative = 0.0
    charged = spent
    mismatches = 0
    accepted = 0
    active = True
    cols = tl.arange(0, BLOCK)
    tl.store(output + row * OUTPUT_STRIDE + cols, -1, cols < OUTPUT_STRIDE)

    for i in range(length):
        if active:
            candidate = tl.load(draft_ids + start + i)
            target = tl.load(target_ids + start + i)
            mismatch = candidate != target
            admissible = True
            regret = 0.0
            if mismatch:
                if (BUDGET == 0.0) | (GATE == 0.0) | (CAP == 0):
                    admissible = False
                else:
                    logits = target_logits + (start + i).to(tl.int64) * LOGIT_STRIDE
                    top = tl.load(logits + target).to(tl.float32)
                    valid = (candidate >= 0) & (candidate < VOCAB_SIZE)
                    proposed = tl.load(logits + candidate, valid, other=-float("inf")).to(tl.float32)
                    regret = top - proposed
                    cumulative += regret
                    mismatches += 1
                    value = (length - i).to(tl.float32)
                    admissible = (regret >= 0.0) & (regret < float("inf"))
                    admissible = admissible & (regret / value <= GATE) & (cumulative <= remaining + 1.0e-9)
                    admissible = admissible & (mismatches <= CAP)
            if admissible:
                tl.store(output + row * OUTPUT_STRIDE + i, candidate)
                accepted = i + 1
                charged += regret
            else:
                active = False

    if accepted < length:
        next_token = tl.load(target_ids + start + accepted).to(tl.int32)
    else:
        next_token = tl.load(bonus_ids + row).to(tl.int32)
    tl.store(output + row * OUTPUT_STRIDE + accepted, next_token)
    tl.store(spent_budget + slot, tl.where(commit, charged, spent))


def select_asd_prefix(
    draft_ids,
    target_logits,
    bonus_ids,
    cu_draft,
    request_meta,
    spent_budget,
    max_draft_tokens,
    *,
    budget,
    gate,
    mismatch_cap,
):
    batch = cu_draft.numel()
    output = torch.empty((batch, max_draft_tokens + 1), dtype=torch.int32, device=draft_ids.device)
    if batch:
        target_ids = target_logits.argmax(dim=-1) if draft_ids.numel() else draft_ids
        _select_asd_prefix[(batch,)](
            draft_ids,
            target_ids,
            target_logits,
            bonus_ids,
            cu_draft,
            request_meta,
            spent_budget,
            output,
            target_logits.stride(0),
            target_logits.size(1),
            output.stride(0),
            budget,
            gate,
            min(mismatch_cap, max_draft_tokens),
            triton.next_power_of_2(max_draft_tokens + 1),
            num_warps=1,
        )
    return output
