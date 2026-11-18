# SPDX-License-Identifier: Apache-2.0
"""Batched scoring for the DFlash2 candidate lattice."""

import torch

from vllm_ascend.ops.triton.spec_decode.tree.mega_tree import best_first_tree
from vllm_ascend.worker.v2.spec_decode.tree.dflash2_builder import DFlash2BeamTreeBuilder


def score_transitions(
    candidate_ids,
    unary_logits,
    projected_hidden,
    root_token_ids,
    predecessor_codebook,
    successor_codebook,
    child_topk,
):
    """Return contiguous child slots and normalized scores shaped [R, D, C, K].

    At depth 0 only parent row 0 is used. At each later depth, parent row j
    corresponds to column j of candidate_ids at the preceding depth.
    """
    num_reqs, depth, candidates = candidate_ids.shape
    rank = projected_hidden.shape[-1]

    # Build one batch with the same C-by-C contraction shape at each depth.
    # Padding the root rows keeps the batch regular. All root rows are equal.
    root_rows = root_token_ids.reshape(num_reqs, 1, 1).expand(-1, 1, candidates)
    parent_ids = torch.cat((root_rows, candidate_ids[:, :-1]), dim=1)
    predecessor = predecessor_codebook.index_select(0, parent_ids.reshape(-1))
    successor = successor_codebook.index_select(0, candidate_ids.reshape(-1))
    predecessor = predecessor.view(num_reqs, depth, candidates, rank)
    successor = successor.view(num_reqs * depth, candidates, rank)

    # Keep the multiply and contraction in the checkpoint dtype as in
    # DFlash2Scorer. Add the unary term only after the contraction rounds.
    weighted = predecessor * projected_hidden.unsqueeze(2)
    correction = torch.bmm(weighted.flatten(0, 1), successor.transpose(1, 2))
    correction = correction.view(num_reqs, depth, candidates, candidates)
    scores = correction.float() + unary_logits.float().unsqueeze(2)
    log_probs = torch.log_softmax(scores, dim=-1)
    values, slots = torch.topk(log_probs, child_topk, dim=-1, sorted=True)
    return slots.to(torch.int32).contiguous(), values.contiguous()


def prepare_transitions(scorer, draft_logits, root_token_ids, draft_hidden, draft_candidate_ids, child_topk):
    """Use the production scorer's checks and projection before batching rows."""
    ctx = scorer.prepare(
        None,
        draft_logits,
        root_token_ids,
        draft_hidden,
        keep_origin_chain=False,
        candidate_ids=draft_candidate_ids,
    )
    candidate_ids = ctx.candidate_ids.contiguous()
    child_slots, child_log_probs = score_transitions(
        candidate_ids,
        ctx.unary_logits,
        ctx.hidden,
        ctx.root_token_ids,
        scorer.predecessor_codebook,
        scorer.successor_codebook,
        child_topk,
    )
    return candidate_ids, child_slots, child_log_probs


class MegaTreeBuilder(DFlash2BeamTreeBuilder):
    """DFlash2 v0.1: batched edge scoring followed by one best-first layout kernel."""

    def build(self, draft_logits, out, *, root_token_ids, draft_hidden=None, draft_candidate_ids=None):
        # Only dispatch shapes measured on A2. Other graph buckets retain the
        # beam implementation, including its depth penalty and origin policy.
        if (
            draft_logits.device.type != "npu"
            or draft_logits.shape != (1, 15, 64)
            or self.budget not in (32, 48)
            or self.keep_origin_chain
            or self.depth_penalty
        ):
            return super().build(
                draft_logits,
                out,
                root_token_ids=root_token_ids,
                draft_hidden=draft_hidden,
                draft_candidate_ids=draft_candidate_ids,
            )
        ids, slots, log_probs = prepare_transitions(
            self.scorer,
            draft_logits,
            root_token_ids,
            draft_hidden,
            draft_candidate_ids,
            self.budget,
        )
        best_first_tree(
            ids,
            slots,
            log_probs,
            out.tokens,
            out.depths,
            out.parents,
            out.first_child,
            out.next_sibling,
            out.visibility,
            out_num_nodes=out.num_nodes,
        )
        # The root alone offers C64 children, so the chosen budget is always filled.
        self.num_selected_nodes = self.budget
        return out
