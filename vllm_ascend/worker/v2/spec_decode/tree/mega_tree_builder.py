# SPDX-License-Identifier: Apache-2.0
"""MegaTree construction for the DFlash2 candidate lattice."""

import torch
from vllm.logger import logger

from vllm_ascend.worker.v2.spec_decode.tree.dflash2_builder import DFlash2BeamTreeBuilder

_MAX_BUDGET = 128


class MegaTreeBuilder(DFlash2BeamTreeBuilder):
    """Use fused operand preparation and a per-request cursor queue."""

    def build(self, draft_logits, out, *, root_token_ids, draft_hidden=None, draft_candidate_ids=None):
        scorer = self.scorer
        codebooks = (scorer.predecessor_codebook, scorer.successor_codebook)
        if (
            draft_logits.device.type != "npu"
            or draft_logits.shape[1:] != (15, 64)
            or self.budget > _MAX_BUDGET
            or out.tokens.shape != (draft_logits.shape[0], self.budget)
            or self.keep_origin_chain
            or self.depth_penalty
            or draft_hidden is None
            or draft_hidden.dtype != torch.bfloat16
            or any(
                book.ndim != 2 or book.shape[-1] != 256 or book.dtype != torch.bfloat16 or not book.is_contiguous()
                for book in codebooks
            )
        ):
            logger.debug_once(
                "MegaTree using the DFlash2 beam builder: depth=%d candidates=%d budget=%d "
                "device=%s hidden_dtype=%s codebooks=%s depth_penalty=%s keep_origin_chain=%s",
                draft_logits.shape[1],
                draft_logits.shape[2],
                self.budget,
                draft_logits.device.type,
                draft_hidden.dtype if draft_hidden is not None else None,
                tuple((tuple(book.shape), book.dtype, book.is_contiguous()) for book in codebooks),
                self.depth_penalty,
                self.keep_origin_chain,
            )
            return super().build(
                draft_logits,
                out,
                root_token_ids=root_token_ids,
                draft_hidden=draft_hidden,
                draft_candidate_ids=draft_candidate_ids,
            )
        # The NPU-only module uses Ascend's indexed codebook gather extension.
        from vllm_ascend.ops.triton.spec_decode.tree.mega_tree import build_tree

        logger.debug_once(
            "MegaTree cursor search: depth=%d candidates=%d budget=%d", *draft_logits.shape[1:], self.budget
        )
        ctx = scorer.prepare(
            None,
            draft_logits,
            root_token_ids,
            draft_hidden,
            keep_origin_chain=False,
            candidate_ids=draft_candidate_ids,
        )
        build_tree(
            ctx.candidate_ids.contiguous(),
            ctx.unary_logits.contiguous(),
            ctx.hidden.contiguous(),
            ctx.root_token_ids.contiguous(),
            *codebooks,
            out,
        )
        self.num_selected_nodes = self.budget
        return out
