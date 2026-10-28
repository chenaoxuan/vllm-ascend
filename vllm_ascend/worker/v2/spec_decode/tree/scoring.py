"""Model scoring adapters. Device tensors never determine host control flow."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_ascend.ops.vocab_parallel_embedding import AscendVocabParallelEmbedding, VocabParallelMode


def select_candidates(logits, width, *, keep_origin_chain=False):
    """Full-vocabulary normalized edge scores; IDs remain in the input vocab."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    scores, ids = log_probs.topk(width, dim=-1)
    if keep_origin_chain:
        origin = logits.argmax(-1)
        present = (ids == origin.unsqueeze(-1)).any(-1)
        ids[..., -1] = torch.where(present, ids[..., -1], origin)
        scores[..., -1] = log_probs.gather(-1, ids[..., -1:])[..., 0]
        origin_child = (ids == origin.unsqueeze(-1)).long().argmax(-1)
    else:
        origin_child = None
    return scores, ids, origin_child


@dataclass
class CandidateLayer:
    scores: torch.Tensor
    token_ids: torch.Tensor
    origin_child: torch.Tensor | None = None


class IndependentScorer:
    """DFlash depth logits without a correction head."""

    name = "independent"
    # Preserve the original prefix/DFlash pruning preference for shallow nodes.
    # This biases global selection only, never edge normalization or expansion.
    default_depth_penalty = 0.2
    # Kept for callers that inspected the scorer-specific setting directly.
    depth_penalty = default_depth_penalty

    def reserve(self, num_reqs, num_nodes, device):
        pass

    def workspace_tensors(self):
        return ()

    def prepare(self, state, logits, roots, hidden, *, keep_origin_chain=False):
        # Independent depths share no parent state. Score them in one batch,
        # preserving the original DFlash kernel count and contiguous inputs.
        scores, ids, origin = select_candidates(logits, state.width, keep_origin_chain=keep_origin_chain)
        return CandidateLayer(scores, ids, origin)

    def score(self, state, ctx, depth_slot, keep_origin_chain):
        scores, ids = ctx.scores[:, depth_slot], ctx.token_ids[:, depth_slot]
        origin = None if ctx.origin_child is None else ctx.origin_child[:, depth_slot]
        return CandidateLayer(scores.unsqueeze(1), ids.unsqueeze(1), None if origin is None else origin.unsqueeze(1))

    def update(self, state, ctx, num_added):
        pass


@dataclass
class MarkovContext:
    logits: torch.Tensor
    frontier_tokens: torch.Tensor


class MarkovScorer(IndependentScorer):
    """DSpark correction in draft vocabulary, followed by target-ID mapping."""

    name = "markov"
    default_depth_penalty = 0.0
    depth_penalty = default_depth_penalty

    def __init__(self, model):
        self.model = model

    def prepare(self, state, logits, roots, hidden, *, keep_origin_chain=False):
        return MarkovContext(logits, roots.unsqueeze(1))

    def score(self, state, ctx, depth_slot, keep_origin_chain):
        logits = ctx.logits[:, depth_slot].unsqueeze(1)
        if self.model is not None:
            emb = self.model.markov_embed(ctx.frontier_tokens)
            bias = self.model.markov_bias(emb.reshape(-1, emb.shape[-1]))
            logits = logits + bias.view(*emb.shape[:-1], -1)
        scores, ids, origin = select_candidates(logits, state.width, keep_origin_chain=keep_origin_chain)
        if self.model is not None:
            ids = self.model.map_draft_to_target(ids)
        return CandidateLayer(scores, ids, origin)

    def update(self, state, ctx, num_added):
        ctx.frontier_tokens = state.tokens[:, state.num_nodes : state.num_nodes + num_added]


class DominoCorrectionScorer:
    """Domino correction head scorer for prefix (DARTree) construction."""

    def __init__(self, draft_model, target_model, hidden_dim=None):
        fc1, self.middle, fc2 = self._unwrap_correction_mlp(draft_model.embed_proj)
        gru = draft_model.prefix_gru
        if gru.num_layers != 1 or gru.bidirectional:
            raise ValueError("Domino correction requires a single-layer unidirectional prefix_gru")
        self.gru_hidden_dim = int(gru.hidden_size)
        if hidden_dim is None:
            hidden_dim = fc1.in_features - self.gru_hidden_dim
        self.hidden_dim = int(hidden_dim)
        if fc1.in_features != self.hidden_dim + self.gru_hidden_dim:
            raise ValueError(
                "Domino correction MLP input dim mismatch: got "
                f"{fc1.in_features}, expected {self.hidden_dim + self.gru_hidden_dim}"
            )
        self.w_z = fc1.weight[:, : self.hidden_dim].detach().contiguous()
        self.w_s = fc1.weight[:, self.hidden_dim :].detach().contiguous()
        self.fc1_bias = fc1.bias.detach() if fc1.bias is not None else None
        self.fc2_weight = fc2.weight.detach().contiguous()
        self.fc2_bias = fc2.bias.detach() if fc2.bias is not None else None
        self.gru_w_ih = gru.weight_ih_l0.detach().contiguous()
        self.gru_w_hh = gru.weight_hh_l0.detach().contiguous()
        self.gru_b_ih = gru.bias_ih_l0.detach() if gru.bias else None
        self.gru_b_hh = gru.bias_hh_l0.detach() if gru.bias else None
        embedding = target_model.model.embed_tokens
        table = F.linear(embedding.weight, self.gru_w_ih, self.gru_b_ih).contiguous()
        if isinstance(embedding, AscendVocabParallelEmbedding):
            # Model-load initialization only: hot-path token IDs stay global.
            # Project locally before gathering the smaller GRU gate table.
            if embedding.parallel_mode is VocabParallelMode.PCP_X_TP:
                raise NotImplementedError("Domino correction does not support PCP-sharded target embeddings")
            if embedding.tp_size > 1:
                group = (
                    embedding.output_reduce_group
                    if embedding.parallel_mode is VocabParallelMode.STANDARD
                    else embedding.token_exchange_group
                )
                if group is None or group.world_size != embedding.tp_size:
                    raise RuntimeError("Domino target embedding has no matching vocabulary-shard group")
                table = group.all_gather(table, dim=0)
            if table.shape[0] != embedding.num_embeddings_padded:
                raise RuntimeError("Domino target embedding gather did not reconstruct the padded vocabulary")
            mapping = embedding.get_sharded_to_full_mapping()
            if mapping is None and embedding.num_added_embeddings:
                # The upstream mapping is None for TP1; base padding still
                # separates the original vocabulary from added-token rows.
                mapping = list(range(embedding.org_vocab_size)) + list(
                    range(
                        embedding.org_vocab_size_padded,
                        embedding.org_vocab_size_padded + embedding.num_added_embeddings,
                    )
                )
            mapping = None if mapping is None else mapping[: embedding.num_embeddings]
            if mapping is not None and any(index != token for token, index in enumerate(mapping)):
                # One explicit model-load H2D transfer; no runtime lookup copy.
                index = torch.tensor(mapping, dtype=torch.long, device=table.device)
                table = table.index_select(0, index)
            else:
                table = table[: embedding.num_embeddings]
        elif not isinstance(embedding, nn.Embedding):
            raise TypeError("Domino correction requires an Ascend vocabulary-parallel or full nn.Embedding table")
        self._gru_input_proj_table = table.contiguous()

    @staticmethod
    def _unwrap_correction_mlp(embed_proj):
        if not isinstance(embed_proj, nn.Sequential):
            raise TypeError("Domino correction requires draft_model.embed_proj to be nn.Sequential")
        modules = list(embed_proj.children())
        if len(modules) < 2:
            raise ValueError("embed_proj must contain at least two layers")
        if not isinstance(modules[0], nn.Linear) or not isinstance(modules[-1], nn.Linear):
            raise ValueError("embed_proj first/last layers must be nn.Linear")
        fc1 = modules[0]
        fc2 = modules[-1]
        middle = nn.Sequential(*modules[1:-1]) if len(modules) > 2 else nn.Identity()
        return fc1, middle, fc2

    def project_z(self, parallel_hiddens: torch.Tensor) -> torch.Tensor:
        return F.linear(parallel_hiddens, self.w_z, self.fc1_bias)

    def update_hidden(self, token_ids: torch.Tensor, h_state: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.reshape(-1)
        gi = self._gru_input_proj_table.index_select(0, token_ids)
        gh = F.linear(h_state, self.gru_w_hh, self.gru_b_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h_state


def _prefix_corrected_candidates(
    scorer,
    z: torch.Tensor,
    parent_hidden: torch.Tensor,
    candidate_vals: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_weight: torch.Tensor,
    candidate_bias: torch.Tensor | None,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Domino shortlist correction for one depth.

    Returns the corrected top-k scores and token IDs over the base top-C set.
    """
    s_proj = F.linear(parent_hidden, scorer.w_s, None)
    mid = scorer.middle(z.unsqueeze(1) + s_proj)
    bias = torch.einsum("rwm,rcm->rwc", mid, candidate_weight)
    if candidate_bias is not None:
        bias = bias + candidate_bias.unsqueeze(1)
    candidate_logits = candidate_vals.unsqueeze(1).to(bias.dtype) + bias
    candidate_logits = candidate_logits.float()
    top_vals, top_ids = torch.topk(candidate_logits, k=k, dim=-1)
    log_z = torch.logsumexp(candidate_logits, dim=-1, keepdim=True)
    top_scores = top_vals - log_z
    width = parent_hidden.size(1)
    cand_ids = candidate_ids.unsqueeze(1).expand(-1, width, -1)
    sel_ids = torch.gather(cand_ids, 2, top_ids)
    return top_scores, sel_ids


@dataclass
class DominoContext:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    candidate_vals: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_weight: torch.Tensor
    candidate_bias: torch.Tensor | None
    z_parts: torch.Tensor


class DominoScorer(IndependentScorer):
    """Top-C correction and GRU state, independent of tree selection policy."""

    name = "domino"

    def __init__(self, correction, prefix_len=0, candidate_size=64):
        self.correction = correction
        self.prefix_len = prefix_len
        self.candidate_size = candidate_size
        self._hidden = None

    def reserve(self, num_reqs, num_nodes, device):
        scorer = self.correction
        shape = (num_reqs, num_nodes, scorer.gru_hidden_dim)
        if self._hidden is None or self._hidden.shape != shape or self._hidden.device != device:
            self._hidden = torch.empty(shape, dtype=scorer.gru_w_hh.dtype, device=device)

    def workspace_tensors(self):
        return () if self._hidden is None else (self._hidden,)

    def prepare(self, state, logits, roots, hidden, *, keep_origin_chain=False):
        if hidden is None:
            raise ValueError("Domino tree scoring requires draft hidden states")
        scorer = self.correction
        r, depth, vocab = logits.shape
        c = state.width if self.candidate_size is None else max(state.width, min(self.candidate_size, vocab))
        vals, ids = torch.topk(logits.float(), k=c, dim=-1)
        flat = ids.reshape(-1)
        weight = scorer.fc2_weight.index_select(0, flat).view(r, depth, c, -1)
        bias = None if scorer.fc2_bias is None else scorer.fc2_bias.index_select(0, flat).view(r, depth, c)
        states = self._hidden[:r, : state.parents.shape[1]]
        states.zero_()
        states[:, 0] = scorer.update_hidden(roots.reshape(-1), states[:, 0])
        return DominoContext(logits, states, vals, ids, weight, bias, scorer.project_z(hidden[:, :depth]))

    def score(self, state, ctx, depth_slot, keep_origin_chain):
        if depth_slot < self.prefix_len:
            scores, ids, origin = select_candidates(
                ctx.logits[:, depth_slot], state.width, keep_origin_chain=keep_origin_chain
            )
            return CandidateLayer(
                scores.unsqueeze(1), ids.unsqueeze(1), None if origin is None else origin.unsqueeze(1)
            )
        h = self.correction.gru_hidden_dim
        parent_hidden = ctx.hidden_states.gather(1, state.frontier.unsqueeze(-1).expand(-1, -1, h))
        scores, ids = _prefix_corrected_candidates(
            self.correction,
            ctx.z_parts[:, depth_slot],
            parent_hidden,
            ctx.candidate_vals[:, depth_slot],
            ctx.candidate_ids[:, depth_slot],
            ctx.candidate_weight[:, depth_slot],
            None if ctx.candidate_bias is None else ctx.candidate_bias[:, depth_slot],
            state.width,
        )
        # The shortlist defines Domino's candidate distribution. Its best
        # retained edge is the greedy edge of that same distribution.
        origin = scores.argmax(-1) if keep_origin_chain else None
        return CandidateLayer(scores, ids, origin)

    def update(self, state, ctx, num_added):
        h = self.correction.gru_hidden_dim
        start = state.num_nodes + 1
        parents = state.parents[:, start : start + num_added]
        previous = ctx.hidden_states.gather(1, parents.unsqueeze(-1).expand(-1, -1, h))
        tokens = state.tokens[:, state.num_nodes : state.num_nodes + num_added]
        updated = self.correction.update_hidden(tokens.reshape(-1), previous.reshape(-1, h))
        ctx.hidden_states[:, start : start + num_added] = updated.reshape(tokens.shape[0], num_added, h)
