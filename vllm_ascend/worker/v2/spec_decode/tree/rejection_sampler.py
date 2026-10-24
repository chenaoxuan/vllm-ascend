import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout

_PAD_TOKEN_ID = -1

# Device-keyed scalar cache for token-id verification.
_neg_one: dict[torch.device, torch.Tensor] = {}


def _scalar(cache: dict, device: torch.device, value, dtype) -> torch.Tensor:
    t = cache.get(device)
    if t is None or t.dtype != dtype:
        t = torch.tensor(value, dtype=dtype, device=device)
        cache[device] = t
    return t


def greedy_tree_reject(
    tree: TreeLayout,
    target_logits: torch.Tensor,
    num_speculative_tokens: int,
    path_node_ids: torch.Tensor,
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Greedy-verify a draft tree by token-id comparison on device.

    Walks ``tokens`` / ``first_child`` / ``next_sibling``. ``target_logits``
    is ``[batch_size, budget + 1, vocab_size]``, indexed by node id (column 0
    is the already-accepted root).

    Returns ``[num_reqs, spec_len + 1]`` accepted token ids including bonus.
    Unused slots are ``-1``. ``path_node_ids`` is filled with accepted draft
    node ids ``[num_reqs, spec_len]`` (``-1`` unused). ``sampled_token_ids`` is
    a preallocated ``[R, spec_len+1]`` buffer (filled with PAD).
    """
    from vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch import use_tree_triton

    if use_tree_triton():
        return _greedy_tree_reject_triton(
            tree,
            target_logits,
            num_speculative_tokens,
            path_node_ids=path_node_ids,
            sampled_token_ids=sampled_token_ids,
        )
    return greedy_tree_reject_torch(
        tree,
        target_logits,
        num_speculative_tokens,
        path_node_ids=path_node_ids,
        sampled_token_ids=sampled_token_ids,
    )


def _greedy_tree_reject_triton(
    tree: TreeLayout,
    target_logits: torch.Tensor,
    num_speculative_tokens: int,
    path_node_ids: torch.Tensor,
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    from vllm_ascend.ops.triton.spec_decode.tree.greedy_reject import (
        greedy_tree_reject_triton,
    )

    tokens = tree.tokens
    num_reqs = tokens.shape[0]
    spec_len = num_speculative_tokens
    target_token_ids = target_logits.argmax(dim=-1)
    sampled_token_ids = sampled_token_ids[:num_reqs, : spec_len + 1]
    sampled_token_ids.fill_(_PAD_TOKEN_ID)
    path_out = path_node_ids[:num_reqs, :spec_len]
    path_out.fill_(-1)
    greedy_tree_reject_triton(
        tokens.contiguous(),
        tree.first_child.contiguous(),
        tree.next_sibling.contiguous(),
        target_token_ids.contiguous(),
        sampled_token_ids,
        path_out,
        spec_len,
    )
    return sampled_token_ids


def greedy_tree_reject_torch(
    tree: TreeLayout,
    target_logits: torch.Tensor,
    num_speculative_tokens: int,
    path_node_ids: torch.Tensor,
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Torch fallback when tree Triton is off."""
    tokens = tree.tokens
    first_child = tree.first_child
    next_sibling = tree.next_sibling
    device = tokens.device
    num_reqs, budget = tokens.shape
    spec_len = num_speculative_tokens
    target_token_ids = target_logits.argmax(dim=-1)
    sampled_token_ids = sampled_token_ids[:num_reqs, : spec_len + 1]
    sampled_token_ids.fill_(_PAD_TOKEN_ID)
    path_out = path_node_ids[:num_reqs, :spec_len]
    path_out.fill_(-1)
    neg_one = _scalar(_neg_one, device, -1, torch.long)

    for req_idx in range(num_reqs):
        current = torch.zeros((), dtype=torch.long, device=device)
        alive = torch.ones((), dtype=torch.bool, device=device)
        for n_out in range(spec_len):
            t = target_token_ids[req_idx, current]
            sampled_token_ids[req_idx, n_out] = torch.where(
                alive, t, sampled_token_ids[req_idx, n_out]
            )
            child = first_child[req_idx, current].to(torch.long)
            found_child = neg_one
            for _ in range(budget):
                valid = alive & (found_child < 0) & (child >= 0)
                slot = (child - 1).clamp(min=0)
                found_child = torch.where(
                    valid & (tokens[req_idx, slot] == t),
                    child,
                    found_child,
                )
                child = torch.where(
                    valid & (found_child < 0),
                    next_sibling[req_idx, child.clamp(min=0)].to(torch.long),
                    child,
                )
            found = alive & (found_child >= 0)
            path_out[req_idx, n_out] = torch.where(found, found_child, path_out[req_idx, n_out])
            current = torch.where(found, found_child, current)
            alive = found
        t = target_token_ids[req_idx, current]
        sampled_token_ids[req_idx, spec_len] = torch.where(
            alive, t, sampled_token_ids[req_idx, spec_len]
        )

    return sampled_token_ids


def _pack_tree_target_logits(
    logits: torch.Tensor,
    cu_num_logits_np: np.ndarray,
    num_reqs: int,
    node_dim: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack flat ``[num_logits, V]`` target logits into ``[R, node_dim, V]``.

    Column ``j`` is target logits at tree node id ``j``. Requests shorter than
    ``node_dim`` keep ``-inf`` in the unused node columns.
    ``cu_num_logits_np`` is the host prefix-sum of per-request logit counts.
    Optional ``out`` reuses a preallocated ``[R, node_dim, V]`` buffer.
    """
    vocab = logits.shape[-1]
    # Host prefix-sum of per-request logit counts. Equal counts that merely
    # *sum* to ``R * node_dim`` (e.g. 8+10) are not a rectangular ``[R, D, V]``.
    counts = np.diff(np.asarray(cu_num_logits_np[: num_reqs + 1], dtype=np.int64))
    if counts.size == num_reqs and np.all(counts == node_dim) and logits.shape[0] == num_reqs * node_dim:
        return logits.view(num_reqs, node_dim, vocab)
    if out is None:
        packed = logits.new_full((num_reqs, node_dim, vocab), float("-inf"))
    else:
        packed = out[:num_reqs, :node_dim]
        packed.fill_(float("-inf"))
    for i in range(num_reqs):
        start = int(cu_num_logits_np[i])
        end = int(cu_num_logits_np[i + 1])
        n = min(end - start, node_dim)
        packed[i, :n] = logits[start : start + n]
    return packed

def _limit_tree_to_verified_nodes(tree, input_batch):
    """Clip each topological prefix on device, preserving reverse sibling order."""
    n_req = input_batch.num_reqs
    logits_cu = input_batch.cu_num_logits[:n_req + 1]
    query_cu = input_batch.query_start_loc[:n_req + 1]
    n_logits = logits_cu[1:] - logits_cu[:-1]
    n_query = query_cu[1:] - query_cu[:-1]
    counts = torch.minimum(tree.num_nodes[:n_req], torch.minimum(n_logits, n_query) - 1).clamp(min=0)
    first = torch.full_like(tree.first_child, -1)
    sibling = torch.full_like(tree.next_sibling, -1)
    for slot in range(tree.parents.shape[1]):
        node = slot + 1
        parent = tree.parents[:n_req, slot].long()
        valid = node <= counts
        ordered = (parent >= 0) & (parent < node)
        # Clamp the gather index and keep invalid nodes out of the links.
        parent = parent.clamp(0, slot).unsqueeze(1)
        previous = first[:n_req].gather(1, parent)
        active = (valid & ordered).unsqueeze(1)
        sibling[:n_req, node] = torch.where(active[:, 0], previous[:, 0], -1)
        first[:n_req].scatter_(1, parent, torch.where(active, node, previous))
    tree.first_child = first
    tree.next_sibling = sibling
    tree.num_nodes = counts.to(tree.num_nodes.dtype)
    return tree

class TreeRejectionSampler(RejectionSampler):
    """Verify draft-tree tokens against target argmax token IDs."""

    path_node_ids: torch.Tensor | None = None

    def __init__(self, sampler, spec_config, device: torch.device):
        super().__init__(sampler, spec_config, device)
        self.device = device
        self._path_node_ids_buf: torch.Tensor | None = None
        self._sampled_buf: torch.Tensor | None = None
        self._packed_logits_buf: torch.Tensor | None = None
        self._buf_cap_reqs = 0
        self._buf_cap_vocab = 0
        self._buf_cap_nodes = 0

    def _ensure_bufs(self, num_reqs: int, node_dim: int, vocab: int) -> None:
        device = self.device
        grow = (
            self._path_node_ids_buf is None
            or num_reqs > self._buf_cap_reqs
            or node_dim > self._buf_cap_nodes
            or vocab > self._buf_cap_vocab
            or self._path_node_ids_buf.device != device
        )
        if not grow:
            return
        self._buf_cap_reqs = max(num_reqs, self._buf_cap_reqs)
        self._buf_cap_nodes = max(node_dim, self._buf_cap_nodes)
        self._buf_cap_vocab = max(vocab, self._buf_cap_vocab)
        r, d, v = self._buf_cap_reqs, self._buf_cap_nodes, self._buf_cap_vocab
        self._path_node_ids_buf = torch.empty(
            (r, self.num_speculative_steps), dtype=torch.long, device=device
        )
        self._sampled_buf = torch.empty(
            (r, self.num_speculative_steps + 1), dtype=torch.long, device=device
        )
        self._packed_logits_buf = torch.empty(
            (r, d, v), dtype=torch.float32, device=device
        )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # topk=1 is a chain: verify against packed input_ids like DSpark,
        # not first_child / greedy_tree_reject.
        if get_ascend_config().tree_spec_config.topk <= 1:
            self.path_node_ids = None
            return super().__call__(logits, input_batch, draft_logits)

        tree = TreeLayout(
            tokens=input_batch.tree_tokens,
            depths=input_batch.tree_depths,
            parents=input_batch.tree_parents,
            num_nodes=input_batch.tree_num_nodes,
            visibility=input_batch.tree_visibility,
            first_child=input_batch.tree_first_child,
            next_sibling=input_batch.tree_next_sibling,
        )
        from vllm_ascend.worker.v2.spec_decode.tree.chain_pack import chain_layout_active

        if chain_layout_active():
            tree = _limit_tree_to_verified_nodes(tree, input_batch)
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None
        num_reqs = input_batch.num_reqs
        node_dim = tree.tokens.shape[1] + 1
        self._ensure_bufs(num_reqs, node_dim, logits.shape[-1])
        target_logits = _pack_tree_target_logits(
            logits,
            input_batch.cu_num_logits_np,
            num_reqs,
            node_dim,
            out=self._packed_logits_buf,
        )
        path_node_ids = self._path_node_ids_buf[:num_reqs]
        path_node_ids.fill_(-1)
        sampled_buf = self._sampled_buf[:num_reqs]
        sampled = greedy_tree_reject(
            tree,
            target_logits,
            self.num_speculative_steps,
            path_node_ids=path_node_ids,
            sampled_token_ids=sampled_buf,
        )
        num_sampled = (sampled != _PAD_TOKEN_ID).sum(dim=-1).to(dtype=torch.int32)
        cu = input_batch.cu_num_logits[: num_reqs + 1]
        num_logits = (cu[1:] - cu[:-1]).to(dtype=num_sampled.dtype)
        is_chunked = (
            input_batch.seq_lens[:num_reqs]
            < self.sampler.req_states.prefill_len.gpu[input_batch.idx_mapping[:num_reqs]]
        )
        zero = num_sampled.new_zeros(())
        no_sample = is_chunked | (num_logits == 0)
        num_sampled = torch.where(no_sample, zero, num_sampled)
        num_rejected = torch.where(no_sample, zero, num_logits - num_sampled)
        # Chunked prefills zero num_sampled; keep those path rows at -1 so
        # KV compact / propose() do not scramble linear prefill state.
        path_node_ids = path_node_ids.masked_fill(
            (num_sampled == 0).unsqueeze(1), -1
        )
        self.path_node_ids = path_node_ids
        input_batch.path_node_ids = path_node_ids
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
