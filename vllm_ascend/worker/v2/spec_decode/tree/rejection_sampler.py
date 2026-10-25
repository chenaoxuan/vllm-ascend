import torch
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout

_PAD_TOKEN_ID = -1


def greedy_tree_reject(
    tree: TreeLayout,
    target_token_ids: torch.Tensor,
    cu_num_logits: torch.Tensor,
    verified_nodes: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    acceptance: TreeAcceptance,
) -> None:
    """Verify flat target argmax IDs, writing the supplied device result buffers.

    Target rows are addressed through ``cu_num_logits``; no padded [R, B, V]
    logits are materialized. ``verified_nodes`` bounds access to each request's
    actual query/logit prefix. Existing sibling order (including ties) is kept.
    """
    if target_token_ids.device.type == "npu":
        from vllm_ascend.ops.triton.spec_decode.tree.greedy_reject import greedy_tree_reject_triton

        greedy_tree_reject_triton(
            tree.tokens,
            tree.first_child,
            tree.next_sibling,
            target_token_ids,
            cu_num_logits,
            verified_nodes,
            sampled_token_ids,
            acceptance.path_node_ids,
            acceptance.num_sampled,
            acceptance.num_accepted,
            acceptance.endpoint_node,
            acceptance.active,
        )
    else:
        greedy_tree_reject_torch(tree, target_token_ids, cu_num_logits, verified_nodes, sampled_token_ids, acceptance)


def greedy_tree_reject_torch(
    tree: TreeLayout,
    target_token_ids: torch.Tensor,
    cu_num_logits: torch.Tensor,
    verified_nodes: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    acceptance: TreeAcceptance,
) -> None:
    """Tensor reference with no device-to-host reads or data-dependent Python."""
    num_reqs, budget = tree.tokens.shape
    spec_len = acceptance.path_node_ids.shape[1]
    device = tree.tokens.device
    sampled_token_ids.fill_(_PAD_TOKEN_ID)
    acceptance.path_node_ids.fill_(-1)
    current = torch.zeros(num_reqs, dtype=torch.long, device=device)
    alive = acceptance.active.clone()
    num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    num_accepted = torch.zeros_like(num_sampled)
    starts = cu_num_logits[:num_reqs].long()
    # An inactive row may have no target logits at all. One sentinel lets the
    # tensor reference gather safely without inspecting a device predicate.
    target_ids = torch.cat((target_token_ids, target_token_ids.new_zeros(1)))
    target_last = target_token_ids.numel()
    rows = torch.arange(num_reqs, device=device)

    for step in range(spec_len):
        target_index = torch.where(alive, starts + current, target_last)
        target = target_ids[target_index]
        sampled_token_ids[:, step] = torch.where(alive, target, _PAD_TOKEN_ID)
        num_sampled += alive.to(torch.int32)
        child = tree.first_child[rows, current].long()
        found = torch.full_like(current, -1)
        for _ in range(budget):
            linked = alive & (found < 0) & (child > 0) & (child <= budget)
            eligible = linked & (child <= verified_nodes)
            token = tree.tokens[rows, (child - 1).clamp(0, max(budget - 1, 0))]
            match = eligible & (token == target)
            found = torch.where(match, child, found)
            next_child = tree.next_sibling[rows, child.clamp(0, budget)].long()
            child = torch.where(linked & ~match, next_child, child)
        matched = alive & (found > 0)
        acceptance.path_node_ids[:, step] = torch.where(matched, found, -1)
        current = torch.where(matched, found, current)
        num_accepted += matched.to(torch.int32)
        alive = matched

    target_index = torch.where(alive, starts + current, target_last)
    target = target_ids[target_index]
    sampled_token_ids[:, spec_len] = torch.where(alive, target, _PAD_TOKEN_ID)
    num_sampled += alive.to(torch.int32)
    acceptance.num_sampled.copy_(num_sampled)
    acceptance.num_accepted.copy_(num_accepted)
    acceptance.endpoint_node.copy_(current)


class TreeRejectionSampler(RejectionSampler):
    """Greedy tree verification with shared accepted-path result buffers."""

    def __init__(self, sampler, spec_config, device: torch.device):
        super().__init__(sampler, spec_config, device)
        self.device = device
        self.acceptance: TreeAcceptance | None = None
        self.path_node_ids: torch.Tensor | None = None
        self._result_buf: TreeAcceptance | None = None
        self._sampled_buf: torch.Tensor | None = None
        self._buf_cap_reqs = 0

    def _ensure_bufs(self, num_reqs: int) -> None:
        if self._result_buf is not None and num_reqs <= self._buf_cap_reqs:
            return
        self._buf_cap_reqs = max(num_reqs, self._buf_cap_reqs)
        num_reqs = self._buf_cap_reqs
        device = self.device
        self._result_buf = TreeAcceptance(
            path_node_ids=torch.empty((num_reqs, self.num_speculative_steps), dtype=torch.long, device=device),
            num_sampled=torch.empty(num_reqs, dtype=torch.int32, device=device),
            num_accepted=torch.empty(num_reqs, dtype=torch.int32, device=device),
            endpoint_node=torch.empty(num_reqs, dtype=torch.long, device=device),
            active=torch.empty(num_reqs, dtype=torch.bool, device=device),
        )
        self._sampled_buf = torch.empty((num_reqs, self.num_speculative_steps + 1), dtype=torch.long, device=device)

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
        invalid_drafts: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # Chain sampling retains upstream stochastic rejection semantics.
        if get_ascend_config().tree_spec_config.topk <= 1:
            self.acceptance = None
            self.path_node_ids = None
            input_batch.path_node_ids = None
            input_batch.tree_acceptance = None
            return super().__call__(logits, input_batch, draft_logits, invalid_drafts)

        tree = TreeLayout(
            tokens=input_batch.tree_tokens,
            depths=input_batch.tree_depths,
            parents=input_batch.tree_parents,
            num_nodes=input_batch.tree_num_nodes,
            visibility=input_batch.tree_visibility,
            first_child=input_batch.tree_first_child,
            next_sibling=input_batch.tree_next_sibling,
        )
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None
        num_reqs = input_batch.num_reqs
        self._ensure_bufs(num_reqs)
        buf = self._result_buf
        acceptance = TreeAcceptance(
            buf.path_node_ids[:num_reqs],
            buf.num_sampled[:num_reqs],
            buf.num_accepted[:num_reqs],
            buf.endpoint_node[:num_reqs],
            buf.active[:num_reqs],
        )
        cu = input_batch.cu_num_logits[: num_reqs + 1]
        num_logits = cu[1:] - cu[:-1]
        query_cu = input_batch.query_start_loc[: num_reqs + 1]
        is_chunked = (
            input_batch.seq_lens[:num_reqs]
            < self.sampler.req_states.prefill_len.gpu[input_batch.idx_mapping[:num_reqs]]
        )
        acceptance.active.copy_(~is_chunked & (num_logits > 0) & (query_cu[1:] > query_cu[:-1]))
        verified_nodes = tree.verified_node_counts(query_cu, cu)
        sampled = self._sampled_buf[:num_reqs]
        greedy_tree_reject(tree, logits.argmax(dim=-1), cu, verified_nodes, sampled, acceptance)
        num_rejected = torch.where(acceptance.active, num_logits - acceptance.num_sampled, 0).to(torch.int32)
        self.acceptance = acceptance
        self.path_node_ids = acceptance.path_node_ids
        input_batch.path_node_ids = acceptance.path_node_ids
        input_batch.tree_acceptance = acceptance
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=num_nans,
            num_sampled=acceptance.num_sampled,
            num_rejected=num_rejected,
        )
