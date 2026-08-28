import torch

from vllm_ascend.worker.v2.spec_decode.tree.utils import TreeLayout

_PAD_TOKEN_ID = -1


def greedy_tree_reject(
    tree: TreeLayout,
    target_logits: torch.Tensor,
    num_speculative_tokens: int,
) -> torch.Tensor:
    """Greedy-verify a draft tree by token-id comparison on device.

    ``target_logits`` is ``[batch_size, budget + 1, vocab_size]``, indexed by
    node id (column 0 is the already-accepted root).

    Returns ``[num_reqs, spec_len + 1]`` accepted token ids including bonus.
    Unused slots are ``-1``.
    """
    tokens = tree.tokens
    parents = tree.parents
    num_nodes = tree.num_nodes
    device = tokens.device
    num_reqs, budget = tokens.shape
    spec_len = num_speculative_tokens
    target_token_ids = target_logits.argmax(dim=-1)
    sampled_token_ids = torch.full(
        (num_reqs, spec_len + 1),
        _PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    node_ids = torch.arange(1, budget + 1, dtype=torch.long, device=device)
    neg_one = torch.tensor(-1, dtype=torch.long, device=device)

    for req_idx in range(num_reqs):  # tl.program_id(0)
        current = torch.zeros((), dtype=torch.long, device=device)
        alive = torch.ones((), dtype=torch.bool, device=device)
        for n_out in range(spec_len):
            t = target_token_ids[req_idx, current]
            sampled_token_ids[req_idx, n_out] = torch.where(
                alive, t, sampled_token_ids[req_idx, n_out]
            )
            child = neg_one
            # TODO: scan all budget slots because child_maps is a host dict
            # and cannot be indexed on device. Pack it into a sibling tensor
            # so the inner loop only walks children of `current`.
            for slot in range(budget):
                hit = (
                    alive
                    & (child < 0)
                    & (num_nodes[req_idx] > slot)
                    & (parents[req_idx, slot] == current)
                    & (tokens[req_idx, slot] == t)
                )
                child = torch.where(hit, node_ids[slot], child)
            found = alive & (child >= 0)
            current = torch.where(found, child, current)
            alive = found
        t = target_token_ids[req_idx, current]
        sampled_token_ids[req_idx, spec_len] = torch.where(
            alive, t, sampled_token_ids[req_idx, spec_len]
        )

    return sampled_token_ids
