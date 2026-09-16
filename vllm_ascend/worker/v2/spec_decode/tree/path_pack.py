from dataclasses import dataclass

import torch
from vllm.logger import init_logger

from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import (
    compact_tree_query_along_path,
)

logger = init_logger("vllm." + __name__)


@dataclass
class Dsv4PathPack:
    """Leaf-path TND layout for DSV4 tree target verify.

    DSA has no working tree mask. Each root-to-leaf path is one causal TND
    sequence so sibling queries do not share a packed-causal SWA stream.
    Token tensors are device; ``num_tokens`` / ``num_paths`` are host counts
    taken at the prepare_inputs boundary.
    """

    num_tokens: int
    num_paths: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    token_req: torch.Tensor
    token_node: torch.Tensor
    node_row: torch.Tensor
    path_query_start_loc: torch.Tensor
    path_seq_lens: torch.Tensor
    path_req_idx: torch.Tensor
    req_query_start_loc: torch.Tensor
    prefix_lens: torch.Tensor
    path_qlens: torch.Tensor
    token_path: torch.Tensor
    token_col: torch.Tensor
    num_reqs: int


def build_dsv4_path_pack(
    tokens: torch.Tensor,
    parents: torch.Tensor,
    first_child: torch.Tensor,
    num_nodes: torch.Tensor,
    depths: torch.Tensor,
    root_token_ids: torch.Tensor,
    prefix_lens: torch.Tensor,
    spec_len: int,
) -> Dsv4PathPack | None:
    """Enumerate beam leaves and pack each root-to-leaf path as a causal chain.

    ``tokens`` / ``parents`` / ``depths`` are ``[R, budget]``. ``first_child``
    is ``[R, budget+1]``. ``num_nodes`` / ``root_token_ids`` / ``prefix_lens``
    are ``[R]``. All device. ``spec_len`` is a host int.
    """
    num_reqs, budget = tokens.shape
    device = tokens.device
    node_dim = budget + 1
    node_ids = torch.arange(node_dim, device=device)
    # Topology only. Scheduler draft pads are -1 placeholders and must not
    # drop real tree nodes; unused budget slots are already outside num_nodes.
    present = node_ids.unsqueeze(0) <= num_nodes.to(dtype=node_ids.dtype).unsqueeze(1)
    present[:, 0] = True
    is_leaf = (first_child < 0) & present
    n_leaf = int(is_leaf.sum().item())  # D2H
    if n_leaf <= 0:
        return None

    leaf_req, leaf_node = is_leaf.nonzero(as_tuple=True)
    max_q = spec_len + 1
    path = leaf_node.new_zeros((n_leaf, max_q))
    path[:, -1] = leaf_node.to(dtype=torch.long)
    req = leaf_req
    for step in range(max_q - 2, -1, -1):
        nid = path[:, step + 1]
        slot = (nid - 1).clamp(min=0)
        par = parents[req, slot].to(dtype=torch.long)
        path[:, step] = torch.where(nid > 0, par, nid)

    leaf_slot = (leaf_node.to(dtype=torch.long) - 1).clamp(min=0)
    zero = leaf_node.new_zeros(n_leaf)
    leaf_depth = torch.where(
        leaf_node > 0,
        depths[req, leaf_slot].to(dtype=torch.long),
        zero,
    )
    qlens = (leaf_depth + 1).to(dtype=torch.int32)
    d = torch.arange(max_q, device=device)
    src_col = (max_q - qlens.to(dtype=torch.long).unsqueeze(1) + d).clamp(
        min=0, max=max_q - 1
    )
    aligned = path.gather(1, src_col)
    mask = d.unsqueeze(0) < qlens.to(dtype=torch.long).unsqueeze(1)
    rows, cols = mask.nonzero(as_tuple=True)
    n_tok = int(qlens.sum().item())  # D2H
    if n_tok <= 0:
        return None

    token_req = req[rows]
    token_node = aligned[rows, cols]
    slot = (token_node - 1).clamp(min=0)
    tok = tokens[token_req, slot].to(dtype=root_token_ids.dtype)
    tok = torch.where(tok >= 0, tok, root_token_ids[token_req])
    input_ids = torch.where(
        token_node == 0,
        root_token_ids[token_req],
        tok,
    )
    positions = prefix_lens[token_req] + cols.to(dtype=prefix_lens.dtype)
    path_qsl = qlens.new_zeros(n_leaf + 1)
    path_qsl[1:] = qlens.cumsum(0)
    path_seq_lens = prefix_lens[req] + qlens.to(dtype=prefix_lens.dtype)

    req_counts = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    ones = torch.ones(n_tok, dtype=torch.int32, device=device)
    req_counts.scatter_add_(0, token_req.to(dtype=torch.int32), ones)
    req_qsl = req_counts.new_zeros(num_reqs + 1)
    req_qsl[1:] = req_counts.cumsum(0)

    node_row = torch.full(
        (num_reqs, node_dim), -1, dtype=torch.long, device=device
    )
    order = torch.arange(n_tok, device=device, dtype=torch.long)
    node_row[token_req.flip(0), token_node.flip(0)] = order.flip(0)

    return Dsv4PathPack(
        num_tokens=n_tok,
        num_paths=n_leaf,
        input_ids=input_ids,
        positions=positions,
        token_req=token_req.to(dtype=torch.long),
        token_node=token_node.to(dtype=torch.long),
        node_row=node_row,
        path_query_start_loc=path_qsl,
        path_seq_lens=path_seq_lens,
        path_req_idx=req.to(dtype=torch.long),
        req_query_start_loc=req_qsl,
        prefix_lens=prefix_lens,
        path_qlens=qlens,
        token_path=rows.to(dtype=torch.long),
        token_col=cols.to(dtype=torch.long),
        num_reqs=num_reqs,
    )


def scatter_path_logits(
    logits: torch.Tensor,
    token_req: torch.Tensor,
    token_node: torch.Tensor,
    num_reqs: int,
    node_dim: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Scatter path-token logits onto ``[R, budget+1, V]`` by tree node id.

    NPU ``index_put`` requires the destination and source dtypes to match;
    target logits are bf16 while the packed buffer is fp32.
    """
    packed = out[:num_reqs, :node_dim]
    packed.fill_(float("-inf"))
    src = logits if logits.dtype == packed.dtype else logits.to(dtype=packed.dtype)
    n = src.shape[0]
    from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
        _step_diag,
        _step_dsa_skip,
        _step_packed,
        _step_serial,
        path_log,
    )

    if _step_diag or (_step_serial and _step_packed):
        mn = mx = -1
        if n > 0:
            mn = int(token_node[:n].amin().item())  # D2H
            mx = int(token_node[:n].amax().item())  # D2H
        path_log(
            "scatter packed_logits_in=%s serial_logits=%s dst_idx min=%s max=%s "
            "serial_ran=%s packed_ran=%s skip_write=%s%s",
            tuple(out.shape),
            tuple(logits.shape),
            mn,
            mx,
            int(_step_serial),
            int(_step_packed),
            int(_step_dsa_skip),
            " BUG_packed_and_serial" if (_step_serial and _step_packed) else "",
        )
    # First (req, node) wins so later leaves cannot overwrite root logits.
    packed[token_req[:n].flip(0), token_node[:n].flip(0)] = src.flip(0)
    return packed


def scatter_path_hidden(
    src: torch.Tensor,
    node_row: torch.Tensor,
    num_reqs: int,
    node_dim: int,
) -> torch.Tensor:
    """Scatter path-token rows onto a rectangular ``[R * (budget+1), ...]``."""
    dest = src.new_zeros((num_reqs * node_dim, *src.shape[1:]))
    valid = node_row[:num_reqs] >= 0
    req_idx, node_idx = valid.nonzero(as_tuple=True)
    dest[req_idx * node_dim + node_idx] = src[node_row[req_idx, node_idx]]
    return dest


def compact_dsv4_path_hidden(
    last_hidden_states: torch.Tensor,
    aux_hidden_states: list[torch.Tensor] | None,
    positions: torch.Tensor,
    node_row: torch.Tensor,
    path_node_ids: torch.Tensor,
    node_dim: int,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor, torch.Tensor]:
    """Scatter path rows to node-id columns, then compact the accepted path.

    Returns rectangular hidden / aux / positions and ``query_start_loc`` of
    length ``R+1`` with stride ``node_dim``. Device-only besides the host
    ``node_dim`` / ``num_reqs`` already in the caller.
    """
    num_reqs = path_node_ids.shape[0]
    hidden = scatter_path_hidden(last_hidden_states, node_row, num_reqs, node_dim)
    aux_out = None
    if aux_hidden_states:
        aux_out = [
            scatter_path_hidden(aux, node_row, num_reqs, node_dim)
            for aux in aux_hidden_states
        ]
    pos = scatter_path_hidden(positions, node_row, num_reqs, node_dim)
    tensors = [hidden]
    if aux_out:
        tensors.extend(aux_out)
    qsl = torch.arange(
        num_reqs + 1, device=hidden.device, dtype=torch.int32
    ) * node_dim
    compact_tree_query_along_path(
        tensors,
        qsl,
        path_node_ids,
        linearize_positions=pos,
    )
    return hidden, aux_out, pos, qsl


def apply_dsv4_path_pack(runner, input_batch) -> None:
    """Rewrite the target batch from packed 1+budget into leaf-path TND."""
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
        dsv4_path_isolation_needed,
    )

    if not dsv4_path_isolation_needed(runner):
        spec = getattr(runner, "speculator", None)
        if bool(getattr(spec, "_dsv4_dspark_draft", False)):
            from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
                _pack_skip_seen,
                path_log,
            )

            reason = "topk==1" if (get_ascend_config().tree_spec_config.topk or 0) <= 1 else "not_dsv4_tree"
            if reason not in _pack_skip_seen:
                _pack_skip_seen.add(reason)
                path_log("pack skip reason=%s", reason)
        return
    if input_batch.has_prefill:
        from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
            _pack_skip_seen,
            path_log,
        )

        if "has_prefill" not in _pack_skip_seen:
            _pack_skip_seen.add("has_prefill")
            path_log("pack skip reason=has_prefill")
        return
    if getattr(input_batch, "tree_tokens", None) is None:
        from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
            _pack_skip_seen,
            path_log,
        )

        if "not_tree" not in _pack_skip_seen:
            _pack_skip_seen.add("not_tree")
            path_log("pack skip reason=not_tree")
        return
    if (get_ascend_config().tree_spec_config.topk or 0) <= 1:
        return

    num_reqs = input_batch.num_reqs
    qsl = input_batch.query_start_loc[: num_reqs + 1]
    root_token_ids = input_batch.input_ids[qsl[:num_reqs].to(dtype=torch.long)]
    idx = input_batch.idx_mapping[:num_reqs]
    prefix_lens = runner.req_states.num_computed_tokens.gpu[idx]
    spec_len = runner.num_speculative_steps
    pack = build_dsv4_path_pack(
        input_batch.tree_tokens[:num_reqs],
        input_batch.tree_parents[:num_reqs],
        input_batch.tree_first_child[:num_reqs],
        input_batch.tree_num_nodes[:num_reqs],
        input_batch.tree_depths[:num_reqs],
        root_token_ids,
        prefix_lens,
        spec_len,
    )
    if pack is None:
        from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import path_log

        path_log("pack skip reason=empty_pack")
        return

    t = pack.num_tokens
    bufs = runner.input_buffers
    # post_update does query_len - num_rejected. Keep the scheduler
    # 1+budget qsl; pack.num_tokens is the expanded leaf-path count.
    logical_qsl = input_batch.query_start_loc[: num_reqs + 1].clone()
    bufs.input_ids[:t].copy_(pack.input_ids.to(dtype=bufs.input_ids.dtype))
    bufs.positions[:t].copy_(pack.positions.to(dtype=bufs.positions.dtype))
    # RoPE stays prefix+col. Isolated writes use CoW slot_mappings or serial
    # official suffix slots, not this packed sibling-colliding layout.
    bufs.slot_positions[:t].copy_(pack.positions.to(dtype=bufs.slot_positions.dtype))
    bufs.is_padding[:t].fill_(False)
    bufs.query_start_loc[: num_reqs + 1].copy_(
        pack.req_query_start_loc.to(dtype=bufs.query_start_loc.dtype)
    )
    req_qsl_np = pack.req_query_start_loc.cpu().numpy()  # D2H
    path_qsl_cpu = pack.path_query_start_loc.cpu()  # D2H
    path_seq_cpu = pack.path_seq_lens.cpu()  # D2H

    input_batch.num_tokens = t
    input_batch.num_tokens_after_padding = t
    input_batch.input_ids = bufs.input_ids[:t]
    input_batch.positions = bufs.positions[:t]
    input_batch.is_padding = bufs.is_padding[:t]
    input_batch.slot_positions = bufs.slot_positions[:t]
    input_batch.query_start_loc = bufs.query_start_loc[: num_reqs + 1]
    input_batch.query_start_loc_np = req_qsl_np
    input_batch.logits_indices = torch.arange(
        t, device=bufs.input_ids.device, dtype=torch.int32
    )
    input_batch.cu_num_logits = input_batch.query_start_loc
    input_batch.cu_num_logits_np = req_qsl_np
    input_batch.expanded_idx_mapping = input_batch.idx_mapping[pack.token_req]
    input_batch.max_query_len = spec_len + 1
    input_batch.tree_path_query_start_loc = pack.path_query_start_loc
    input_batch.tree_path_query_start_loc_cpu = path_qsl_cpu
    input_batch.tree_path_seq_lens = pack.path_seq_lens
    input_batch.tree_path_seq_lens_cpu = path_seq_cpu
    input_batch.tree_path_req_idx = pack.path_req_idx
    input_batch.tree_path_node_ids = pack.token_node
    input_batch.tree_path_token_req = pack.token_req
    input_batch.tree_node_row = pack.node_row
    input_batch.tree_prefix_lens = pack.prefix_lens
    input_batch.tree_path_kv_isolated = False
    input_batch.tree_path_block_tables = None
    input_batch.tree_path_slot_mappings = None
    from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
        ensure_dsv4_path_verifier,
    )

    verifier = ensure_dsv4_path_verifier(runner)
    if verifier is not None:
        verifier.prepare_batch(input_batch, pack)
        verifier._post_query_start_loc = logical_qsl
        if getattr(verifier, "_diag_left", 0) > 0:
            from vllm_ascend.worker.v2.spec_decode.tree.dsv4_path_verify import (
                path_log,
                tree_token_head,
                tree_token_stats,
            )

            nqp = getattr(getattr(runner, "speculator", None), "num_query_per_req", None)
            tmin, tmax, nneg = tree_token_stats(input_batch.tree_tokens[:num_reqs])

            path_log(
                "pack query_len=%s num_tokens=%s num_leaves=%s num_query_per_req=%s "
                "skip_compressed_cache_write=%s tree_path_kv_isolated=%s cow_ready=%s "
                "tree_tokens min=%s max=%s neg_count=%s head=%s",
                spec_len + 1,
                pack.num_tokens,
                pack.num_paths,
                nqp,
                int(not bool(getattr(input_batch, "tree_path_kv_isolated", False))),
                int(bool(getattr(input_batch, "tree_path_kv_isolated", False))),
                int(verifier.cow_ready),
                tmin,
                tmax,
                nneg,
                tree_token_head(input_batch.tree_tokens[:num_reqs]),
            )
