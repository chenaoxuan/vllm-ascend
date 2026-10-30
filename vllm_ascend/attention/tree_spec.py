from typing import Any

import torch
from vllm.config import VllmConfig

from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def align_up(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def tree_spec_enabled(vllm_config: VllmConfig | None = None) -> bool:
    """True when ``additional_config.tree_spec_config.enabled`` is set."""
    if vllm_config is not None:
        tree_config = (getattr(vllm_config, "additional_config", None) or {}).get("tree_spec_config") or {}
        return bool(tree_config.get("enabled", False))
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().tree_spec_config.enabled)
    except RuntimeError:
        return False


def tree_query_len() -> int | None:
    """Storage/visibility capacity B+1, independent of scheduled target rows.

    Target capture and dispatch use TreeShape.target_query_len. This helper
    sizes root-inclusive visibility storage, including for a truncated tree.
    """
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        tree_cfg = get_ascend_config().tree_spec_config
        if not tree_cfg.enabled:
            return None
        return 1 + int(tree_cfg.budget)
    except (RuntimeError, AssertionError):
        return None


def tree_decode_threshold(vllm_config: VllmConfig | None = None) -> int:
    """Target decode capacity; DSV4 topk=1 retains its K+1 chain width."""
    if vllm_config is not None:
        from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape

        shape = TreeShape.from_config(vllm_config)
        if shape is not None:
            return shape.target_query_len
        spec = vllm_config.speculative_config
        return 1 + int(spec.num_speculative_tokens) if spec is not None else 1
    query_len = tree_query_len()
    return query_len if query_len is not None else 1


def tree_treat_short_extends_as_decodes(
    common_attn_metadata,
    *,
    tree_enabled: bool,
    extra_disable: bool = False,
) -> bool:
    """Short prefills stay prefills when tree target metadata is available.

    Draft propose does not populate ``is_prefilling``; keep the default
    short-extend-as-decode path there so split does not require the flag.
    """
    if extra_disable:
        return False
    if not tree_enabled:
        return True
    return getattr(common_attn_metadata, "is_prefilling", None) is None


def dummy_tree_visibility(
    num_decode: int,
    device: torch.device,
    budget: int | None = None,
) -> torch.Tensor:
    """Identity visibility for FULL capture when the dummy batch has no tree."""
    if budget is None:
        query_len = tree_query_len()
        budget = query_len - 1 if query_len is not None else 1
    vis = torch.eye(budget, dtype=torch.bool, device=device)
    return vis.unsqueeze(0).expand(num_decode, -1, -1).contiguous()


def tree_verify_shape(num_tokens: int) -> tuple[int, int] | None:
    """``(num_decode, 1+budget)`` when the batch is tree-verify shaped."""
    tree_q = tree_query_len()
    if tree_q is None or tree_q <= 0 or num_tokens < tree_q or num_tokens % tree_q != 0:
        return None
    n_dec = num_tokens // tree_q
    if n_dec <= 0:
        return None
    return n_dec, tree_q


def dummy_tree_mask_for_capture(num_tokens: int, num_reqs: int) -> bool:
    """Dummy FULL capture only needs the tree layout for ``k × (1+budget)``."""
    shaped = tree_verify_shape(num_tokens)
    return shaped is not None and shaped[0] == num_reqs


def seq_lens_are_tree_query(seq_lens: torch.Tensor | None, num_decode: int) -> bool:
    """Dummy FULL capture sets seq_len == query_len; tree verify is 1+budget."""
    tree_q = tree_query_len()
    if tree_q is None or seq_lens is None or num_decode <= 0:
        return False
    sl = seq_lens[:num_decode]
    if sl.numel() != num_decode or sl.device.type != "cpu":
        return False
    return all(v == tree_q for v in sl.tolist())


def is_draft_forward() -> bool:
    """True only inside a draft forward that already has a forward context.

    Graph ``prepare_attn`` runs before ``set_forward_context``; treat that as
    target so dummy tree visibility can still be injected.
    """
    from vllm.forward_context import is_forward_context_available

    return is_forward_context_available() and bool(_EXTRA_CTX.is_draft_model)


def need_dummy_tree_visibility_for_capture() -> bool:
    """Target FULL capture only; draft decode must keep its own mask/indices."""
    from vllm.forward_context import is_forward_context_available

    if not is_forward_context_available():
        return False
    if not _EXTRA_CTX.capturing or is_draft_forward():
        return False
    return tree_query_len() is not None


def is_target_tree_step(common_attn_metadata) -> bool:
    """True when the current target forward should apply tree visibility."""
    if not tree_spec_enabled() or is_draft_forward():
        return False
    if common_attn_metadata.tree_visibility is not None:
        return True
    return need_dummy_tree_visibility_for_capture()


class TreeSpecAttnAdapter:
    """Target-side tree metadata. Kernel dispatch stays in each backend."""

    def configure_builder(self, builder, vllm_config: VllmConfig, device: torch.device) -> None:
        return

    def build_target_inputs(self, builder, common_attn_metadata, *, num_decodes: int) -> Any:
        return None


class GqaTreeSpecAdapter(TreeSpecAttnAdapter):
    """FIA 4D/3D custom mask is filled later in AttentionMaskBuilder."""

    def configure_builder(self, builder, vllm_config: VllmConfig, device: torch.device) -> None:
        query_len = tree_decode_threshold(vllm_config)
        builder.attn_mask_builder.configure_tree_mask(
            max_num_decode=int(vllm_config.scheduler_config.max_num_seqs),
            query_len=query_len,
            kv_len=align_up(int(vllm_config.model_config.max_model_len), 128),
        )


class DsaTreeSpecAdapter(TreeSpecAttnAdapter):
    """Paged ori sparse indices: SWA prefix window union ancestor draft slots."""

    def configure_builder(self, builder, vllm_config: VllmConfig, device: torch.device) -> None:
        window = int(vllm_config.model_config.hf_config.sliding_window)
        query_len = tree_decode_threshold(vllm_config)
        budget = query_len - 1
        index_width = align_up(window + budget, 128)
        scheduler_config = vllm_config.scheduler_config
        max_rows = max(
            int(scheduler_config.max_num_batched_tokens),
            int(scheduler_config.max_num_seqs) * query_len,
        )
        compilation_config = vllm_config.compilation_config
        capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
        if capture_sizes:
            max_rows = max(max_rows, int(compilation_config.max_cudagraph_capture_size) * query_len)
        builder.tree_ori_indices_buffer = torch.full(
            (max_rows, 1, index_width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        builder.tree_ori_index_width = index_width

    def build_target_inputs(self, builder, common_attn_metadata, *, num_decodes: int) -> torch.Tensor | None:
        from vllm_ascend.ascend_config import get_ascend_config

        # topk=1 is a chain. DSA SWA/causal already matches chain spec decode.
        if get_ascend_config().tree_spec_config.topk <= 1:
            return None
        if num_decodes <= 0 or not is_target_tree_step(common_attn_metadata):
            return None
        vis = common_attn_metadata.tree_visibility
        device = builder.device
        if vis is None:
            vis = dummy_tree_visibility(num_decodes, device)
        elif vis.device != device:
            vis = vis.to(device, non_blocking=True)  # H2D
        num_tokens = int(common_attn_metadata.num_input_tokens or 0)
        if num_tokens <= 0:
            num_tokens = int(builder.num_actual_tokens or 0)
        if num_tokens <= 0:
            return None
        window = int(builder.model_config.hf_config.sliding_window)
        arguments = dict(
            block_table=builder.block_table,
            seq_lens=builder.seq_lens,
            query_start_loc=common_attn_metadata.query_start_loc[: common_attn_metadata.num_reqs + 1],
            tree_visibility=vis,
            num_decodes=num_decodes,
            num_tokens=num_tokens,
            window_size=window,
            storage_block_size=int(builder.storage_block_size),
            buffer=builder.tree_ori_indices_buffer,
        )
        from vllm_ascend.worker.v2.spec_decode.tree import chain_pack

        layout = chain_pack._LAYOUT
        if layout is not None and layout.get("graph_requested", False):
            # DSV4 FULL target capture records visibility/index construction
            # before attention; host metadata preparation only binds inputs.
            arguments["buffer"] = builder.tree_ori_indices_buffer[:num_tokens]
            layout.setdefault("graph_tree_indices", []).append(arguments)
            return arguments["buffer"]
        return build_tree_ori_sparse_indices(**arguments)


def _positions_to_slots(positions, block_table, req_ids, block_size):
    nblocks = block_table.shape[1]
    block_nums = torch.div(positions, block_size, rounding_mode="floor")
    safe_nums = block_nums.clamp(min=0, max=nblocks - 1).to(torch.long)
    block_ids = torch.gather(block_table[req_ids], 1, safe_nums)
    valid = (positions >= 0) & (block_nums < nblocks) & (block_ids > 0)
    slots = block_ids * block_size + torch.remainder(positions, block_size)
    return torch.where(valid, slots, -1).to(torch.int32)


def build_tree_ori_sparse_indices(
    block_table,
    seq_lens,
    query_start_loc,
    tree_visibility,
    num_decodes,
    num_tokens,
    window_size,
    storage_block_size,
    buffer,
):
    device = buffer.device
    n_req = int(seq_lens.shape[0])
    budget = int(tree_visibility.shape[-1])
    if n_req == 0 or num_tokens == 0:
        buffer.fill_(-1)
        return buffer[:num_tokens]
    if budget <= 0 or not 0 < num_decodes <= n_req:
        raise RuntimeError("Invalid tree visibility/decode batch")
    if tree_visibility.shape[0] < num_decodes:
        raise RuntimeError("Missing tree visibility rows")
    qsl = query_start_loc[: n_req + 1].to(torch.long)
    qlens = qsl[1:] - qsl[:-1]
    prefixes = seq_lens.to(torch.long) - qlens
    token = torch.arange(num_tokens, device=device)
    req = torch.searchsorted(qsl[1:], token, right=True).clamp(max=n_req - 1)
    offset = token - qsl[req]
    in_query = (offset >= 0) & (offset < qlens[req])
    decode = (req < num_decodes) & in_query
    vis = tree_visibility[:num_decodes].to(device=device, dtype=torch.bool)
    # Each row includes this node and all non-root ancestors.
    node_depth = vis.sum(dim=-1, dtype=torch.long)
    vis_req = req.clamp(max=num_decodes - 1)
    vis_row = (offset - 1).clamp(min=0, max=budget - 1)
    draft_query = decode & (offset >= 1) & (offset <= budget)
    depth = torch.where(draft_query, node_depth[vis_req, vis_row], 0)
    logical_pos = prefixes[req] + torch.where(decode, depth, offset)
    window_start = (logical_pos - int(window_size) + 1).clamp(min=0)

    # The prefix segment includes the root at P, if P is still in the window.
    prefix_end = torch.where(decode, prefixes[req], logical_pos)
    prefix_len = (prefix_end - window_start + 1).clamp(min=0, max=window_size)
    col = torch.arange(window_size, device=device)
    prefix_pos = window_start[:, None] + col[None, :]
    prefix_slots = _positions_to_slots(prefix_pos, block_table, req, storage_block_size)
    prefix_valid = (col[None, :] < prefix_len[:, None]) & in_query[:, None]
    prefix_slots = prefix_slots.masked_fill(~prefix_valid, -1)

    # Physical slots use flat node IDs; visibility uses logical tree depths.
    draft_depth = node_depth[vis_req]
    draft_logical_pos = prefixes[req, None] + draft_depth
    draft_valid = vis[vis_req, vis_row] & draft_query[:, None]
    draft_valid &= draft_logical_pos >= window_start[:, None]
    draft_valid &= (draft_logical_pos <= logical_pos[:, None]) & (draft_depth > 0)
    node_id = torch.arange(1, budget + 1, device=device)
    physical_pos = prefixes[req, None] + node_id[None, :]
    draft_slots = _positions_to_slots(physical_pos, block_table, req, storage_block_size)
    draft_slots = draft_slots.masked_fill(~draft_valid, -1)

    packed = torch.cat([prefix_slots, draft_slots], dim=-1)
    valid = packed >= 0
    ncol = packed.shape[-1]
    order = torch.arange(ncol, device=device)
    key = order[None, :] + (~valid).to(torch.long) * ncol
    packed = packed.gather(1, key.argsort(dim=-1))
    width = int(buffer.shape[-1])
    if width < window_size:
        raise RuntimeError("Tree SWA buffer is smaller than the model window")
    if ncol < width:
        packed = torch.cat([packed, packed.new_full((num_tokens, width - ncol), -1)], dim=1)
    else:
        packed = packed[:, :width]
    buffer[:num_tokens].copy_(packed[:, None, :])
    buffer[num_tokens:].fill_(-1)
    return buffer[:num_tokens]
