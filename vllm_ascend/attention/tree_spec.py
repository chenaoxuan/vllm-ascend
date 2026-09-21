from typing import Any

import torch
from vllm.config import VllmConfig

from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def align_up(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def tree_spec_enabled(vllm_config: VllmConfig | None = None) -> bool:
    """True when ``additional_config.tree_spec_config.enabled`` is set."""
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().tree_spec_config.enabled)
    except RuntimeError:
        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        tree_cfg = additional_config.get("tree_spec_config") or {}
        return bool(tree_cfg.get("enabled", False))


def tree_query_len() -> int | None:
    """1 + budget from tree_spec_config; does not need current vLLM config."""
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        tree_cfg = get_ascend_config().tree_spec_config
        if not tree_cfg.enabled:
            return None
        return 1 + int(tree_cfg.budget)
    except (RuntimeError, AssertionError):
        return None


def tree_decode_threshold(vllm_config: VllmConfig | None = None) -> int:
    """Target decode query length: ``1 + budget`` when tree spec is on."""
    query_len = tree_query_len()
    if query_len is not None:
        return query_len
    spec = getattr(vllm_config, "speculative_config", None) if vllm_config is not None else None
    if spec is not None:
        return 1 + int(spec.num_speculative_tokens)
    return 1


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
        return build_tree_ori_sparse_indices(
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


def _positions_to_slots(
    positions: torch.Tensor,
    block_table: torch.Tensor,
    req_ids: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map absolute token positions ``[T, W]`` to paged slot ids."""
    nblocks = block_table.shape[1]
    block_nums = torch.div(positions, block_size, rounding_mode="floor")
    safe_nums = block_nums.clamp(min=0, max=nblocks - 1).to(torch.long)
    block_offsets = torch.remainder(positions, block_size)
    req_tables = block_table[req_ids]
    block_ids = torch.gather(req_tables, 1, safe_nums)
    return (block_ids * block_size + block_offsets).to(torch.int32)


def build_tree_ori_sparse_indices(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    tree_visibility: torch.Tensor,
    num_decodes: int,
    num_tokens: int,
    window_size: int,
    storage_block_size: int,
    buffer: torch.Tensor,
) -> torch.Tensor:
    """Per-token ori slots for tree target verify (and mixed-batch prefills).

    Decode queries see the trailing SWA window of the committed prefix
    (including the root slot) plus draft slots allowed by ``tree_visibility``.
    Prefill tokens in a mixed batch keep a causal sliding window.
    Valid slots are left-packed: the A3 SWA kernel stops at the first -1.
    Returned view is a leading slice of ``buffer`` so ACLGraph keeps a stable
    pointer.
    """
    device = buffer.device
    num_reqs = seq_lens.shape[0]
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    prefix_lens = seq_lens - query_lens
    budget = int(tree_visibility.shape[-1])
    query_len = 1 + budget
    vis = tree_visibility[:num_decodes]
    if vis.shape[0] < num_decodes:
        vis = vis[: vis.shape[0]]

    query_lens = query_lens.to(torch.long)
    req_ids = torch.repeat_interleave(
        torch.arange(num_reqs, device=device, dtype=torch.long),
        query_lens,
    )
    if req_ids.numel() < num_tokens:
        pad_n = num_tokens - req_ids.numel()
        fill = req_ids[-1] if req_ids.numel() > 0 else torch.zeros((), device=device, dtype=torch.long)
        req_ids = torch.cat([req_ids, fill.reshape(1).expand(pad_n)])
    elif req_ids.numel() > num_tokens:
        req_ids = req_ids[:num_tokens]

    token_offsets = torch.arange(num_tokens, device=device, dtype=torch.long) - query_start_loc[req_ids]
    in_query = token_offsets < query_lens[req_ids]
    positions = prefix_lens[req_ids] + token_offsets
    is_decode = (req_ids < num_decodes) & in_query

    decode_start = (prefix_lens - window_size + 1).clamp(min=0)[req_ids]
    prefill_start = (positions - window_size + 1).clamp(min=0)
    starts = torch.where(is_decode, decode_start, prefill_start)
    decode_end = prefix_lens[req_ids]
    ends = torch.where(is_decode, decode_end, positions)
    vis_len = (ends - starts + 1).clamp(min=0)

    prefix_cols = torch.arange(window_size, device=device)
    prefix_pos = starts.unsqueeze(1) + prefix_cols.unsqueeze(0)
    prefix_valid = (prefix_cols.unsqueeze(0) < vis_len.unsqueeze(1)) & in_query.unsqueeze(1)
    prefix_slots = _positions_to_slots(prefix_pos, block_table, req_ids, storage_block_size)
    prefix_slots = prefix_slots.where(prefix_valid, prefix_slots.new_full((), -1))

    vis_req = req_ids.clamp(max=max(vis.shape[0] - 1, 0))
    vis_row = (token_offsets - 1).clamp(min=0, max=budget - 1)
    vis_rows = vis[vis_req, vis_row]
    draft_query = is_decode & (token_offsets >= 1) & (token_offsets < query_len)
    draft_valid = vis_rows & draft_query.unsqueeze(1)
    draft_cols = torch.arange(budget, device=device)
    draft_pos = prefix_lens[req_ids].unsqueeze(1) + 1 + draft_cols.unsqueeze(0)
    draft_slots = _positions_to_slots(draft_pos, block_table, req_ids, storage_block_size)
    draft_slots = draft_slots.where(draft_valid, draft_slots.new_full((), -1))

    packed = torch.cat([prefix_slots, draft_slots], dim=-1)
    # A3 SWA GetOriSparseActualSeqLen stops at the first -1, so valid prefix
    # and ancestor draft slots must be left-packed. A hole in the SWA window
    # would drop every later draft slot and collapse pos1+ logits.
    valid = packed >= 0
    ncol = packed.shape[-1]
    order = torch.arange(ncol, device=device, dtype=torch.float32)
    key = order + (~valid).to(order.dtype) * float(ncol)
    packed = packed.gather(1, key.argsort(dim=-1))
    index_width = buffer.shape[-1]
    if packed.shape[-1] < index_width:
        pad = packed.new_full((num_tokens, index_width - packed.shape[-1]), -1)
        packed = torch.cat([packed, pad], dim=-1)
    elif packed.shape[-1] > index_width:
        packed = packed[:, :index_width]
    packed = packed.unsqueeze(1)
    buffer[:num_tokens].copy_(packed)
    if buffer.shape[0] > num_tokens:
        buffer[num_tokens:].fill_(-1)
    return buffer[:num_tokens]
