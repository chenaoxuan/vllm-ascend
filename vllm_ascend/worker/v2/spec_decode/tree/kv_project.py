import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import (
    iter_unique_kv_cache_tensors,
)

logger = init_logger("vllm." + __name__)


try:
    from vllm.v1.kv_cache_interface import CircularBufferSpec as _CircularBufferSpec
except ImportError:
    _CircularBufferSpec = None


def _spec_is_circular_buffer(spec) -> bool:
    if _CircularBufferSpec is None or spec is None:
        return False
    if isinstance(spec, _CircularBufferSpec):
        return True
    first = getattr(spec, "first_spec", None)
    return first is not None and isinstance(first, _CircularBufferSpec)


def _group_is_linear_token_cache(group) -> bool:
    """True when compact can treat the group as 1 token per slot."""
    names = getattr(group, "layer_names", ()) or ()
    if any("state_cache" in name for name in names):
        return False
    spec = group.kv_cache_spec
    if _spec_is_circular_buffer(spec) or getattr(spec, "cache_role", None):
        return False
    inner = getattr(spec, "kv_cache_specs", None)
    specs = list(inner.values()) if inner else [spec]
    for item in specs:
        if _spec_is_circular_buffer(item) or getattr(item, "cache_role", None):
            return False
        ratio = getattr(item, "tokens_per_state", None)
        if ratio is None:
            ratio = getattr(item, "compress_ratio", 1)
        if ratio is not None and int(ratio) > 1:
            return False
    return True


def needs_causal_kv_repair(runner) -> bool:
    """True when a KV group cannot be compacted by paged slot after tree verify."""
    groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", None)
    if not groups:
        return False
    return any(not _group_is_linear_token_cache(group) for group in groups)


class TreeKvCompact:
    """Move accepted-path target KV from tree slots onto the linear prefix.

    Verify wrote ori KV at packed ``prefix+node``. GQA keeps the root at
    ``prefix`` and lands drafts at ``prefix+1..``. DSV4 packed verify
    recomputes the root at ``prefix``; official sampled tokens start there,
    so drafts land at ``prefix+0..`` (``dst_from_zero``). C4/indexer/state
    are not compacted.

    Single-stream: ``run`` after target+reject and before
    ``num_computed`` increments. One ACLGraph per ``num_reqs`` so the
    per-layer gather/scatter is one replay instead of many launches.
    ``idx_mapping`` / ``path_node_ids`` are copied into persistent
    buffers; block tables and ``num_computed`` are live tensors captured
    by address. Graph gather uses preallocated scratch. Missing gears or
    capture failure fall back to the same eager ops.
    """

    def __init__(self, runner):
        self.runner = runner
        self.device = runner.device
        self.spec_len = runner.num_speculative_steps
        self.max_num_reqs = runner.max_num_reqs
        self._graphs: dict[int, object] = {}
        self._groups = None
        self._held = None
        self._hold = False
        self._idx = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        self._path = torch.full(
            (self.max_num_reqs, self.spec_len),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._depth = (
            torch.arange(self.spec_len, device=self.device, dtype=torch.int32) + 1
        )
        self._depth0 = torch.arange(self.spec_len, device=self.device, dtype=torch.int32)
        self._dst_off = self._depth

    def capture(self, sizes: list[int]) -> None:
        if not sizes or self.runner.model_config.enforce_eager:
            return
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "NPUGraph"):
            return
        from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
        from vllm.platforms import current_platform

        self._bind_groups()
        self._idx.fill_(0)
        self._path.fill_(-1)
        self._hold = False
        pool = current_platform.get_global_graph_pool()
        for num_reqs in sizes:
            try:
                validate_cudagraph_capturing_enabled()
                with torch.inference_mode():
                    self._compact(num_reqs)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, pool=pool):
                        self._compact(num_reqs)
                self._graphs[num_reqs] = graph
                logger.info("Captured tree KV compact ACLGraph num_reqs=%s", num_reqs)
            except Exception as exc:
                logger.warning(
                    "Tree KV compact ACLGraph capture failed for num_reqs=%s; "
                    "eager fallback for this size. %s",
                    num_reqs,
                    exc,
                )

    def run(
        self,
        idx_mapping: torch.Tensor,
        path_node_ids: torch.Tensor,
        *,
        dst_from_zero: bool = False,
    ) -> None:
        num_reqs = path_node_ids.shape[0]
        self._idx[:num_reqs].copy_(idx_mapping[:num_reqs])
        self._path[:num_reqs].copy_(path_node_ids[:num_reqs])
        self._dst_off = self._depth0 if dst_from_zero else self._depth
        self._hold = dst_from_zero
        graph = self._graphs.get(num_reqs)
        if graph is not None and not dst_from_zero:
            graph.replay()
            return
        self._bind_groups()
        self._compact(num_reqs)

    def scatter_held(self) -> None:
        """Re-apply the last gather onto dest slots (after a later overwrite)."""
        if not self._held:
            return
        for cache, scratch, dst_flat in self._held:
            tail = cache.shape[2:]
            flat = cache.reshape(cache.shape[0] * cache.shape[1], *tail)
            flat.index_copy_(0, dst_flat, scratch)

    def _bind_groups(self) -> None:
        if self._groups is not None:
            return
        ctx = self.runner.compilation_config.static_forward_context
        groups = []
        seen: set[int] = set()
        for group_id, group in enumerate(self.runner.kv_cache_config.kv_cache_groups):
            if not _group_is_linear_token_cache(group):
                continue
            caches: list[torch.Tensor] = []
            gathers: list[torch.Tensor] = []
            for layer_name in group.layer_names:
                layer = ctx.get(layer_name)
                if layer is None:
                    continue
                for tensor in iter_unique_kv_cache_tensors(
                    getattr(layer, "kv_cache", None)
                ):
                    ptr = tensor.untyped_storage().data_ptr()
                    if ptr in seen:
                        continue
                    seen.add(ptr)
                    caches.append(tensor)
                    tail = tensor.shape[2:]
                    gathers.append(
                        torch.empty(
                            (self.max_num_reqs * self.spec_len, *tail),
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                    )
            if not caches:
                continue
            groups.append(
                (
                    caches,
                    self.runner.block_tables.block_tables[group_id].gpu,
                    self.runner.block_tables.kernel_block_sizes[group_id],
                    gathers,
                )
            )
        self._groups = groups

    def _compact(self, num_reqs: int) -> None:
        if not self._groups:
            self._held = None
            return
        idx = self._idx[:num_reqs]
        path = self._path[:num_reqs]
        num_computed = self.runner.req_states.num_computed_tokens.gpu
        nslot = num_reqs * self.spec_len
        safe_idx = idx.clamp(min=0)
        prefix = num_computed[safe_idx]
        dst_pos = prefix.unsqueeze(1) + self._dst_off
        src_pos = prefix.unsqueeze(1) + path.clamp(min=0).to(dtype=prefix.dtype)
        valid = (path >= 0) & (idx >= 0).unsqueeze(1)
        src_pos = torch.where(valid, src_pos, dst_pos)
        max_pos = self.runner.max_model_len - 1
        dst_pos = dst_pos.clamp(max=max_pos)
        src_pos = src_pos.clamp(max=max_pos)
        req_f = safe_idx.unsqueeze(1).expand(num_reqs, self.spec_len)
        hold = self._hold
        held = []
        for caches, block_table, block_size, gathers in self._groups:
            max_block = block_table.shape[1] - 1
            src_bi = torch.div(src_pos, block_size, rounding_mode="floor").clamp(
                min=0, max=max_block
            )
            dst_bi = torch.div(dst_pos, block_size, rounding_mode="floor").clamp(
                min=0, max=max_block
            )
            src_block = block_table[req_f, src_bi]
            dst_block = block_table[req_f, dst_bi]
            src_flat = (src_block * block_size + src_pos % block_size).to(
                dtype=torch.long
            ).reshape(-1)
            dst_flat = (dst_block * block_size + dst_pos % block_size).to(
                dtype=torch.long
            ).reshape(-1)
            dst_keep = dst_flat.clone() if hold else None
            for cache, gather in zip(caches, gathers):
                tail = cache.shape[2:]
                flat = cache.reshape(cache.shape[0] * cache.shape[1], *tail)
                scratch = gather[:nslot]
                torch.index_select(flat, 0, src_flat, out=scratch)
                flat.index_copy_(0, dst_flat, scratch)
                if hold:
                    held.append((cache, scratch, dst_keep))
        self._held = held if hold else None


def run_short_causal_forward(
    runner,
    input_batch: AscendInputBatch,
    *,
    model=None,
    model_kwargs: dict | None = None,
):
    """Eager causal target forward. Caller owns input-buffer save/restore."""
    n = input_batch.num_tokens
    eplb = runner.eplb
    if eplb is not None:
        eplb.prepare_forward(runner.model_config, n)
    block_tables, slot_mappings = runner.prepare_attn(input_batch)
    attn_metadata = runner.model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        block_tables,
        slot_mappings,
        runner.attn_groups,
        runner.kv_cache_config,
    )
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, runner.kv_cache_config
    )
    fwd_model = model if model is not None else runner.model
    inputs = {
        "input_ids": input_batch.input_ids,
        "positions": input_batch.positions,
        "inputs_embeds": None,
        "intermediate_tensors": None,
        **runner.model_state.prepare_inputs(input_batch, runner.req_states),
    }
    if model_kwargs:
        inputs.update(model_kwargs)
        inputs["input_ids"] = input_batch.input_ids
        inputs["positions"] = input_batch.positions
    batch_descriptor = BatchDescriptor(num_tokens=n)
    with set_forward_context(
        attn_metadata,
        runner.vllm_config,
        num_tokens=n,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=batch_descriptor,
        slot_mapping=slot_mappings_by_layer,
        is_padding=input_batch.is_padding,
    ):
        return fwd_model(**inputs)


def commit_dsv4_accepted_chain(
    runner,
    idx_mapping: torch.Tensor,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
) -> None:
    """Write the accepted chain onto official prefix KV after packed verify.

    Packed tree verify skips C4/indexer/state and writes ori with SWA-C4
    hidden. A later causal re-forward repairs C4/state and the bonus ori
    slot; accepted ori slots are restored from the tree-verify gather so
    the next SWA-C4 verify does not mix true-C4 keys with SWA-C4 queries.
    Draft hidden/aux use the repaired MTP rows.
    """
    sampler = runner.rejection_sampler
    path = sampler.path_node_ids
    compact = runner.tree_kv_compact
    getter = getattr(runner.model, "get_mtp_target_hidden_states", None)
    if compact is not None and path is not None:
        compact.run(idx_mapping, path, dst_from_zero=True)
    spec = runner.speculator
    meta = repair_tree_kv_causal(runner, idx_mapping, sampled_tokens, num_sampled)
    if compact is not None and path is not None:
        compact.scatter_held()
    if meta is None or spec is None:
        return
    n, pos, qsl, aux = meta
    spec._dsv4_commit_n = n
    spec._dsv4_commit_pos = pos
    spec._dsv4_commit_qsl = qsl
    spec._dsv4_commit_aux = [a.clone() for a in aux] if aux else None
    mtp = getter() if callable(getter) else None
    spec._dsv4_commit_hidden = mtp[:n].clone() if mtp is not None else None


def repair_tree_kv_causal(
    runner,
    idx_mapping: torch.Tensor,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
):
    """Causal re-forward of accepted tokens onto official linear KV.

    Packed tree verify skipped C4/indexer/state; this writes them on the
    accepted chain. Caller saves/restores shared input buffers.

    Returns ``(n, positions, query_start_loc, aux)`` or ``None``. ``n`` is
    a host int.
    """
    orig_reqs = sampled_tokens.shape[0]
    if orig_reqs <= 0:
        return None
    idx = idx_mapping[:orig_reqs]
    keep = num_sampled[:orig_reqs] > 0
    n_keep = int(keep.sum().item())  # D2H
    if n_keep <= 0:
        return None
    num_reqs = orig_reqs
    if n_keep < orig_reqs:
        idx = idx[keep]
        sampled_tokens = sampled_tokens[keep]
        num_sampled = num_sampled[keep]
        num_reqs = n_keep
    safe_idx = idx.clamp(min=0)
    device = sampled_tokens.device
    max_q = sampled_tokens.shape[1]
    local = torch.arange(max_q, device=device)
    mask = local.unsqueeze(0) < num_sampled.unsqueeze(1)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    qsl[1:] = num_sampled.to(dtype=torch.int32).cumsum(0)
    n = int(qsl[-1].item())  # D2H
    if n <= 0:
        return None

    prefix = runner.req_states.num_computed_tokens.gpu[safe_idx]
    ids = sampled_tokens.masked_select(mask)
    pos = (prefix.unsqueeze(1) + local.to(dtype=prefix.dtype)).masked_select(mask)
    seq_lens = prefix + num_sampled.to(dtype=prefix.dtype)

    bufs = runner.input_buffers
    old_qsl = bufs.query_start_loc[: orig_reqs + 1].clone()
    old_n = int(old_qsl[-1].item())  # D2H
    old_n = max(old_n, n)
    old_ids = bufs.input_ids[:old_n].clone()
    old_pos = bufs.positions[:old_n].clone()
    old_seq = bufs.seq_lens[:orig_reqs].clone()
    old_seq_np = bufs.seq_lens_np[:orig_reqs].copy()
    old_pad = bufs.is_padding[:old_n].clone()

    idx_np = safe_idx.cpu().numpy()  # D2H
    qsl_np = qsl.cpu().numpy()  # D2H
    seq_np = seq_lens.cpu().numpy().astype(np.int32, copy=False)  # D2H
    scheduled_np = num_sampled.cpu().numpy().astype(np.int32, copy=False)  # D2H
    computed_np = prefix.cpu().numpy().astype(np.int32, copy=False)  # D2H

    bufs.input_ids[:n].copy_(ids.to(dtype=bufs.input_ids.dtype))
    bufs.positions[:n].copy_(pos.to(dtype=bufs.positions.dtype))
    bufs.query_start_loc[: num_reqs + 1].copy_(qsl)
    bufs.seq_lens[:num_reqs].copy_(seq_lens.to(dtype=bufs.seq_lens.dtype))
    bufs.seq_lens_np[:num_reqs] = seq_np
    bufs.is_padding[:n].fill_(False)
    update_cos_sin(bufs.positions[:n])

    prefill_np = runner.req_states.prefill_len.np[idx_np]
    input_batch = AscendInputBatch(
        req_ids=[""] * num_reqs,
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=idx,
        idx_mapping_np=idx_np,
        expanded_idx_mapping=idx,
        expanded_local_pos=torch.zeros(
            num_reqs, dtype=torch.int32, device=device
        ),
        num_scheduled_tokens=scheduled_np,
        num_tokens=n,
        num_tokens_after_padding=n,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=bufs.query_start_loc[: num_reqs + 1],
        query_start_loc_np=qsl_np,
        seq_lens=bufs.seq_lens[:num_reqs],
        seq_lens_cpu_upper_bound=torch.from_numpy(np.array(seq_np, copy=True)),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=computed_np,
        prefill_len_np=prefill_np,
        num_computed_prefill_tokens_np=runner.req_states.num_computed_prefill_tokens[
            idx_np
        ],
        is_prefilling_np=np.zeros(num_reqs, dtype=bool),
        has_prefill=False,
        input_ids=bufs.input_ids[:n],
        positions=bufs.positions[:n],
        is_padding=bufs.is_padding[:n],
        logits_indices=bufs.query_start_loc[1 : num_reqs + 1] - 1,
        cu_num_logits=bufs.query_start_loc[: num_reqs + 1],
        cu_num_logits_np=qsl_np,
        has_structured_output_reqs=False,
        prompt_lens=None,
        max_query_len=int(scheduled_np.max()) if scheduled_np.size else 0,
        seq_lens_np=bufs.seq_lens_np[:num_reqs],
        attn_state=AscendAttentionState.ChunkedPrefill,
        tree_visibility=None,
        slot_positions=None,
    )

    out = None
    try:
        out = run_short_causal_forward(runner, input_batch)
    finally:
        bufs.input_ids[:old_n].copy_(old_ids)
        bufs.positions[:old_n].copy_(old_pos)
        bufs.query_start_loc[: orig_reqs + 1].copy_(old_qsl)
        bufs.seq_lens[:orig_reqs].copy_(old_seq)
        bufs.seq_lens_np[:orig_reqs] = old_seq_np
        bufs.is_padding[:old_n].copy_(old_pad)
        update_cos_sin(bufs.positions[:old_n])
    if out is None:
        return None
    _, aux = _split_model_output(out)
    return n, pos, qsl, aux


def _split_model_output(output):
    if isinstance(output, tuple):
        hidden, rest = output[0], output[1]
        if rest is None:
            return hidden, None
        if isinstance(rest, torch.Tensor):
            return hidden, [rest]
        return hidden, list(rest)
    return output, None
