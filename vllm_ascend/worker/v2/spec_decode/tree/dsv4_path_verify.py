from __future__ import annotations

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.ops.rotary_embedding import update_cos_sin
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import (
    iter_unique_kv_cache_tensors,
)
from vllm_ascend.worker.v2.spec_decode.tree.kv_project import (
    _group_is_linear_token_cache,
    run_short_causal_forward,
)

logger = init_logger("vllm." + __name__)

# First 1–2 tree-verify windows: enough to reconstruct serial vs packed.
_DIAG_WINDOWS = 2
_step_serial = False
_step_packed = False
_step_dsa_skip = False
_step_dsa_logged = False
_step_diag = False
_pack_skip_seen: set[str] = set()
# Host bool. None until TP is up so install can still log once per process.
_path_log_rank0: bool | None = None
_swa_idx_parts: list[tuple] = []


def _path_log_enabled() -> bool:
    """DSV4-PATH INFO on TP rank 0 only. Rank is a host int, never a device tensor."""
    global _path_log_rank0
    if _path_log_rank0 is not None:
        return _path_log_rank0
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            model_parallel_is_initialized,
        )

        if not model_parallel_is_initialized():
            return True
        _path_log_rank0 = get_tensor_model_parallel_rank() == 0
    except Exception:
        return True
    return _path_log_rank0


def path_log_enabled() -> bool:
    """Whether TP0 diagnostics are active; host-only."""
    return _path_log_enabled()


def path_log(msg: str, *args) -> None:
    if not _path_log_enabled():
        return
    logger.info("DSV4-PATH " + msg, *args)


def _reset_step_flags(*, diag: bool = False) -> None:
    global _step_serial, _step_packed, _step_dsa_skip, _step_dsa_logged, _step_diag
    _step_serial = False
    _step_packed = False
    _step_dsa_skip = False
    _step_dsa_logged = False
    _step_diag = diag


def _graph_mode_str(runner) -> str:
    cc = getattr(runner, "compilation_config", None)
    if cc is None:
        cc = getattr(getattr(runner, "vllm_config", None), "compilation_config", None)
    mode = getattr(cc, "cudagraph_mode", None)
    eager = bool(getattr(getattr(runner, "model_config", None), "enforce_eager", False))
    return f"{mode} eager={int(eager)}"


def _isolation_skip_reason(runner) -> str:
    from vllm_ascend.worker.v2.spec_decode import (
        dflash_tree_spec_enabled,
        dsv4_dspark_draft,
    )

    vllm_config = getattr(runner, "vllm_config", None)
    if not dflash_tree_spec_enabled(vllm_config):
        return "not_tree"
    from vllm_ascend.ascend_config import get_ascend_config

    if (get_ascend_config().tree_spec_config.topk or 0) <= 1:
        return "topk==1"
    speculator = getattr(runner, "speculator", None)
    if bool(getattr(speculator, "_dsv4_dspark_draft", False)):
        return "needed"
    if dsv4_dspark_draft(vllm_config):
        return "needed"
    return "not_dsv4"


def note_dsa_skip_write(
    *,
    skip_write: bool,
    isolated: bool,
    has_path_qsl: bool,
    num_tokens: int,
) -> None:
    """Log DSA skip-write vs serial intercept. BUG if both serial and packed ran."""
    global _step_dsa_skip, _step_dsa_logged
    _step_dsa_skip = skip_write
    both = _step_serial and _step_packed
    if not _step_diag and not both:
        return
    if _step_dsa_logged and not both:
        return
    path_log(
        "dsa skip_write=%s isolated=%s path_qsl=%s num_tokens=%s "
        "serial_ran=%s packed_ran=%s%s",
        int(skip_write),
        int(isolated),
        int(has_path_qsl),
        num_tokens,
        int(_step_serial),
        int(_step_packed),
        " BUG_packed_and_serial" if both else "",
    )
    _step_dsa_logged = True


def is_dsv4_path_verify_proxy(model) -> bool:
    return type(model).__name__ == "_Dsv4PathVerifyProxy"


def _model_is_isolated(model) -> bool:
    if model is None:
        return False
    if is_dsv4_path_verify_proxy(model):
        return True
    inner = getattr(model, "_model", None)
    if is_dsv4_path_verify_proxy(inner):
        return True
    orig = getattr(model, "original_model", None)
    if is_dsv4_path_verify_proxy(orig):
        return True
    getter = getattr(model, "get_original_model", None)
    if callable(getter):
        orig = getter()
        if is_dsv4_path_verify_proxy(orig):
            return True
    return False


def dsv4_path_isolation_needed(runner) -> bool:
    """True when DSV4 tree target verify must isolate leaf-path KV."""
    from vllm_ascend.worker.v2.spec_decode import (
        dflash_tree_spec_enabled,
        dsv4_dspark_draft,
    )

    vllm_config = getattr(runner, "vllm_config", None)
    if not dflash_tree_spec_enabled(vllm_config):
        return False
    from vllm_ascend.ascend_config import get_ascend_config

    if (get_ascend_config().tree_spec_config.topk or 0) <= 1:
        return False
    speculator = getattr(runner, "speculator", None)
    if bool(getattr(speculator, "_dsv4_dspark_draft", False)):
        return True
    return dsv4_dspark_draft(vllm_config)


def ensure_dsv4_path_verifier(runner):
    """Create the verifier and wrap ``runner.model`` when isolation is needed.

    Called from both ``load_model`` and ``initialize_kv_cache`` so a missed
    load_model gate cannot leave packed TND without serial/CoW intercept.
    """
    if not dsv4_path_isolation_needed(runner):
        if not getattr(runner, "_dsv4_path_skip_logged", False):
            runner._dsv4_path_skip_logged = True
            path_log(
                "wrap/install skip reason=%s graph=%s",
                _isolation_skip_reason(runner),
                _graph_mode_str(runner),
            )
        return getattr(runner, "dsv4_path_verifier", None)
    if getattr(runner, "dsv4_path_verifier", None) is None:
        runner.dsv4_path_verifier = Dsv4PathVerifier(runner)
        path_log("wrap/install verifier=created graph=%s", _graph_mode_str(runner))
    model = getattr(runner, "model", None)
    if model is not None and not _model_is_isolated(model):
        runner.model = wrap_model_for_dsv4_path_verify(runner, model)
        path_log(
            "wrap/install intercept_ready=1 bare_model=%s graph=%s",
            type(model).__name__,
            _graph_mode_str(runner),
        )
    return runner.dsv4_path_verifier


def wrap_model_for_dsv4_path_verify(runner, model):
    """Intercept eager ``model()`` so serial leaf verify can replace packed TND."""
    runner._dsv4_bare_model = model
    return _Dsv4PathVerifyProxy(runner, model)


class _Dsv4PathVerifyProxy:
    def __init__(self, runner, model):
        object.__setattr__(self, "_runner", runner)
        object.__setattr__(self, "_model", model)

    def __call__(self, *args, **kwargs):
        global _step_serial, _step_packed, _step_dsa_skip, _step_dsa_logged, _step_diag
        runner = object.__getattribute__(self, "_runner")
        model = object.__getattribute__(self, "_model")
        verifier = getattr(runner, "dsv4_path_verifier", None)
        if verifier is None:
            path_log("wrap intercept=0 reason=no_verifier graph=%s", _graph_mode_str(runner))
            return model(*args, **kwargs)
        if verifier._in_isolated_forward:
            return model(*args, **kwargs)
        pack = verifier._pending_pack
        reason = verifier.serial_skip_reason()
        intercept = reason is None
        _step_diag = pack is not None and verifier._diag_left > 0
        if pack is not None:
            verifier._wrap_hit = True
            _step_serial = intercept
            _step_packed = not intercept
            _step_dsa_skip = False
            _step_dsa_logged = False
            nqp = getattr(getattr(runner, "speculator", None), "num_query_per_req", None)
            path_log(
                "wrap intercept=%s reason=%s cow_ready=%s isolated=%s "
                "num_tokens=%s num_leaves=%s num_query_per_req=%s graph=%s",
                int(intercept),
                reason or "serial",
                int(verifier.cow_ready),
                int(bool(getattr(verifier._pending_batch, "tree_path_kv_isolated", False))),
                pack.num_tokens,
                pack.num_paths,
                nqp,
                _graph_mode_str(runner),
            )
        elif intercept or verifier._diag_left > 0:
            path_log(
                "wrap intercept=%s reason=%s graph=%s",
                int(intercept),
                reason or "serial",
                _graph_mode_str(runner),
            )
        try:
            if intercept:
                return verifier.run_serial(model, kwargs)
            return model(*args, **kwargs)
        finally:
            if pack is not None and verifier._diag_left > 0:
                verifier._diag_left -= 1
            verifier.clear_pending()

    def __getattr__(self, name):
        # ``model.forward`` must not skip isolation; GPU eager uses ``model()``
        # but piecewise / EP helpers may grab ``.forward`` off the module.
        if name == "forward":
            return self.__call__
        return getattr(object.__getattribute__(self, "_model"), name)

    def __setattr__(self, name, value):
        if name in ("_runner", "_model"):
            object.__setattr__(self, name, value)
            return
        setattr(object.__getattribute__(self, "_model"), name, value)


class Dsv4PathVerifier:
    """Leaf-path KV isolation for DSV4 tree target verify.

    Host metadata (path counts, CPU prefix sums) is taken at the prepare
    boundary. Token / block-table tensors stay on device.
    """

    def __init__(self, runner):
        self.runner = runner
        self.scratch_extra = 0
        self.group_orig_nblocks: list[int] = []
        self.pages_per_path: list[int] = []
        self.kernel_bs_host: list[int] = []
        self.slot_mapping_enabled_host: list[bool] = []
        self.max_paths = 0
        self.cow_ready = False
        self._scratch_attempted = False
        self._pending_pack = None
        self._pending_batch = None
        self._in_isolated_forward = False
        self._diag_left = _DIAG_WINDOWS
        self._wrap_hit = False
        self._expect_verify = False
        self._cur_leaves: list[int] = []
        # Scheduler 1+budget query_start_loc. Path pack rewrites qsl to
        # leaf-token counts; post_update must keep the logical length.
        self._post_query_start_loc = None

    def install_scratch(self) -> None:
        """Grow each unique KV tensor by private tail pages. Fail closed to serial."""
        self._scratch_attempted = True
        runner = self.runner
        from vllm_ascend.ascend_config import get_ascend_config

        budget = int(get_ascend_config().tree_spec_config.budget or 0)
        spec_len = int(runner.num_speculative_steps)
        max_paths = int(runner.max_num_reqs) * max(budget, 1)
        kernel_bs = _host_kernel_block_sizes(runner)
        if not kernel_bs:
            path_log(
                "cow skip reason=empty_kernel_block_sizes serial=1"
            )
            return
        self.kernel_bs_host = kernel_bs
        pages_per_path = [
            1 + (spec_len + b - 1) // b for b in self.kernel_bs_host
        ]
        extra = max_paths * max(pages_per_path)
        slot_en = getattr(runner.block_tables, "_slot_mapping_enabled", None)
        if slot_en is None:
            slot_en = [True] * len(self.kernel_bs_host)
        self.slot_mapping_enabled_host = [bool(x) for x in list(slot_en)]
        orig = []
        for group in runner.kv_cache_config.kv_cache_groups:
            n = None
            for layer_name in group.layer_names:
                layer = runner.compilation_config.static_forward_context.get(layer_name)
                for tensor in iter_unique_kv_cache_tensors(getattr(layer, "kv_cache", None)):
                    n = tensor.shape[0] if n is None else min(n, tensor.shape[0])
            orig.append(int(n or 0))
        if extra <= 0 or any(n <= 0 for n in orig) or extra > min(orig):
            path_log(
                "cow skip reason=scratch_size extra=%s orig=%s serial=1",
                extra,
                orig,
            )
            return
        if not _expand_unique_kv_block0(runner, extra):
            path_log("cow skip reason=overlapping_kv_views serial=1")
            return
        self.scratch_extra = extra
        self.group_orig_nblocks = orig
        self.pages_per_path = pages_per_path
        self.max_paths = max_paths
        self.cow_ready = True
        path_log(
            "cow extra_blocks=%s max_paths=%s pages_per_path=%s",
            extra,
            max_paths,
            pages_per_path,
        )

    def prepare_batch(self, input_batch, pack) -> None:
        """Attach path pack; fork CoW tables when scratch is live."""
        if not self.cow_ready and not self._scratch_attempted:
            self.install_scratch()
        self._pending_pack = pack
        self._pending_batch = input_batch
        self._expect_verify = True
        self._wrap_hit = False
        _reset_step_flags(diag=self._diag_left > 0)
        input_batch.tree_path_kv_isolated = False
        input_batch.tree_path_block_tables = None
        input_batch.tree_path_slot_mappings = None
        if not self.cow_ready or pack.num_paths > self.max_paths:
            return
        forked, slots = _fork_path_kv(self, pack, input_batch)
        if forked is None:
            return
        input_batch.tree_path_block_tables = forked
        input_batch.tree_path_slot_mappings = slots
        input_batch.tree_path_kv_isolated = True

    def serial_skip_reason(self) -> str | None:
        """None means serial should intercept this ``model()``. Else why not."""
        if self._in_isolated_forward:
            return "nested_isolated_forward"
        batch = self._pending_batch
        if batch is None:
            return "no_pending_batch"
        if getattr(batch, "tree_path_query_start_loc", None) is None:
            return "not_tree_path_pack"
        if getattr(batch, "has_prefill", False):
            return "has_prefill"
        if getattr(batch, "is_dummy", False):
            return "dummy"
        if bool(getattr(batch, "tree_path_kv_isolated", False)):
            return "cow_isolated"
        return None

    def should_run_serial(self) -> bool:
        return self.serial_skip_reason() is None

    def clear_pending(self) -> None:
        self._pending_pack = None
        self._pending_batch = None

    def run_serial(self, model, kwargs):
        """One causal write-then-attend per leaf, then roll suffix KV/state back."""
        pack = self._pending_pack
        batch = self._pending_batch
        runner = self.runner
        self._in_isolated_forward = True
        try:
            return _serial_leaf_forwards(runner, model, kwargs, batch, pack)
        finally:
            self._in_isolated_forward = False


def _host_kernel_block_sizes(runner) -> list[int]:
    """Host kernel page sizes at KV-init. Empty means CoW cannot size scratch."""
    raw = getattr(getattr(runner, "block_tables", None), "kernel_block_sizes", None)
    if raw is None:
        raw = getattr(runner, "kernel_block_sizes", None)
    if raw is None:
        return []
    if torch.is_tensor(raw):
        if raw.numel() == 0:
            return []
        raw = raw.detach().tolist()  # D2H at KV init
    elif len(raw) == 0:
        return []
    out: list[int] = []
    for x in list(raw):
        if isinstance(x, (list, tuple)):
            x = x[0] if x else 1
        if torch.is_tensor(x):
            x = x.detach().tolist()  # D2H at KV init
            if isinstance(x, list):
                x = x[0] if x else 1
        out.append(max(int(x), 1))
    return out


def _expand_unique_kv_block0(runner, extra_blocks: int) -> bool:
    ctx = runner.compilation_config.static_forward_context
    groups: dict[int, list[torch.Tensor]] = {}
    for layer in ctx.values():
        for tensor in iter_unique_kv_cache_tensors(getattr(layer, "kv_cache", None)):
            ptr = tensor.untyped_storage().data_ptr()
            groups.setdefault(ptr, []).append(tensor)
    mapping: dict[int, torch.Tensor] = {}
    for ptr, tensors in groups.items():
        sigs = {(t.shape, t.stride(), t.storage_offset()) for t in tensors}
        if len(sigs) > 1:
            return False
        src = tensors[0]
        extra = src.new_zeros((extra_blocks, *src.shape[1:]))
        mapping[ptr] = torch.cat([src, extra], dim=0)
    for layer in ctx.values():
        cache = getattr(layer, "kv_cache", None)
        if cache is None:
            continue
        layer.kv_cache = _remap_kv_cache(cache, mapping)
    return True


def _remap_kv_cache(cache, mapping: dict[int, torch.Tensor]):
    if isinstance(cache, torch.Tensor):
        return mapping.get(cache.untyped_storage().data_ptr(), cache)
    if isinstance(cache, list):
        return [_remap_kv_cache(x, mapping) for x in cache]
    if isinstance(cache, tuple):
        return tuple(_remap_kv_cache(x, mapping) for x in cache)
    return cache


def _fork_path_kv(verifier: Dsv4PathVerifier, pack, input_batch):
    runner = verifier.runner
    num_paths = pack.num_paths
    path_req = pack.path_req_idx
    prefix = pack.prefix_lens[path_req]
    qlens = pack.path_qlens.to(dtype=prefix.dtype)
    end_pos = prefix + qlens - 1
    gathered = runner.block_tables.gather_block_tables(
        input_batch.idx_mapping[: pack.num_reqs],
        num_reqs_padded=pack.num_reqs,
    )
    n_tok = pack.num_tokens
    n_groups = len(gathered)
    if n_groups != len(verifier.group_orig_nblocks):
        return None
    slot_out = torch.full(
        (n_groups, n_tok),
        PAD_SLOT_ID,
        dtype=torch.long,
        device=pack.positions.device,
    )
    forked_tables = []
    path_ids = torch.arange(num_paths, device=path_req.device, dtype=torch.long)
    groups = runner.kv_cache_config.kv_cache_groups
    for g, src_table in enumerate(gathered):
        # C4/state/circular: never remap table columns as token pages (bs=2 OOB).
        if g >= len(groups) or not _group_is_linear_token_cache(groups[g]):
            forked_tables.append(src_table[path_req].clone())
            continue
        orig = verifier.group_orig_nblocks[g]
        bs = verifier.kernel_bs_host[g]
        pages_cap = verifier.pages_per_path[g]
        src_rows = src_table[path_req]
        forked = src_rows.clone()
        page0 = torch.div(prefix, bs, rounding_mode="floor")
        page1 = torch.div(end_pos, bs, rounding_mode="floor")
        max_bi = forked.shape[1] - 1
        scratch_base = orig + path_ids * pages_cap
        for k in range(pages_cap):
            page = (page0 + k).clamp(min=0, max=max_bi).to(dtype=torch.long)
            use = (page0 + k) <= page1
            new_id = (scratch_base + k).to(dtype=forked.dtype)
            cur = forked.gather(1, page.unsqueeze(1)).squeeze(1)
            forked.scatter_(1, page.unsqueeze(1), torch.where(use, new_id, cur).unsqueeze(1))
        _copy_mixed_pages(runner, g, src_rows, forked, page0, prefix, bs)
        forked_tables.append(forked)
        if verifier.slot_mapping_enabled_host[g]:
            pos = pack.positions.to(dtype=torch.long)
            bi = torch.div(pos, bs, rounding_mode="floor").clamp(min=0, max=max_bi)
            off = pos % bs
            block_ids = forked[pack.token_path, bi]
            slot_out[g] = block_ids.to(dtype=torch.long) * bs + off
    return tuple(forked_tables), slot_out


def _copy_mixed_pages(runner, group_id, src_rows, forked, page0, prefix, bs):
    need = (prefix % bs) != 0
    max_bi = src_rows.shape[1] - 1
    page = page0.clamp(min=0, max=max_bi).to(dtype=torch.long)
    src_ids = src_rows.gather(1, page.unsqueeze(1)).squeeze(1).to(dtype=torch.long)
    dst_ids = forked.gather(1, page.unsqueeze(1)).squeeze(1).to(dtype=torch.long)
    group = runner.kv_cache_config.kv_cache_groups[group_id]
    ctx = runner.compilation_config.static_forward_context
    seen: set[int] = set()
    for layer_name in group.layer_names:
        layer = ctx.get(layer_name)
        for cache in iter_unique_kv_cache_tensors(getattr(layer, "kv_cache", None)):
            ptr = cache.untyped_storage().data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            src = cache.index_select(0, src_ids)
            dst = cache.index_select(0, dst_ids)
            mask = need.reshape((-1,) + (1,) * (src.ndim - 1))
            cache.index_copy_(0, dst_ids, torch.where(mask, src, dst))


def _mtp_hidden_buffer(runner):
    getter = getattr(runner.model, "get_mtp_target_hidden_states", None)
    if not callable(getter):
        return None
    return getter()


def _serial_leaf_forwards(runner, model, kwargs, batch, pack):
    n_tok = pack.num_tokens
    hidden_out = None
    aux_out = None
    mtp_buf = _mtp_hidden_buffer(runner)
    mtp_out = None
    path_req_cpu = pack.path_req_idx.cpu()  # D2H
    qsl_cpu = pack.path_query_start_loc.cpu()  # D2H
    num_paths = pack.num_paths
    num_reqs = pack.num_reqs
    paths_by_req: list[list[int]] = [[] for _ in range(num_reqs)]
    for p in range(num_paths):
        paths_by_req[int(path_req_cpu[p])].append(p)
    max_wave = max((len(p) for p in paths_by_req), default=0)
    bufs = runner.input_buffers
    orig_n = int(batch.num_tokens)
    old_ids = bufs.input_ids[:orig_n].clone()
    old_pos = bufs.positions[:orig_n].clone()
    old_qsl = bufs.query_start_loc[: batch.num_reqs + 1].clone()
    old_seq = bufs.seq_lens[: batch.num_reqs].clone()
    old_seq_np = bufs.seq_lens_np[: batch.num_reqs].copy()
    old_pad = bufs.is_padding[:orig_n].clone()

    diag = bool(getattr(getattr(runner, "dsv4_path_verifier", None), "_diag_left", 0))
    if diag:
        _log_serial_leaf_summary(pack, qsl_cpu, num_paths)

    for wave in range(max_wave):
        chosen = [paths_by_req[r][wave] for r in range(num_reqs) if wave < len(paths_by_req[r])]
        if not chosen:
            continue
        verifier = getattr(runner, "dsv4_path_verifier", None)
        if verifier is not None:
            verifier._cur_leaves = chosen
        snap = _snapshot_suffix(runner, batch, pack, chosen)
        if diag:
            _log_snapshot_breakdown(snap)
        wave_hidden = _forward_path_wave(
            runner, model, kwargs, batch, pack, chosen, path_req_cpu, qsl_cpu
        )
        hid, aux = _split_model_output(wave_hidden)
        if hidden_out is None:
            hidden_out = hid.new_zeros((n_tok, *hid.shape[1:]))
        if mtp_buf is not None and mtp_out is None:
            mtp_out = mtp_buf.new_zeros((n_tok, mtp_buf.shape[-1]))
        offset = 0
        for p in chosen:
            s = int(qsl_cpu[p])
            e = int(qsl_cpu[p + 1])
            n = e - s
            hidden_out[s:e] = hid[offset : offset + n]
            if mtp_out is not None:
                mtp_out[s:e] = mtp_buf[offset : offset + n]
            if aux:
                if aux_out is None:
                    aux_out = [a.new_zeros((n_tok, *a.shape[1:])) for a in aux]
                for i, a in enumerate(aux):
                    aux_out[i][s:e] = a[offset : offset + n]
            offset += n
        snap.restore()
        if diag:
            path_log(
                "serial restore ok=1 n_pieces=%s linear=%s circular=%s group_ids=%s",
                len(snap.pieces),
                snap.n_linear,
                snap.n_circular,
                list(snap.circular_group_ids),
            )

    bufs.input_ids[:orig_n].copy_(old_ids)
    bufs.positions[:orig_n].copy_(old_pos)
    bufs.query_start_loc[: batch.num_reqs + 1].copy_(old_qsl)
    bufs.seq_lens[: batch.num_reqs].copy_(old_seq)
    bufs.seq_lens_np[: batch.num_reqs] = old_seq_np
    bufs.is_padding[:orig_n].copy_(old_pad)
    update_cos_sin(bufs.positions[:orig_n])
    if mtp_out is not None and mtp_buf is not None:
        mtp_buf[:n_tok].copy_(mtp_out)
    if hidden_out is None:
        sample = kwargs.get("inputs_embeds")
        if sample is None:
            sample = pack.input_ids
        hidden_out = sample.new_zeros((n_tok, *getattr(sample, "shape", (n_tok,))[1:]))
    if aux_out is None:
        return hidden_out
    return hidden_out, aux_out


def _split_model_output(output):
    if isinstance(output, tuple):
        hidden, rest = output[0], output[1]
        if rest is None:
            return hidden, None
        if isinstance(rest, torch.Tensor):
            return hidden, [rest]
        return hidden, list(rest)
    return output, None


def _forward_path_wave(
    runner, model, kwargs, batch, pack, chosen: list[int], path_req_cpu, qsl_cpu
):
    device = pack.input_ids.device
    pieces_ids = []
    pieces_pos = []
    qlens = []
    req_locals = []
    for p in chosen:
        s = int(qsl_cpu[p])
        e = int(qsl_cpu[p + 1])
        pieces_ids.append(pack.input_ids[s:e])
        pieces_pos.append(pack.positions[s:e])
        qlens.append(e - s)
        req_locals.append(int(path_req_cpu[p]))
    n = sum(qlens)
    num_sub = len(chosen)
    ids = torch.cat(pieces_ids, dim=0).clamp(min=0)
    pos = torch.cat(pieces_pos, dim=0)
    qsl = torch.zeros(num_sub + 1, dtype=torch.int32, device=device)
    qsl[1:] = torch.tensor(qlens, dtype=torch.int32, device=device).cumsum(0)  # H2D
    req_idx = torch.tensor(req_locals, dtype=torch.long, device=device)  # H2D
    idx = batch.idx_mapping[req_idx]
    prefix = pack.prefix_lens[req_idx]
    qlen_t = torch.tensor(qlens, dtype=prefix.dtype, device=device)  # H2D
    seq_lens = prefix + qlen_t
    idx_np = idx.cpu().numpy()  # D2H
    qsl_np = qsl.cpu().numpy()  # D2H
    seq_np = seq_lens.cpu().numpy().astype(np.int32, copy=False)  # D2H
    scheduled_np = qlen_t.cpu().numpy().astype(np.int32, copy=False)  # D2H
    computed_np = prefix.cpu().numpy().astype(np.int32, copy=False)  # D2H

    bufs = runner.input_buffers
    bufs.input_ids[:n].copy_(ids.to(dtype=bufs.input_ids.dtype))
    bufs.positions[:n].copy_(pos.to(dtype=bufs.positions.dtype))
    bufs.query_start_loc[: num_sub + 1].copy_(qsl)
    bufs.seq_lens[:num_sub].copy_(seq_lens.to(dtype=bufs.seq_lens.dtype))
    bufs.seq_lens_np[:num_sub] = seq_np
    bufs.is_padding[:n].fill_(False)
    update_cos_sin(bufs.positions[:n])

    input_batch = AscendInputBatch(
        req_ids=[""] * num_sub,
        num_reqs=num_sub,
        num_reqs_after_padding=num_sub,
        idx_mapping=idx,
        idx_mapping_np=idx_np,
        expanded_idx_mapping=idx,
        expanded_local_pos=torch.zeros(num_sub, dtype=torch.int32, device=device),
        num_scheduled_tokens=scheduled_np,
        num_tokens=n,
        num_tokens_after_padding=n,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=bufs.query_start_loc[: num_sub + 1],
        query_start_loc_np=qsl_np,
        seq_lens=bufs.seq_lens[:num_sub],
        seq_lens_cpu_upper_bound=torch.from_numpy(np.array(seq_np, copy=True)),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=computed_np,
        prefill_len_np=runner.req_states.prefill_len.np[idx_np],
        num_computed_prefill_tokens_np=runner.req_states.num_computed_prefill_tokens[
            idx_np
        ],
        is_prefilling_np=np.zeros(num_sub, dtype=bool),
        has_prefill=False,
        input_ids=bufs.input_ids[:n],
        positions=bufs.positions[:n],
        is_padding=bufs.is_padding[:n],
        logits_indices=bufs.query_start_loc[1 : num_sub + 1] - 1,
        cu_num_logits=bufs.query_start_loc[: num_sub + 1],
        cu_num_logits_np=qsl_np,
        has_structured_output_reqs=False,
        prompt_lens=None,
        max_query_len=int(scheduled_np.max()) if scheduled_np.size else 0,
        seq_lens_np=bufs.seq_lens_np[:num_sub],
        attn_state=AscendAttentionState.ChunkedPrefill,
        tree_visibility=None,
        slot_positions=bufs.positions[:n],
        tree_path_kv_isolated=True,
    )
    model_kwargs = dict(kwargs)
    model_kwargs["input_ids"] = input_batch.input_ids
    model_kwargs["positions"] = input_batch.positions
    model_kwargs["inputs_embeds"] = None
    return run_short_causal_forward(
        runner,
        input_batch,
        model=model,
        model_kwargs=model_kwargs,
    )


class _SuffixSnapshot:
    def __init__(
        self,
        pieces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        n_linear: int = 0,
        n_circular: int = 0,
        circular_group_ids: tuple[int, ...] = (),
    ):
        self.pieces = pieces
        self.n_linear = n_linear
        self.n_circular = n_circular
        self.circular_group_ids = circular_group_ids

    def restore(self) -> None:
        for cache, idx, data in self.pieces:
            cache.index_copy_(0, idx, data)


def _log_snapshot_breakdown(snap: _SuffixSnapshot) -> None:
    miss = ""
    if snap.n_circular == 0:
        miss = " BUG_circular=0_skip_write=0"
    path_log(
        "serial snapshot n_pieces=%s linear=%s circular=%s group_ids=%s%s",
        len(snap.pieces),
        snap.n_linear,
        snap.n_circular,
        list(snap.circular_group_ids),
        miss,
    )


def _cache_view_key(cache: torch.Tensor) -> tuple:
    """Identity of one KV view. Shared DSV4 backing must not collapse layers."""
    return (
        cache.untyped_storage().data_ptr(),
        cache.storage_offset(),
        tuple(cache.shape),
        tuple(cache.stride()),
        cache.dtype,
    )


def _log_serial_leaf_summary(pack, qsl_cpu, num_paths: int) -> None:
    hist: dict[tuple[int, int], int] = {}
    n0 = first0 = last0 = -1
    for p in range(num_paths):
        s = int(qsl_cpu[p])
        e = int(qsl_cpu[p + 1])
        n_leaf = e - s
        last_pos = first_pos = -1
        if n_leaf > 0:
            first_pos = int(pack.positions[s].item())  # D2H
            last_pos = int(pack.positions[e - 1].item())  # D2H
        hist[(n_leaf, last_pos)] = hist.get((n_leaf, last_pos), 0) + 1
        if p == 0:
            n0, first0, last0 = n_leaf, first_pos, last_pos
    path_log(
        "serial leaf0 n_tok=%s first_pos=%s last_pos=%s n_leaves=%s hist=%s",
        n0,
        first0,
        last0,
        num_paths,
        sorted(hist.items()),
    )


def tree_token_stats(tokens: torch.Tensor) -> tuple[int, int, int]:
    """min / max / neg_count of a tree token buffer. Diag D2H only."""
    if tokens.numel() == 0:
        return -1, -1, 0
    tmin = int(tokens.amin().item())  # D2H
    tmax = int(tokens.amax().item())  # D2H
    nneg = int((tokens < 0).sum().item())  # D2H
    return tmin, tmax, nneg


def tree_token_head(tokens: torch.Tensor, k: int = 4) -> list[int]:
    """First ``k`` tree token ids. Diag D2H on TP0 only."""
    if not _path_log_enabled() or tokens.numel() == 0:
        return []
    flat = tokens.reshape(-1)
    n = min(k, int(flat.numel()))
    return [int(flat[i].item()) for i in range(n)]  # D2H


def pos_span(pos: torch.Tensor | None, n: int) -> tuple[int, int]:
    """Position min/max of ``pos[:n]``. Diag D2H on TP0 only."""
    if not _path_log_enabled() or pos is None or n <= 0 or pos.numel() == 0:
        return -1, -1
    sl = pos[:n]
    return int(sl.amin().item()), int(sl.amax().item())  # D2H


def host_i0(t: torch.Tensor | np.ndarray | None) -> int:
    """First element as a host int. Diag D2H on TP0 only."""
    if not _path_log_enabled() or t is None:
        return -1
    if isinstance(t, np.ndarray):
        return int(t.reshape(-1)[0]) if t.size else -1
    if t.numel() == 0:
        return -1
    return int(t.reshape(-1)[0].item())  # D2H


def host_slot_head(t: torch.Tensor | None, n: int) -> list[int]:
    """First ``n`` 1D slots. Diag D2H on TP0 only."""
    if not _path_log_enabled() or t is None or t.numel() == 0 or n <= 0:
        return []
    flat = t.reshape(-1)
    k = min(n, int(flat.numel()))
    return [int(x) for x in flat[:k].tolist()]  # D2H


def host_float_head(t: torch.Tensor | None, n: int) -> list[float]:
    """First ``n`` scalar values. Diag D2H on TP0 only."""
    if not _path_log_enabled() or t is None or t.numel() == 0 or n <= 0:
        return []
    flat = t.detach().float().reshape(-1)
    k = min(n, int(flat.numel()))
    return [round(float(x), 3) for x in flat[:k].tolist()]  # D2H


def tensor_rms(t: torch.Tensor | None) -> float:
    """Mean RMS of ``t``. Diag D2H on TP0 only."""
    if not _path_log_enabled() or t is None or t.numel() == 0:
        return -1.0
    return float(t.detach().float().square().mean().sqrt().item())  # D2H


def tensor_signature(t: torch.Tensor | None) -> list[float]:
    """First scalar plus first/last row means. Diag D2H on TP0 only."""
    if not _path_log_enabled() or t is None or t.numel() == 0:
        return []
    rows = t.detach().float().reshape(t.shape[0], -1)
    sig = torch.stack((rows[0, 0], rows[0].mean(), rows[-1].mean()))
    return [round(float(x), 5) for x in sig.tolist()]  # D2H


def log_draft_source(
    hidden: torch.Tensor | None,
    aux: list[torch.Tensor] | None,
    n_tok: int,
    layer_ids: list[int] | None = None,
) -> None:
    """Target hidden rows consumed by DSpark context projection."""
    if not _path_log_enabled():
        return
    hidden = None if hidden is None else hidden[:n_tok]
    active_aux = [] if not aux else [a[:n_tok] for a in aux]
    active_layer_ids = [] if layer_ids is None else layer_ids
    aux_parts = [
        f"{active_layer_ids[i] if i < len(active_layer_ids) else i}:"
        f"r={tensor_rms(a):.4f},sig={tensor_signature(a)}"
        for i, a in enumerate(active_aux)
    ]
    path_log(
        "draft_src used=%s n=%s hidden_rms=%.4f hidden_sig=%s aux=[%s]",
        "aux" if active_aux else "hidden",
        n_tok,
        tensor_rms(hidden),
        tensor_signature(hidden),
        " | ".join(aux_parts),
    )


def log_tree_tokens(tokens: torch.Tensor, num_nodes: int = -1) -> None:
    """tree_tokens min/max/head. Diag D2H on TP0 only."""
    flush_swa_idx_log()
    if not _path_log_enabled():
        return
    tmin, tmax, nneg = tree_token_stats(tokens)
    path_log(
        "tree_tokens min=%s max=%s neg_count=%s head=%s num_nodes=%s",
        tmin,
        tmax,
        nneg,
        tree_token_head(tokens),
        num_nodes,
    )


def log_markov_tree(
    root_tokens: torch.Tensor,
    base_top1: torch.Tensor,
    corrected_top1: torch.Tensor,
    spine_tokens: torch.Tensor,
    spine_margins: torch.Tensor,
) -> None:
    """One request's base-vs-Markov greedy spine. Diag D2H only."""
    flush_swa_idx_log()
    if not _path_log_enabled() or root_tokens.numel() == 0:
        return
    base = host_slot_head(base_top1[0], int(base_top1.shape[-1]))
    corrected = host_slot_head(corrected_top1[0], int(corrected_top1.shape[-1]))
    spine = host_slot_head(spine_tokens[0], int(spine_tokens.shape[-1]))
    path_log(
        "markov root=%s base=%s corrected=%s spine=%s margin=%s changed=%s",
        host_i0(root_tokens),
        base,
        corrected,
        spine,
        host_float_head(spine_margins[0], int(spine_margins.shape[-1])),
        sum(a != b for a, b in zip(base, corrected)),
    )


def log_tree_plan(
    tokens: torch.Tensor,
    depths: torch.Tensor,
    parents: torch.Tensor,
    first_child: torch.Tensor,
    num_nodes: torch.Tensor,
    spec_len: int,
) -> None:
    """One request's depth budget and root branches. Diag D2H only."""
    if not _path_log_enabled() or tokens.numel() == 0:
        return
    n = min(host_i0(num_nodes), int(tokens.shape[1]))
    tok = host_slot_head(tokens[0], n)
    dep = host_slot_head(depths[0], n)
    par = host_slot_head(parents[0], n)
    child = host_slot_head(first_child[0], n + 1)
    depth_hist = [sum(d == level for d in dep) for level in range(1, spec_len + 1)]
    roots = [t for t, p in zip(tok, par) if p == 0]
    leaf_hist = [
        (level, sum(d == level and child[i + 1] < 0 for i, d in enumerate(dep)))
        for level in range(1, spec_len + 1)
    ]
    path_log(
        "tree_plan depth=%s roots=%s leaf_depth=%s",
        depth_hist,
        roots[:8],
        [(d, nleaf) for d, nleaf in leaf_hist if nleaf],
    )


def log_tree_accept(
    tokens: torch.Tensor,
    depths: torch.Tensor,
    path_node_ids: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    spec_len: int,
) -> None:
    """Accepted node path and whether it used a non-spine branch. Diag D2H only."""
    if not _path_log_enabled() or path_node_ids.numel() == 0:
        return
    sampled_n = host_i0(num_sampled)
    nodes = [x for x in host_slot_head(path_node_ids[0], spec_len) if x >= 0]
    tree_tokens = host_slot_head(tokens[0], int(tokens.shape[1]))
    tree_depths = host_slot_head(depths[0], int(depths.shape[1]))
    path_tokens = [tree_tokens[node - 1] for node in nodes]
    path_depths = [tree_depths[node - 1] for node in nodes]
    off = [i for i, (node, depth) in enumerate(zip(nodes, path_depths)) if node != depth]
    sampled = host_slot_head(sampled_token_ids[0], max(sampled_n, 0))
    path_log(
        "accept_path sampled=%s nodes=%s depth=%s tokens=%s off_spine=%s "
        "first_off=%s out=%s",
        sampled_n,
        nodes,
        path_depths,
        path_tokens,
        len(off),
        off[0] if off else -1,
        sampled,
    )


def _slot_1d_span(slot_mapping: torch.Tensor | None, n_tok: int) -> tuple[int, int, torch.Tensor | None]:
    """Valid slot min/max plus 1D view. Diag D2H on TP0 only."""
    if slot_mapping is None or slot_mapping.numel() == 0 or n_tok <= 0:
        return -1, -1, slot_mapping
    sm = slot_mapping[:n_tok]
    slot_1d = sm[:, 0] if sm.ndim == 2 else sm
    valid = slot_1d >= 0
    if not int(valid.any().item()):  # D2H
        return -1, -1, slot_1d
    cols = slot_1d[valid]
    return int(cols.amin().item()), int(cols.amax().item()), slot_1d  # D2H


def log_swa_indices(
    indices: torch.Tensor | None,
    *,
    compressor_ratio: int = -1,
    block_table: torch.Tensor | None = None,
    gkey: str = "",
    block_size: int = -1,
) -> None:
    """Buffer one SWA index row; ``flush_swa_idx_log`` emits a single TP0 line."""
    if not _path_log_enabled() or indices is None or indices.numel() == 0:
        return
    row = indices.reshape(indices.shape[0], -1)[0]
    vals = [int(x) for x in row.tolist()]  # D2H
    valid = [x for x in vals if x >= 0]
    nvalid = len(valid)
    first = valid[:8]
    vmin = min(valid) if valid else -1
    vmax = max(valid) if valid else -1
    bt0 = host_i0(None if block_table is None else block_table.reshape(-1))
    _swa_idx_parts.append((first, nvalid, vmin, vmax, compressor_ratio, bt0, gkey, block_size))


def flush_swa_idx_log() -> None:
    """Emit buffered SWA index rows as one TP0 line."""
    global _swa_idx_parts
    if not _path_log_enabled() or not _swa_idx_parts:
        return
    parts = _swa_idx_parts
    _swa_idx_parts = []
    if len(parts) == 1:
        first, nvalid, vmin, vmax, ratio, bt0, gkey, bs = parts[0]
        path_log(
            "swa_idx first=%s nvalid=%s vmin=%s vmax=%s ratio=%s bt0=%s gkey=%s bs=%s",
            first,
            nvalid,
            vmin,
            vmax,
            ratio,
            bt0,
            gkey,
            bs,
        )
        return
    chunks = []
    for first, nvalid, vmin, vmax, ratio, bt0, gkey, bs in parts:
        chunks.append(
            f"first={first} nvalid={nvalid} vmin={vmin} vmax={vmax} "
            f"ratio={ratio} bt0={bt0} gkey={gkey} bs={bs}"
        )
    path_log("swa_idx %s", " | ".join(chunks))


def log_draft_meta(
    *,
    prefix_len: int,
    num_computed: int,
    batch_seq0: int,
    buf_seq0_pre: int,
    buf_seq0: int,
    nqp: int,
    query0: int,
    bonus: int,
    tree0: int,
    tree_head: list[int],
    qpos0: int,
    ctx_slots: list[int],
    qslots: list[int],
    sequential: int,
    gid: int,
    ngroups: int,
    layer_gidx: int | list[int],
    layer_gidx_set: int,
    prefilling: int = -1,
    causal: int = -1,
    has_prefill: int = -1,
    force_prefill: int = -1,
) -> None:
    """One TP0 line after draft prepare. SWA/context slots, not group-0. Diag D2H only."""
    flush_swa_idx_log()
    if not _path_log_enabled():
        return
    swa_prefix0 = buf_seq0 - nqp if buf_seq0 >= 0 and nqp >= 0 else -1
    path_log(
        "draft_meta prefix_len=%s num_computed=%s batch_seq0=%s "
        "buf_seq0_pre=%s buf_seq0=%s swa_prefix0=%s nqp=%s "
        "query0=%s bonus=%s tree0=%s head=%s qpos0=%s "
        "ctx_slots=%s qslots=%s sequential=%s "
        "gid=%s ngroups=%s layer_gidx=%s layer_gidx_set=%s "
        "prefilling=%s causal=%s has_prefill=%s force_prefill=%s",
        prefix_len,
        num_computed,
        batch_seq0,
        buf_seq0_pre,
        buf_seq0,
        swa_prefix0,
        nqp,
        query0,
        bonus,
        tree0,
        tree_head,
        qpos0,
        ctx_slots,
        qslots,
        sequential,
        gid,
        ngroups,
        layer_gidx,
        layer_gidx_set,
        prefilling,
        causal,
        has_prefill,
        force_prefill,
    )


def log_draft_query(
    *,
    qids: list[int],
    qpos: list[int],
    q_h_rms: float,
    ctx_n: int,
    ctx_h_rms: float,
    ctx_head_rms: float,
    ctx_tail_rms: float,
    q_h_sig: list[float],
    ctx_sig: list[float],
    mask_id: int,
) -> None:
    """One TP0 line: draft query ids/pos and hidden fingerprints. Diag D2H only."""
    if not _path_log_enabled():
        return
    path_log(
        "draft_q ids=%s pos=%s h_rms=%.4f ctx_n=%s ctx_h_rms=%.4f "
        "ctx_head_rms=%.4f ctx_tail_rms=%.4f h_sig=%s ctx_sig=%s mask_id=%s",
        qids,
        qpos,
        q_h_rms,
        ctx_n,
        ctx_h_rms,
        ctx_head_rms,
        ctx_tail_rms,
        q_h_sig,
        ctx_sig,
        mask_id,
    )


def log_draft_kv_write(
    n_tok: int,
    positions: torch.Tensor | None,
    slot_mapping: torch.Tensor | None = None,
    num_computed: int = -1,
    slot_mapping_b: torch.Tensor | None = None,
    extra_n: int = 0,
    hidden: torch.Tensor | None = None,
) -> None:
    """One TP0 line for a DSpark context-KV scatter. Diag D2H only."""
    flush_swa_idx_log()
    if not _path_log_enabled():
        return
    pmin, pmax = pos_span(positions, n_tok)
    full = int(n_tok > 0 and pmin == 0 and n_tok == pmax + 1)
    mode = "full_prefix" if full else "chunk"
    smin, smax, slot_1d = _slot_1d_span(slot_mapping, n_tok)
    bmin, bmax, slot_b = _slot_1d_span(slot_mapping_b, n_tok)
    path_log(
        "draft_kv write n_tok=%s pos=%s..%s slot=%s..%s head=%s "
        "slot_b=%s..%s head_b=%s extra_n=%s mode=%s "
        "full_prefix=%s num_computed=%s implied_len=%s ctx_rms=%s ctx_sig=%s",
        n_tok,
        pmin,
        pmax,
        smin,
        smax,
        host_slot_head(slot_1d, min(n_tok, 8)),
        bmin,
        bmax,
        host_slot_head(slot_b, min(n_tok, 8)),
        extra_n,
        mode,
        full,
        num_computed,
        pmax + 1 if pmax >= 0 else -1,
        tensor_rms(None if hidden is None else hidden[:n_tok]),
        tensor_signature(None if hidden is None else hidden[:n_tok]),
    )


def log_draft_kv_layers(rows: list[tuple]) -> None:
    """One TP0 line: per DSpark layer scatter. ``rows`` are host tuples."""
    flush_swa_idx_log()
    if not _path_log_enabled() or not rows:
        return
    parts = []
    for row in rows:
        i, pfx, ratio, gidx, slots, wrote = row[:6]
        ptr = row[6] if len(row) > 6 else -1
        skip = row[7] if len(row) > 7 else 0
        kv = row[8] if len(row) > 8 else None
        cache_probe = row[9] if len(row) > 9 else None
        slot0 = host_i0(None if slots is None else slots.reshape(-1))
        ptr_s = f"{ptr & 0xFFFFFFFF:x}" if ptr != -1 else "-"
        cache_dtype = (
            "-" if cache_probe is None else str(cache_probe.dtype).removeprefix("torch.")
        )
        parts.append(
            f"i={i} pfx={pfx} r={ratio} gidx={gidx} slot0={slot0} "
            f"wrote={int(wrote)} ptr={ptr_s} skip={int(skip)} "
            f"kv_rms={tensor_rms(kv):.4f} kv_sig={tensor_signature(kv)} "
            f"cache={cache_dtype}:{tensor_signature(cache_probe)}"
        )
    path_log("kv_layers %s", " | ".join(parts))


def _native_allocated_block_ids(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Physical block ids in the request rows. Not prefix/block_size pages.

    Circular C4/state is a 1–2 block ring; unused table columns are 0. Drop 0
    when any real id is present, but keep a lone 0 (valid first pool block).
    """
    sel = torch.unique(table[idx].reshape(-1).to(dtype=torch.long))
    sel = sel[sel >= 0]
    nonzero = sel[sel > 0]
    if nonzero.numel() > 0:
        return nonzero
    return sel


def _linear_suffix_block_ids(
    table: torch.Tensor,
    idx: torch.Tensor,
    prefix: torch.Tensor,
    spec_len: int,
    bs: int,
) -> torch.Tensor:
    max_block = table.shape[1] - 1
    page0 = torch.div(prefix, bs, rounding_mode="floor")
    n_pages = 2 + spec_len // max(bs, 1)
    off = torch.arange(n_pages, device=table.device, dtype=torch.long)
    pages = (page0.unsqueeze(1) + off.unsqueeze(0)).clamp(min=0, max=max_block)
    req_f = idx.unsqueeze(1).expand_as(pages)
    sel = torch.unique(table[req_f, pages].to(dtype=torch.long).reshape(-1))
    return sel[sel > 0]


def _snapshot_suffix(runner, batch, pack, chosen: list[int]) -> _SuffixSnapshot:
    """Clone SWA suffix pages and the whole allocated C4/state ring.

    DSV4 layers share one backing allocation. Key views by offset/shape, not
    ``data_ptr``: merging would restore SWA pages onto the wrong circular
    bytes. Circular groups use physical table ids (typically 1–2 blocks).
    """
    device = pack.input_ids.device
    path_ids = torch.tensor(chosen, dtype=torch.long, device=device)  # H2D
    req_local = pack.path_req_idx[path_ids]
    idx = batch.idx_mapping[req_local]
    prefix = pack.prefix_lens[req_local]
    spec_len = int(runner.num_speculative_steps)
    ctx = runner.compilation_config.static_forward_context
    verifier = getattr(runner, "dsv4_path_verifier", None)
    bs_list = list(getattr(verifier, "kernel_bs_host", None) or ())
    if not bs_list:
        bs_list = _host_kernel_block_sizes(runner)
    view_cache: dict[tuple, torch.Tensor] = {}
    view_sels: dict[tuple, list[torch.Tensor]] = {}
    view_circular: dict[tuple, bool] = {}
    view_groups: dict[tuple, set[int]] = {}
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        table = runner.block_tables.block_tables[group_id].gpu
        is_linear = _group_is_linear_token_cache(group)
        if is_linear:
            bs = bs_list[group_id] if group_id < len(bs_list) else 1
            sel = _linear_suffix_block_ids(table, idx, prefix, spec_len, bs)
        else:
            sel = _native_allocated_block_ids(table, idx)
        if sel.numel() == 0:
            continue
        for layer_name in group.layer_names:
            layer = ctx.get(layer_name)
            for cache in iter_unique_kv_cache_tensors(getattr(layer, "kv_cache", None)):
                key = _cache_view_key(cache)
                prev = view_cache.get(key)
                if prev is None or cache.numel() > prev.numel():
                    view_cache[key] = cache
                view_sels.setdefault(key, []).append(sel)
                view_circular[key] = view_circular.get(key, False) or (not is_linear)
                view_groups.setdefault(key, set()).add(group_id)
    pieces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    n_linear = 0
    n_circular = 0
    circular_groups: set[int] = set()
    for key, cache in view_cache.items():
        sel = torch.unique(torch.cat(view_sels[key]))
        sel = sel[sel >= 0]
        nonzero = sel[sel > 0]
        sel = nonzero if nonzero.numel() > 0 else sel
        if sel.numel() == 0:
            continue
        data = cache.index_select(0, sel)
        pieces.append((cache, sel, data.clone()))
        if view_circular[key]:
            n_circular += 1
            circular_groups.update(view_groups[key])
        else:
            n_linear += 1
    return _SuffixSnapshot(
        pieces,
        n_linear=n_linear,
        n_circular=n_circular,
        circular_group_ids=tuple(sorted(circular_groups)),
    )
