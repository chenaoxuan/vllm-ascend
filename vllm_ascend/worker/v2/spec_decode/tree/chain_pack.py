"""Native tree verification plans and accepted-node commitment.

Persistent history stays shared during verification. Node rows remain private
until the accepted branch is frozen and committed on device.
"""

from __future__ import annotations

from contextlib import contextmanager
from math import prod

import torch

from vllm_ascend.ops.triton.spec_decode.tree.node_pool import build_ancestors
from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance

_LAYOUT: dict | None = None
_PLAN: dict | None = None
_KV_STACK: list[list[torch.Tensor]] = []


class TreeHostMetadata:
    """CPU-only block IDs supplied by the scheduler, owned by the tree runner.

    This observes request updates without changing framework block allocation,
    staged device writes or synchronization. New requests replace old slot data.
    """

    def __init__(self):
        self.blocks: dict[str, list[list[int]]] = {}

    def add_requests(self, requests):
        for request in requests:
            self.blocks[request.req_id] = [list(ids) for ids in request.block_ids]

    def update_requests(self, requests, live_req_ids):
        self.blocks = {key: value for key, value in self.blocks.items() if key in live_req_ids}
        for req_id, groups in zip(requests.req_ids, requests.new_block_ids):
            if groups is not None:
                for saved, new in zip(self.blocks[req_id], groups):
                    saved.extend(new)

    def snapshot(self, req_ids, block_tables, cache_config):
        result = {}
        for group_id, group in enumerate(cache_config.kv_cache_groups):
            width = block_tables.block_tables[group_id].gpu.shape[1]
            expansion = block_tables.blocks_per_kv_block[group_id]
            rows, lengths = [], []
            for req_id in req_ids:
                ids = [
                    block * expansion + offset for block in self.blocks[req_id][group_id] for offset in range(expansion)
                ]
                if len(ids) > width:
                    raise RuntimeError("Tree CPU block-table mirror exceeds device row capacity")
                lengths.append(len(ids))
                rows.append(ids + [0] * (width - len(ids)))
            for layer_name in group.layer_names:
                result[layer_name] = rows, lengths
        return result


class _LayerBuffers:
    """Module-owned node storage reused between serialized verification rounds.

    Every acquire has distinct storage. Reset only after attention and accepted
    node commitment have been enqueued on the same stream. Plan inputs and
    cached TopK must never use this pool.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple, list[torch.Tensor]] = {}
        self._used: dict[tuple, int] = {}

    def acquire(
        self,
        kind: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        key = (kind, torch.device(device), dtype, tuple(shape[1:]))
        buffers = self._buffers.setdefault(key, [])
        index = self._used.get(key, 0)
        self._used[key] = index + 1
        if index == len(buffers):
            buffers.append(torch.empty(shape, dtype=dtype, device=device))
        elif buffers[index].shape[0] < shape[0]:
            buffers[index] = torch.empty(shape, dtype=dtype, device=device)
        return buffers[index][: shape[0]]

    def reset(self) -> None:
        self._used.clear()


def _upload_metadata(
    device: torch.device,
    *fields: tuple[list, torch.dtype, tuple[int, ...]],
    host_buffers: list[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Upload immutable plan metadata once, independently of layer scratch.

    Typed views retain each field's dtype and alignment. The NumPy views
    reference CPU storage only; no device values are read on host.
    """
    offsets, nbytes = [], 0
    for _, dtype, shape in fields:
        size = dtype.itemsize
        begin = (nbytes + size - 1) // size * size
        nbytes = begin + prod(shape) * size
        offsets.append((begin, nbytes))
    host = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=torch.device(device).type == "npu")
    target = torch.empty(nbytes, dtype=torch.uint8, device=device)
    result = []
    for (values, dtype, shape), (begin, end) in zip(fields, offsets):
        if end > begin:
            host[begin:end].view(dtype).view(shape).numpy()[...] = values
        result.append(target[begin:end].view(dtype).view(shape))
    # H2D: the runner retains this immutable pinned source with the round until
    # the next metadata boundary has completed. Layer scratch never owns it.
    host_buffers.append(host)
    target.copy_(host, non_blocking=True)
    return tuple(result)


def _key(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def set_tree_chain_layout(
    parents, num_nodes, depths, query_start_loc, *, query_start_loc_cpu, start_pos_cpu, max_depth, host_tables
) -> dict:
    global _LAYOUT, _PLAN
    if _PLAN is not None and not _PLAN.get("committed", False):
        raise RuntimeError("Previous tree verification was not committed")
    # The runner owns this round. Snapshot device inputs because proposal and
    # metadata buffers can reuse their addresses before the next verification.
    _LAYOUT = dict(
        parents=parents.clone(),
        num_nodes=num_nodes.clone(),
        depths=depths.clone(),
        qsl=query_start_loc.clone(),
        plan=None,
        qsl_cpu=[int(value) for value in query_start_loc_cpu],
        start_cpu=[int(value) for value in start_pos_cpu],
        max_depth=int(max_depth),
        host_tables=host_tables,
    )
    _PLAN = None
    return _LAYOUT


def clear_tree_chain_layout() -> None:
    global _LAYOUT, _PLAN
    if _PLAN is not None and not _PLAN.get("committed", False):
        raise RuntimeError("Cannot discard uncommitted tree state")
    _LAYOUT = None
    _PLAN = None


def chain_layout_active() -> bool:
    return _LAYOUT is not None


@contextmanager
def linear_prefill_scope():
    """Run ordinary prefill kernels without exposing the decode node plan."""
    global _LAYOUT, _PLAN
    layout, plan = _LAYOUT, _PLAN
    if _KV_STACK:
        raise RuntimeError("Cannot enter prefill inside a tree compressor scope")
    _LAYOUT = _PLAN = None
    try:
        yield
    finally:
        _LAYOUT, _PLAN = layout, plan


@contextmanager
def chain_kv_scope(caches):
    _KV_STACK.append([cache for cache in caches if cache is not None])
    try:
        yield
    finally:
        _KV_STACK.pop()


def _build_plan(num_tokens: int, device) -> dict | None:
    """Build a fixed-capacity device plan; host inputs were already CPU metadata."""
    global _PLAN
    if _PLAN is not None:
        return _PLAN
    if _LAYOUT is None:
        return None
    layout = _LAYOUT
    qsl, prefixes = layout["qsl_cpu"], layout["start_cpu"]
    n_req, depth_limit = len(prefixes), layout["max_depth"]
    if (
        n_req == 0
        or depth_limit <= 0
        or len(qsl) != n_req + 1
        or qsl[0] != 0
        or qsl[-1] > num_tokens
        or any(end <= begin for begin, end in zip(qsl, qsl[1:]))
        or min(prefixes) < 0
    ):
        raise RuntimeError("Invalid host tree query boundaries/capacity")
    req_index = [req for req in range(n_req) for _ in range(qsl[req], qsl[req + 1])]
    prefix_of = [prefixes[req] for req in req_index]
    n = len(req_index)
    host_buffers = []
    req_ids, host_qsl, start = _upload_metadata(
        device,
        (req_index, torch.long, (n,)),
        (qsl, torch.int32, (n_req + 1,)),
        (prefixes, torch.int32, (n_req,)),
        host_buffers=host_buffers,
    )
    tokens = torch.arange(n, dtype=torch.long, device=device)
    ancestor_ids, _, depths, hidden_ids, _ = build_ancestors(
        layout["parents"],
        layout["depths"],
        layout["num_nodes"],
        req_ids,
        host_qsl,
        depth_limit,
    )
    lens = (depths + 1).to(torch.int32)
    # FULL tree graphs read the current device prefixes on every replay.
    start = layout.get("graph_prefixes", start).index_select(0, req_ids)
    _PLAN = {
        "n": n,
        "n_req": n_req,
        "req_cu": qsl,
        "prefixes": prefix_of,
        "req_ids": req_ids,
        "depths": depths,
        "ancestors": ancestor_ids,
        "hidden_ids": hidden_ids,
        "max_q": depth_limit + 1,
        "max_kv": layout.get("graph_capacity", max(prefixes) + depth_limit + 1),
        "query_cu": torch.arange(n + 1, dtype=torch.int32, device=device),
        "start": start,
        "seqused": start + lens,
        "token_index": tokens,
        "host_uploads": host_buffers,
        "cache_views": {},
        "index_topk": {},
        "query_metadata": {},
        "node_kv": {},
        "node_records": [],
        "committed": False,
    }
    layout["plan"] = _PLAN
    return _PLAN


def _new_cache_views(buffers: _LayerBuffers, caches, num_pages):
    """Preserve aliases such as packed FP8 key/scale/full-cache views."""
    groups, result = [], [None] * len(caches)
    order = sorted(range(len(caches)), key=lambda i: caches[i].data_ptr())
    for i in order:
        cache = caches[i]
        if cache.ndim < 2 or any(stride < 0 for stride in cache.stride()):
            raise RuntimeError("Unsupported cache tensor strides")
        size = cache.element_size()
        step = int(cache.stride(0)) * size
        span = size + sum((int(n) - 1) * int(s) * size for n, s in zip(cache.shape[1:], cache.stride()[1:]))
        if step <= 0 or span > step:
            raise RuntimeError("Overlapping/invalid physical cache pages")
        storage = cache.untyped_storage().data_ptr()
        group = next(
            (
                g
                for g in groups
                if g["storage"] == storage
                and g["step"] == step
                and cache.data_ptr() - g["base"] >= 0
                and cache.data_ptr() - g["base"] + span <= step
            ),
            None,
        )
        if group is None:
            group = {
                "storage": storage,
                "step": step,
                "base": cache.data_ptr(),
                "bytes": buffers.acquire("cache", (num_pages * step,), torch.uint8, cache.device),
            }
            # Clear the full active workspace, including null page and padding.
            group["bytes"].zero_()
            groups.append(group)
        offset = cache.data_ptr() - group["base"]
        if offset % size:
            raise RuntimeError("Misaligned packed-cache view")
        result[i] = torch.empty(0, dtype=cache.dtype, device=cache.device).set_(
            group["bytes"].untyped_storage(),
            offset // size,
            (num_pages, *cache.shape[1:]),
            tuple(cache.stride()),
        )
    return result


def tree_cache_view(cache):
    """Resolve the cache matching temporary compressor slot IDs/block tables."""
    if cache is None or _LAYOUT is None:
        return cache
    if _PLAN is None or _key(cache) not in _PLAN["cache_views"]:
        raise RuntimeError("Tree cache writer/reader has no private workspace")
    return _PLAN["cache_views"][_key(cache)]


def cache_tree_topk(buffer, indices, offset=0):
    if _PLAN is None or buffer is None or offset != 0:
        raise RuntimeError("Invalid packed TopK cache write")
    if int(indices.shape[0]) != _PLAN["n"]:
        raise RuntimeError("TopK output does not use unique endpoint query rows")
    # The indexer owns a fresh output tensor; subsequent layers only read it.
    _PLAN["index_topk"][_key(buffer)] = indices


def get_tree_topk(buffer, offset=0):
    if _PLAN is None or buffer is None or offset != 0:
        raise RuntimeError("Invalid packed TopK cache read")
    indices = _PLAN["index_topk"].get(_key(buffer))
    if indices is None:
        raise RuntimeError("No TopK producer for this tree verification step")
    return indices


def finish_tree_layer() -> None:
    """Release layer cache bindings after all native consumers are queued."""
    if _PLAN is None:
        return
    if _KV_STACK:
        raise RuntimeError("Tree layer finished inside a compressor scope")
    _PLAN["cache_views"].clear()


def prepare_winner(acceptance: TreeAcceptance):
    """Freeze accepted node pages before original-KV compaction changes history."""
    from vllm_ascend.worker.v2.spec_decode.tree.node_pool import prepare_nodes

    if _PLAN is None:
        raise RuntimeError("Missing tree verification plan/accepted path")
    plan = _PLAN
    if plan["committed"] or plan["node_kv"]:
        raise RuntimeError("Tree state committed twice or a layer was not finalized")
    n_req = plan["n_req"]
    if acceptance.path_node_ids.shape[0] != n_req:
        raise RuntimeError("Accepted-path batch differs from verification batch")
    qsl = _LAYOUT["qsl"][: n_req + 1]
    # The root is endpoint zero when a bonus is sampled without any draft.
    winner = qsl[:-1] + acceptance.endpoint_node
    return prepare_nodes(plan, winner, acceptance.active)


def commit_winner(prepared):
    """Commit the frozen node pages after original-KV compaction."""
    from vllm_ascend.worker.v2.spec_decode.tree.node_pool import commit_nodes

    if _PLAN is None or _PLAN["committed"] or _PLAN["node_kv"]:
        raise RuntimeError("Missing/unfinalized tree plan or duplicate commitment")
    commit_nodes(prepared)
    _PLAN["committed"] = True
    _PLAN["node_records"].clear()
