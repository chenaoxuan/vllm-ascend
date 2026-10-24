# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in DSV4 node pool. Device data is indexed by immutable tree ancestry."""

from math import prod

import torch
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.spec_decode.tree.node_commit import commit_node_pages, freeze_node_pages
from vllm_ascend.ops.triton.spec_decode.tree.node_compressor import aggregate_nodes


class NodeWorkspace:
    """Module-owned device storage reused across serialized verification rounds.

    Nodes/cache views stay live through accepted-chain commitment. Different
    modules and repeated invocations within one round must use distinct pools.
    All reuse is ordered on the same tree execution stream.
    """

    def __init__(self):
        from vllm_ascend.worker.v2.spec_decode.tree.chain_pack import _LayerBuffers

        self.buffers = _LayerBuffers()
        self.views = {}

    def cache_views(self, caches, num_pages):
        from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

        key = tuple(cp._key(cache) for cache in caches)
        saved = self.views.get(key)
        if saved is None or saved[0].shape[0] < num_pages:
            # Cache storage is independent of resettable aggregation scratch.
            saved = cp._new_cache_views(cp._LayerBuffers(), caches, num_pages)
            self.views[key] = saved
        else:
            # Keep null pages and unused rows deterministic on every round.
            for view in saved:
                view[:num_pages].zero_()
        return [view[:num_pages] for view in saved]


def node_workspace(module, plan):
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

    owners = plan.setdefault("node_workspace_owners", set())
    if id(module) in owners:
        return NodeWorkspace()
    owners.add(id(module))
    graph_workspaces = (cp._LAYOUT or {}).get("graph_workspaces")
    if graph_workspaces is not None:
        # Graph gears must retain their own addresses, including when an eager
        # prefill or a different graph gear runs between two replays.
        workspace = graph_workspaces.get(id(module))
        if workspace is None:
            workspace = graph_workspaces[id(module)] = NodeWorkspace()
        workspace.buffers.reset()
        return workspace
    workspace = module.__dict__.get("_tree_node_workspace")
    if workspace is None:
        workspace = NodeWorkspace()
        module._tree_node_workspace = workspace
    workspace.buffers.reset()
    return workspace


def compress_nodes(module, hidden, state, metadata):
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

    plan = cp._build_plan(hidden.shape[0], hidden.device)
    workspace = node_workspace(module, plan)
    cache_meta = metadata.cache.req_metadata
    state_meta = metadata.state.req_metadata
    ratio, head = module.compress_ratio, module.head_dim
    n, device = plan["n"], hidden.device
    key = ("projection", ratio)
    saved = plan["query_metadata"].get(key)
    if saved is None:
        # Projection uses one independent row per node. Its logical residue
        # preserves APE exactly; the disposable native aggregation sees only
        # null history. Triton performs the real branch aggregation afterwards.
        residue = ((plan["start"] + plan["depths"]) % ratio).to(torch.int32)
        table = torch.zeros((n, ratio), dtype=torch.int32, device=device)
        table.scatter_(1, residue.long().unsqueeze(1), (plan["token_index"] + 1).int().unsqueeze(1))
        saved = (
            residue,
            table,
            torch.ones((n, module.rope_head_dim), device=device),
            torch.zeros((n, module.rope_head_dim), device=device),
        )
        plan["query_metadata"][key] = saved
    residue, projection_table, dummy_cos, dummy_sin = saved
    nodes = workspace.buffers.acquire("nodes", (n + 1, 1, 1, 2 * module.coff * head), state.dtype, device)
    nodes.zero_()
    module._compress(
        hidden.index_select(0, plan["hidden_ids"]),
        nodes,
        projection_table,
        plan["query_cu"],
        residue,
        dummy_cos,
        dummy_sin,
    )
    full_cos, full_sin = cache_meta.full_compress_cos, cache_meta.full_compress_sin
    cos = full_cos.view(full_cos.shape[0], full_cos.shape[-1])
    sin = full_sin.view(full_sin.shape[0], full_sin.shape[-1])
    output = aggregate_nodes(
        state,
        nodes,
        state_meta.block_table,
        plan,
        module.norm.weight,
        cos,
        sin,
        ratio,
        head,
        module.rope_head_dim,
        module.norm_eps,
        hidden.dtype,
        workspace=workspace.buffers,
    )
    # Retain each node once until acceptance. History is never copied per branch.
    record_nodes(plan, state, nodes[1:], state_meta, 1)

    # Reserve block zero and one physical row per node. Persistent prefix
    # pages remain in the original cache and are addressed directly by readers.
    caches = cp._KV_STACK[-1]
    bs = int(cache_meta.storage_block_size)
    views = workspace.cache_views(caches, 1 + triton.cdiv(n, bs))
    bindings = plan.setdefault("node_kv", {})
    for original, view in zip(caches, views):
        rows = view[1:].reshape(-1, *view.shape[2:])[:n]
        binding = {"original": original, "rows": rows, "table": cache_meta.block_table}
        for tensor in (original, view):
            cp_key = cp._key(tensor)
            plan["cache_views"][cp_key] = view
            bindings[cp_key] = binding
        record_nodes(plan, original, rows, cache_meta, ratio)
    slot_key = ("node_slots", ratio, bs)
    slots = plan["query_metadata"].get(slot_key)
    if slots is None:
        emit = (plan["start"] + plan["depths"] + 1) % ratio == 0
        slots = torch.stack((plan["token_index"] // bs + 1, plan["token_index"] % bs), 1).int()
        slots = torch.where(emit.unsqueeze(1), slots, -1)
        plan["query_metadata"][slot_key] = slots
    return output, slots


def attend_nodes(query, original, compressed, metadata, topk, sinks, ratio, window, scale):
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

    if get_ascend_config().tree_spec_config.attention_backend == "native":
        from vllm_ascend.ops.triton.spec_decode.tree.native_attention import native_attention as node_attention
    else:
        from vllm_ascend.ops.triton.spec_decode.tree.node_attention import node_attention

    plan = cp._PLAN
    binding = plan["node_kv"][cp._key(compressed)]
    n = plan["n"]
    output = node_attention(
        query[:n],
        original,
        binding["original"],
        binding["rows"],
        metadata.block_table,
        binding["table"],
        plan,
        topk,
        sinks,
        ratio,
        window,
        scale,
    )
    if query.shape[0] > n:
        output = torch.cat((output, output.new_zeros((query.shape[0] - n, *output.shape[1:]))))
    cp.finish_tree_layer()
    plan["node_kv"].clear()
    return output


def record_nodes(plan, original, snapshot, metadata, ratio):
    """Keep node rows and device block mappings alive; no data copy or readback."""
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

    real, _ = cp._LAYOUT["host_tables"][metadata.cache_group_key]
    bs = original.shape[1]
    # A scheduler snapshot shares the same CPU rows across layers in a group.
    # The cache is round-local, so request replacement cannot reuse stale pages.
    key = (id(real), bs, ratio)
    page_sets = plan.setdefault("node_page_sets", {})
    pages = page_sets.get(key)
    if pages is None:
        touched = set()
        for req, (begin_node, end_node) in enumerate(zip(plan["req_cu"], plan["req_cu"][1:])):
            if begin_node == end_node:
                continue
            start = plan["prefixes"][begin_node]
            length = min(plan["max_q"], end_node - begin_node)
            begin = start if ratio == 1 else start // ratio
            end = start + length if ratio == 1 else (start + length) // ratio
            if end > begin:
                first, last = begin // bs, min((end - 1) // bs + 1, len(real[req]))
                touched.update(page for page in real[req][first:last] if page > 0)
        pages = tuple(sorted(touched))
        page_sets[key] = pages
    plan.setdefault("node_records", []).append((original, snapshot, metadata.block_table, pages, ratio))


def prepare_nodes(plan, winner, active):
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack as cp

    descriptors, total, max_rows = [], 0, 1
    for original, snapshot, table, pages, ratio in plan.get("node_records", []):
        size = original.element_size()
        row_bytes = prod(original.shape[2:]) * size
        if snapshot.stride(0) * snapshot.element_size() != row_bytes:
            raise RuntimeError("Node snapshots must have contiguous rows")
        for page in pages:
            descriptors.append(
                (
                    original.data_ptr() + page * original.stride(0) * size,
                    original.stride(1) * size,
                    row_bytes,
                    original.shape[1],
                    snapshot.data_ptr(),
                    table.data_ptr(),
                    table.shape[1],
                    page,
                    total,
                    ratio,
                )
            )
            total += original.shape[1] * row_bytes
            max_rows = max(max_rows, original.shape[1])
    (desc,) = cp._upload_metadata(
        winner.device, (descriptors, torch.int64, (len(descriptors), 10)), host_buffers=plan["host_uploads"]
    )
    staging = torch.empty(total, dtype=torch.uint8, device=winner.device)
    enabled = torch.empty(len(descriptors), dtype=torch.bool, device=winner.device)
    if descriptors:
        freeze_node_pages[(len(descriptors), max_rows)](
            desc,
            plan["ancestors"],
            plan["depths"],
            plan["start"],
            plan["req_ids"],
            winner,
            active,
            staging,
            enabled,
            plan["n"],
            plan["max_q"],
            triton.next_power_of_2(plan["n"]),
            256,
        )
    return desc, staging, enabled, max_rows


def commit_nodes(prepared):
    desc, staging, enabled, max_rows = prepared
    if desc.shape[0]:
        commit_node_pages[(desc.shape[0], max_rows)](desc, staging, enabled, 256)
