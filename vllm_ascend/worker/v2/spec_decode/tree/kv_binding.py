"""Bind eager packed-V4 original KV to the current target SWA metadata.

The chain layout owns these records, so set/clear_tree_chain_layout also
resets their lifetime. No diagnostic module, group-number assumption, or
storage-pointer-only deduplication is involved. A shared cache pool can have
multiple disjoint live page mappings. Inactive table capacity is not ownership.
"""
from __future__ import annotations


def _require(condition, message):
    if not condition:
        raise RuntimeError('TREE_ORI_BINDING: ' + message)


def _layout():
    from vllm_ascend.worker.v2.spec_decode.tree import chain_pack
    return chain_pack._LAYOUT


def _host(tensor):
    return tensor.detach().cpu().tolist()


def _view_key(tensor):
    return (str(tensor.device), tensor.data_ptr(), tuple(tensor.shape),
            tuple(tensor.stride()), str(tensor.dtype))


def record_original_kv(layer_name, cache, metadata, num_tokens):
    """Record the flat target cache/table BEFORE temporary chain expansion."""
    layout = _layout()
    if layout is None:
        return
    tree_indices = getattr(metadata, 'tree_ori_indices', None)
    if (getattr(metadata, 'dspark_swa_indices', None) is not None
            and tree_indices is None):
        return  # Draft-only SWA metadata never supplies target bindings.
    _require(tree_indices is not None, 'active tree target has no SWA tree indices')
    context = layout.get('_ori_commit_bindings')
    if context is None:
        _require(layout.get('idx') is not None, 'tree layout has no request-state mapping')
        idx = [int(v) for v in _host(layout['idx'])]
        starts = [int(v) for v in _host(layout['start'])]
        qsl = [int(v) for v in _host(layout['qsl'])]
        _require(len(idx) > 0 and len(starts) == len(idx)
                 and len(qsl) == len(idx) + 1, 'inconsistent tree layout lengths')
        _require(qsl[0] == 0 and all(b > a for a, b in zip(qsl, qsl[1:])),
                 'invalid tree query boundaries')
        _require(len(set(idx)) == len(idx) and min(idx) >= 0,
                 'duplicate/negative request-state indices')
        _require(min(starts) >= 0, 'negative target prefix')
        context = dict(idx=idx, starts=starts, qsl=qsl,
                       bindings={}, bound=False)
        layout['_ori_commit_bindings'] = context
    _require(not context['bound'], 'target cache observed after this layout was bound')
    n = len(context['idx'])
    _require(int(num_tokens) >= context['qsl'][-1], 'target hidden rows are truncated')
    table = metadata.block_table
    bs = int(metadata.storage_block_size)
    _require(cache.ndim >= 2 and bs > 0 and int(cache.shape[1]) == bs,
             'target original-KV page size disagrees with SWA metadata')
    _require(table.ndim == 2 and int(table.shape[0]) >= n,
             'SWA table does not contain every target batch row')
    _require(str(cache.device) == str(table.device), 'cache/table devices differ')
    table = table[:n]
    # Validate values on every observation: identical metadata pointers do
    # not imply identical contents when builders reuse buffers.
    _require(_host(metadata.query_start_loc[:n + 1]) == context['qsl'],
             'SWA query order differs from the active target layout')
    lengths = _host(metadata.seq_lens[:n])
    expected = [p + b - a for p, a, b in
                zip(context['starts'], context['qsl'], context['qsl'][1:])]
    _require(lengths == expected, 'SWA physical sequence lengths differ from target layout')
    # Snapshot for each cache binding: a metadata builder may reuse the SAME
    # table storage for a later layer/group with different contents.
    saved_table = table.detach().clone()
    host_table = tuple(tuple(int(page) for page in row) for row in _host(saved_table))
    live = []
    for row, (prefix, end) in enumerate(zip(context['starts'], expected)):
        columns = (end + bs - 1) // bs
        _require(columns <= len(host_table[row]), 'target sequence exceeds SWA table capacity')
        live.append(host_table[row][:columns])
        # Freed SWA history may contain null pages. Every page containing a
        # current verification row must exist, including the root's page.
        _require(all(0 < host_table[row][col] < int(cache.shape[0])
                     for col in range(prefix // bs, columns)),
                 'target query references an unallocated/out-of-range SWA page')
    live = tuple(live)
    cache_key = _view_key(cache)
    key = (cache_key, live)
    group = str(getattr(metadata, 'cache_group_key', '') or '')
    owner = 'group:' + group if group else 'layer:' + str(layer_name)
    record = dict(cache=cache, cache_key=cache_key, table=saved_table,
        table_key=key, block_size=bs,
        layers={str(layer_name)}, owners={owner}, live=live,
        pages={page for row in live for page in row if page > 0})
    # Only columns outside each request's physical sequence are ignored.
    # Prefix columns are retained conservatively, even if a SWA window could
    # have expired them; this avoids guessing which aliased history is live.
    peers = [previous for previous in context['bindings'].values()
             if previous['cache_key'] == cache_key]
    for previous in peers:
        if previous['live'] == live:
            continue
        if previous['layers'] & record['layers'] or previous['owners'] & record['owners']:
            _require(False, 'owner_mapping_changed')
        if previous['pages'] & record['pages']:
            _require(False, 'overlapping_live_pages')
    previous = context['bindings'].get(key)
    if previous is not None:
        previous['layers'].update(record['layers'])
        previous['owners'].update(record['owners'])
        return
    context['bindings'][key] = record


def bind_original_kv(compactor, num_reqs):
    """Return the existing compactor group format using observed SWA tables.

Forward metadata is batch-row indexed. The existing compactor addresses
request-state rows: explicitly scatter the table rows through idx_mapping.
"""
    runner = compactor.runner
    _require(bool(runner.model_config.enforce_eager),
             'metadata-bound packed V4 compaction requires enforce_eager=True')
    layout = _layout()
    _require(layout is not None, 'packed V4 compaction has no active target layout')
    context = layout.get('_ori_commit_bindings')
    _require(context is not None and context['bindings'],
             'no target original-KV bindings; install the DSA forward hook')
    _require(not context['bound'], 'this target layout was already bound for commitment')
    _require(num_reqs is not None and int(num_reqs) == len(context['idx']),
             'commit request count differs from target verification')
    n = int(num_reqs)
    idx = [int(v) for v in _host(compactor._idx[:n])]
    _require(idx == context['idx'], 'commit request order differs from target verification')
    _require(max(idx) < runner.max_num_reqs, 'request-state index exceeds capacity')
    computed = _host(runner.req_states.num_computed_tokens.gpu)
    _require([int(computed[i]) for i in idx] == context['starts'],
             'computed prefix changed before original-KV commitment')
    paths = _host(compactor._path[:n])
    for row, begin, end in zip(paths, context['qsl'], context['qsl'][1:]):
        ended = False
        for value in row:
            node = int(value)
            if node == -1:
                ended = True
            else:
                _require(not ended and 0 < node < end - begin,
                         'accepted node is absent from this request verification')
    draft_names = set(getattr(runner.speculator, 'draft_attn_layer_names', ()) or ())
    groups, remapped, group_lookup = [], {}, {}
    for record in context['bindings'].values():
        _require(not (record['layers'] & draft_names), 'drafter cache in target bindings')
        cache, table, bs = record['cache'], record['table'], record['block_size']
        table_key = record['table_key']
        if table_key not in remapped:
            state_table = table.new_zeros((runner.max_num_reqs, table.shape[1]))
            rows = compactor._idx[:n].to(device=table.device).long()
            state_table.index_copy_(0, rows, table)
            remapped[table_key] = state_table
        group_key = (table_key, bs)
        if group_key not in group_lookup:
            group_lookup[group_key] = len(groups)
            groups.append(([], remapped[table_key], bs, []))
        groups[group_lookup[group_key]][0].append(cache)
    context['bound'] = True
    return groups
