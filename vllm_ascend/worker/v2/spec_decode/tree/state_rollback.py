"""GDN state rollback for draft-tree speculative decoding.

``tree_spec_config`` verifies a draft *tree*: the target model runs
``1 + budget`` tokens per request (the already-accepted root plus ``budget``
non-root nodes) and acceptance walks one root-to-leaf path through them. GDN
caches one recurrent state per page, so the state layout has to follow the tree
instead of the linear MTP chain:

* state slot ``j`` (0-based) holds the recurrent state *after* tree node ``j``;
  slot 0 is the root, whose state is the committed prefix state, and slot
  ordering matches node id ordering (every builder guarantees
  ``parent_id < node_id``);
* node ``j`` starts from its parent's slot, so dropping the rejected siblings is
  a pure index change -- no state copy and no recomputation;
* after sampling, the next step resumes from the last *accepted* node, which is
  what :func:`compute_tree_resume_column` computes for
  ``MambaHybridModelState.postprocess_state``.

The number of slots (``1 + budget``) is what
``vllm_ascend.worker.v2.attn_utils.get_kv_cache_spec`` widens ``MambaSpec`` to and
what ``AscendGDNAttentionMetadataBuilder`` sizes its spec buffers by. It is
larger than the chain width (``1 + num_speculative_tokens``), hence the state
memory and per-step state traffic grow with the tree width.

Only the Qwen GDN path (``MambaAttentionBackendEnum.GDN_ATTN``) is wired up so
far; :func:`validate_tree_state_slots` rejects a draft tree on any other state
backend (KDA, Mamba1/2, short-conv, pure linear attention) instead of silently
serving stale states.

``tree_spec_config.gdn_state_update`` selects *how* the state follows the tree;
``"snapshot"`` (the default, and the method this module implements) is the
per-node snapshot described above. The setting is read, and validated, on this
GDN path only, so it stays inert for any other state backend -- see
:data:`GDN_STATE_UPDATE_METHODS` and :func:`validate_tree_state_slots`.

The kernel side is tree-agnostic: :func:`compute_tree_init_state_indices` turns
the ``[R, slots]`` page table plus the tree's parent ids into a per-token table
of *page ids to load the initial state from*, which the recurrent op consumes
through its optional ``init_state_indices`` input. Node ``t`` therefore loads
its parent's page and writes its own, both indexed by the same slot table.
"""

from dataclasses import dataclass, replace

import torch

# MAX_MTP in
# csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h
# The kernel returns without touching the state once the per-request sequence
# length exceeds it, i.e. the state silently goes stale, so a single call can
# verify at most this many tokens (root + nodes) per request. The draft tree
# itself may be wider: :func:`compute_tree_verify_plan` cuts it into chunks of
# this width and chains the chunks through the state pages.
GDN_RECURRENT_MAX_VERIFY_TOKENS = 16


@dataclass(frozen=True)
class GdnStateUpdateMethod:
    """What a ``tree_spec_config.gdn_state_update`` method provides.

    ``state_init``: the recurrent state of tree node ``j`` is loaded from its
    parent's slot through the optional ``init_state_indices`` input of
    ``npu_recurrent_gated_delta_rule``
    (csrc/attention/recurrent_gated_delta_rule).

    ``conv``: the short-conv half follows the tree end to end -- the tree walk
    (:func:`compute_tree_conv_rows`), the widened state page that keeps this
    step's node activations and the history re-anchored on the node the sampler
    accepted (:func:`compute_tree_prefix_source`, applied by
    ``ops/gdn.tree_causal_conv1d``, sized by ``_widen_mamba_spec_for_tree``).
    """

    state_init: bool
    conv: bool
    description: str


# Only implemented methods are registered, and ``tree_spec_config.gdn_state_update``
# is read -- and validated against this registry -- on the GDN path only, so the
# setting stays inert (never an error) for every other state backend. See
# :func:`validate_tree_state_slots` and :func:`selected_gdn_state_update_method`.
#
# A new method has to touch three places: :func:`tree_state_slot_count` (state
# layout), ``AscendGDNAttentionMetadataBuilder._build_tree_init_state_indices``
# (per-node initial states) and ``ops/gdn.tree_causal_conv1d`` (short conv).
GDN_STATE_UPDATE_METHODS: dict[str, GdnStateUpdateMethod] = {
    "snapshot": GdnStateUpdateMethod(
        state_init=True,
        conv=True,
        description=(
            "one recurrent state snapshot per tree node: node j loads its parent's "
            "slot and writes its own, and the short-conv page keeps this step's node "
            "activations with the history re-anchored on the node the sampler accepted"
        ),
    ),
}
DEFAULT_GDN_STATE_UPDATE = "snapshot"


def selected_gdn_state_update_method() -> GdnStateUpdateMethod:
    """The draft-tree GDN state update method ``tree_spec_config`` selects.

    The name is resolved here instead of in ``TreeSpecConfig`` because the setting
    only means something for GDN state backends: a model that does not use GDN
    never reaches this function, so an unknown value cannot fail there.
    """
    from vllm_ascend.ascend_config import get_ascend_config

    name = getattr(
        get_ascend_config().tree_spec_config,
        "gdn_state_update",
        DEFAULT_GDN_STATE_UPDATE,
    )
    method = GDN_STATE_UPDATE_METHODS.get(name)
    if method is None:
        raise ValueError(
            f"tree_spec_config.gdn_state_update must be one of {tuple(GDN_STATE_UPDATE_METHODS)}, got {name!r}"
        )
    return method


def tree_state_slot_count(vllm_config) -> int | None:
    """Number of GDN state slots needed per request.

    Returns ``None`` when no draft tree is active, i.e. the caller must keep the
    linear-chain layout. ``1 + budget`` is the layout of the ``"snapshot"`` state
    update method, which caches one state per tree node; a method with a different
    layout changes this function (see :data:`GDN_STATE_UPDATE_METHODS`).
    """
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.worker.v2.spec_decode import dflash_tree_spec_enabled

    if not getattr(vllm_config, "use_v2_model_runner", False):
        # Only the v2 model runner hosts the tree; a leftover tree config must
        # not change the state layout of the linear MTP path.
        return None
    # Check the shared tree gate before reading global state, so ordinary
    # model-state construction retains its original initialization contract.
    if not dflash_tree_spec_enabled(vllm_config):
        return None
    ascend_config = get_ascend_config()
    tree_config = getattr(ascend_config, "tree_spec_config", None)
    if tree_config is None:
        return None
    budget = tree_config.budget
    if budget is None:
        return None
    if getattr(getattr(vllm_config, "cache_config", None), "mamba_cache_mode", None) == "align":
        # The align pre-copy kernel migrates the running state page and shifts
        # its short-conv window by ``num_accepted_tokens - 1``. A draft tree
        # cannot express that: the cursor is a node id and a node's window is
        # its own root-to-node chain instead of the previous batch entry, so the
        # shifted window would silently corrupt the conv history. The recurrent
        # state does not care: every node keeps its own page.
        raise NotImplementedError(
            "tree_spec_config is not supported with mamba_cache_mode='align' "
            "yet; the align pre-copy shift of the short-conv state has no "
            "draft-tree equivalent. Disable mamba_cache_mode='align' or "
            "tree_spec_config."
        )
    return 1 + budget


def validate_tree_state_slots(slots: int, mamba_type) -> None:
    """Reject draft trees that the current state path cannot serve yet.

    A tree wider than the recurrent kernel's per-call width is *not* rejected:
    :func:`compute_tree_verify_plan` splits the verify into
    :data:`GDN_RECURRENT_MAX_VERIFY_TOKENS`-wide chunks of node columns and the
    chunks are chained through the state pages, so ``slots`` is only bounded by
    the state memory the layout allocates.
    """
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    if slots < 1:
        raise ValueError(f"a draft tree verifies at least one token per request, got slots={slots}.")
    if mamba_type is not MambaAttentionBackendEnum.GDN_ATTN:
        raise NotImplementedError(
            f"tree_spec_config is enabled but mamba_type={mamba_type} does not "
            "support draft-tree state rollback yet; only GDN_ATTN (Qwen GDN) is "
            "wired up. Disable tree_spec_config for this model."
        )
    # Only now, with a GDN state backend in hand, does
    # ``tree_spec_config.gdn_state_update`` mean anything: an unknown method, or a
    # method that does not carry one half of the state across the tree, is refused
    # here -- never for a model that does not use GDN.
    method = selected_gdn_state_update_method()
    if not method.state_init:
        raise NotImplementedError(
            "the configured tree_spec_config.gdn_state_update does not carry the "
            "recurrent state across a draft tree yet: the recurrent kernel would "
            "read each node's own slot instead of its parent slot through "
            "init_state_indices. Disable tree_spec_config or pick another "
            "gdn_state_update."
        )
    if not method.conv:
        raise NotImplementedError(
            "the configured tree_spec_config.gdn_state_update does not carry the "
            "short-conv half across a draft tree yet: causal_conv1d would convolve "
            "each node over the previous token in the batch instead of its own "
            "root-to-node chain, so non-chain nodes would verify wrong. Disable "
            "tree_spec_config or pick another gdn_state_update."
        )


def tree_conv_width(
    mamba_type,
    conv_state_shape: tuple[int, ...],
    num_spec: int,
) -> int | None:
    """Short-conv kernel width behind a GDN conv-state shape.

    The conv state is ``(width - 1 + num_spec) x conv_dim`` and the conv
    dimension (``2 * key_dim + value_dim``) is orders of magnitude larger than
    the state length, so the smaller side of the 2-D state shape is the length
    to undo the speculation width on. Returns ``None`` for layers without a
    short conv (Mamba2, KDA) or an unexpected shape.
    """
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    if mamba_type is not MambaAttentionBackendEnum.GDN_ATTN:
        return None
    if len(conv_state_shape) != 2:
        return None
    width = min(conv_state_shape) - num_spec + 1
    if width < 2:
        raise ValueError(
            f"cannot recover the short-conv width from the state shape "
            f"{tuple(conv_state_shape)} with num_speculative_tokens={num_spec}."
        )
    return width


def compute_tree_resume_column(
    path_node_ids: torch.Tensor,
    num_sampled: torch.Tensor,
) -> torch.Tensor:
    """Return the state slot the next step must resume from, as an int32 tensor.

    ``path_node_ids[req, k]`` is the k-th accepted draft node of the tree walk
    (``-1`` past the accepted path, see ``greedy_tree_reject`` /
    ``block_tree_reject``) and ``num_sampled`` counts accepted tokens including
    the trailing resample, so the last accepted *node* sits at index
    ``num_sampled - 2``. ``num_sampled == 1`` (no accepted draft, e.g. chunked
    prefill) keeps the root slot, which matches the linear-chain neutral value
    ``num_accepted_tokens = 1``.
    """
    sampled = num_sampled.to(torch.long)
    last_index = (sampled - 2).clamp(min=0).unsqueeze(1)
    last_node = path_node_ids.to(torch.long).gather(1, last_index).squeeze(1)
    last_node = torch.where(sampled >= 2, last_node, torch.zeros_like(last_node))
    return (last_node + 1).to(torch.int32)


def tree_row_mask(
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> torch.Tensor:
    """Rows of the batch whose step ran the tree op, in batch order, as a bool.

    ``num_sampled`` alone cannot tell the two rows apart that both report ``1``
    with an all ``-1`` path: a tree step that accepted the root only, and a row
    the scheduler gave no draft token (which the linear operator commits). They
    differ in how many tokens were verified: the tree op handles every row that
    checked more than its own token, while a single-token row keeps the
    committed short-conv history the linear operator left in place. The samplers
    set ``num_rejected = num_logits - num_sampled``
    (see ``spec_decode/tree/rejection_sampler.py``), so the sum is exactly the
    number of verified tokens and needs no extra tensor from the sampler.
    """
    return (num_sampled + num_rejected) > 1


def compute_tree_init_state_indices(
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor,
) -> torch.Tensor:
    """Per-token state page the recurrent kernel must load its initial state from.

    Args:
        state_indices: ``[R, slots]`` state pages written by this step, column
            ``j`` holding tree node ``j``'s state (``j == 0`` is the root, whose
            page carries the committed prefix state).
        num_accepted_tokens: ``[R]`` resume cursor of the *previous* step, i.e.
            the accepted node id plus one; ``slot = num_accepted - 1`` is the
            page holding the state the next token continues from.
        tree_parents: ``[R, budget]`` parent node id of node ``slot + 1``
            (``0`` = root), straight from the tree layout.
        tree_num_nodes: ``[R]`` number of non-root nodes; ``0`` marks a row that
            was folded in from a prompt chunk rather than built as a tree.

    Returns:
        ``[R, slots]`` int32 page ids, in the same request/column order as
        ``state_indices``. The caller packs them token-major, because a row may
        verify fewer than ``slots`` tokens: see
        ``gdn_attn_builder._pack_tree_rows_to_tokens``.

    A row folded in from a prompt chunk is a linear sequence whose token ``t``
    follows token ``t - 1``: pointing the table at that token's own page makes
    the kernel's "parent page == previous token page" test hit and keeps its
    in-register carry, i.e. the exact linear-chain behaviour.
    """
    rows, slots = state_indices.shape
    if slots - 1 > tree_parents.shape[1]:
        raise ValueError(
            f"state table has {slots} slots but the tree layout only carries "
            f"{tree_parents.shape[1]} parents; tree_spec_config.budget must "
            "match the state slot count."
        )
    parents = tree_parents[:rows, : slots - 1].to(torch.long)
    # Chain rows: node ``t`` (1-based) continues the token before it.
    chain_parents = torch.arange(1, slots, device=state_indices.device, dtype=torch.long) - 1
    parents = torch.where(tree_num_nodes[:rows, None] > 0, parents, chain_parents[None, :])
    num_accepted = num_accepted_tokens[:rows].to(torch.long)
    accepted_slot = (num_accepted - 1).clamp(min=0).unsqueeze(1)
    root_init = torch.gather(state_indices, 1, accepted_slot)
    node_init = torch.gather(state_indices, 1, parents)
    init_state_indices = torch.cat((root_init, node_init), dim=1)
    return init_state_indices.to(torch.int32)


def tree_node_parents_depths(
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor,
    slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node parent ids and depths of the draft tree.

    Args:
        tree_parents: ``[R, budget]`` parent node id of node ``t + 1``, straight
            from the tree layout (see :func:`compute_tree_init_state_indices`).
        tree_num_nodes: ``[R]`` number of non-root nodes; ``0`` marks a row that
            was folded in from a prompt chunk and is a linear sequence.
        slots: number of nodes per request, i.e. ``1 + budget``.

    Returns:
        ``(parents, depths)``, both ``[R, slots]`` int64; the root's parent is
        ``-1``. Folded rows are chained (node ``t`` follows ``t - 1``) so they
        keep the batch-order window of the linear MTP chain.
    """
    rows = tree_parents.shape[0]
    device = tree_parents.device
    parents = torch.full((rows, slots), -1, dtype=torch.long, device=device)
    parents[:, 1:] = tree_parents[:rows, : slots - 1].to(torch.long)
    chain_parents = torch.arange(slots, dtype=torch.long, device=device) - 1
    parents = torch.where(tree_num_nodes[:rows, None] > 0, parents, chain_parents.expand(rows, slots))
    depths = torch.zeros((rows, slots), dtype=torch.long, device=device)
    for node in range(1, slots):
        # Every builder guarantees ``parent_id < node_id``, so a single forward
        # sweep resolves the whole depth table.
        parent_of_node = parents[:, node].clamp(min=0).unsqueeze(1)
        depths[:, node] = depths.gather(1, parent_of_node).squeeze(1) + 1
    return parents, depths


def _tree_conv_tap_rows(
    chain: torch.Tensor,
    depths: torch.Tensor,
    own_rows: torch.Tensor,
    request_ids: torch.Tensor,
    node_row_table: torch.Tensor,
    history_base: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Rows of the ``width`` conv taps of every target, oldest tap first.

    Args:
        chain: ``[N, width - 1]`` walk-up ancestors, column ``k`` being the
            ``k + 1``-th ancestor (parent first).
        depths: ``[N]`` depth of every target.
        own_rows: ``[N]`` row of the target's own activation.
        request_ids: ``[N]`` request each target belongs to.
        node_row_table: ``[R, slots]`` row of every node's activation.
        history_base: ``[R]`` row of the oldest committed activation.

    The rows index ``cat([committed, activations])``: the committed
    activations of the previous steps first (oldest first), this step's
    activations after.  A tap that falls before the root resolves to the
    committed history, which keeps the window of shallow nodes identical to the
    linear-chain window.
    """
    taps = []
    for tap in range(width - 1):
        # Tap ``width - 2`` is the parent, ``width - 3`` the grandparent, ...
        ancestor = node_row_table[request_ids, chain[:, width - 2 - tap].clamp(min=0)]
        committed = history_base[request_ids] + (depths + tap).clamp(max=width - 2)
        taps.append(torch.where(depths >= (width - 1 - tap), ancestor, committed))
    taps.append(own_rows)
    return torch.stack(taps, dim=1)


def compute_tree_conv_rows(
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor,
    width: int,
    node_row_offset: int,
    row_widths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token conv window of every node of a draft tree.

    Args:
        tree_parents: ``[R, budget]`` tree parents, as above.
        tree_num_nodes: ``[R]`` non-root node count, as above.
        width: short-conv kernel width; the window holds ``width`` taps.
        node_row_offset: first row of this step's activations inside the gather
            source; the ``R * (width - 1)`` committed rows precede it.
        row_widths: ``[R]`` tokens each request verifies, i.e. ``1 +`` its node
            count. ``None`` assumes the full ``1 + budget`` width. A row may
            verify fewer tokens -- the scheduler truncates a draft near
            ``max_model_len`` and the tree search can run out of candidates --
            and the layer reads the table by global token offset, so the rows
            are packed back to back.

    Returns:
        ``[R * slots, width]`` rows of the taps (oldest first) consumed by
        ``tree_causal_conv1d``, indexing
        ``cat([committed history, this step's activations])``. Only the first
        ``row_widths.sum()`` rows are meaningful; the layer slices the table by
        the token count of the spec batch, and for the full width every row is
        meaningful.

    Node ``t`` convolves over its own root-to-node chain: its parent, its
    grandparent, ... and, once the chain runs past the root, the committed
    activations of the previous steps.  Sibling and unrelated nodes never enter
    a node's window, which is what the linear-chain conv kernel gets wrong.
    """
    rows = tree_parents.shape[0]
    slots = tree_parents.shape[1] + 1
    device = tree_parents.device
    if row_widths is None:
        row_widths = torch.full((rows,), slots, dtype=torch.long, device=device)
    else:
        # Validated before the device transfer (a CPU tensor here), because a
        # device-side ``bool(...)`` would synchronize the stream.
        row_widths = row_widths.reshape(-1)
        if row_widths.numel() != rows:
            raise ValueError(f"row_widths must hold {rows} widths, got {row_widths.numel()}.")
        if bool(((row_widths < 1) | (row_widths > slots)).any()):
            raise ValueError(f"every draft-tree row verifies between 1 and {slots} tokens, got {row_widths.tolist()}.")
        row_widths = row_widths.to(device=device, dtype=torch.long)
    parents, depths = tree_node_parents_depths(tree_parents, tree_num_nodes, slots)
    # Walk-up ancestors: column ``k`` is the ``k + 1``-th ancestor. ``clamp``
    # keeps the gather in range where an ancestor is missing; those entries are
    # discarded by the depth test in _tree_conv_tap_rows.
    chain = [parents]
    for _ in range(width - 2):
        chain.append(torch.gather(parents, 1, chain[-1].clamp(min=0)))
    chain = torch.stack(chain, dim=2)

    # Token-major packing: the ``p``-th token of the step is node
    # ``p - start(r)`` of row ``r``. The tables are built on the fixed
    # ``R * slots`` capacity -- so the result keeps its shape and the cost does
    # not depend on the widths -- and only the first ``row_widths.sum()`` rows
    # hold windows the layer reads.
    starts = torch.cumsum(row_widths, dim=0) - row_widths
    ends = starts + row_widths
    positions = torch.arange(rows * slots, device=device)
    token_rows = (positions.unsqueeze(1) >= ends.unsqueeze(0)).sum(dim=1).clamp(max=rows - 1)
    token_cols = (positions - starts[token_rows]).clamp(max=slots - 1)
    node_row_table = node_row_offset + starts.unsqueeze(1) + torch.arange(slots, device=device).unsqueeze(0)

    window_rows = _tree_conv_tap_rows(
        chain=chain[token_rows, token_cols],
        depths=depths[token_rows, token_cols],
        own_rows=node_row_offset + positions,
        request_ids=token_rows,
        node_row_table=node_row_table,
        history_base=torch.arange(rows, device=device) * (width - 1),
        width=width,
    )
    return window_rows.to(torch.int32)


def compute_tree_prefix_source(
    path_node_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Committed-history columns the next step has to read its window from.

    The committed history is the ``width - 1`` activations *ending at* the token
    the next step's root continues from, i.e. the last node the sampler accepted
    (see :func:`compute_tree_resume_column`). ``num_sampled`` counts the accepted
    tokens including the trailing resample, so that node sits at tree depth
    ``num_sampled - 1``: depth ``0`` is the root, once no draft node was accepted.
    ``path_node_ids[req, k]`` holds the accept-walk node of depth ``k + 1``, i.e.
    the last accepted node at index ``num_sampled - 2``.

    A tap of the window is therefore resolved by its tree depth ``d``:

    * ``d >= 1``: the accepted node of that depth, stored by this step in node
      column ``width - 1 + node id``;
    * ``d == 0``: the root, node column ``width - 1`` -- its activation is this
      step's first token, so it is *not* one of the committed columns;
    * ``d < 0``: the activation ``-d`` tokens before the root, i.e. committed
      column ``width - 1 + d``, which already holds it.

    A row the sampler left without a path (``num_sampled == 0``, a prompt chunk
    folded into the spec batch) maps the committed columns onto themselves: the
    chain op already left the history its root continues from.

    Args:
        path_node_ids: ``[R, budget]`` accepted draft nodes of the tree walk.
        num_sampled: ``[R]`` accepted token count including the resample.
        width: short-conv kernel width.

    Returns:
        ``[R, width - 1]`` int32 conv-state columns, oldest first.
    """
    rows = num_sampled.shape[0]
    device = num_sampled.device
    # Depth of the last accepted node; ``-1`` marks a row the sampler skipped, so
    # that its taps land on the committed columns themselves.
    last_depth = (num_sampled.to(torch.long) - 1).clamp(min=-1).unsqueeze(1)
    depth = last_depth - (width - 2) + torch.arange(width - 1, device=device).unsqueeze(0)
    # Only the node taps (``depth >= 1``) are looked up in the path: its entry
    # ``depth - 1`` is exactly the node of that depth. The lower entries are
    # masked out below, so the clamped index only has to stay in range.
    node_ids = path_node_ids[:rows].to(torch.long).gather(1, (depth - 1).clamp(min=0))
    return torch.where(depth >= 1, width - 1 + node_ids, width - 1 + depth).to(torch.int32)


# ---------------------------------------------------------------------------
# Chunked verify of a draft tree wider than the recurrent kernel's call width
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeVerifyChunk:
    """One ``npu_recurrent_gated_delta_rule`` call of a wide draft-tree verify.

    The kernel verifies at most :data:`GDN_RECURRENT_MAX_VERIFY_TOKENS` tokens per
    request in one call (its per-token row buffers are sized by that constant and
    it *returns without touching the state* above it). A wider tree is therefore
    cut into node columns and chained through the state pages: node ``j`` always
    loads its parent's page through ``init_state_indices``, which the chunk that
    verified the parent already wrote, and every tree builder guarantees
    ``parent_id < node_id``. Chunk ``c`` owns node columns
    ``[c * width, (c + 1) * width)`` of every row, so its tokens are always a
    *prefix* of those columns.

    The chunk's token dim is ``rows * width`` and is addressed as a padded grid:
    row ``r`` holds the nodes it owns in cells ``[r * width, r * width + length)``
    and the remaining cells of that row are padding. The kernel never reads
    padding -- ``cu_seqlens`` gives every row its prefix length and a row of
    length 0 is skipped -- so the op sees the same shapes whatever the per-row
    widths are, which is what a full CUDA graph needs.

    Attributes:
        token_indices: ``[rows * width]`` int64, gather into the step's *packed*
            token dim (``packed[p]`` is token ``p``, see
            ``AscendGDNAttentionMetadataBuilder._pack_tree_rows_to_tokens``).
        state_indices: ``[rows * width]`` int32 ssm pages of those tokens, ``-1``
            outside the chunk. A negative index makes the kernel return without
            updating any state (``Process``), so the sentinel turns a malformed
            plan -- a cell outside the row's own prefix, whose state page the
            kernel would read as the parent of a later token -- into a loud no-op
            instead of a silent rollback of another row's state. Padding itself is
            never read: the kernel walks each row's
            ``[cu_seqlens[row], cu_seqlens[row + 1])`` range and only compares a
            token with its predecessor inside that range.
        init_state_indices: ``[rows * width]`` int32 parent pages, ``-1`` outside;
            the first cell of every row prefix has to carry a real page, because
            the kernel loads it unconditionally and returns on a negative one.
        out_indices: ``[rows * width]`` int64 rows of the op output buffer the op
            output belongs to. Padding cells point past the layer's real tokens
            (``scratch_base``), so their undefined output cannot reach a token.
        cu_seqlens: ``[rows + 1]`` int32 token count of every row, in the kernel's
            ``actual_seq_lengths`` convention: entry 0 is an unused base and entry
            ``row + 1`` is how many tokens row ``row`` verifies (the kernel
            accumulates them into token offsets itself).
        width: node columns this chunk covers (the kernel's call width).
    """

    token_indices: torch.Tensor
    state_indices: torch.Tensor
    init_state_indices: torch.Tensor
    out_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    width: int


@dataclass(frozen=True)
class TreeVerifyPlan:
    """Chunked recurrent verify of one draft-tree step.

    Built by :func:`compute_tree_verify_plan`; consumed by
    ``vllm_ascend.ops.gdn`` (the chunk loop) and by
    ``AscendGDNAttentionMetadataBuilder`` (stable buffers for graph capture).
    """

    rows: int
    slots: int
    width: int
    chunks: tuple[TreeVerifyChunk, ...]

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def chunk_tokens(self) -> int:
        """Token dim of every chunk: ``rows * width``."""
        return self.rows * self.width

    def copy_into(self, buffers: "TreeVerifyPlanBuffers") -> "TreeVerifyPlan":
        """Write this plan into stable buffers and return the matching views.

        A full CUDA graph reads its inputs by address, so the chunk tensors the
        captured ops were recorded with have to be refreshed in place on every
        replay instead of being reallocated (the plan is rebuilt per step, the
        buffers live in the builder).
        """
        size = self.chunk_tokens
        chunks = []
        for index, chunk in enumerate(self.chunks):
            token_indices = buffers.token_indices[index][:size]
            token_indices.copy_(chunk.token_indices)
            state_indices = buffers.state_indices[index][:size]
            state_indices.copy_(chunk.state_indices)
            init_state_indices = buffers.init_state_indices[index][:size]
            init_state_indices.copy_(chunk.init_state_indices)
            out_indices = buffers.out_indices[index][:size]
            out_indices.copy_(chunk.out_indices)
            cu_seqlens = buffers.cu_seqlens[index][: self.rows + 1]
            cu_seqlens.copy_(chunk.cu_seqlens)
            chunks.append(
                replace(
                    chunk,
                    token_indices=token_indices,
                    state_indices=state_indices,
                    init_state_indices=init_state_indices,
                    out_indices=out_indices,
                    cu_seqlens=cu_seqlens,
                )
            )
        return TreeVerifyPlan(rows=self.rows, slots=self.slots, width=self.width, chunks=tuple(chunks))


@dataclass
class TreeVerifyPlanBuffers:
    """Stable per-chunk tensors of a :class:`TreeVerifyPlan` (see ``copy_into``).

    Every field carries one row per chunk, sized for the graph's row capacity, so
    a replay can slice the live row count off the front.
    """

    token_indices: torch.Tensor
    state_indices: torch.Tensor
    init_state_indices: torch.Tensor
    out_indices: torch.Tensor
    cu_seqlens: torch.Tensor

    def zero_lengths(self) -> None:
        """Turn every recorded chunk into a no-op call.

        A replay that carries no draft tree (idle, prefill) must not run the
        recorded chunks over the stale indices of the previous step: with every
        row length zero the kernel skips each of them, exactly like the zeroed
        ``spec_actual_seq_lengths`` of the linear path.
        """
        self.cu_seqlens.zero_()


def tree_verify_chunk_count(
    slots: int,
    width: int = GDN_RECURRENT_MAX_VERIFY_TOKENS,
) -> int:
    """Number of recurrent-op calls a ``1 + budget = slots`` verify needs."""
    if width < 1:
        raise ValueError(f"the verify chunk width must be positive, got {width}.")
    if slots < 1:
        raise ValueError(f"a verify step holds at least one token per request, got slots={slots}.")
    return -(-slots // width)


def allocate_tree_verify_plan_buffers(
    slots: int,
    rows: int,
    device: torch.device,
    width: int = GDN_RECURRENT_MAX_VERIFY_TOKENS,
) -> TreeVerifyPlanBuffers:
    """Allocate the stable chunk tensors of a ``rows``-row, ``slots``-wide verify."""
    chunks = tree_verify_chunk_count(slots, width)
    tokens = rows * width
    return TreeVerifyPlanBuffers(
        token_indices=torch.empty((chunks, tokens), dtype=torch.long, device=device),
        state_indices=torch.empty((chunks, tokens), dtype=torch.int32, device=device),
        init_state_indices=torch.empty((chunks, tokens), dtype=torch.int32, device=device),
        out_indices=torch.empty((chunks, tokens), dtype=torch.long, device=device),
        cu_seqlens=torch.empty((chunks, rows + 1), dtype=torch.int32, device=device),
    )


def compute_tree_verify_plan(
    row_widths: torch.Tensor,
    state_indices: torch.Tensor,
    init_state_indices: torch.Tensor,
    scratch_base: int,
    width: int = GDN_RECURRENT_MAX_VERIFY_TOKENS,
) -> TreeVerifyPlan | None:
    """Cut a draft-tree verify into recurrent-op calls of at most ``width`` tokens.

    Args:
        row_widths: ``[rows]`` verified token count of every spec row (a padded
            graph row has width 0). The rows are the model's *tight* token order:
            row ``r`` owns tokens ``[sum(widths[:r]), sum(widths[:r + 1]))``.
        state_indices: ``[rows, slots]`` int32 ssm state pages, in the packed
            token-major layout the op consumes (``reshape(-1)[p]`` is token ``p``).
        init_state_indices: ``[rows, slots]`` int32 parent pages, same layout.
        scratch_base: first row of the caller's output buffer that no verified
            token uses (``num_actual_tokens``); chunk padding is scattered there.
        width: node columns per call, i.e. the kernel's ``MAX_MTP``.

    Returns:
        ``None`` when the whole tree fits one call (the caller keeps the plain
        single-call path), otherwise one :class:`TreeVerifyChunk` per call.

    The widths are trusted, not re-validated: they come from the builder, which
    checks every speculative row against the tree width on CPU data (a ``bool()``
    here would sync the device once per layer per step) and passes 0 for its
    padded graph rows. A width past ``slots`` makes the gather below fail loudly;
    ``state_indices.shape`` may be any ``[rows, slots]`` pair.
    """
    if state_indices.dim() != 2 or init_state_indices.shape != state_indices.shape:
        raise ValueError(
            f"the state tables must be a pair of [rows, slots] tensors, got "
            f"{tuple(state_indices.shape)} and {tuple(init_state_indices.shape)}."
        )
    rows, slots = state_indices.shape
    if slots <= width:
        return None
    if scratch_base < 0:
        raise ValueError(f"the output scratch base must be non-negative, got {scratch_base}.")
    device = state_indices.device
    widths = torch.as_tensor(row_widths, device=device, dtype=torch.long).reshape(-1)
    if widths.numel() != rows:
        raise ValueError(f"got {widths.numel()} row widths for {rows} state rows.")

    token_starts = torch.cumsum(widths, dim=0) - widths
    chunk_starts = torch.arange(0, slots, width, device=device, dtype=torch.long)
    cells = torch.arange(width, device=device, dtype=torch.long)
    # Node columns of every row inside every chunk: [chunks, rows, width].
    row_ends = torch.minimum(chunk_starts[:, None] + width, widths[None, :])
    row_lengths = (row_ends - chunk_starts[:, None]).clamp(min=0)
    owned = cells[None, None, :] < row_lengths[:, :, None]
    tokens = token_starts[None, :, None] + chunk_starts[:, None, None] + cells[None, None, :]
    # Padding cells are only ever gathered, never read by the kernel, so index 0
    # keeps the gather in bounds.
    tokens = torch.where(owned, tokens, torch.zeros_like(tokens))

    flat_state = state_indices.reshape(-1)
    flat_init = init_state_indices.reshape(-1)
    minus_one = torch.tensor(-1, dtype=torch.int32, device=device)
    scratch = torch.arange(rows * width, device=device, dtype=torch.long) + scratch_base
    chunks = []
    for index in range(chunk_starts.numel()):
        chunk_tokens = tokens[index].reshape(-1)
        chunk_owned = owned[index].reshape(-1)
        chunks.append(
            TreeVerifyChunk(
                token_indices=chunk_tokens,
                state_indices=torch.where(
                    chunk_owned,
                    flat_state.index_select(0, chunk_tokens),
                    minus_one,
                ),
                init_state_indices=torch.where(
                    chunk_owned,
                    flat_init.index_select(0, chunk_tokens),
                    minus_one,
                ),
                out_indices=torch.where(chunk_owned, chunk_tokens, scratch),
                cu_seqlens=torch.cat(
                    (
                        torch.zeros(1, dtype=torch.int32, device=device),
                        row_lengths[index].to(torch.int32),
                    )
                ),
                width=width,
            )
        )
    return TreeVerifyPlan(rows=rows, slots=slots, width=width, chunks=tuple(chunks))
