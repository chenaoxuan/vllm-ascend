"""Accepted-path commitment and query transport at the model boundary."""

from dataclasses import replace

from vllm_ascend.worker.v2.spec_decode.tree import chain_pack
from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.kv_binding import bind_original_kv
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path


class TreeCommitCoordinator:
    """Apply one acceptance result before advancing request state.

    Ordinary target caches use token-row transport. DSV4 also freezes its
    compressed/state deltas before that transport and commits them afterward.
    The runner does not inspect model-specific verification state. A single
    execution stream orders verification, commitment, and the next proposal.
    """

    def __init__(self, runner):
        self.compactor = runner.tree_kv_compact
        self.has_node_state = runner.speculator.packed_dsv4_verify()
        if self.has_node_state:
            self.compactor.bind_groups = bind_original_kv

    def commit(self, idx_mapping, acceptance: TreeAcceptance) -> None:
        if not self.has_node_state:
            self.compactor.run(idx_mapping, acceptance)
            return

        layout = chain_pack._LAYOUT
        if layout is None:
            raise RuntimeError("DSV4 tree acceptance has no verification layout")
        mixed = layout.get("mixed_batch")
        if mixed is not None:
            rows = mixed.decode.request_indices
            idx_mapping = idx_mapping.index_select(0, rows)
            acceptance = acceptance.select_rows(rows)
        if acceptance.path_node_ids.shape[0] == 0:
            return

        full_graph = layout.get("full_graph")
        if full_graph is not None:
            full_graph.commit(idx_mapping, acceptance)
            return
        prepared = chain_pack.prepare_winner(acceptance)
        self.compactor.run(idx_mapping, acceptance)
        chain_pack.commit_winner(prepared)

    def compact_queries(self, tensors, query_start_loc, acceptance, positions, *, layout=None) -> None:
        """Prepare the next draft's accepted hidden/aux rows and positions.

        Mixed-batch prefill rows retain their original sequence. FULL target
        graphs own stable query-transport graphs; all paths use the same row
        movement algorithm and the same acceptance contract.
        """
        full_graph = None
        if self.has_node_state and layout is not None:
            full_graph = layout.get("full_graph")
            mixed = layout.get("mixed_batch")
            if mixed is not None:
                acceptance = replace(acceptance, active=acceptance.active & mixed.decode_mask)
        if full_graph is not None:
            full_graph.compact_queries(tensors, query_start_loc, acceptance, positions)
        else:
            compact_tree_query_along_path(tensors, query_start_loc, acceptance, positions)
