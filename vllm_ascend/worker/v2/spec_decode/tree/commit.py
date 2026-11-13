"""Accepted-path commitment and query transport at the model boundary."""

from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_tree_query_along_path


class TreeCommitCoordinator:
    """Commit accepted token caches and stage recurrent state on one stream."""

    def __init__(self, runner):
        from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState
        from vllm_ascend.worker.v2.spec_decode.tree.qwen_query_graph import QwenTreeQueryGraphs

        self.compactor = runner.tree_kv_compact
        self.qwen_query_graphs = None
        self.stage_model_state = None
        state = runner.model_state
        if isinstance(state, AscendMambaHybridModelState):
            self.stage_model_state = state.stage_tree_acceptance
            state.tree_graph_stats = runner.speculator.tree_graph_stats
            if runner.compilation_config.cudagraph_mode in (
                CUDAGraphMode.FULL,
                CUDAGraphMode.FULL_DECODE_ONLY,
            ):
                self.qwen_query_graphs = QwenTreeQueryGraphs(runner, stats=state.tree_graph_stats)

    def commit(self, idx_mapping, acceptance: TreeAcceptance, *, num_rejected=None) -> None:
        if self.stage_model_state is not None:
            if num_rejected is None:
                raise ValueError("GDN tree commitment requires the verifier's rejected-token counts")
            self.stage_model_state(acceptance, num_rejected)
        self.compactor.run(idx_mapping, acceptance)

    def compact_queries(self, tensors, query_start_loc, acceptance, positions, *, input_batch=None) -> None:
        """Prepare accepted hidden/aux rows and positions for the next draft."""
        if self.qwen_query_graphs is not None and self.qwen_query_graphs.eligible(input_batch):
            self.qwen_query_graphs.compact_queries(tensors, query_start_loc, acceptance, positions)
        else:
            compact_tree_query_along_path(tensors, query_start_loc, acceptance, positions)
