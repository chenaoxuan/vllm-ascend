from vllm.v1.attention.backend import AttentionMetadataBuilder


def _init_reorder_batch_threshold(
    self,
    reorder_batch_threshold: int | None = 1,
    supports_spec_as_decode: bool = False,
    supports_dcp_with_varlen: bool = False,
) -> None:
    self.reorder_batch_threshold = reorder_batch_threshold
    if self.reorder_batch_threshold is not None and supports_spec_as_decode:
        # If the backend supports spec-as-decode kernels, then we can set
        # the reorder_batch_threshold based on the number of speculative
        # tokens from the config.
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is not None and speculative_config.num_speculative_tokens is not None:
            if hasattr(speculative_config, "method") and speculative_config.method == "dflash":
                max_num_queries_for_spec = 1 + speculative_config.num_speculative_tokens
            else:
                max_num_queries_for_spec = (
                    1 + (2 if speculative_config.parallel_drafting else 1) * speculative_config.num_speculative_tokens
                )
            self.reorder_batch_threshold = max(
                self.reorder_batch_threshold,
                max_num_queries_for_spec,
            )

    if self.vllm_config.parallel_config.decode_context_parallel_size > 1 and not supports_dcp_with_varlen:
        self.reorder_batch_threshold = 1


AttentionMetadataBuilder._init_reorder_batch_threshold = _init_reorder_batch_threshold
