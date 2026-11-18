"""Opt-in GDN/SSM tree speculative-decoding interface.

This adapter deliberately subclasses the existing tree speculator.  The
normal tree path remains unchanged; enabling ``tree_spec_config.gdn_ssm_enabled``
only selects this class so the GDN/SSM path can be measured independently.
The target-side GDN metadata and Triton recurrent operator are already owned by
the Mamba hybrid model state and are therefore not duplicated here.
"""

from __future__ import annotations

from typing import Any

from vllm.config import VllmConfig

from vllm_ascend.worker.v2.spec_decode.tree.speculator import AscendTreeSpeculator


class AscendGdnSsmTreeSpeculator(AscendTreeSpeculator):
    """Explicit GDN/SSM tree interface backed by the existing tree host."""

    _speculator_name = "DFlashGdnSsmTree"
    gdn_ssm_tree_enabled = True

    def __init__(self, vllm_config: VllmConfig, device):
        super().__init__(vllm_config, device)
        self.gdn_ssm_tree_interface = "gdn_ssm_tree_v1"

    def get_gdn_ssm_tree_interface(self) -> str:
        """Return the stable interface name for diagnostics and profiling."""
        return self.gdn_ssm_tree_interface

    def get_gdn_ssm_tree_capabilities(self) -> dict[str, Any]:
        """Describe the target-side path without changing the base tree API."""
        return {
            "interface": self.gdn_ssm_tree_interface,
            "tree_host": type(self).__mro__[1].__name__,
            "gdn_metadata": "SharedGDNTreeMetadata",
            "state_rollback": "tree_verify_plan",
            "recurrent_kernel": "gdn_tree_op_or_triton_fallback",
            "budget": int(self.budget),
            "topk": int(self.topk),
        }
