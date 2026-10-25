from dataclasses import dataclass

import torch


@dataclass
class TreeAcceptance:
    """Device-side result shared by verification and accepted-state transport.

    ``path_node_ids`` is [R, K], excludes the root and uses -1 for its unused
    suffix. Counts and endpoints are [R]; ``num_sampled`` includes the bonus
    token, while ``num_accepted`` counts draft nodes only. An active root-only
    result has one sampled token, zero accepted drafts and endpoint zero.
    Buffers belong to the verifier and remain valid until its next call.
    """

    path_node_ids: torch.Tensor
    num_sampled: torch.Tensor
    num_accepted: torch.Tensor
    endpoint_node: torch.Tensor
    active: torch.Tensor

    @classmethod
    def from_path(cls, path_node_ids: torch.Tensor, num_sampled: torch.Tensor) -> "TreeAcceptance":
        """Build the result on device when an existing caller supplies a path.

        The endpoint is the final valid *position* in the path. Node IDs need
        not be monotonic, so reducing their values with max would be incorrect.
        """
        active = num_sampled > 0
        valid = (path_node_ids >= 0) & active.unsqueeze(1)
        num_accepted = valid.sum(dim=1).to(torch.int32)
        if path_node_ids.shape[1] == 0:
            endpoint = torch.zeros_like(num_sampled, dtype=path_node_ids.dtype)
        else:
            columns = torch.arange(path_node_ids.shape[1], device=path_node_ids.device)
            last = torch.where(valid, columns, -1).amax(dim=1)
            endpoint = path_node_ids.gather(1, last.clamp_min(0).unsqueeze(1)).squeeze(1)
            endpoint = torch.where(last >= 0, endpoint, 0)
        return cls(path_node_ids, num_sampled, num_accepted, endpoint, active)

    def select_rows(self, indices: torch.Tensor) -> "TreeAcceptance":
        """Select the same device-side request rows from every result field."""
        return TreeAcceptance(
            self.path_node_ids.index_select(0, indices),
            self.num_sampled.index_select(0, indices),
            self.num_accepted.index_select(0, indices),
            self.endpoint_node.index_select(0, indices),
            self.active.index_select(0, indices),
        )
