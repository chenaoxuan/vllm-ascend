import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT_DIM = 5
SCHEMA = "occupancy_v1"
HIDDEN = 16


def compute_phi(log_q: torch.Tensor, depth) -> torch.Tensor:
    """Build occupancy features from C-wide log-softmax.

    ``log_q`` is ``[..., C]``. ``depth`` is a Python int or a tensor
    broadcastable to ``log_q[..., 0]``. Returns ``[..., C, FEAT_DIM]``:
    log_q, gap-to-second, entropy, depth, depth==1.
    """
    c = log_q.size(-1)
    if c >= 2:
        second = log_q.topk(2, dim=-1).values[..., 1]
    else:
        second = log_q[..., 0]
    gap = log_q - second.unsqueeze(-1)
    entropy = -(log_q.exp() * log_q).sum(dim=-1, keepdim=True).expand_as(log_q)
    if torch.is_tensor(depth):
        d = depth.to(dtype=log_q.dtype)
        while d.ndim < log_q.ndim - 1:
            d = d.unsqueeze(-1)
        d = d.expand(log_q.shape[:-1])
    else:
        d = log_q.new_full(log_q.shape[:-1], float(depth))
    d = d.unsqueeze(-1).expand_as(log_q)
    d1 = (d == 1).to(dtype=log_q.dtype)
    return torch.stack((log_q, gap, entropy, d, d1), dim=-1)


class OccupancyHead(nn.Module):
    """Tiny MLP: FEAT_DIM -> 16 -> SiLU -> 1. Shared by train and prefix."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(FEAT_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 1)
        self.reset_near_logq()

    def reset_near_logq(self) -> None:
        with torch.no_grad():
            self.fc1.weight.zero_()
            self.fc1.bias.zero_()
            self.fc1.weight[0, 0] = 1.0
            self.fc1.weight[1, 3] = -0.2
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
            self.fc2.weight[0, 0] = 1.0

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(phi))).squeeze(-1)


def occupancy_ell(head: nn.Module, phi: torch.Tensor) -> torch.Tensor:
    """Log-occupancy ``[..., C]``; monotone in ``head(phi)`` so topk matches."""
    return F.logsigmoid(head(phi))


def rank_by_occupancy(
    log_q: torch.Tensor,
    cand_ids: torch.Tensor,
    k: int,
    head: nn.Module,
    depth,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ell_topk, sel_ids)`` over the last dim of ``log_q`` / ``cand_ids``."""
    ell = occupancy_ell(head, compute_phi(log_q, depth))
    top_vals, top_ids = torch.topk(ell, k=min(k, ell.size(-1)), dim=-1)
    sel_ids = torch.gather(cand_ids, -1, top_ids)
    return top_vals, sel_ids


def load_occupancy_ckpt(
    path: str,
    *,
    topk: int,
    budget: int,
    candidate_size: int,
    spec_num: int | None = None,
) -> tuple[OccupancyHead, dict] | tuple[None, None]:
    """Load a trained head if ``meta`` matches this run. Host IO, not hot path."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        meta = obj.get("meta") if isinstance(obj, dict) else None
        if (
            not isinstance(obj, dict)
            or obj.get("schema") != SCHEMA
            or not isinstance(meta, dict)
            or int(obj.get("feat_dim", -1)) != FEAT_DIM
        ):
            return None, None
        if (
            int(meta.get("C", -1)) != int(candidate_size)
            or int(meta.get("topk", -1)) != int(topk)
            or int(meta.get("budget", -1)) != int(budget)
        ):
            return None, None
        if spec_num is not None and int(meta.get("spec_num", -1)) != int(spec_num):
            return None, None
        head = OccupancyHead()
        head.load_state_dict(obj["state_dict"])
        head.eval()
        return head, meta
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return None, None
