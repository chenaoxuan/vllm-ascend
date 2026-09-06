import time
from pathlib import Path

import torch

from vllm_ascend.worker.v2.spec_decode.tree.training_tree.features import SCHEMA

_pending_stash: dict | None = None


def set_pending_stash(stash: dict | None) -> None:
    global _pending_stash
    _pending_stash = stash


def take_pending_stash() -> dict | None:
    global _pending_stash
    stash = _pending_stash
    _pending_stash = None
    return stash


def empty_parent_stash(
    num_reqs: int,
    max_nodes: int,
    candidate_count: int,
    device: torch.device,
    *,
    meta: dict,
    dump_path: str,
) -> dict:
    """Device buffers indexed by final parent node id ``0 .. max_nodes-1``."""
    return {
        "log_q": torch.zeros(
            num_reqs, max_nodes, candidate_count, dtype=torch.float32, device=device
        ),
        "cand_ids": torch.zeros(
            num_reqs, max_nodes, candidate_count, dtype=torch.int32, device=device
        ),
        "depth": torch.zeros(num_reqs, max_nodes, dtype=torch.int16, device=device),
        "valid": torch.zeros(num_reqs, max_nodes, dtype=torch.bool, device=device),
        "meta": meta,
        "dump_path": dump_path,
    }


def scatter_parent_rows(
    stash: dict,
    parent_ids: torch.Tensor,
    log_q: torch.Tensor,
    cand_ids: torch.Tensor,
    depth: int,
    valid_parent: torch.Tensor,
) -> None:
    """Write one depth's ``[R, width, C]`` rows into the parent-indexed stash."""
    num_reqs, width, c = log_q.shape
    max_nodes = stash["log_q"].shape[1]
    device = log_q.device
    req = torch.arange(num_reqs, device=device).unsqueeze(1).expand(num_reqs, width)
    pid = parent_ids.clamp(min=0, max=max_nodes - 1)
    write = valid_parent
    stash["log_q"][req, pid] = torch.where(
        write.unsqueeze(-1), log_q, stash["log_q"][req, pid]
    )
    stash["cand_ids"][req, pid] = torch.where(
        write.unsqueeze(-1), cand_ids.to(torch.int32), stash["cand_ids"][req, pid]
    )
    stash["depth"][req, pid] = torch.where(
        write, torch.full_like(stash["depth"][req, pid], depth), stash["depth"][req, pid]
    )
    stash["valid"][req, pid] = stash["valid"][req, pid] | write


def remap_stash_parents(stash: dict, old_to_new: torch.Tensor, new_n: int) -> dict:
    """Reindex stash from pre-prune ids to final node ids. Root 0 stays 0."""
    log_q = stash["log_q"]
    num_reqs, old_n, c = log_q.shape
    new_n = max(int(new_n), 1)
    device = log_q.device
    # Extra slot is a sink so unkept writes do not clobber root (nid 0).
    out = empty_parent_stash(
        num_reqs,
        new_n + 1,
        c,
        device,
        meta=stash["meta"],
        dump_path=stash["dump_path"],
    )
    old_ids = torch.arange(old_n, device=device).unsqueeze(0).expand(num_reqs, old_n)
    in_range = old_ids < old_to_new.shape[1]
    new_ids = old_to_new.gather(1, old_ids.clamp(max=old_to_new.shape[1] - 1))
    keep = stash["valid"] & in_range & ((old_ids == 0) | (new_ids > 0))
    req = torch.arange(num_reqs, device=device).unsqueeze(1).expand(num_reqs, old_n)
    sink = new_n
    nid = torch.where(
        keep, new_ids.clamp(min=0, max=new_n - 1), new_ids.new_full(new_ids.shape, sink)
    )
    out["log_q"][req, nid] = torch.where(
        keep.unsqueeze(-1), log_q, out["log_q"][req, nid]
    )
    out["cand_ids"][req, nid] = torch.where(
        keep.unsqueeze(-1), stash["cand_ids"], out["cand_ids"][req, nid]
    )
    out["depth"][req, nid] = torch.where(keep, stash["depth"], out["depth"][req, nid])
    out["valid"][req, nid] = out["valid"][req, nid] | keep
    out["log_q"] = out["log_q"][:, :new_n]
    out["cand_ids"] = out["cand_ids"][:, :new_n]
    out["depth"] = out["depth"][:, :new_n]
    out["valid"] = out["valid"][:, :new_n]
    return out


def flush_occupancy_dump(
    stash: dict,
    path_node_ids: torch.Tensor,
    gold_token_ids: torch.Tensor,
) -> Path | None:
    """Keep gold-parent rows, pack ``occupancy_v1``, ``torch.save`` one shard.

    ``path_node_ids`` is ``[R, spec_len]`` accepted children (``-1`` unused).
    ``gold_token_ids`` is ``[R, spec_len]`` greedy target ids at each parent.
    """
    log_q = stash["log_q"]
    cand_ids = stash["cand_ids"]
    depth = stash["depth"]
    valid = stash["valid"]
    num_reqs, _p, c = log_q.shape
    spec_len = gold_token_ids.shape[1]
    device = log_q.device
    parent = torch.zeros(num_reqs, spec_len, dtype=torch.long, device=device)
    if spec_len > 1:
        parent[:, 1:] = path_node_ids[:, :-1].clamp(min=0)
    alive = gold_token_ids >= 0
    if spec_len > 1:
        alive = alive.clone()
        alive[:, 1:] = alive[:, 1:] & (path_node_ids[:, :-1] >= 0)
    parent = parent.clamp(max=log_q.shape[1] - 1)
    req = torch.arange(num_reqs, device=device).unsqueeze(1).expand(num_reqs, spec_len)
    row_ok = valid[req, parent] & alive
    flat = row_ok.reshape(-1)
    n_keep = int(flat.to(torch.int32).sum().detach().to("cpu"))  # D2H
    if n_keep == 0:
        return None
    packed = {
        "schema": SCHEMA,
        "meta": dict(stash["meta"]),
        # D2H: dump boundary only.
        "log_q": log_q[req, parent].reshape(-1, c)[flat].detach().to("cpu", torch.float16),
        "cand_ids": cand_ids[req, parent].reshape(-1, c)[flat].detach().to("cpu", torch.int32),
        "gold_id": gold_token_ids.reshape(-1)[flat].detach().to("cpu", torch.int32),
        "depth": depth[req, parent].reshape(-1)[flat].detach().to("cpu", torch.int16),
    }
    out_dir = Path(stash["dump_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"shard_{time.time_ns()}.pt"
    torch.save(packed, dest)
    return dest
