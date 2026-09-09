#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from vllm_ascend.platform import ModelConfig
from vllm_ascend.utils import singleton


def _generate_attn_mask(max_seq_len, dtype):
    # Construct lower triangle matrix.
    mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
    # Create upper triangle matrix used to mark mask positions.
    mask_flag = ~mask_flag
    # Currently for fp16 dtype, the mask value should be set to -inf.
    # TODO: Eliminate this part in the future.
    mask_value = float("-inf") if dtype == torch.float16 else 1
    attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype).masked_fill_(mask_flag, mask_value)
    return attn_mask

def align_up(value, alignment=128):
    return ((value + alignment - 1) // alignment) * alignment


def _tree_query_len() -> int | None:
    """1 + budget from tree_spec_config; does not need current vLLM config."""
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        tree_cfg = get_ascend_config().tree_spec_config
        if not tree_cfg.enabled:
            return None
        return 1 + int(tree_cfg.budget)
    except (RuntimeError, AssertionError):
        return None


def _tree_spec_mask_caps() -> tuple[int, int, int] | None:
    """Max (num_decode, query_len, kv_len) when tree spec is enabled."""
    query_len = _tree_query_len()
    if query_len is None:
        return None
    max_num_seqs = 1
    max_model_len = query_len
    try:
        from vllm.config import get_current_vllm_config

        vcfg = get_current_vllm_config()
        max_num_seqs = int(vcfg.scheduler_config.max_num_seqs)
        max_model_len = int(vcfg.model_config.max_model_len)
    except (RuntimeError, AssertionError):
        pass
    return (
        max_num_seqs,
        query_len,
        align_up(max_model_len, 128),
    )


def _dummy_tree_visibility(
    num_decode: int,
    device: torch.device,
    budget: int | None = None,
) -> torch.Tensor:
    """Identity visibility for FULL capture when the dummy batch has no tree."""
    if budget is None:
        query_len = _tree_query_len()
        budget = query_len - 1 if query_len is not None else 1
    vis = torch.eye(budget, dtype=torch.bool, device=device)
    return vis.unsqueeze(0).expand(num_decode, -1, -1).contiguous()


def tree_fia_bsnd_shape(num_tokens: int) -> tuple[int, int] | None:
    """``(num_decode, 1+budget)`` when the batch is tree-verify shaped."""
    tree_q = _tree_query_len()
    if tree_q is None or tree_q <= 0 or num_tokens < tree_q or num_tokens % tree_q != 0:
        return None
    n_dec = num_tokens // tree_q
    if n_dec <= 0:
        return None
    return n_dec, tree_q


def dummy_tree_mask_for_capture(num_tokens: int, num_reqs: int) -> bool:
    """Dummy FULL capture only needs the 4D tree mask for ``k × (1+budget)``."""
    bsnd = tree_fia_bsnd_shape(num_tokens)
    return bsnd is not None and bsnd[0] == num_reqs


def _seq_lens_are_tree_query(
    seq_lens: torch.Tensor | None, num_decode: int
) -> bool:
    """Dummy FULL capture sets seq_len == query_len; tree verify is 1+budget."""
    tree_q = _tree_query_len()
    if tree_q is None or seq_lens is None or num_decode <= 0:
        return False
    sl = seq_lens[:num_decode]
    if sl.numel() != num_decode or sl.device.type != "cpu":
        return False
    return all(v == tree_q for v in sl.tolist())


def _need_dummy_tree_mask_for_capture() -> bool:
    """Target FULL capture only; draft decode must keep its own FIA mask."""
    from vllm.forward_context import is_forward_context_available

    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    if not is_forward_context_available():
        return False
    if not _EXTRA_CTX.capturing or _EXTRA_CTX.is_draft_model:
        return False
    return _tree_spec_mask_caps() is not None

@singleton
class AttentionMaskBuilder:
    def __init__(self, device: torch.device):
        self.attn_mask_cache = None
        self._seq_len_cached = 0
        self.device = device
        self.chunked_prefill_attn_mask = None
        # Growable tree-decode FIA mask; reused via fill_ + slice.
        self._tree_attn_mask: torch.Tensor | None = None
        self._tree_mask_caps = (0, 0, 0)  # (num_decode, query_len, kv_len)
        self._tree_spec_caps: tuple[int, int, int] | None = None
        # Contiguous [B, Q, KV] for FIA BSND + sparse_mode=0. Capture weak-refs
        # this buffer; do not return a temporary cat/contiguous tensor.
        self._tree_fia_bsnd: torch.Tensor | None = None

    def configure_tree_mask(
        self, max_num_decode: int, query_len: int, kv_len: int
    ) -> None:
        """Pin max mask shape from VllmConfig (capture does not always have it)."""
        self._tree_spec_caps = (max_num_decode, query_len, kv_len)
        self._tree_fia_bsnd = torch.empty(
            max_num_decode, query_len, kv_len, dtype=torch.bool, device=self.device
        )

    def _sync_tree_fia_bsnd(self, num_decode: int, query_len: int, kv_len: int) -> None:
        """Copy 4D [B, 1, Q, KV] into the 3D FIA buffer before graph replay.

        v2 run_fullgraph replays first and updates FIA params after, so the
        captured 3D pointer must already hold this step's tree mask.
        """
        if self._tree_attn_mask is None:
            return
        src = self._tree_attn_mask[:num_decode, 0, :query_len, :kv_len]
        if (
            self._tree_fia_bsnd is None
            or self._tree_fia_bsnd.shape[0] < num_decode
            or self._tree_fia_bsnd.shape[1] < query_len
            or self._tree_fia_bsnd.shape[2] < kv_len
            or self._tree_fia_bsnd.device != src.device
            or self._tree_fia_bsnd.dtype != src.dtype
        ):
            cap = self._tree_spec_caps
            self._tree_fia_bsnd = torch.empty(
                max(num_decode, cap[0] if cap else num_decode),
                max(query_len, cap[1] if cap else query_len),
                max(kv_len, cap[2] if cap else kv_len),
                dtype=src.dtype,
                device=src.device,
            )
        self._tree_fia_bsnd[:num_decode, :query_len, :kv_len].copy_(src)
        if self._tree_fia_bsnd.shape[0] > num_decode:
            self._tree_fia_bsnd[num_decode:, :query_len, :kv_len].fill_(True)

    def as_bsnd_fia_mask(
        self,
        attn_mask: torch.Tensor | None,
        batch_size: int,
        q_len: int,
    ) -> torch.Tensor | None:
        """Return the 3D [B, Q, KV] slice FIA captured (stable pointer).

        Content is written in ``_sync_tree_fia_bsnd`` during prepare_attn.
        Do not allocate a new tensor here — that would dangle the graph weak-ref.
        """
        buf = self._tree_fia_bsnd
        if buf is not None and buf.shape[0] >= batch_size and buf.shape[1] >= q_len:
            return buf[:batch_size, :q_len]
        if attn_mask is None or attn_mask.ndim < 3:
            return None
        src = attn_mask[:, 0] if attn_mask.ndim == 4 else attn_mask
        src_b, src_q, src_kv = src.shape
        if src_q < q_len:
            return None
        self._tree_fia_bsnd = torch.empty(
            max(batch_size, src_b), src_q, src_kv, dtype=src.dtype, device=src.device
        )
        n = min(src_b, batch_size)
        self._tree_fia_bsnd[:n, :src_q, :src_kv].copy_(src[:n])
        if n < batch_size:
            self._tree_fia_bsnd[n:batch_size, :src_q, :src_kv].fill_(True)
        return self._tree_fia_bsnd[:batch_size, :src_q, :src_kv]

    def _resolved_tree_caps(self) -> tuple[int, int, int] | None:
        return self._tree_spec_caps or _tree_spec_mask_caps()

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype):
        if self.attn_mask_cache is None or max_seq_len > self._seq_len_cached:
            self.attn_mask_cache = _generate_attn_mask(max_seq_len, dtype)
            self._seq_len_cached = max_seq_len
        assert self.attn_mask_cache is not None, "Something is wrong in generate_attn_mask."
        if self.attn_mask_cache.dtype != dtype:
            self.attn_mask_cache = self.attn_mask_cache.to(dtype)
        return self.attn_mask_cache[:max_seq_len, :max_seq_len].contiguous().to(self.device, non_blocking=True)

    def get_splitfuse_attn_mask(self) -> torch.Tensor:
        if self.chunked_prefill_attn_mask is None:
            self.chunked_prefill_attn_mask = (
                torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8).to(self.device)
            )
        return self.chunked_prefill_attn_mask

    def get_attention_mask(self, causal: bool,
                           model_config: ModelConfig,
                           tree_visibility: torch.Tensor | None = None,
                           seq_lens: torch.Tensor | None = None,
                           num_decode: int = 0,
                           num_tokens: int = 0,
                           for_capture: bool = False,
                           ):
        if not causal:
            # FIA applies any provided mask as defaultMask (sparse_mode=0),
            # which would wrongly mask out the upper triangle for
            # bidirectional attention, so non-causal attention must not
            # carry a mask here. The 310P mask builder overrides this
            # because its attention operators require an explicit
            # non-masking mask instead.
            return None

        if num_decode > 0:
            tree_shaped = (
                num_tokens > 0 and dummy_tree_mask_for_capture(num_tokens, num_decode)
            )
            if (
                tree_visibility is None
                and (
                    tree_shaped
                    if for_capture
                    else (
                        _need_dummy_tree_mask_for_capture()
                        and _seq_lens_are_tree_query(seq_lens, num_decode)
                    )
                )
            ):
                tree_visibility = _dummy_tree_visibility(num_decode, self.device)
            if tree_visibility is not None:
                # Dummy FULL gears that are not k × (1+budget) must keep the
                # 2D splitfuse mask. TND + sparse 3 rejects a 4D tree mask.
                if for_capture and not tree_shaped:
                    return self.get_splitfuse_attn_mask()
                return self.get_tree_attention_mask(
                    tree_visibility, seq_lens, num_decode
                )

        if model_config.runner_type == "pooling":
            return self.get_attn_mask(2048, torch.bool)

        return self.get_splitfuse_attn_mask()
    
    def get_tree_attention_mask(self, tree_visibility: torch.Tensor,
                                seq_lens: torch.Tensor,
                                num_decode):
        from vllm_ascend.worker.v2.spec_decode.tree.timer import (
            tree_time,
            tree_timer_begin_step,
        )

        tree_timer_begin_step()
        with tree_time("build_attn_mask"):
            return self._get_tree_attention_mask_impl(
                tree_visibility, seq_lens, num_decode
            )

    def _get_tree_attention_mask_impl(self, tree_visibility: torch.Tensor,
                                     seq_lens: torch.Tensor,
                                     num_decode):
        from vllm_ascend.worker.v2.spec_decode.tree.triton_dispatch import (
            use_tree_triton,
        )

        max_nodes = tree_visibility.shape[-1]
        query_len = 1 + max_nodes
        max_caps = self._resolved_tree_caps()
        if max_caps is not None:
            query_len = max(query_len, max_caps[1])
        num_mask = min(num_decode, tree_visibility.shape[0], seq_lens.shape[0])
        # seq_lens is host metadata (CPU). Caps already cover max_model_len.
        if max_caps is not None:
            kv_len = max_caps[2]
            alloc_b = max(num_decode, max_caps[0])
            alloc_q = max(query_len, max_caps[1])
            alloc_kv = kv_len
        else:
            kv_len = align_up(int(seq_lens[:num_mask].max()), 128)
            alloc_b, alloc_q, alloc_kv = num_decode, query_len, kv_len
        # Paged FIA custom mask is (B, 1, Q_S, KV_S). A 3D (B, Q, Kv) tensor
        # looks like (Q, Kv) at bs=1 but shares the leading request's mask at bs>1.
        # Allocate once to max caps so FULL capture/replay keep a stable pointer.
        cap_b, cap_q, cap_kv = self._tree_mask_caps
        if (
            self._tree_attn_mask is None
            or alloc_b > cap_b
            or alloc_q > cap_q
            or alloc_kv > cap_kv
            or self._tree_attn_mask.device != self.device
        ):
            self._tree_attn_mask = torch.empty(
                alloc_b, 1, alloc_q, alloc_kv, dtype=torch.bool, device=self.device
            )
            self._tree_mask_caps = (alloc_b, alloc_q, alloc_kv)
            cap_kv = alloc_kv
        # Use the allocated KV width so graph capture binds a fixed shape.
        attn_mask = self._tree_attn_mask[:num_decode, :, :query_len, :cap_kv]
        # Triton needs a contiguous bool→int8 view; sliced caps (kv_len < cap)
        # are non-contiguous and must stay on the torch path.
        mask_slice = attn_mask[:num_mask]
        if use_tree_triton() and mask_slice.is_contiguous():
            from vllm_ascend.ops.triton.spec_decode.tree.attention_mask import (
                fill_tree_attention_mask_triton,
            )

            prev = seq_lens[:num_mask].to(dtype=torch.int32)
            if prev.device != self.device:
                prev = prev.to(self.device, non_blocking=True)  # H2D
            prev = prev - query_len
            fill_tree_attention_mask_triton(
                mask_slice, tree_visibility[:num_mask], prev
            )
            if num_decode > num_mask:
                attn_mask[num_mask:].fill_(True)
            self._sync_tree_fia_bsnd(num_decode, query_len, cap_kv)
            return attn_mask
        attn_mask.fill_(True)
        for i in range(num_mask):
            req_mask = attn_mask[i, 0]
            prev_kv_len = int(seq_lens[i]) - query_len
            req_mask[:, : prev_kv_len + 1] = False
            # tree_visibility: True = can attend; FIA bool mask: True = masked out.
            # Draft columns are slot-indexed (contiguous j). KV slot_mapping uses
            # unique token-index coordinates while RoPE positions stay depth-based.
            draft_start = prev_kv_len + 1
            req_mask[1:, draft_start : draft_start + max_nodes] = ~tree_visibility[i]
        self._sync_tree_fia_bsnd(num_decode, query_len, cap_kv)
        return attn_mask
