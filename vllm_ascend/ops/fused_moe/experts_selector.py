#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
from collections.abc import Callable

import torch
import torch.nn.functional as F
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import split_tensor_along_first_dim
from vllm_ascend.utils import get_weight_prefetch_method


def _ensure_dflash_layout(context, device: torch.device) -> bool:
    """Build once per forward: all_rows, req_boundaries, arange, exposed rows.

    Returns False when there is no suffix draft token to prune.
    """
    if getattr(context, "dflash_all_rows", None) is not None:
        return context.dflash_exposed_global_rows.numel() > 0

    all_rows_list = []
    req_boundaries = [0]
    for _req_id, rows in context.dflash_verify_rows:
        if rows.numel() == 0:
            continue
        all_rows_list.append(rows)
        req_boundaries.append(req_boundaries[-1] + rows.numel())

    if not all_rows_list:
        empty = torch.empty(0, dtype=torch.long, device=device)
        context.dflash_all_rows = empty
        context.dflash_req_boundaries = [0]
        context.dflash_req_boundaries_gpu = torch.zeros(1, dtype=torch.int32, device=device)
        context.dflash_arange = empty
        context.dflash_exposed_global_rows = empty
        return False

    all_rows = torch.cat(all_rows_list)
    arange = torch.arange(all_rows.numel(), dtype=torch.long, device=device)
    prefix_k = context.dflash_prefix_k
    exposed_local = []
    for i in range(len(req_boundaries) - 1):
        start = req_boundaries[i]
        end = req_boundaries[i + 1]
        protected = min(prefix_k, end - start)
        if protected < end - start:
            exposed_local.append(arange[start + protected : end])

    context.dflash_all_rows = all_rows
    context.dflash_req_boundaries = req_boundaries
    context.dflash_req_boundaries_gpu = torch.tensor(req_boundaries, dtype=torch.int32, device=device)
    context.dflash_arange = arange
    if exposed_local:
        context.dflash_exposed_global_rows = all_rows[torch.cat(exposed_local)]
    else:
        context.dflash_exposed_global_rows = torch.empty(0, dtype=torch.long, device=device)
    return context.dflash_exposed_global_rows.numel() > 0


def _get_dflash_union_mask(context, num_experts: int, device: torch.device) -> torch.Tensor:
    mask = getattr(context, "dflash_union_mask", None)
    if mask is None or mask.numel() != num_experts or mask.device != device:
        mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
        context.dflash_union_mask = mask
    else:
        mask.zero_()
    return mask


def _apply_dflash_pruning(
    router_logits: torch.Tensor,
    top_k: int,
    scoring_func: str,
    use_grouped_topk: bool,
    custom_routing_function: Callable | None,
) -> torch.Tensor:
    context = get_forward_context()
    mode = getattr(context, "dflash_mode", "baseline")
    verify_rows = getattr(context, "dflash_verify_rows", ())
    prefix_k = getattr(context, "dflash_prefix_k", None)
    escape_rank = getattr(context, "dflash_escape_rank", 1)

    if mode == "baseline" or not verify_rows:
        return router_logits
    if mode != "union":
        raise ValueError(f"unsupported DFlash prefix mode: {mode!r}")
    if prefix_k is None or prefix_k < 0:
        raise ValueError("DFlash prefix k must be non-negative")
    if escape_rank < 1 or escape_rank > top_k:
        raise ValueError("DFlash escape rank must be in [1, top_k]")
    if scoring_func != "softmax" or use_grouped_topk or custom_routing_function is not None:
        return router_logits
    if router_logits.ndim != 2 or not router_logits.dtype.is_floating_point:
        return router_logits

    if not _ensure_dflash_layout(context, router_logits.device):
        return router_logits

    _, num_experts = router_logits.shape
    selected_k = min(top_k, num_experts)
    all_rows = context.dflash_all_rows
    req_boundaries = context.dflash_req_boundaries
    exposed_global_rows = context.dflash_exposed_global_rows
    all_topk_ids = torch.topk(router_logits[all_rows], selected_k, dim=-1).indices
    apply_dflash_union_mask(
        router_logits,
        all_topk_ids,
        req_boundaries,
        exposed_global_rows,
        context,
        prefix_k,
        escape_rank,
        num_experts,
    )

    return router_logits


def _apply_dflash_union_mask_torch(
    router_logits: torch.Tensor,
    all_topk_ids: torch.Tensor,
    req_boundaries: list[int],
    exposed_global_rows: torch.Tensor,
    union_mask: torch.Tensor,
    prefix_k: int,
    escape_rank: int,
) -> None:
    """Native PyTorch stage-2: build union_mask and write -inf on suffix rows."""
    allowed_expert_ids = []
    for i in range(len(req_boundaries) - 1):
        start = req_boundaries[i]
        end = req_boundaries[i + 1]
        num_rows_in_req = end - start
        req_ids = all_topk_ids[start:end]
        protected = min(prefix_k, num_rows_in_req)

        if protected > 0:
            allowed_expert_ids.append(req_ids[:protected].reshape(-1))
        if protected < num_rows_in_req:
            allowed_expert_ids.append(req_ids[protected:, :escape_rank].reshape(-1))

    union_mask[torch.cat(allowed_expert_ids)] = True
    exposed_logits = router_logits[exposed_global_rows]
    allowed = union_mask.unsqueeze(0).expand(exposed_logits.shape[0], -1)
    router_logits[exposed_global_rows] = torch.where(
        allowed, exposed_logits, torch.finfo(router_logits.dtype).min
    )


def apply_dflash_union_mask(
    router_logits: torch.Tensor,
    all_topk_ids: torch.Tensor,
    req_boundaries: list[int],
    exposed_global_rows: torch.Tensor,
    context,
    prefix_k: int,
    escape_rank: int,
    num_experts: int,
) -> None:
    """Stage-2 interface. Prefers the fused Triton kernels; native torch is kept."""
    union_mask = _get_dflash_union_mask(context, num_experts, router_logits.device)
    from vllm.triton_utils import HAS_TRITON

    if HAS_TRITON:
        from vllm_ascend.ops.triton.dflash_moe_prune import apply_dflash_union_mask_triton

        bounds = getattr(context, "dflash_req_boundaries_gpu", None)
        if bounds is None:
            bounds = torch.tensor(req_boundaries, dtype=torch.int32, device=router_logits.device)
            context.dflash_req_boundaries_gpu = bounds
        max_rows = 0
        for i in range(len(req_boundaries) - 1):
            max_rows = max(max_rows, req_boundaries[i + 1] - req_boundaries[i])
        apply_dflash_union_mask_triton(
            router_logits,
            all_topk_ids,
            bounds,
            exposed_global_rows,
            union_mask,
            prefix_k,
            escape_rank,
            max_rows,
        )
        return

    _apply_dflash_union_mask_torch(
        router_logits,
        all_topk_ids,
        req_boundaries,
        exposed_global_rows,
        union_mask,
        prefix_k,
        escape_rank,
    )


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor=1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    indices_type: torch.dtype | None = None,
    mix_placement: bool = False,
    num_logical_experts: int = -1,
    num_shared_experts: int = 0,
    num_experts: int = -1,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
):
    """
    Fused experts with select experts.

    Args:
        router_logits: router logits of shape (num_tokens, hidden_size).
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        top_k: number of top k experts.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.
        indices_type: dtype of indices
        num_experts: Number of experts.

    Returns:
        topk_weights: router weights of shape (num_tokens, top_k).
        topk_ids: selected expert IDs of shape (num_tokens, top_k).
    """
    router_logits = _apply_dflash_pruning(
        router_logits=router_logits,
        top_k=top_k,
        scoring_func=scoring_func,
        use_grouped_topk=use_grouped_topk,
        custom_routing_function=custom_routing_function,
    )

    is_support_npu_moe_gating_top_k = check_npu_moe_gating_top_k(
        hidden_states=hidden_states,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        scoring_func=scoring_func,
        custom_routing_function=custom_routing_function,
    )

    if is_support_npu_moe_gating_top_k:
        topk_weights, topk_ids = _select_experts_with_fusion_ops(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            tid2eid=tid2eid,
            input_ids=input_ids,
        )
    else:
        topk_weights, topk_ids = _native_select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            tid2eid=None,
            input_ids=None,
        )
        # Apply routed scaling factor to weights
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor
    if mix_placement:
        shared_expert_routing_factor = 1.0 if is_support_npu_moe_gating_top_k else (1 / routed_scaling_factor)
        batch_size = topk_ids.shape[0]
        pad_shared_expert_ids = torch.arange(
            num_logical_experts, num_logical_experts + num_shared_experts, dtype=topk_ids.dtype, device=topk_ids.device
        ).repeat(batch_size, 1)

        pad_shared_expert_weights = torch.full(
            (topk_weights.shape[0], num_shared_experts),
            shared_expert_routing_factor,
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )

        topk_ids = torch.cat([topk_ids, pad_shared_expert_ids], dim=1)
        topk_weights = torch.cat([topk_weights, pad_shared_expert_weights], dim=1)

    return topk_weights, topk_ids


def check_npu_moe_gating_top_k(
    hidden_states: torch.Tensor,
    top_k: int,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    scoring_func: str = "softmax",
    custom_routing_function: Callable | None = None,
):
    if scoring_func == "sigmoid" and not renormalize:  # sigmoid + renorm=0 is not supported in current branch
        return False
    if custom_routing_function is not None:
        return False
    if scoring_func != "softmax" and scoring_func != "sigmoid" and scoring_func != "sqrtsoftplus":
        return False
    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    if not (
        num_expert_group > 0
        and hidden_states.shape[-1] % num_expert_group == 0
        and hidden_states.shape[-1] // num_expert_group > 2
    ):
        return False
    if topk_group < 1 or topk_group > num_expert_group:
        return False
    if top_k < 1 or top_k > (hidden_states.shape[-1] / (num_expert_group * topk_group)):
        return False
    if topk_group * hidden_states.shape[-1] / num_expert_group < top_k:  # noqa: SIM103
        return False
    return True


def _native_grouped_topk(
    topk_weights: torch.Tensor,
    num_expert_group: int | None,
    topk_group: int | None,
):
    topk_group = 0 if topk_group is None else topk_group
    num_expert_group = 0 if num_expert_group is None else num_expert_group

    num_token = topk_weights.shape[0]
    grouped_weights = topk_weights.view(num_token, num_expert_group, -1).max(dim=-1).values
    topk_group_indices = torch.topk(grouped_weights.to(torch.float32), k=topk_group, dim=-1, sorted=False)[1]
    topk_group_mask = torch.zeros_like(grouped_weights)
    topk_group_mask.scatter_(1, topk_group_indices, 1)
    topk_weight_mask = (
        topk_group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, topk_weights.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )
    topk_weights = topk_weights.masked_fill(~topk_weight_mask.bool(), 0.0)

    return topk_weights


def _renormalize_topk_weights(
    topk_weights: torch.Tensor,
    renormalize: bool,
):
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights


def _select_expert_use_group_topk(
    topk_weights: torch.Tensor,
    topk_group: int | None,
    renormalize: bool,
    top_k: int,
    num_expert_group: int | None,
    e_score_correction_bias: torch.Tensor | None,
):
    assert topk_group is not None
    assert num_expert_group is not None

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_weights = topk_weights
        topk_weights = topk_weights + e_score_correction_bias.unsqueeze(0)

    # TODO: Change to npu_group_topk when the latest CANN and NNAL is available
    # >>> torch_npu._npu_group_topk(topk_weights, group_num=num_expert_group, k=topk_group)
    topk_weights = _native_grouped_topk(topk_weights, num_expert_group, topk_group)
    # TODO bfloat16 is not supported in torch.topk with ge graph.
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(topk_weights.to(torch.float32), k=top_k, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_weights.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(topk_weights.to(torch.float32), k=top_k, dim=-1, sorted=False)
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = _renormalize_topk_weights(topk_weights, renormalize)
    return topk_weights, topk_ids


def _select_experts_with_fusion_ops(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    topk_group: int | None,
    num_expert_group: int | None,
    scoring_func: str = "softmax",
    routed_scaling_factor=1.0,
    tid2eid=None,
    input_ids=None,
):
    topk_group = topk_group if topk_group is not None else 1
    num_expert_group = num_expert_group if num_expert_group is not None else 1
    renorm = int(renormalize)
    if scoring_func == "sqrtsoftplus":
        if tid2eid is not None:
            forward_context = get_forward_context()
            input_ids = forward_context.input_ids.to(torch.int64)
            # tid2eid_ones = torch.ones(tid2eid.shape[0],tid2eid.shape[1],device=router_logits.device,dtype=torch.int32)
            tid2eid_ones = tid2eid.to(torch.int32)
            if forward_context.moe_comm_type == MoECommType.ALLGATHER:
                prepare_finalize = forward_context.moe_comm_method.prepare_finalize
                input_ids = prepare_finalize.all_gather_input_id_with_dp_group(input_ids)
            else:
                input_ids = forward_context.moe_comm_method.pad_and_split_input_ids(input_ids)

            if forward_context.flash_comm_v1_enabled and forward_context.moe_comm_type != MoECommType.ALLGATHER:
                # Process for Flash Comm V1
                tp_size = get_tp_group().world_size
                tp_rank = get_tp_group().rank_in_group
                splitted_input = split_tensor_along_first_dim(input_ids, num_partitions=tp_size)
                input_ids = splitted_input[tp_rank].contiguous()
            input_ids = torch.where(input_ids == -1, 0, input_ids)
        else:
            input_ids = None
            tid2eid_ones = None
        topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
            x=router_logits,
            k=top_k,
            bias=e_score_correction_bias,
            input_ids=input_ids,
            tid2eid=tid2eid_ones,
            k_group=topk_group,
            group_count=num_expert_group,
            routed_scaling_factor=routed_scaling_factor,
            eps=1e-20,
            group_select_mode=1,
            # The hash custom op currently rejects renorm != 0. Apply
            # norm_topk_prob in Python below before returning to MoE compute.
            renorm=0,
            norm_type=2,
            out_flag=False,
        )
        return topk_weights, topk_ids
    norm_type = 0 if scoring_func == "softmax" else 1
    if e_score_correction_bias is not None and e_score_correction_bias.dtype != router_logits.dtype:
        e_score_correction_bias = e_score_correction_bias.to(router_logits.dtype)
    topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
        router_logits,
        k=top_k,
        k_group=topk_group,
        group_count=num_expert_group,
        group_select_mode=1,
        renorm=renorm,
        norm_type=norm_type,  # 0: softmax; 1: sigmoid
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=1e-20,
        bias_opt=e_score_correction_bias,
    )

    return topk_weights, topk_ids


def _native_select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    use_hash: bool = False,
    tid2eid: dict[int, int] | None = None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k experts based on router logits.

    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        router_logits: Router logits of shape (num_tokens, num_experts).
        top_k: Number of experts to select.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.

    Returns:
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).

    Raises:
        ValueError: If an unsupported scoring function is provided.
    """

    if scoring_func == "softmax":
        topk_weights = router_logits.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        topk_weights = router_logits.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        topk_weights = F.softplus(router_logits).sqrt()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if use_grouped_topk:
        topk_weights, topk_ids = _select_expert_use_group_topk(
            topk_weights=topk_weights,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
        )
        return topk_weights * routed_scaling_factor, topk_ids

    if e_score_correction_bias is not None:
        topk_weights = topk_weights + e_score_correction_bias

    if custom_routing_function is not None:
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
        # Required by npu_moe_init_routing
        topk_ids = topk_ids.to(torch.int32)
        return topk_weights, topk_ids

    topk_weights, topk_ids = topk_weights.topk(top_k, dim=-1)
    topk_weights = topk_weights.to(hidden_states.dtype)

    # Required by npu_moe_init_routing
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = _renormalize_topk_weights(topk_weights, renormalize)
    topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids


def zero_experts_compute(
    expert_indices: torch.Tensor,
    expert_scales: torch.Tensor,
    num_experts: int,
    zero_expert_type: str,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if zero_expert_type == "identity":
        zero_expert_mask = expert_indices < num_experts
        zero_expert_scales = expert_scales.clone()
        zero_expert_scales = torch.where(zero_expert_mask, 0.0, zero_expert_scales)

        hidden_states = hidden_states.unsqueeze(1)
        zero_expert_scales = zero_expert_scales.unsqueeze(2)
        result = hidden_states * zero_expert_scales
        result = result.sum(dim=1)

    normal_expert_mask = expert_indices >= num_experts
    expert_indices = torch.where(normal_expert_mask, 0, expert_indices)
    expert_scales = torch.where(normal_expert_mask, 0.0, expert_scales)

    return expert_indices, expert_scales, result
