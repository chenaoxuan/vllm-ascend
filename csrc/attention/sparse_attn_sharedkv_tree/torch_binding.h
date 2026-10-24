// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
#include <torch/extension.h>
#include <torch/library.h>
#pragma once
#include "aclnn_torch_adapter/op_api_common.h"

#ifdef TREE_ATTENTION_STANDALONE
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
#endif

namespace {
std::tuple<at::Tensor, at::Tensor> tree_attention_output(const at::Tensor &q, std::string layout,
    bool return_softmax_lse)
{
    for (size_t i = 0; i < q.sizes().size(); i++) {
        TORCH_CHECK(q.size(i) > 0,
            "All values within query's shape should be greater "
            "than 0, but shape[",
            i,
            "] is ",
            q.size(i));
    }
    at::Tensor output = at::empty_symint(q.sym_sizes(), q.options());
    at::Tensor softmax_lse;
    if (return_softmax_lse) {
        c10::SymDimVector lse_sizes(q.sym_sizes().begin(), q.sym_sizes().end());
        lse_sizes.back() = c10::SymInt(1);
        softmax_lse = at::empty_symint(lse_sizes, q.options().dtype(c10::ScalarType::Float));
    } else {
        softmax_lse = at::empty({0}, q.options().dtype(c10::ScalarType::Float));
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_attn_sharedkv_tree_npu(const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata,
    const c10::optional<at::Tensor> &cmp_tree_kv, const c10::optional<at::Tensor> &tree_ancestors,
    const c10::optional<at::Tensor> &tree_prefix,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left,
    int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv, bool return_softmax_lse)
{
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    std::tuple<at::Tensor, at::Tensor> output = tree_attention_output(q, layout_q_str, return_softmax_lse);
    at::Tensor attn_out = std::get<0>(output);
    at::Tensor softmax_lse = std::get<1>(output);
    int64_t ori_kv_stride = 0;
    int64_t cmp_kv_stride = 0;
    if (ori_kv.has_value()){
        const at::Tensor& tmp_kv = *ori_kv;
        ori_kv_stride = tmp_kv.stride(0);
    }
    if (cmp_kv.has_value()){
        const at::Tensor& tmp_kv = *cmp_kv;
        cmp_kv_stride = tmp_kv.stride(0);
    }

    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    EXEC_NPU_CMD(aclnnSparseAttnSharedkvTree, q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
        ori_block_table, cmp_block_table, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_kv, sinks,
        metadata, cmp_tree_kv, tree_ancestors, tree_prefix, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_kv_stride, cmp_kv_stride, ori_win_left, ori_win_right, layout_q_ptr,
        layout_kv_ptr, return_softmax_lse, attn_out, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(attn_out, softmax_lse);
}
std::tuple<at::Tensor, at::Tensor> npu_sparse_attn_sharedkv_tree_meta(const at::Tensor &q, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata,
    const c10::optional<at::Tensor> &cmp_tree_kv, const c10::optional<at::Tensor> &tree_ancestors,
    const c10::optional<at::Tensor> &tree_prefix,
    double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left,
    int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv, bool return_softmax_lse)
{
    std::string layout_q_str = std::string(layout_q);
    std::tuple<at::Tensor, at::Tensor> output = tree_attention_output(q, layout_q_str, return_softmax_lse);

    return output;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)
{
    ops.def(
        "npu_sparse_attn_sharedkv_tree("
            "Tensor q, *, "
            "Tensor? ori_kv=None, "
            "Tensor? cmp_kv=None, "
            "Tensor? ori_sparse_indices=None, "
            "Tensor? cmp_sparse_indices=None, "
            "Tensor? ori_block_table=None, "
            "Tensor? cmp_block_table=None, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "Tensor? sinks=None, "
            "Tensor? metadata=None, "
            "Tensor? cmp_tree_kv=None, "
            "Tensor? tree_ancestors=None, "
            "Tensor? tree_prefix=None, "
            "float softmax_scale=0, "
            "int cmp_ratio=0, "
            "int ori_mask_mode=4, "
            "int cmp_mask_mode=3, "
            "int ori_win_left=128, "
            "int ori_win_right=0, "
            "str layout_q=\"BSND\", "
            "str layout_kv=\"PA_ND\", "
            "bool return_softmax_lse=False"
        ") -> (Tensor out, Tensor softmax_lse)"
        );
    ops.impl("npu_sparse_attn_sharedkv_tree", torch::kPrivateUse1, &npu_sparse_attn_sharedkv_tree_npu);

    ops.impl("npu_sparse_attn_sharedkv_tree", torch::kMeta, &npu_sparse_attn_sharedkv_tree_meta);
}
