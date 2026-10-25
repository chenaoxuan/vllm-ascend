// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"
#ifdef TREE_COMPRESSOR_STANDALONE
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
#endif
namespace {
at::Tensor tree_compressor_output(const at::Tensor &state, const at::Tensor &nodes,
    const at::Tensor &norm, const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &table, const at::Tensor &prefix, const at::Tensor &ancestors,
    const at::Tensor &depths, const at::Tensor &requests, int64_t ratio, int64_t rope, double eps) {
    TORCH_CHECK(ratio == 4 || ratio == 128, "tree compressor ratio must be 4 or 128");
    TORCH_CHECK(rope == 64 && eps > 0, "tree compressor requires rope_dim=64 and positive epsilon");
    TORCH_CHECK(norm.dim() == 1 && (norm.size(0) == 128 || norm.size(0) == 512), "unsupported head dimension");
    int64_t d=norm.size(0), c=ratio==4?2:1;
    TORCH_CHECK(ancestors.dim()==2 && ancestors.size(0)>0 && ancestors.size(1)>0, "invalid ancestors");
    int64_t n=ancestors.size(0);
    TORCH_CHECK(state.dim()>=3 && state.size(1)>0 && state.stride(1)>=2*c*d, "invalid state layout");
    TORCH_CHECK(nodes.is_contiguous() && nodes.numel()==(n+1)*2*c*d, "nodes must include the reserved zero row");
    TORCH_CHECK(table.dim()==2 && table.is_contiguous(), "table must be contiguous 2D");
    for (const auto &v : {state,nodes,norm,cos,sin})
        TORCH_CHECK(v.scalar_type()==at::kFloat && v.device()==state.device(), "state and weights must be colocated float32");
    for (const auto &v : {prefix,table})
        TORCH_CHECK(v.scalar_type()==at::kInt && v.is_contiguous() && v.device()==state.device(), "prefix/table must be colocated int32");
    for (const auto &v : {ancestors,depths,requests})
        TORCH_CHECK(v.scalar_type()==at::kLong && v.is_contiguous() && v.device()==state.device(), "tree indices must be colocated int64");
    TORCH_CHECK(prefix.numel()==n && depths.numel()==n && requests.numel()==n, "tree row counts differ");
    TORCH_CHECK(cos.dim()==2 && sin.dim()==2 && cos.size(1)>=rope && sin.size(1)>=rope &&
        cos.stride(1)==1 && sin.stride(1)==1 && norm.is_contiguous(), "invalid norm/RoPE layout");
    return at::empty({n,d}, state.options().dtype(at::kBFloat16));
}
at::Tensor tree_compressor_npu(const at::Tensor &state, const at::Tensor &nodes,
    const at::Tensor &norm, const at::Tensor &cos, const at::Tensor &sin,
    const at::Tensor &table, const at::Tensor &prefix, const at::Tensor &ancestors,
    const at::Tensor &depths, const at::Tensor &requests, int64_t ratio, int64_t rope, double eps) {
    auto out=tree_compressor_output(state,nodes,norm,cos,sin,table,prefix,ancestors,depths,requests,ratio,rope,eps);
    int64_t pageStride=state.stride(0), rowStride=state.stride(1), cosStride=cos.stride(0), sinStride=sin.stride(0);
    EXEC_NPU_CMD(aclnnCompressorTree,state,nodes,norm,cos,sin,table,prefix,ancestors,depths,requests,
        ratio,rope,eps,pageStride,rowStride,cosStride,sinStride,out);
    return out;
}
}
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def("compressor_tree(Tensor state, Tensor nodes, Tensor norm, Tensor cos, Tensor sin, "
        "Tensor table, Tensor prefix, Tensor ancestors, Tensor depths, Tensor requests, "
        "int ratio, int rope_dim, float epsilon) -> Tensor");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops) { ops.impl("compressor_tree", &tree_compressor_npu); }
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) { ops.impl("compressor_tree", &tree_compressor_output); }
