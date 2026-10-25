// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"
namespace {
void tree_project_check(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
    at::Tensor &nodes, const at::Tensor &ape, const at::Tensor &positions, const at::Tensor &cu) {
    TORCH_CHECK(x.dim()==2 && x.scalar_type()==at::kBFloat16 && x.is_contiguous(), "tree projection requires contiguous TH BF16");
    TORCH_CHECK(ape.dim()==2 && (ape.size(0)==4 || ape.size(0)==128), "invalid compression ratio");
    int64_t c=ape.size(0)==4?2:1, d=ape.size(1)/c;
    TORCH_CHECK(d==128 || d==512, "unsupported head dimension");
    for (const auto &w : {wkv,wgate})
        TORCH_CHECK(w.dim()==2 && w.size(0)==c*d && w.size(1)==x.size(1) &&
            w.scalar_type()==at::kBFloat16 && w.device()==x.device() && w.is_contiguous(), "invalid projection weights");
    TORCH_CHECK(nodes.is_contiguous() && nodes.scalar_type()==at::kFloat &&
        nodes.numel()==(x.size(0)+1)*2*c*d && nodes.device()==x.device(), "invalid node state");
    TORCH_CHECK(ape.is_contiguous() && ape.scalar_type()==at::kFloat && ape.device()==x.device(), "invalid APE");
    TORCH_CHECK(positions.dim()==1 && positions.numel()==x.size(0) && cu.dim()==1 && cu.numel()==x.size(0)+1,
        "one independent query row per node is required");
    for (const auto &v : {positions,cu})
        TORCH_CHECK(v.is_contiguous() && v.scalar_type()==at::kInt && v.device()==x.device(), "indices must be colocated int32");
}
void tree_project_npu(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
    at::Tensor &nodes, const at::Tensor &ape, const at::Tensor &positions, const at::Tensor &cu) {
    tree_project_check(x,wkv,wgate,nodes,ape,positions,cu);
    int64_t ratio=ape.size(0), c=ratio==4?2:1, d=ape.size(1)/c, n=x.size(0);
    auto state=nodes.view({n+1,1,2*c*d});
    // Metadata-only placeholders keep the existing native tiler ABI. The tree
    // projection kernel never reads these norm/RoPE/table values or writes cmp.
    auto norm=ape.select(0,0).narrow(0,0,d);
    auto rope=at::empty({n,64},ape.options());
    auto table=positions.view({n,1});
    auto cmp=at::empty({n,d},x.options());
    c10::optional<at::Tensor> used=c10::nullopt;
    int64_t ropeDim=64, rotary=2, cacheMode=1, stride=state.stride(0);
    double eps=1e-6;
    EXEC_NPU_CMD(aclnnCompressorTreeProject,x,wkv,wgate,state,ape,norm,rope,rope,table,cu,used,positions,
        ropeDim,ratio,c,eps,rotary,cacheMode,stride,cmp);
}
}
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def("compressor_tree_project(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) nodes, "
            "Tensor ape, Tensor positions, Tensor cu) -> ()");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops) { ops.impl("compressor_tree_project", &tree_project_npu); }
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) { ops.impl("compressor_tree_project", &tree_project_check); }
