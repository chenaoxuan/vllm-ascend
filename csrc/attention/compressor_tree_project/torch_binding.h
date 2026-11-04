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
// Fused tree aggregation inputs (port of the standalone compressor_tree contract).
void tree_agg_check(const at::Tensor &state, const at::Tensor &prefix, const at::Tensor &ancestors,
    const at::Tensor &depths, const at::Tensor &requests, const at::Tensor &norm,
    const at::Tensor &cos, const at::Tensor &sin, const at::Tensor &table, int64_t n) {
    TORCH_CHECK(state.dim()>=3 && state.size(1)>0 && state.scalar_type()==at::kFloat, "invalid aggregation state");
    TORCH_CHECK(state.stride(1)>=norm.size(0), "state rows must hold both projected halves");
    TORCH_CHECK(prefix.numel()==n && prefix.is_contiguous() && prefix.scalar_type()==at::kInt, "invalid prefix");
    TORCH_CHECK(ancestors.dim()==2 && ancestors.size(0)==n && ancestors.size(1)>0 && ancestors.is_contiguous() &&
        ancestors.scalar_type()==at::kLong, "invalid ancestors");
    TORCH_CHECK(depths.numel()==n && depths.is_contiguous() && depths.scalar_type()==at::kLong, "invalid depths");
    TORCH_CHECK(requests.numel()==n && requests.is_contiguous() && requests.scalar_type()==at::kLong, "invalid requests");
    TORCH_CHECK(norm.dim()==1 && (norm.size(0)==128 || norm.size(0)==512) && norm.is_contiguous() &&
        norm.scalar_type()==at::kFloat, "invalid aggregation norm");
    TORCH_CHECK(cos.dim()==2 && sin.dim()==2 && cos.size(1)>=64 && sin.size(1)>=64 &&
        cos.stride(1)==1 && sin.stride(1)==1 && cos.scalar_type()==at::kFloat && sin.scalar_type()==at::kFloat,
        "invalid aggregation RoPE layout");
    TORCH_CHECK(table.dim()==2 && table.is_contiguous() && table.scalar_type()==at::kInt, "invalid aggregation block table");
    for (const auto &v : {state,prefix,ancestors,depths,requests,norm,cos,sin,table})
        TORCH_CHECK(v.device()==state.device(), "aggregation tensors must be colocated");
}
at::Tensor tree_project_meta(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
    at::Tensor &nodes, const at::Tensor &ape, const at::Tensor &positions, const at::Tensor &cu,
    const c10::optional<at::Tensor> &state, const c10::optional<at::Tensor> &prefix,
    const c10::optional<at::Tensor> &ancestors, const c10::optional<at::Tensor> &depths,
    const c10::optional<at::Tensor> &requests, const c10::optional<at::Tensor> &normW,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const c10::optional<at::Tensor> &table) {
    int64_t c=ape.size(0)==4?2:1, d=ape.size(1)/c;
    return at::empty({x.size(0),d},x.options());
}
at::Tensor tree_project_npu(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
    at::Tensor &nodes, const at::Tensor &ape, const at::Tensor &positions, const at::Tensor &cu,
    const c10::optional<at::Tensor> &state, const c10::optional<at::Tensor> &prefix,
    const c10::optional<at::Tensor> &ancestors, const c10::optional<at::Tensor> &depths,
    const c10::optional<at::Tensor> &requests, const c10::optional<at::Tensor> &normW,
    const c10::optional<at::Tensor> &cos, const c10::optional<at::Tensor> &sin,
    const c10::optional<at::Tensor> &table) {
    tree_project_check(x,wkv,wgate,nodes,ape,positions,cu);
    int64_t ratio=ape.size(0), c=ratio==4?2:1, d=ape.size(1)/c, n=x.size(0);
    auto stateView=nodes.view({n+1,1,2*c*d});
    // Metadata-only placeholders keep the existing native tiler ABI. The tree
    // projection kernel never reads these norm/RoPE/table values or writes cmp.
    auto norm=ape.select(0,0).narrow(0,0,d);
    auto rope=at::empty({n,64},ape.options());
    auto tablePh=positions.view({n,1});
    c10::optional<at::Tensor> used=c10::nullopt;
    int64_t ropeDim=64, rotary=2, cacheMode=1, stride=stateView.stride(0);
    double eps=1e-6;
    int64_t aggMaxDepth=0, aggWidth=0, aggPageStride=0, aggRowStride=0, aggCosStride=0, aggSinStride=0;
    if (state.has_value()) {
        // Fused tree aggregation phase: all nine inputs must be present together.
        TORCH_CHECK(prefix.has_value() && ancestors.has_value() && depths.has_value() && requests.has_value() &&
                    normW.has_value() && cos.has_value() && sin.has_value() && table.has_value(),
                    "tree aggregation inputs must be passed together with state");
        tree_agg_check(state.value(),prefix.value(),ancestors.value(),depths.value(),requests.value(),
            normW.value(),cos.value(),sin.value(),table.value(),n);
        aggMaxDepth=ancestors.value().size(1);
        aggWidth=table.value().size(1);
        aggPageStride=state.value().stride(0);
        aggRowStride=state.value().stride(1);
        aggCosStride=cos.value().stride(0);
        aggSinStride=sin.value().stride(0);
    }
    auto cmp=at::empty({n,d},x.options());
    EXEC_NPU_CMD(aclnnCompressorTreeProject,x,wkv,wgate,stateView,ape,norm,rope,rope,tablePh,cu,used,positions,
        state,prefix,ancestors,depths,requests,normW,cos,sin,table,
        ropeDim,ratio,c,eps,rotary,cacheMode,stride,
        aggMaxDepth,aggWidth,aggPageStride,aggRowStride,aggCosStride,aggSinStride,
        cmp);
    return cmp;
}
}
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def("compressor_tree_project(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) nodes, "
            "Tensor ape, Tensor positions, Tensor cu, "
            "Tensor? state=None, Tensor? prefix=None, Tensor? ancestors=None, Tensor? depths=None, "
            "Tensor? requests=None, Tensor? norm_weight=None, Tensor? cos=None, Tensor? sin=None, "
            "Tensor? table=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops) { ops.impl("compressor_tree_project", &tree_project_npu); }
TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) { ops.impl("compressor_tree_project", &tree_project_meta); }
