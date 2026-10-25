// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
#pragma once
#include <torch/extension.h>
#include <torch/library.h>
#include "aclnn_torch_adapter/op_api_common.h"

#ifdef TREE_INDEXER_STANDALONE
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
#endif

namespace {
at::Tensor tree_indexer_meta(const at::Tensor &q, const at::Tensor &k, const at::Tensor &w,
    const at::Tensor &qs, const at::Tensor &ks, const at::Tensor &nodes, const at::Tensor &nodeScale,
    const at::Tensor &ancestor_indices, const at::Tensor &prefix, const at::Tensor &req, const at::Tensor &table,
    const at::Tensor &cu, const at::Tensor &lengths, const at::Tensor &residual,
    const at::Tensor &metadata, int64_t topk) {
    TORCH_CHECK(q.dim()==3 && q.size(1)==64 && q.size(2)==128 && q.scalar_type()==at::kChar,
        "tree indexer requires INT8 [nodes,64,128] queries");
    int64_t n=q.size(0);
    TORCH_CHECK(n>0 && topk>0 && topk<=2048, "invalid tree indexer size");
    for (const auto &t : {q,w,qs,nodes,nodeScale,ancestor_indices,prefix,req,table,cu,lengths,residual,metadata})
        TORCH_CHECK(t.device()==q.device() && t.is_contiguous(), "tree inputs must be contiguous and colocated");
    TORCH_CHECK(k.device()==q.device() && ks.device()==q.device(), "history must be colocated");
    TORCH_CHECK(k.dim()==4 && k.size(2)==1 && k.size(3)==128 && k.scalar_type()==at::kChar &&
        k.stride(1)==128 && k.stride(3)==1, "invalid paged key history");
    TORCH_CHECK(ks.dim()==4 && ks.size(0)==k.size(0) && ks.size(1)==k.size(1) &&
        ks.size(2)==1 && ks.size(3)==1 && ks.stride(1)==1 && ks.scalar_type()==at::kHalf,
        "invalid paged scale history");
    TORCH_CHECK(w.numel()==n*64 && qs.numel()==n*64 && w.scalar_type()==at::kHalf && qs.scalar_type()==at::kHalf,
        "weights and query scales must be FP16 [nodes,64]");
    TORCH_CHECK(nodes.dim()==2 && nodes.size(0)==n && nodes.size(1)==128 && nodes.scalar_type()==at::kChar &&
        nodeScale.numel()==n && nodeScale.scalar_type()==at::kHalf, "invalid node key/scale pool");
    TORCH_CHECK(ancestor_indices.dim()==2 && ancestor_indices.size(0)==n && ancestor_indices.size(1)>0 && ancestor_indices.scalar_type()==at::kLong &&
        req.numel()==n && req.scalar_type()==at::kLong, "invalid tree ancestry/request indices");
    TORCH_CHECK(prefix.numel()==n && lengths.numel()==n && residual.numel()==n && cu.numel()==n+1 &&
        table.dim()==2 && metadata.numel()==1024, "invalid tree metadata shapes");
    for (const auto &t : {prefix,table,cu,lengths,residual,metadata})
        TORCH_CHECK(t.scalar_type()==at::kInt, "tree sequence metadata must be INT32");
    return at::empty({n,1,topk},q.options().dtype(at::kInt));
}

at::Tensor tree_indexer_npu(const at::Tensor &q, const at::Tensor &k, const at::Tensor &w,
    const at::Tensor &qs, const at::Tensor &ks, const at::Tensor &nodes, const at::Tensor &nodeScale,
    const at::Tensor &ancestor_indices, const at::Tensor &prefix, const at::Tensor &req, const at::Tensor &table,
    const at::Tensor &cu, const at::Tensor &lengths, const at::Tensor &residual,
    const at::Tensor &metadata, int64_t topk) {
    auto out=tree_indexer_meta(q,k,w,qs,ks,nodes,nodeScale,ancestor_indices,prefix,req,table,cu,lengths,residual,metadata,topk);
    auto values=at::empty({0},q.options().dtype(at::kBFloat16));
    auto candidateOut=at::empty({0},q.options().dtype(at::kInt));
    c10::optional<at::Tensor> none=c10::nullopt;
    int64_t quant=2,maxQ=1,mask=3,ratio=4,returnValue=0,candidateMode=3,candidateBlocks=2048,candidateSize=8;
    int64_t keyStride=k.stride(0),scaleStride=ks.stride(0);
    auto keyScale=ks.squeeze(-1);
    char qLayout[]="TND",kLayout[]="PA_BBND";
    EXEC_NPU_CMD(aclnnQuantLightningIndexerTree,q,k,w,qs,keyScale,cu,none,none,lengths,residual,table,none,metadata,none,
        nodes,nodeScale,ancestor_indices,prefix,req,topk,quant,maxQ,qLayout,kLayout,mask,ratio,returnValue,
        candidateMode,candidateBlocks,candidateSize,keyStride,scaleStride,out,values,candidateOut);
    return out;
}
}
TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
    ops.def("quant_lightning_indexer_tree(Tensor query, Tensor history, Tensor weights, Tensor query_scale, "
        "Tensor history_scale, Tensor nodes, Tensor node_scale, Tensor ancestors, Tensor prefixes, Tensor requests, "
        "Tensor table, Tensor cu, Tensor lengths, Tensor residual, Tensor metadata, int topk) -> Tensor");
    ops.impl("quant_lightning_indexer_tree",torch::kPrivateUse1,&tree_indexer_npu);
    ops.impl("quant_lightning_indexer_tree",torch::kMeta,&tree_indexer_meta);
}
