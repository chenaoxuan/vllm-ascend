/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_tree.cpp
 * \brief
 */


#include "kernel_operator.h"
#include "arch22/quant_lightning_indexer_tree_kernel_arch22.h"
#include "quant_lightning_indexer_tree_template_tiling_key.h"
using namespace QLITreeKernel;
using namespace optiling::detail;
template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void quant_lightning_indexer_tree(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR queryScale, GM_ADDR keyScale,
    GM_ADDR cuSeqlensQ, GM_ADDR cuSeqlensK, GM_ADDR sequsedQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK,
    GM_ADDR blockTable, GM_ADDR outputIdxOffset, GM_ADDR metadata, GM_ADDR candidateTopkIndex,
    GM_ADDR treeKey, GM_ADDR treeScale, GM_ADDR treeAncestors, GM_ADDR treePrefix, GM_ADDR treeRequests,
    GM_ADDR sparseIndices, GM_ADDR sparseValues, GM_ADDR candidateTopkIndexOut, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    auto user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(QLITreeTilingData, data, tiling);
    QLITreePreload<QLITreeType<int8_t, int8_t, float, uint16_t, int32_t, PAGE_ATTENTION,
        LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T)>> op;
    op.Init(query,key,weights,queryScale,keyScale,cuSeqlensQ,cuSeqlensK,sequsedQ,sequsedK,cmpResidualK,
        blockTable,outputIdxOffset,metadata,candidateTopkIndex,sparseIndices,sparseValues,candidateTopkIndexOut,
        user,&data,&pipe);
    if ASCEND_IS_AIC {
        op.matmulService.InitTree(treeKey,treeAncestors,treePrefix,treeRequests,data.treeDepth,data.treeNodes);
    } else {
        op.vectorService.InitTree(treeScale,treeAncestors,treePrefix,treeRequests,data.treeDepth,data.treeNodes);
    }
    op.Process();
}
