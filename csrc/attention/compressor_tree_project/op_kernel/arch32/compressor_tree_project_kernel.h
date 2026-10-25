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
 * \file compressor_tree_project_kernel.h
 * \brief
 */

#ifndef COMPRESSOR_TREE_PROJECT_KERNEL
#define COMPRESSOR_TREE_PROJECT_KERNEL

#include "compressor_tree_project_comm.h"
#include "compressor_tree_project_template_tiling_key.h"
#include "compressor_tree_project_kernel_perf.h"
#include "compressor_tree_project_tiling_data.h"
#include "compressor_tree_project_tools.h"

using namespace AscendC;

namespace CompressorTreeProject {

template <typename COMP>
class CompressorTreeProjectKernel {
public:
    __aicore__ inline CompressorTreeProjectKernel(TPipe* pipe, const optiling::CompressorTreeProjectTilingData* __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData) {}

    __aicore__ inline void Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *stateCache,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *stateBlockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *workspace);
    __aicore__ inline void Process();

    // ==============================TilingData&TPipe==============================
    TPipe* pipe_;
    const optiling::CompressorTreeProjectTilingData* __restrict tilingData_;
};

template <typename COMP>
__aicore__ inline void CompressorTreeProjectKernel<COMP>::Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *stateCache,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *stateBlockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut,
        __gm__ uint8_t *workspace)
{

}


template <typename COMP>
__aicore__ inline void CompressorTreeProjectKernel<COMP>::Process()
{

}

} // namespace CompressorTreeProject

#endif // COMPRESSOR_TREE_PROJECT_KERNEL
