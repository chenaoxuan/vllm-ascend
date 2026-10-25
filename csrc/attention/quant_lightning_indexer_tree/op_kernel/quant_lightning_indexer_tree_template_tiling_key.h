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
 * \file quant_lightning_indexer_tree_template_tiling_key.h
 * \brief
 */

#ifndef QUANT_LIGHTNING_INDEXER_TREE_TEMPLATE_TILING_KEY_H
#define QUANT_LIGHTNING_INDEXER_TREE_TEMPLATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define QLITree_TPL_INT8 2
#define QLITree_TPL_INT32 3
#define QLITree_TPL_FLOAT8_E4M3FN 36
#define QLITree_TPL_HIFLOAT8 34
#define QLITree_TPL_FLOAT4_E2M1 40
#define QLITree_LAYOUT_BSND 0
#define QLITree_LAYOUT_TND 1
#define QLITree_LAYOUT_PA_BBND 2

#define ASCENDC_TPL_4_BW 4

ASCENDC_TPL_ARGS_DECL(QuantLightningIndexerTree,
    ASCENDC_TPL_DTYPE_DECL(DT_Q, QLITree_TPL_INT8), ASCENDC_TPL_DTYPE_DECL(DT_K, QLITree_TPL_INT8),
    ASCENDC_TPL_DTYPE_DECL(DT_OUT, QLITree_TPL_INT32), ASCENDC_TPL_BOOL_DECL(PAGE_ATTENTION, 1),
    ASCENDC_TPL_UINT_DECL(Q_LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, QLITree_LAYOUT_TND),
    ASCENDC_TPL_UINT_DECL(K_LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, QLITree_LAYOUT_PA_BBND));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_DTYPE_SEL(DT_Q, QLITree_TPL_INT8), ASCENDC_TPL_DTYPE_SEL(DT_K, QLITree_TPL_INT8),
    ASCENDC_TPL_DTYPE_SEL(DT_OUT, QLITree_TPL_INT32), ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 1),
    ASCENDC_TPL_UINT_SEL(Q_LAYOUT_T, ASCENDC_TPL_UI_LIST, QLITree_LAYOUT_TND),
    ASCENDC_TPL_UINT_SEL(K_LAYOUT_T, ASCENDC_TPL_UI_LIST, QLITree_LAYOUT_PA_BBND)));
#endif
