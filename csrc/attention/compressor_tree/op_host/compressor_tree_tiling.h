// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(CompressorTreeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, dim);
    TILING_DATA_FIELD_DEF(uint32_t, ratio);
    TILING_DATA_FIELD_DEF(uint32_t, rope);
    TILING_DATA_FIELD_DEF(uint32_t, block);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, maxDepth);
    TILING_DATA_FIELD_DEF(uint32_t, cores);
    TILING_DATA_FIELD_DEF(uint64_t, pageStride);
    TILING_DATA_FIELD_DEF(uint64_t, rowStride);
    TILING_DATA_FIELD_DEF(uint64_t, cosStride);
    TILING_DATA_FIELD_DEF(uint64_t, sinStride);
    TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CompressorTree, CompressorTreeTilingData)
struct CompressorTreeCompileInfo {};
}
