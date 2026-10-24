// SPDX-License-Identifier: Apache-2.0
#include "compressor_tree_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
namespace optiling {
static ge::graphStatus Tile(gert::TilingContext *ctx) {
    auto attrs = ctx->GetAttrs();
    const auto &anc = ctx->GetInputShape(7)->GetStorageShape();
    const auto &state = ctx->GetInputShape(0)->GetStorageShape();
    uint32_t n = anc.GetDim(0), dim = ctx->GetInputShape(2)->GetStorageShape().GetDim(0);
    uint32_t ratio = *attrs->GetInt(0), rope = *attrs->GetInt(1);
    if (!n || (dim != 128 && dim != 512) || (ratio != 4 && ratio != 128) || rope != 64) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    uint32_t cores = std::min(n, platform.GetCoreNumAiv());
    CompressorTreeTilingData t;
    t.set_n(n); t.set_dim(dim); t.set_ratio(ratio); t.set_rope(rope);
    t.set_block(state.GetDim(1));
    t.set_width(ctx->GetInputShape(5)->GetStorageShape().GetDim(1));
    t.set_maxDepth(anc.GetDim(1)); t.set_cores(cores);
    t.set_pageStride(*attrs->GetInt(3)); t.set_rowStride(*attrs->GetInt(4));
    t.set_cosStride(*attrs->GetInt(5)); t.set_sinStride(*attrs->GetInt(6));
    t.set_eps(*attrs->GetFloat(2));
    ctx->SetBlockDim(cores); ctx->SetTilingKey(1);
    *ctx->GetWorkspaceSizes(1) = 0;
    auto raw = ctx->GetRawTilingData();
    t.SaveToBuffer(raw->GetData(), raw->GetCapacity()); raw->SetDataSize(t.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Parse(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(CompressorTree).Tiling(Tile).TilingParse<CompressorTreeCompileInfo>(Parse);
}
