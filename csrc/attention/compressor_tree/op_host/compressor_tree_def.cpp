// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
#include "register/op_def_registry.h"
namespace ops {
class CompressorTree : public OpDef {
public:
    explicit CompressorTree(const char *name) : OpDef(name) {
        for (auto input : {"state", "nodes", "norm", "cos", "sin"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).IgnoreContiguous();
        }
        for (auto input : {"table", "prefix"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        }
        for (auto input : {"ancestors", "depths", "requests"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        }
        this->Output("output").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("ratio").Int();
        this->Attr("rope_dim").Int();
        this->Attr("epsilon").Float();
        this->Attr("page_stride").Int();
        this->Attr("row_stride").Int();
        this->Attr("cos_stride").Int();
        this->Attr("sin_stride").Int();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(CompressorTree);
}
