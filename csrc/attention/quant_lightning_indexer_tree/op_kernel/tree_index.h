// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
namespace QLITreeKernel {
struct TreeIndex {
    AscendC::GlobalTensor<int64_t> ancestors, requests;
    AscendC::GlobalTensor<int32_t> prefixes;
    uint32_t width, nodes;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR p, GM_ADDR r, uint32_t w, uint32_t n) {
        ancestors.SetGlobalBuffer((__gm__ int64_t *)a);
        prefixes.SetGlobalBuffer((__gm__ int32_t *)p);
        requests.SetGlobalBuffer((__gm__ int64_t *)r);
        width=w; nodes=n;
    }
    __aicore__ inline int64_t Node(uint32_t row, int64_t k) const {
        int64_t step=(k+1)*4-1-prefixes.GetValue(row);
        if (step<0 || step>=width) return -1;
        int64_t node=ancestors.GetValue(row*width+step);
        return node>=0 && node<nodes ? node : -1;
    }
};
}
