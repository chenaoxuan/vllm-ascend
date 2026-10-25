// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project
#include "kernel_operator.h"
using namespace AscendC;
namespace {
struct TreeTiling {
    uint32_t n, dim, ratio, rope, block, width, maxDepth, cores;
    uint64_t pageStride, rowStride, cosStride, sinStride;
    float eps;
};
template<HardEvent E> __aicore__ inline void Sync() {
    auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
    SetFlag<E>(id); WaitFlag<E>(id);
}
class TreeReduce {
public:
    __aicore__ inline void Init(GM_ADDR state, GM_ADDR nodes, GM_ADDR norm, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR table, GM_ADDR prefix, GM_ADDR ancestors, GM_ADDR depths, GM_ADDR requests,
        GM_ADDR output, const TreeTiling &t) {
        t_ = t;
        state_.SetGlobalBuffer((__gm__ float*)state); nodes_.SetGlobalBuffer((__gm__ float*)nodes);
        norm_.SetGlobalBuffer((__gm__ float*)norm); cos_.SetGlobalBuffer((__gm__ float*)cos);
        sin_.SetGlobalBuffer((__gm__ float*)sin); table_.SetGlobalBuffer((__gm__ int32_t*)table);
        prefix_.SetGlobalBuffer((__gm__ int32_t*)prefix); ancestors_.SetGlobalBuffer((__gm__ int64_t*)ancestors);
        depths_.SetGlobalBuffer((__gm__ int64_t*)depths); requests_.SetGlobalBuffer((__gm__ int64_t*)requests);
        output_.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        pipe_.InitBuffer(buf_, 12 * t.dim * sizeof(float));
        pipe_.InitBuffer(outBuf_, t.dim * sizeof(bfloat16_t));
        pipe_.InitBuffer(indexBuf_, t.rope * sizeof(uint32_t));
        auto all = buf_.Get<float>();
        score_ = all; kv_ = all[t.dim]; max_ = all[2*t.dim]; denominator_ = all[3*t.dim];
        value_ = all[4*t.dim]; temp_ = all[5*t.dim]; weight_ = all[6*t.dim];
        cosine_ = all[7*t.dim]; sine_ = all[8*t.dim]; paired_ = all[9*t.dim];
        scratch_ = all[10*t.dim]; reduced_ = all[11*t.dim];
        indices_ = indexBuf_.Get<uint32_t>();
        for (uint32_t i=0; i<t.rope; ++i) indices_.SetValue(i, (i ^ 1U) * sizeof(float));
        Sync<HardEvent::S_V>();
        DataCopy(weight_, norm_, t.dim);
        Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline void Run() {
        uint32_t d = t_.dim, c = t_.ratio == 4 ? 2 : 1;
        auto out = outBuf_.Get<bfloat16_t>();
        for (uint32_t node=GetBlockIdx(); node<t_.n; node+=t_.cores) {
            int64_t start = prefix_.GetValue(node), depth = depths_.GetValue(node), pos = start + depth;
            Duplicate(value_, 0.0f, d);
            if ((pos+1) % t_.ratio == 0) {
                Duplicate(max_, -3.402823466e+38f, d);
                Duplicate(denominator_, 0.0f, d);
                // Same C4 interleaving as the native compressor: previous/current
                // windows use different projected halves. Invalid rows are never read.
                for (uint32_t w=0; w<t_.ratio*c; ++w) {
                    int64_t p = pos+1-t_.ratio*c+w/c+(w%c)*t_.ratio;
                    if (!Address(node, start, depth, p, w%c, c)) continue;
                    Sync<HardEvent::V_MTE2>();
                    Load(score_, true, c);
                    Sync<HardEvent::MTE2_V>();
                    Max(max_, max_, score_, d);
                    PipeBarrier<PIPE_V>();
                }
                for (uint32_t w=0; w<t_.ratio*c; ++w) {
                    int64_t p = pos+1-t_.ratio*c+w/c+(w%c)*t_.ratio;
                    if (!Address(node, start, depth, p, w%c, c)) continue;
                    Sync<HardEvent::V_MTE2>();
                    Load(score_, true, c); Load(kv_, false, c);
                    Sync<HardEvent::MTE2_V>();
                    Sub(score_, score_, max_, d); PipeBarrier<PIPE_V>();
                    Exp(score_, score_, d); PipeBarrier<PIPE_V>();
                    Add(denominator_, denominator_, score_, d);
                    Mul(temp_, score_, kv_, d); PipeBarrier<PIPE_V>();
                    Add(value_, value_, temp_, d); PipeBarrier<PIPE_V>();
                }
                Div(value_, value_, denominator_, d); PipeBarrier<PIPE_V>();
                Mul(temp_, value_, value_, d); PipeBarrier<PIPE_V>();
                ReduceSum(reduced_, temp_, scratch_, d);
                Sync<HardEvent::V_S>();
                float mean = reduced_.GetValue(0) / d + t_.eps;
                Duplicate(temp_, mean, d); PipeBarrier<PIPE_V>();
                Sqrt(temp_, temp_, d); PipeBarrier<PIPE_V>();
                Div(value_, value_, temp_, d); PipeBarrier<PIPE_V>();
                Mul(value_, value_, weight_, d); PipeBarrier<PIPE_V>();
                uint32_t rd = t_.rope, begin = d-rd;
                int64_t ropePos = pos+1-t_.ratio;
                Sync<HardEvent::V_MTE2>();
                DataCopy(cosine_, cos_[ropePos*t_.cosStride], rd);
                DataCopy(sine_, sin_[ropePos*t_.sinStride], rd);
                Sync<HardEvent::MTE2_V>();
                Gather(paired_, value_[begin], indices_, 0U, rd); PipeBarrier<PIPE_V>();
                // Sign changes are a vector multiply; constants live only in UB.
                Sync<HardEvent::V_S>();
                for (uint32_t i=0; i<rd; ++i) scratch_.SetValue(i, (i%2 == 0) ? -1.0f : 1.0f);
                Sync<HardEvent::S_V>();
                Mul(paired_, paired_, scratch_, rd); PipeBarrier<PIPE_V>();
                Mul(paired_, paired_, sine_, rd);
                Mul(value_[begin], value_[begin], cosine_, rd); PipeBarrier<PIPE_V>();
                Add(value_[begin], value_[begin], paired_, rd); PipeBarrier<PIPE_V>();
            }
            Cast(out, value_, RoundMode::CAST_RINT, d);
            Sync<HardEvent::V_MTE3>();
            DataCopy(output_[node*d], out, d);
            Sync<HardEvent::MTE3_V>();
        }
    }
private:
    __aicore__ inline bool Address(uint32_t node, int64_t start, int64_t depth, int64_t pos, uint32_t half, uint32_t c) {
        if (pos < 0 || pos > start+depth) return false;
        isNode_ = pos >= start;
        if (isNode_) {
            int64_t step = pos-start;
            if (step >= t_.maxDepth) return false;
            int64_t ancestor = ancestors_.GetValue(node*t_.maxDepth+step);
            if (ancestor < 0 || ancestor >= t_.n) return false;
            offset_ = (ancestor+1)*2*c*t_.dim + half*t_.dim;
        } else {
            int64_t col = pos/t_.block;
            if (col >= t_.width) return false;
            int64_t page = table_.GetValue(requests_.GetValue(node)*t_.width+col);
            if (page < 0) return false;
            offset_ = page*t_.pageStride+(pos%t_.block)*t_.rowStride+half*t_.dim;
        }
        return true;
    }
    __aicore__ inline void Load(LocalTensor<float> dst, bool score, uint32_t c) {
        uint64_t offset = offset_ + (score ? c*t_.dim : 0);
        if (isNode_) DataCopy(dst, nodes_[offset], t_.dim);
        else DataCopy(dst, state_[offset], t_.dim);
    }
    TreeTiling t_; TPipe pipe_;
    TBuf<TPosition::VECCALC> buf_, outBuf_, indexBuf_;
    GlobalTensor<float> state_, nodes_, norm_, cos_, sin_;
    GlobalTensor<int32_t> table_, prefix_;
    GlobalTensor<int64_t> ancestors_, depths_, requests_;
    GlobalTensor<bfloat16_t> output_;
    LocalTensor<float> score_, kv_, max_, denominator_, value_, temp_, weight_, cosine_, sine_, paired_, scratch_, reduced_;
    LocalTensor<uint32_t> indices_;
    bool isNode_; uint64_t offset_;
};
}
extern "C" __global__ __aicore__ void compressor_tree(GM_ADDR state, GM_ADDR nodes, GM_ADDR norm,
    GM_ADDR cos, GM_ADDR sin, GM_ADDR table, GM_ADDR prefix, GM_ADDR ancestors, GM_ADDR depths,
    GM_ADDR requests, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TreeTiling);
    GET_TILING_DATA_WITH_STRUCT(TreeTiling, t, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(1)) {
        TreeReduce op;
        op.Init(state,nodes,norm,cos,sin,table,prefix,ancestors,depths,requests,output,t);
        op.Run();
    }
}
