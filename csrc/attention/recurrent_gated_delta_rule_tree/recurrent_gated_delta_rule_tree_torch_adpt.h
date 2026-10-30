/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef RECURRENT_GATED_DELTA_RULE_TREE_TORCH_ADPT_H
#define RECURRENT_GATED_DELTA_RULE_TREE_TORCH_ADPT_H

namespace vllm_ascend {

inline bool tree_gdn_supports_state_page_stride()
{
    static const bool supported = []() {
        auto run = ::GetOpApiFuncAddr("aclnnRecurrentGatedDeltaRuleTree");
        auto workspace = ::GetOpApiFuncAddr("aclnnRecurrentGatedDeltaRuleTreeGetWorkspaceSize");
        Dl_info runInfo{}, workspaceInfo{};
        if (run == nullptr || workspace == nullptr ||
            ::dladdr(run, &runInfo) == 0 || ::dladdr(workspace, &workspaceInfo) == 0 ||
            runInfo.dli_fbase != workspaceInfo.dli_fbase || workspaceInfo.dli_fname == nullptr) {
            return false;
        }
        // Inspect the selected provider. A newer lower-priority vendor must
        // not certify an older operator found earlier in the search path.
        auto handle = ::dlopen(workspaceInfo.dli_fname, RTLD_LAZY | RTLD_LOCAL);
        if (handle == nullptr) {
            return false;
        }
        auto capability = ::dlsym(handle, "aclnnRecurrentGatedDeltaRuleTreeGetCapabilities");
        Dl_info capabilityInfo{};
        using CapabilityFn = uint64_t (*)(void);
        constexpr uint64_t STATE_PAGE_STRIDE_CAPABILITY = 1ULL;
        const bool hasStride = capability != nullptr &&
            ::dladdr(capability, &capabilityInfo) != 0 &&
            capabilityInfo.dli_fbase == workspaceInfo.dli_fbase &&
            (reinterpret_cast<CapabilityFn>(capability)() & STATE_PAGE_STRIDE_CAPABILITY) != 0;
        ::dlclose(handle);
        return hasStride;
    }();
    return supported;
}

at::Tensor npu_recurrent_gated_delta_rule_tree(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const c10::optional<at::Tensor>& beta,
    const c10::optional<double> scale,
    const c10::optional<at::Tensor>& actual_seq_lengths,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    const c10::optional<at::Tensor>& init_state_indices)
{
    TORCH_CHECK(scale.has_value(), "scale cannot be empty.");

    auto options = value.options().dtype(at::ScalarType::BFloat16);
    at::Tensor output = at::empty(value.sizes(), options);
    float scale_real = static_cast<float>(scale.value());
    // Hybrid KV pages may pad only the page stride. The tree kernel consumes
    // this stride directly, so verification touches only its live state pages.
    TORCH_CHECK(state.dim() == 4, "tree GDN state must have four dimensions");
    TORCH_CHECK(state.stride(3) == 1 &&
                state.stride(2) == state.size(3) &&
                state.stride(1) == state.size(2) * state.size(3) &&
                state.stride(0) >= state.size(1) * state.size(2) * state.size(3),
                "tree GDN state must be dense within each non-overlapping page");
    if (state.stride(0) != state.size(1) * state.size(2) * state.size(3)) {
        TORCH_CHECK(tree_gdn_supports_state_page_stride(),
                    "Loaded tree GDN operator lacks state-page stride support. Rebuild/reinstall "
                    "the tree operator and set its vendor before the old vendor in "
                    "ASCEND_CUSTOM_OPP_PATH before importing torch_npu/vllm_ascend. "
                    "Restart the Python process after changing the runtime.");
    }

    EXEC_NPU_CMD(aclnnRecurrentGatedDeltaRuleTree,
                 query,
                 key,
                 value,
                 beta,
                 state,
                 actual_seq_lengths,
                 ssm_state_indices,
                 g,
                 gk,
                 num_accepted_tokens,
                 init_state_indices,
                 scale_real,
                 output);
    return output;
}

} // namespace vllm_ascend
#endif
