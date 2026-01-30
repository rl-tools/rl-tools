#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_RUNNING_NORMALIZER_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_RUNNING_NORMALIZER_PERSIST_H
#include "running_normalizer.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer, GROUP& group) {
        save(device, normalizer.mean, group, "mean");
        save(device, normalizer.std, group, "std");
        Tensor<tensor::Specification<typename SPEC::TI, typename SPEC::TI, tensor::Shape<typename SPEC::TI, 1>>> age_tensor;
        malloc(device, age_tensor);
        set(device, age_tensor, normalizer.age, 0);
        save(device, age_tensor, group, "age");
        free(device, age_tensor);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer, GROUP& group) {
        bool success = load(device, normalizer.mean, group, "mean");
        success &= load(device, normalizer.std, group, "std");
        Tensor<tensor::Specification<typename SPEC::TI, typename SPEC::TI, tensor::Shape<typename SPEC::TI, 1>>> age_tensor;
        malloc(device, age_tensor);
        success &= load(device, age_tensor, group, "age");
        normalizer.age = get(device, age_tensor, 0);
        free(device, age_tensor);
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
