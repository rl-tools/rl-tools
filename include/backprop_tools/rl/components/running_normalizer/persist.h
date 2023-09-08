#include "../../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_RL_COMPONENTS_RUNNING_NORMALIZER_PERSIST_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_RL_COMPONENTS_RUNNING_NORMALIZER_PERSIST_H

#include "running_normalizer.h"
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools{
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, rl::components::RunningNormalizer<SPEC> normalizer, HighFive::Group group) {
        save(device, normalizer.mean, group, "mean");
        save(device, normalizer.std, group, "std");
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
