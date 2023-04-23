#ifndef BACKPROP_TOOLS_RL_COMPONENTS_RUNNING_NORMALIZER_PERSIST_H
#define BACKPROP_TOOLS_RL_COMPONENTS_RUNNING_NORMALIZER_PERSIST_H

#include "running_normalizer.h"
namespace backprop_tools{
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, rl::components::RunningNormalizer<SPEC> normalizer, HighFive::Group group) {
        save(device, normalizer.mean, group, "mean");
        save(device, normalizer.std, group, "std");
    }
}
#endif
