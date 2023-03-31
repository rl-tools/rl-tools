#ifndef LAYER_IN_C_RL_COMPONENTS_RUNNING_NORMALIZER_PERSIST_H
#define LAYER_IN_C_RL_COMPONENTS_RUNNING_NORMALIZER_PERSIST_H

#include "running_normalizer.h"
namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, rl::components::RunningNormalizer<SPEC> normalizer, HighFive::Group group) {
        lic::save(device, normalizer.mean, group, "mean");
        lic::save(device, normalizer.std, group, "std");
    }
}
#endif
