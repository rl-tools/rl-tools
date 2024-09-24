#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_TD3_SAMPLING_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_TD3_SAMPLING_PERSIST_H
#include "../../../containers/matrix/persist.h"
#include "layer.h"
#include "../../../utils/persist.h"
#include <iostream>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, nn::layers::td3_sampling::LayerForward<SPEC>& layer, HighFive::Group group) { }
    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, nn::layers::td3_sampling::LayerForward<SPEC>& layer, HighFive::Group group) {
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
