#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_ADAM_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_ADAM_PERSIST_H
#include "adam.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::optimizers::Adam<SPEC>& optimizer, GROUP& group) {
        save_binary(device, optimizer.parameters._data, 1, group, "parameters");
        save(device, optimizer.age, group, "age");
        save(device, optimizer.first_order_moment_bias_correction, group, "first_order_moment_bias_correction");
        save(device, optimizer.second_order_moment_bias_correction, group, "second_order_moment_bias_correction");
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::optimizers::Adam<SPEC>& optimizer, GROUP& group) {
        bool success = load_binary(device, optimizer.parameters._data, 1, group, "parameters");
        success &= load(device, optimizer.age, group, "age");
        success &= load(device, optimizer.first_order_moment_bias_correction, group, "first_order_moment_bias_correction");
        success &= load(device, optimizer.second_order_moment_bias_correction, group, "second_order_moment_bias_correction");
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
