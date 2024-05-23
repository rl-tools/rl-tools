#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_CAPABILITY_PERSIST_CODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_CAPABILITY_PERSIST_CODE_H
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    std::string to_string(nn::layer_capability::Forward){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::layer_capability::Forward";
    }
    template <auto BATCH_SIZE>
    std::string to_string(nn::layer_capability::Backward<BATCH_SIZE>){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::layer_capability::Backward<" + std::to_string(BATCH_SIZE) + ">";
    }
    template <typename T_PARAMETER_TYPE, auto BATCH_SIZE>
    std::string to_string(nn::layer_capability::Gradient<T_PARAMETER_TYPE, BATCH_SIZE>){
        return "RL_TOOLS""_NAMESPACE_WRAPPER ::rl_tools::nn::layer_capability::Gradient<"+ get_type_string(T_PARAMETER_TYPE{}) + std::string(", ") + std::to_string(BATCH_SIZE) + ">";
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

