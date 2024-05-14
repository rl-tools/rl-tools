


#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_NN_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_NN_H
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn{
    enum class LayerCapability{
        Forward, // just forward
        Backward, // forward + backward wrt to the input
        Gradient // forward + backward wrt to the input + backward wrt to the weights
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#include "activation_functions.h"
#include "layers/layers.h"
#include "optimizers/adam/adam.h"

#endif
