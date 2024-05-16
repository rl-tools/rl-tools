#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_CAPABILITY_CAPABILITY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_CAPABILITY_CAPABILITY_H
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

#include "../parameters/parameters.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn{
    enum class LayerCapability{
        Forward, // just forward
        Backward, // forward + backward wrt to the input
        Gradient // forward + backward wrt to the input + backward wrt to the weights
    };
    /*
     Tags that can also carry the parameter type.
     The LayerCapability determines some fields in the layers (e.g. the intermediate outputs are required for gradient calculations, while for the backward pass wrt. the input only the pre-activations might be needed)
     The PARAMETER_TYPE determines the type of the parameters (e.g. adam requries first and second order moments in addition to the gradient)
     This Capability system allows the switching of models for e.g. checkpointing: We are training a full model with gradients, and optimizers state then convert it to a forward only model (just the parameters) and save it as a checkpoint.
    */
    namespace layer_capability{
        struct Forward{
            static constexpr LayerCapability TAG = LayerCapability::Forward;
            using PARAMETER_TYPE = nn::parameters::Plain;
        };
        struct Backward{
            static constexpr LayerCapability TAG = LayerCapability::Backward;
            using PARAMETER_TYPE = nn::parameters::Plain;
        };
        template <typename T_PARAMETER_TYPE>
        struct Gradient{
            static constexpr LayerCapability TAG = LayerCapability::Gradient;
            using PARAMETER_TYPE = T_PARAMETER_TYPE;
            static_assert(!utils::typing::is_same_v<T_PARAMETER_TYPE, nn::parameters::Plain>);
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

