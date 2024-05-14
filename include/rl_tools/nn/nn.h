


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
    /*
     Tags that can also carry the parameter type.
     The LayerCapability determines some fields in the layers (e.g. the intermediate outputs are required for gradient calculations, while for the backward pass wrt. the input only the pre-activations might be needed)
     The PARAMETER_TYPE determines the type of the parameters (e.g. adam requries first and second order moments in addition to the gradient)
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


#include "activation_functions.h"
#include "layers/layers.h"
#include "optimizers/adam/adam.h"

#endif
