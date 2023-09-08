#include "../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_NN_PARAMETERS_PARAMETERS_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_NN_PARAMETERS_PARAMETERS_H

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::nn::parameters{
    struct Plain{
        // todo: evaluate replacing the instance mechanism with a tag similar to the container type tags
        template <typename CONTAINER>
        struct instance{
            CONTAINER parameters;
        };
    };
    struct Gradient{
        template <typename CONTAINER>
        struct instance: Plain::instance<CONTAINER>{
            CONTAINER gradient;
        };
    };
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
