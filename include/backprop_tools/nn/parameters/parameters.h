#include "../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_NN_PARAMETERS_PARAMETERS_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_NN_PARAMETERS_PARAMETERS_H

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::nn::parameters{
    namespace categories{
        struct Weights{};
        struct Biases{};
    }

    struct Plain{
        // todo: evaluate replacing the instance mechanism with a tag similar to the container type tags
        template <typename T_CONTAINER, typename T_CATEGORY_TAG>
        struct spec {
            using CONTAINER = T_CONTAINER;
            using CATEGORY_TAG = T_CATEGORY_TAG;
        };
        template <typename T_SPEC>
        struct instance{
            using SPEC = T_SPEC;
            using CONTAINER = typename SPEC::CONTAINER;
            CONTAINER parameters;
        };
    };
    struct Gradient: Plain{
        template <typename T_SPEC>
        struct instance: Plain::instance<T_SPEC>{
            using SPEC = T_SPEC;
            using CONTAINER = typename SPEC::CONTAINER;
            CONTAINER gradient;
        };
    };
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
