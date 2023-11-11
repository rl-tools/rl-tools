#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_PARAMETERS_PARAMETERS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_PARAMETERS_PARAMETERS_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::parameters{
    namespace groups{
        struct Normal{};
        struct Input{};
        struct Output{};
    }

    namespace categories{
        struct Weights{};
        struct Biases{};
    }

    struct Plain{
        // todo: evaluate replacing the instance mechanism with a tag similar to the container type tags
        template <typename T_CONTAINER, typename T_GROUP_TAG, typename T_CATEGORY_TAG>
        struct spec {
            using CONTAINER = T_CONTAINER;
            using GROUP_TAG = T_GROUP_TAG;
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
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
