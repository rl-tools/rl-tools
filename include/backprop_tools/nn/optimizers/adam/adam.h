#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_ADAM_ADAM_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_ADAM_ADAM_H

#include "../../../nn/parameters/parameters.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::optimizers{
    namespace adam{
        template<typename T_T, typename T_TI>
        struct DefaultParametersTF {
            using T = T_T;
            using TI = T_TI;
            static constexpr T ALPHA = 0.001;
            static constexpr T BETA_1 = 0.9;
            static constexpr T BETA_2 = 0.999;
            static constexpr T EPSILON = 1e-7;
            static constexpr T WEIGHT_DECAY = 0;
            static constexpr T WEIGHT_DECAY_INPUT = 0;
            static constexpr T WEIGHT_DECAY_OUTPUT = 0;
            static constexpr T BIAS_LR_FACTOR = 1;

        };
        template<typename T_T, typename T_TI>
        struct DefaultParametersTorch {
            using T = T_T;
            using TI = T_TI;
            static constexpr T ALPHA = 0.001;
            static constexpr T BETA_1 = 0.9;
            static constexpr T BETA_2 = 0.999;
            static constexpr T EPSILON = 1e-8;
            static constexpr T WEIGHT_DECAY = 0;
            static constexpr T WEIGHT_DECAY_INPUT = 0;
            static constexpr T WEIGHT_DECAY_OUTPUT = 0;
            static constexpr T BIAS_LR_FACTOR = 1;
        };
        template<typename T_T, typename T_TI>
        struct DefaultParameters{
            using T = T_T;
            using TI = T_TI;
            static constexpr T ALPHA = 0.001;
            static constexpr T BETA_1 = 0.9;
            static constexpr T BETA_2 = 0.999;
            static constexpr T EPSILON = 1e-8;
            static constexpr T WEIGHT_DECAY = 0;
            static constexpr T WEIGHT_DECAY_INPUT = 0;
            static constexpr T WEIGHT_DECAY_OUTPUT = 0;
            static constexpr T BIAS_LR_FACTOR = 1;
        };
    }
    template<typename T_PARAMETERS>
    struct Adam{
        using PARAMETERS = T_PARAMETERS;
        using T = typename PARAMETERS::T;
        using TI = typename PARAMETERS::TI;
        typename PARAMETERS::T first_order_moment_bias_correction;
        typename PARAMETERS::T second_order_moment_bias_correction;
        T alpha = PARAMETERS::ALPHA;
        TI age = 1;
    };


}
RL_TOOLS_NAMESPACE_WRAPPER_END
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::parameters {
    struct Adam: Gradient{
        template <typename T_SPEC>
        struct instance: Gradient::instance<T_SPEC>{
            using SPEC = T_SPEC;
            using CONTAINER = typename SPEC::CONTAINER;
            CONTAINER gradient_first_order_moment;
            CONTAINER gradient_second_order_moment;
        };
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif