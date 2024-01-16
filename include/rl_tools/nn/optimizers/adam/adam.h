#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_ADAM_ADAM_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_ADAM_ADAM_H

#include "../../../nn/parameters/parameters.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::optimizers{
    namespace adam{
        template<typename T_T>
        struct Parameters{
            using T = T_T;
            T alpha;
            T beta_1;
            T beta_2;
            T epsilon;
            T weight_decay;
            T weight_decay_input;
            T weight_decay_output;
            T bias_lr_factor;
        };
        template <typename T>
        constexpr Parameters<T> default_parameters_tensorflow = {
            0.001,
            0.9,
            0.999,
            1e-7,
            0,
            0,
            0,
            1,
        };
        template<typename T>
        constexpr Parameters<T> default_parameters_torch = {
            0.001,
            0.9,
            0.999,
            1e-8,
            0,
            0,
            0,
            1,
        };
        template <typename T_T, typename T_TI, bool T_ENABLE_WEIGHT_DECAY = false, bool T_ENABLE_BIAS_LR_FACTOR = false>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            static constexpr bool ENABLE_WEIGHT_DECAY = T_ENABLE_WEIGHT_DECAY;
            static constexpr bool ENABLE_BIAS_LR_FACTOR = T_ENABLE_BIAS_LR_FACTOR;
        };
    }
    template<typename T_SPEC>
    struct Adam{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        adam::Parameters<T> parameters = adam::default_parameters_tensorflow<T>;
        T first_order_moment_bias_correction;
        T second_order_moment_bias_correction;
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