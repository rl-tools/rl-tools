#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_ADAM_ADAM_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_ADAM_ADAM_H

#include "../../../nn/parameters/parameters.h"
// Note the Adam operations are divided into "./instance/operations_xxx." operations which are operations on the parameters and "./operations_xxx.h" which are operations on the optimizer (and possibly a `nn_model`).
// So the instance operations should be imported before any imports that use parameters (particularly `nn/layers` and `nn_models`). Then the chain is `optimizer operation` (e.g. gradient descent weight update) calls `nn_model` update calls e.g. `nn::layers::dense` update calls `parameters::Adam::instance` update. Hence, the instance operations need to be included first

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
            T epsilon_sqrt;
            T weight_decay;
            T weight_decay_input;
            T weight_decay_output;
            T bias_lr_factor;
        };
        template <typename T>
        struct DEFAULT_PARAMETERS_TENSORFLOW{
            static constexpr T ALPHA = 0.001;
            static constexpr T BETA_1 = 0.9;
            static constexpr T BETA_2 = 0.999;
            static constexpr T EPSILON = 1e-7;
            static constexpr T EPSILON_SQRT = 1e-7;
            static constexpr bool ENABLE_WEIGHT_DECAY = false;
            static constexpr T WEIGHT_DECAY = 0;
            static constexpr T WEIGHT_DECAY_INPUT = 0;
            static constexpr T WEIGHT_DECAY_OUTPUT = 0;
            static constexpr bool ENABLE_BIAS_LR_FACTOR = false;
            static constexpr T BIAS_LR_FACTOR = 1;
            static constexpr bool ENABLE_GRADIENT_CLIPPING = false;
            static constexpr T GRADIENT_CLIP_VALUE = 1;
        };
        template <typename T>
        struct DEFAULT_PARAMETERS_PYTORCH: DEFAULT_PARAMETERS_TENSORFLOW<T>{
            static constexpr T EPSILON = 1e-8;
            static constexpr T EPSILON_SQRT = 1e-8;
        };
        template <typename T_T, typename T_TI, typename T_DEFAULT_PARAMETERS=DEFAULT_PARAMETERS_TENSORFLOW<T_T>, bool T_DYNAMIC_ALLOCATION = true>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using DEFAULT_PARAMETERS = T_DEFAULT_PARAMETERS;
            static constexpr bool ENABLE_WEIGHT_DECAY = DEFAULT_PARAMETERS::ENABLE_WEIGHT_DECAY;
            static constexpr bool ENABLE_BIAS_LR_FACTOR = DEFAULT_PARAMETERS::ENABLE_BIAS_LR_FACTOR;
            static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
        };
    }
    template<typename T_SPEC>
    struct Adam{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using DEFAULT_PARAMETERS = typename SPEC::DEFAULT_PARAMETERS;
        adam::Parameters<T> parameters = {
            DEFAULT_PARAMETERS::ALPHA,
            DEFAULT_PARAMETERS::BETA_1,
            DEFAULT_PARAMETERS::BETA_2,
            DEFAULT_PARAMETERS::EPSILON,
            DEFAULT_PARAMETERS::EPSILON_SQRT,
            DEFAULT_PARAMETERS::WEIGHT_DECAY,
            DEFAULT_PARAMETERS::WEIGHT_DECAY_INPUT,
            DEFAULT_PARAMETERS::WEIGHT_DECAY_OUTPUT,
            DEFAULT_PARAMETERS::BIAS_LR_FACTOR
        };
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1>, SPEC::DYNAMIC_ALLOCATION>> first_order_moment_bias_correction;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1>, SPEC::DYNAMIC_ALLOCATION>> second_order_moment_bias_correction;
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>, SPEC::DYNAMIC_ALLOCATION>> age;
    };


}
RL_TOOLS_NAMESPACE_WRAPPER_END
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::parameters {
    struct Adam{
        template <typename T_CONTAINER, typename T_GROUP_TAG, typename T_CATEGORY_TAG>
        struct spec {
            using CONTAINER = T_CONTAINER;
            using GROUP_TAG = T_GROUP_TAG;
            using CATEGORY_TAG = T_CATEGORY_TAG;
        };
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