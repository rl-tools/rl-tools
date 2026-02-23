#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_LAMB_LAMB_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_LAMB_LAMB_H

#include "../adam/adam.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::optimizers{
    namespace lamb{
        template<typename T_T>
        struct Parameters{
            using T = T_T;
            T upper_bound_trust_ratio;
            T lower_bound_trust_ratio;
        };
        template <typename TYPE_POLICY>
        struct DEFAULT_PARAMETERS{
            using T = typename TYPE_POLICY::DEFAULT;
            static constexpr T UPPER_BOUND_TRUST_RATIO = 10.0;
            static constexpr T LOWER_BOUND_TRUST_RATIO = 0.0;
        };
        template <typename T_TYPE_POLICY, typename T_TI, typename T_ADAM_DEFAULT_PARAMETERS = adam::DEFAULT_PARAMETERS_TENSORFLOW<T_TYPE_POLICY>, typename T_LAMB_DEFAULT_PARAMETERS = DEFAULT_PARAMETERS<T_TYPE_POLICY>, bool T_DYNAMIC_ALLOCATION = true>
        struct Specification: adam::Specification<T_TYPE_POLICY, T_TI, T_ADAM_DEFAULT_PARAMETERS, T_DYNAMIC_ALLOCATION>{
            using LAMB_DEFAULT_PARAMETERS = T_LAMB_DEFAULT_PARAMETERS;
        };
    }
    template<typename T_SPEC>
    struct Lamb: Adam<T_SPEC>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using LAMB_DEFAULT_PARAMETERS = typename SPEC::LAMB_DEFAULT_PARAMETERS;
        using LAMB_PARAMETERS = lamb::Parameters<T>;
        Tensor<tensor::Specification<LAMB_PARAMETERS, TI, tensor::Shape<TI, 1>, SPEC::DYNAMIC_ALLOCATION>> lamb_parameters;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
