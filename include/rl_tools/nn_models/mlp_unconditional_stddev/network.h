#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_NETWORK_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_NETWORK_H

#include "../../nn_models/mlp/network.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::mlp_unconditional_stddev {

    template <typename T_SPEC, template <typename> typename T_BASE = nn_models::mlp::NeuralNetworkForward>
    struct NeuralNetworkForward: T_BASE<T_SPEC>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using LOG_STD_CONTAINER_SPEC = matrix::Specification<T, TI, 1, SPEC::OUTPUT_DIM>;
        using LOG_STD_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<LOG_STD_CONTAINER_SPEC>;
        using LOG_STD_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template spec<LOG_STD_CONTAINER_TYPE, nn::parameters::groups::Output, nn::parameters::categories::Weights>;
        typename SPEC::PARAMETER_TYPE::template instance<LOG_STD_PARAMETER_SPEC> log_std;
        template <typename TT_SPEC>
        using BASE = T_BASE<TT_SPEC>;
    };
    template <typename SPEC, template <typename> typename BASE = nn_models::mlp::NeuralNetworkBackward>
    struct NeuralNetworkBackward: NeuralNetworkForward<SPEC, BASE>{};
    template <typename SPEC, template <typename> typename BASE = nn_models::mlp::NeuralNetworkGradient>
    struct NeuralNetworkGradient: NeuralNetworkBackward<SPEC, BASE>{};

    template<typename CAPABILITY, typename SPEC>
    using _NeuralNetwork =
    typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward,
            NeuralNetworkForward<nn_models::mlp::CapabilitySpecification<CAPABILITY, SPEC>>,
    typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward,
            NeuralNetworkBackward<nn_models::mlp::CapabilitySpecification<CAPABILITY, SPEC>>,
    typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient,
            NeuralNetworkGradient<nn_models::mlp::CapabilitySpecification<CAPABILITY, SPEC>>, void>>>;

    template<typename T_CAPABILITY, typename T_SPEC>
    struct NeuralNetwork: _NeuralNetwork<T_CAPABILITY, T_SPEC>{
        template <typename TT_CAPABILITY>
        using CHANGE_CAPABILITY = NeuralNetwork<TT_CAPABILITY, T_SPEC>;
    };

    template <typename T_SPEC>
    struct BindSpecification{
        template <typename CAPABILITY>
        using NeuralNetwork = nn_models::mlp_unconditional_stddev::NeuralNetwork<CAPABILITY, T_SPEC>;
    };


}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
