#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_OPERATIONS_GENERIC_H

#include "network.h"
#include "../../nn_models/mlp/operations_generic.h"



RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, template <typename> typename BASE>
    void malloc(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC, BASE>& m){
        malloc(device, (nn_models::mlp::NeuralNetworkForward<SPEC>&)m);
        malloc(device, m.log_std);
    }
    template <typename DEVICE, typename SPEC, template <typename> typename BASE>
    void free(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC, BASE>& m){
        free(device, (nn_models::mlp::NeuralNetworkForward<SPEC>&)m);
        free(device, m.log_std);
    }
    template <typename DEVICE, typename SPEC, template <typename> typename BASE, typename RNG>
    void init_weights(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SPEC, BASE>& m, RNG& rng){
        init_weights(device, (nn_models::mlp::NeuralNetworkForward<SPEC>&)m, rng);
        set_all(device, m.log_std.parameters, 0);
    }
    template<typename DEVICE, typename SPEC, template <typename> typename BASE, typename ADAM_PARAMETERS>
    void update(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkGradient<SPEC, BASE>& network, nn::optimizers::Adam<ADAM_PARAMETERS>& optimizer) {
        using T = typename SPEC::T;
        update(device, network.log_std, optimizer);
        update(device, (nn_models::mlp::NeuralNetworkGradient<SPEC>&)network, optimizer);
    }
    template<typename DEVICE, typename SPEC, template <typename> typename BASE>
    void zero_gradient(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkGradient<SPEC, BASE>& network) {
        zero_gradient(device, (nn_models::mlp::NeuralNetworkGradient<SPEC>&)network);
        zero_gradient(device, network.log_std);
    }

    template<typename DEVICE, typename SPEC, template <typename> typename BASE, typename OPTIMIZER>
    void _reset_optimizer_state(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkGradient<SPEC, BASE>& network, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, (nn_models::mlp::NeuralNetworkGradient<SPEC>&)network, optimizer);
        _reset_optimizer_state(device, network.log_std, optimizer);
    }

    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC, template <typename> typename SOURCE_BASE, template <typename> typename TARGET_BASE>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn_models::mlp_unconditional_stddev::NeuralNetworkForward<SOURCE_SPEC, SOURCE_BASE>& source, nn_models::mlp_unconditional_stddev::NeuralNetworkForward<TARGET_SPEC, TARGET_BASE>& target){
        static_assert(rl_tools::nn_models::mlp::check_spec_memory<SOURCE_SPEC, TARGET_SPEC>, "The source and target network must have the same structure");
        copy(source_device, target_device, (nn_models::mlp::NeuralNetworkForward<SOURCE_SPEC>&)source, (nn_models::mlp::NeuralNetworkForward<TARGET_SPEC>&)target);
        copy(source_device, target_device, source.log_std, target.log_std);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif