#include "network.h"
#include <layer_in_c/nn_models/mlp/operations_generic.h>


namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& m){
        malloc(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)m);
        malloc(device, m.log_std);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& m){
        free(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)m);
        free(device, m.log_std);
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void init_weights(DEVICE device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& m, RNG& rng){
        init_weights(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)m, rng);
        set_all(device, m.log_std.parameters, 0);
    }
    template<typename DEVICE, typename SPEC, typename ADAM_PARAMETERS>
    void update(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& network, nn::optimizers::Adam<ADAM_PARAMETERS>& optimizer) {
        using T = typename SPEC::T;
        optimizer.first_order_moment_bias_correction  = 1/(1 - math::pow(typename DEVICE::SPEC::MATH(), ADAM_PARAMETERS::BETA_1, (T)network.age));
        optimizer.second_order_moment_bias_correction = 1/(1 - math::pow(typename DEVICE::SPEC::MATH(), ADAM_PARAMETERS::BETA_2, (T)network.age));
        update(device, network.log_std, optimizer);
        update(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)network, optimizer);
    }
    template<typename DEVICE, typename SPEC>
    void zero_gradient(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& network) {
        zero_gradient(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)network);
        zero_gradient(device, network.log_std);
    }

    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void reset_optimizer_state(DEVICE& device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& network, OPTIMIZER& optimizer) {
        reset_optimizer_state(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)network, optimizer);
        reset_optimizer_state(device, network.log_std, optimizer);
    }

    template<typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<TARGET_SPEC>& target, const nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SOURCE_SPEC>& source){
        static_assert(layer_in_c::nn_models::mlp::check_spec_memory<typename TARGET_SPEC::STRUCTURE_SPEC, typename SOURCE_SPEC::STRUCTURE_SPEC>, "The target and source network must have the same structure");
        copy(target_device, source_device, (nn_models::mlp::NeuralNetworkAdam<TARGET_SPEC>&)target, (nn_models::mlp::NeuralNetworkAdam<SOURCE_SPEC>&)source);
        copy(target_device, source_device, target.log_std, source.log_std);
    }
}