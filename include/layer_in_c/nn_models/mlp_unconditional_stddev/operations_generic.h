#include "network.h"
#include <layer_in_c/nn_models/mlp/operations_generic.h>


namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& m){
        malloc(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)m);
        malloc(device, m.action_log_std);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& m){
        free(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)m);
        free(device, m.action_log_std);
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void init_weights(DEVICE device, nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>& m, RNG& rng){
        init_weights(device, (nn_models::mlp::NeuralNetworkAdam<SPEC>&)m, rng);
        set_all(device, m.action_log_std, 0);
    }
}