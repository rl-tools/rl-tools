#ifndef LAYER_IN_C_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_NETWORK_H
#define LAYER_IN_C_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_NETWORK_H

#include <layer_in_c/nn_models/mlp/network.h>

namespace layer_in_c::nn_models::mlp_unconditional_stddev {
    template<typename SPEC>
    struct NeuralNetworkAdam: public mlp::NeuralNetworkAdam<SPEC>{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        nn::parameters::Adam::instance<Matrix<matrix::Specification<T, TI, 1, SPEC::OUTPUT_DIM>>> action_log_std;
    };


}

#endif
