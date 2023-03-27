#ifndef LAYER_IN_C_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_NETWORK_H
#define LAYER_IN_C_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_NETWORK_H

#include <layer_in_c/nn_models/mlp/network.h>

namespace layer_in_c::nn_models::mlp_unconditional_stddev {

    template<template <typename> typename BASE, typename SPEC>
    struct NEURAL_NETWORK_FACTORY: public BASE<SPEC>{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        nn::parameters::Adam::instance<Matrix<matrix::Specification<T, TI, 1, SPEC::OUTPUT_DIM>>> log_std;
    };

    template <typename SPEC>
    using NeuralNetwork = NEURAL_NETWORK_FACTORY<nn_models::mlp::NeuralNetwork, SPEC>;
    template <typename SPEC>
    using NeuralNetworkAdam = NEURAL_NETWORK_FACTORY<nn_models::mlp::NeuralNetworkAdam, SPEC>;


}

#endif
