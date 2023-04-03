#ifndef LAYER_IN_C_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_NETWORK_H
#define LAYER_IN_C_NN_MODELS_MLP_UNCONDITIONAL_STDDEV_NETWORK_H

#include <layer_in_c/nn_models/mlp/network.h>

namespace layer_in_c::nn_models::mlp_unconditional_stddev {

    template<template <typename> typename BASE, typename SPEC>
    struct NEURAL_NETWORK_FACTORY: public BASE<SPEC>{
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using LOG_STD_CONTAINER_SPEC = matrix::Specification<T, TI, 1, SPEC::OUTPUT_DIM>;
        using LOG_STD_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<LOG_STD_CONTAINER_SPEC>;
    };

    template <typename SPEC>
    struct NeuralNetwork: NEURAL_NETWORK_FACTORY<nn_models::mlp::NeuralNetwork, SPEC>{
        nn::parameters::Plain::instance<typename NeuralNetwork::LOG_STD_CONTAINER_TYPE> log_std;
    };
    template <typename SPEC>
    struct NeuralNetworkAdam: NEURAL_NETWORK_FACTORY<nn_models::mlp::NeuralNetworkAdam, SPEC>{
        nn::parameters::Adam::instance<typename NeuralNetworkAdam::LOG_STD_CONTAINER_TYPE> log_std;
    };


}

#endif
