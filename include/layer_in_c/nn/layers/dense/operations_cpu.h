#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_H

#include <layer_in_c/nn/layers/dense/layer.h>
#include <layer_in_c/utils/generic/polyak.h>

#include "operations_generic.h"

#include <random>

// Specializations
namespace layer_in_c{
    template<typename LS, typename RNG>
    void init_kaiming(nn::layers::dense::Layer<devices::CPU, LS>& layer, RNG& rng) {
        typedef typename LS::T T;
        T negative_slope = std::sqrt((T)5);
        T gain = std::sqrt((T)2.0 / (1 + negative_slope * negative_slope));
        T fan = LS::INPUT_DIM;
        T std = gain / std::sqrt(fan);
        T weight_bound = std::sqrt((T)3.0) * std;
        auto weight_distribution = std::uniform_real_distribution<T>(-weight_bound, weight_bound);
        T bias_bound = 1/std::sqrt((T)LS::INPUT_DIM);
        auto bias_distribution = std::uniform_real_distribution<T>(-bias_bound, bias_bound);

        for(index_t i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.biases[i] = bias_distribution(rng);
            for(index_t j = 0; j < LS::INPUT_DIM; j++) {
                layer.weights[i][j] = weight_distribution(rng);
            }
        }
    }

}

#endif
