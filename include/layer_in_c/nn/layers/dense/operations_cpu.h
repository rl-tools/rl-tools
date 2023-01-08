#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_H

#include "operations_generic.h"
#include <type_traits>

namespace layer_in_c{
    template<typename DEV_SPEC, typename LS, typename RNG>
    void init_kaiming(nn::layers::dense::Layer<devices::CPU<DEV_SPEC>, LS>& layer, RNG& rng) {
        typedef typename LS::T T;
        logging::text(layer.device.logger, "Initializing layer with the Kaiming scheme");
        T negative_slope = math::sqrt(typename DEV_SPEC::MATH(), (T)5);
        T gain = math::sqrt(typename DEV_SPEC::MATH(), (T)2.0 / (1 + negative_slope * negative_slope));
        T fan = LS::INPUT_DIM;
        T std = gain / math::sqrt(typename DEV_SPEC::MATH(), fan);
        T weight_bound = math::sqrt(typename DEV_SPEC::MATH(), (T)3.0) * std;
        T bias_bound = 1/math::sqrt(typename DEV_SPEC::MATH(), (T)LS::INPUT_DIM);
        for(index_t i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.biases[i] = utils::random::uniform_real_distribution(typename DEV_SPEC::RANDOM(), -bias_bound, bias_bound, rng);
            for(index_t j = 0; j < LS::INPUT_DIM; j++) {
                layer.weights[i][j] = utils::random::uniform_real_distribution(typename DEV_SPEC::RANDOM(), -weight_bound, weight_bound, rng);
            }
        }
    }
}

#endif
