#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_CPU_H

#include "operations_generic.h"

namespace layer_in_c{
    template<typename DEV_SPEC, typename LS, typename RNG>
    void init_kaiming(devices::CPU<DEV_SPEC>& device, nn::layers::dense::Layer<LS>& layer, RNG& rng) {
        using T = typename LS::T;
        using TI = typename LS::TI;
        logging::text(device, device.logger, "Initializing layer using the Kaiming scheme");
        T negative_slope = math::sqrt(typename DEV_SPEC::MATH(), (T)5);
        T gain = math::sqrt(typename DEV_SPEC::MATH(), (T)2.0 / (1 + negative_slope * negative_slope));
        T fan = LS::INPUT_DIM;
        T std = gain / math::sqrt(typename DEV_SPEC::MATH(), fan);
        T weight_bound = math::sqrt(typename DEV_SPEC::MATH(), (T)3.0) * std;
        T bias_bound = 1/math::sqrt(typename DEV_SPEC::MATH(), (T)LS::INPUT_DIM);
        for(TI i = 0; i < LS::OUTPUT_DIM; i++) {
            set(layer.biases, 0, i, random::uniform_real_distribution(typename DEV_SPEC::RANDOM(), -bias_bound, bias_bound, rng));
            for(TI j = 0; j < LS::INPUT_DIM; j++) {
                set(layer.weights, i, j, random::uniform_real_distribution(typename DEV_SPEC::RANDOM(), -weight_bound, weight_bound, rng));
            }
        }
    }
}

#endif
