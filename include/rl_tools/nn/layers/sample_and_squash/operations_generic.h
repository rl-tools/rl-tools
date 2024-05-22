#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_SAMPLE_AND_SQUASH_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_SAMPLE_AND_SQUASH_OPERATIONS_GENERIC_H
#include "../../../nn/activation_functions.h"
#include "../../../utils/generic/typing.h"
#include "../../../containers.h"


#include "layer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::sample_and_squash::LayerForward<SPEC>){

    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::sample_and_squash::LayerBackward<SPEC>& layer){
        malloc(device, static_cast<nn::layers::sample_and_squash::LayerForward<SPEC>&>(layer));
        malloc(device, layer.noise);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer){
        malloc(device, static_cast<nn::layers::sample_and_squash::LayerBackward<SPEC>&>(layer));
        malloc(device, layer.output);
    }
    template<typename DEVICE>
    void malloc(DEVICE& device, nn::layers::sample_and_squash::Buffer& buffer) { } // no-op
    template<typename DEVICE>
    void free(DEVICE& device, nn::layers::sample_and_squash::Buffer& buffer) { } // no-op
    template <typename DEVICE, typename SPEC, typename RNG>
    void init_weights(DEVICE& device, nn::layers::sample_and_squash::LayerForward<SPEC>& layer, RNG& rng){

    }
    template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG>
    void evaluate(const DEVICE& device, const nn::layers::sample_and_squash::LayerForward<SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::sample_and_squash::Buffer& buffer, RNG& rng){
        static_assert(INPUT_SPEC::COLS == 2*SPEC::DIM);
        static_assert(OUTPUT_SPEC::COLS == SPEC::DIM);
        static_assert(INPUT_SPEC::ROWS == OUTPUT_SPEC::ROWS);
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        using PARAMETERS = typename SPEC::PARAMETERS;
        for(TI row_i = 0; row_i < INPUT_SPEC::ROWS; row_i++){
            for(TI col_i = 0; col_i < SPEC::DIM; col_i++){
                T mean = get(input, row_i, col_i);
                T log_std = get(input, row_i, SPEC::DIM + col_i);
                T log_std_clipped = math::clamp(device.math, log_std, (T)PARAMETERS::LOG_STD_LOWER_BOUND, (T)PARAMETERS::LOG_STD_UPPER_BOUND);
                T std = math::exp(device.math, log_std_clipped);
                T output_val = random::normal_distribution::sample(device.random, mean, std, rng);
                set(output, row_i, col_i, output_val);
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
