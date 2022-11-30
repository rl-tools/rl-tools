#ifndef LAYER_IN_C_NN_LAYERS_DENSE_LAYER_H
#define LAYER_IN_C_NN_LAYERS_DENSE_LAYER_H
#include <layer_in_c/nn/activation_functions.h>
#include <layer_in_c/utils/polyak.h>

namespace layer_in_c::nn::layers::dense {
    template<typename T_T, int T_INPUT_DIM, int T_OUTPUT_DIM, nn::activation_functions::ActivationFunction T_ACTIVATION_FUNCTION>
    struct LayerSpec {
        typedef T_T T;
        static constexpr int INPUT_DIM = T_INPUT_DIM;
        static constexpr int OUTPUT_DIM = T_OUTPUT_DIM;
        static constexpr nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION = T_ACTIVATION_FUNCTION;
    };
    template<typename T_SPEC>
    struct Layer {
        typedef T_SPEC SPEC;
        static constexpr int INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr int OUTPUT_DIM = SPEC::OUTPUT_DIM;
        typename SPEC::T weights[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T biases[SPEC::OUTPUT_DIM];
    };
    template<typename SPEC>
    struct LayerBackward : public Layer<SPEC> {
        typename SPEC::T output[SPEC::OUTPUT_DIM];
    };
    template<typename SPEC>
    struct LayerBackwardGradient : public LayerBackward<SPEC> {
        typename SPEC::T d_weights[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_biases[SPEC::OUTPUT_DIM];
    };
    template<typename T>
    struct DefaultSGDParameters {
    public:
        static constexpr T ALPHA = 0.001;

    };
    template<typename SPEC, typename PARAMETERS>
    struct LayerBackwardSGD : public LayerBackwardGradient<SPEC> {
    };

    template<typename SPEC, typename PARAMETERS>
    struct LayerBackwardAdam : public LayerBackwardGradient<SPEC> {
        typename SPEC::T d_weights_first_order_moment[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_weights_second_order_moment[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_biases_first_order_moment[SPEC::OUTPUT_DIM];
        typename SPEC::T d_biases_second_order_moment[SPEC::OUTPUT_DIM];
    };
}

#endif