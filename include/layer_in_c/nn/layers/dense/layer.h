#ifndef LAYER_IN_C_NN_LAYERS_DENSE_LAYER_H
#define LAYER_IN_C_NN_LAYERS_DENSE_LAYER_H
#include <layer_in_c/nn/activation_functions.h>
#include <layer_in_c/utils/generic/typing.h>

namespace layer_in_c::nn::layers::dense {
    template<typename T_T, typename T_TI, T_TI T_INPUT_DIM, T_TI T_OUTPUT_DIM, nn::activation_functions::ActivationFunction T_ACTIVATION_FUNCTION>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        static constexpr auto INPUT_DIM = T_INPUT_DIM;
        static constexpr auto OUTPUT_DIM = T_OUTPUT_DIM;
        static constexpr nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION = T_ACTIVATION_FUNCTION;
        // Summary
        static constexpr auto NUM_WEIGHTS = OUTPUT_DIM * INPUT_DIM + OUTPUT_DIM;
    };
    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec_memory =
            utils::typing::is_same_v<typename SPEC_1::T, typename SPEC_2::T>
            && SPEC_1::INPUT_DIM == SPEC_2::INPUT_DIM
            && SPEC_1::OUTPUT_DIM == SPEC_2::OUTPUT_DIM;
    template<typename SPEC_1, typename SPEC_2>
    constexpr bool check_spec =
        check_spec_memory<SPEC_1, SPEC_2>
        && SPEC_1::ACTIVATION_FUNCTION == SPEC_2::ACTIVATION_FUNCTION;

    template<typename T_SPEC>
    struct Layer {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
        static constexpr TI NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        typename SPEC::T weights[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T biases[SPEC::OUTPUT_DIM];

    };
    template<typename SPEC>
    struct LayerBackward : public Layer<SPEC> {
        typename SPEC::T pre_activations[SPEC::OUTPUT_DIM];
    };
    template<typename SPEC>
    struct LayerBackwardGradient : public LayerBackward<SPEC> {
        typename SPEC::T output[SPEC::OUTPUT_DIM];
        typename SPEC::T d_weights[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_biases[SPEC::OUTPUT_DIM];
    };
    template<typename T>
    struct DefaultSGDParameters {
    public:
        static constexpr T ALPHA = 0.001;

    };
    template<typename SPEC, typename PARAMETERS>
    struct LayerBackwardSGD : public LayerBackwardGradient<SPEC> {};

    template<typename SPEC, typename PARAMETERS>
    struct LayerBackwardAdam : public LayerBackwardGradient<SPEC> {
        typename SPEC::T d_weights_first_order_moment[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_weights_second_order_moment[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_biases_first_order_moment[SPEC::OUTPUT_DIM];
        typename SPEC::T d_biases_second_order_moment[SPEC::OUTPUT_DIM];
    };
}

#endif