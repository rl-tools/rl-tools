#ifndef LAYER_IN_C_NN_LAYERS_DENSE_LAYER_H
#define LAYER_IN_C_NN_LAYERS_DENSE_LAYER_H
#include <layer_in_c/nn/activation_functions.h>

namespace layer_in_c::nn::layers::dense {
    template<typename T_T, auto T_INPUT_DIM, auto T_OUTPUT_DIM, nn::activation_functions::ActivationFunction T_ACTIVATION_FUNCTION>
    struct LayerSpecification {
        typedef T_T T;
        static constexpr auto INPUT_DIM = T_INPUT_DIM;
        static constexpr auto OUTPUT_DIM = T_OUTPUT_DIM;
        static constexpr nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION = T_ACTIVATION_FUNCTION;
        // Summary
        static constexpr auto NUM_WEIGHTS = OUTPUT_DIM * INPUT_DIM + OUTPUT_DIM;
    };
    template<typename DEVICE, typename T_SPEC>
    struct Layer {
        typedef T_SPEC SPEC;
        static constexpr typename DEVICE::index_t INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr typename DEVICE::index_t OUTPUT_DIM = SPEC::OUTPUT_DIM;
        static constexpr typename DEVICE::index_t NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        typename SPEC::T weights[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T biases[SPEC::OUTPUT_DIM];

        DEVICE& device;
        explicit Layer(DEVICE& device) : device(device) {}
    };
    template<typename DEVICE, typename SPEC>
    struct LayerBackward : public Layer<DEVICE, SPEC> {
        typename SPEC::T pre_activations[SPEC::OUTPUT_DIM];
        explicit LayerBackward(DEVICE& device) : Layer<DEVICE, SPEC>(device) {}
    };
    template<typename DEVICE, typename SPEC>
    struct LayerBackwardGradient : public LayerBackward<DEVICE, SPEC> {
        typename SPEC::T output[SPEC::OUTPUT_DIM];
        typename SPEC::T d_weights[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_biases[SPEC::OUTPUT_DIM];
        explicit LayerBackwardGradient(DEVICE& device) : LayerBackward<DEVICE, SPEC>(device) {}
    };
    template<typename T>
    struct DefaultSGDParameters {
    public:
        static constexpr T ALPHA = 0.001;

    };
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    struct LayerBackwardSGD : public LayerBackwardGradient<DEVICE, SPEC> {
        explicit LayerBackwardSGD(DEVICE& device) : LayerBackwardGradient<DEVICE, SPEC>(device) {}
    };

    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    struct LayerBackwardAdam : public LayerBackwardGradient<DEVICE, SPEC> {
        typename SPEC::T d_weights_first_order_moment[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_weights_second_order_moment[SPEC::OUTPUT_DIM][SPEC::INPUT_DIM];
        typename SPEC::T d_biases_first_order_moment[SPEC::OUTPUT_DIM];
        typename SPEC::T d_biases_second_order_moment[SPEC::OUTPUT_DIM];
        explicit LayerBackwardAdam(DEVICE& device) : LayerBackwardGradient<DEVICE, SPEC>(device) {}
    };
}

#endif