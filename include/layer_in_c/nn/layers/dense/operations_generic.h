#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_GENERIC_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_GENERIC_H

#include <layer_in_c/nn/layers/dense/layer.h>
#include <layer_in_c/utils/generic/polyak.h>
#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif

namespace layer_in_c{
    template<typename DEVICE, typename LS, typename RNG>
    void init_kaiming(nn::layers::dense::Layer<DEVICE, LS>& layer, RNG& rng) {
        typedef typename LS::T T;
        T negative_slope = math::sqrt(typename DEVICE::SPEC::MATH(), (T)5);
        T gain = math::sqrt(typename DEVICE::SPEC::MATH(), (T)2.0 / (1 + negative_slope * negative_slope));
        T fan = LS::INPUT_DIM;
        T std = gain / math::sqrt(typename DEVICE::SPEC::MATH(), fan);
        T weight_bound = math::sqrt(typename DEVICE::SPEC::MATH(), (T)3.0) * std;
        T bias_bound = 1/math::sqrt(typename DEVICE::SPEC::MATH(), (T)LS::INPUT_DIM);
        for(index_t i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.biases[i] = utils::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -bias_bound, bias_bound, rng);
            for(index_t j = 0; j < LS::INPUT_DIM; j++) {
                layer.weights[i][j] = utils::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -weight_bound, weight_bound, rng);
            }
        }
    }
    // evaluating a layer does not change its state (like pre_activations and outputs). Before using backward, to fill the state, use the forward method instead
    template<typename DEVICE, typename T, typename SPEC>
    FUNCTION_PLACEMENT void evaluate(const nn::layers::dense::Layer<DEVICE, SPEC>& layer, const T input[SPEC::INPUT_DIM], T output[SPEC::OUTPUT_DIM]) {
        // Warning do not use the same buffer for input and output!
        for(index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            output[i] = layer.biases[i];
            for(index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                output[i] += layer.weights[i][j] * input[j];
            }
            output[i] = nn::activation_functions::activation<typename DEVICE::SPEC::MATH, T, SPEC::ACTIVATION_FUNCTION>(output[i]);
        }
    }

    template<typename DEVICE, typename T, typename SPEC>
    FUNCTION_PLACEMENT void forward(nn::layers::dense::LayerBackward<DEVICE, SPEC>& layer, const T input[SPEC::INPUT_DIM], T output[SPEC::OUTPUT_DIM]) {
        // Warning do not use the same buffer for input and output!
        for(index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            layer.pre_activations[i] = layer.biases[i];
            for(index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                layer.pre_activations[i] += layer.weights[i][j] * input[j];
            }
            output[i] = nn::activation_functions::activation<T, SPEC::ACTIVATION_FUNCTION>(layer.pre_activations[i]);
        }
    }

    template<typename DEVICE, typename T, typename SPEC>
    FUNCTION_PLACEMENT void forward(nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>& layer, const T input[SPEC::INPUT_DIM]) {
        // Warning do not use the same buffer for input and output!
        for(index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            layer.pre_activations[i] = layer.biases[i];
            for(index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                layer.pre_activations[i] += layer.weights[i][j] * input[j];
            }
            layer.output[i] = nn::activation_functions::activation<typename DEVICE::SPEC::MATH, T, SPEC::ACTIVATION_FUNCTION>(layer.pre_activations[i]);
        }
    }
    template<typename DEVICE, typename T, typename SPEC>
    [[deprecated("Calling forward with an output buffer on a layer requiring the gradient is not recommended. Consider using forward without an output buffer to avoid copies instead.")]]
    FUNCTION_PLACEMENT void forward(nn::layers::dense::LayerBackwardGradient<DEVICE, SPEC>& layer, const T input[SPEC::INPUT_DIM], T output[SPEC::OUTPUT_DIM]) {
        // compile time warning if used
        forward(layer, input);
        for(index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            output[i] = layer.output[i];
        }
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void backward(nn::layers::dense::LayerBackward<DEVICE, SPEC>& layer, const typename SPEC::T d_output[SPEC::OUTPUT_DIM], typename SPEC::T d_input[SPEC::INPUT_DIM]) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        for(index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            typename SPEC::T d_pre_activation = nn::activation_functions::d_activation_d_x<typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(layer.pre_activations[i]) * d_output[i];
            for(index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                if(i == 0){
                    d_input[j] = 0;
                }
                d_input[j] += layer.weights[i][j] * d_pre_activation;
            }
        }
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void backward(nn::layers::dense::LayerBackward<DEVICE, SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], const typename SPEC::T d_output[SPEC::OUTPUT_DIM], typename SPEC::T d_input[SPEC::INPUT_DIM]) {
        backward(layer, d_output, d_input);

    }

    template<typename DEVICE, typename LS>
    FUNCTION_PLACEMENT void backward(nn::layers::dense::LayerBackwardGradient<DEVICE, LS>& layer, const typename LS::T input[LS::INPUT_DIM], const typename LS::T d_output[LS::OUTPUT_DIM], typename LS::T d_input[LS::INPUT_DIM]) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        // todo: think about storing gradient in column major order to avoid iterating over the minor dimension
        for(index_t i = 0; i < LS::OUTPUT_DIM; i++) {
            typename LS::T d_pre_activation = nn::activation_functions::d_activation_d_x<typename DEVICE::SPEC::MATH, typename LS::T, LS::ACTIVATION_FUNCTION>(layer.pre_activations[i]) * d_output[i];
            layer.d_biases[i] += d_pre_activation;
            for(index_t j = 0; j < LS::INPUT_DIM; j++) {
                if(i == 0){
                    d_input[j] = 0;
                }
                d_input[j] += layer.weights[i][j] * d_pre_activation;
                layer.d_weights[i][j] += d_pre_activation * input[j];
            }
        }
    }
    template<typename DEVICE, typename LS>
    FUNCTION_PLACEMENT void zero_gradient(nn::layers::dense::LayerBackwardGradient<DEVICE, LS>& layer) {
        for(index_t i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.d_biases[i] = 0;
            for(index_t j = 0; j < LS::INPUT_DIM; j++) {
                layer.d_weights[i][j] = 0;
            }
        }
    }
    template<typename DEVICE, typename LS, typename PARAMETERS>
    FUNCTION_PLACEMENT void update_layer(nn::layers::dense::LayerBackwardSGD<DEVICE, LS, PARAMETERS>& layer){
        for(index_t i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.biases[i] -= PARAMETERS::ALPHA * layer.d_biases[i];
            for(index_t j = 0; j < LS::INPUT_DIM; j++) {
                layer.weights[i][j] -= PARAMETERS::ALPHA * layer.d_weights[i][j];
            }
        }
    }

    template<typename DEVICE, typename LS, typename PARAMETERS>
    FUNCTION_PLACEMENT void reset_optimizer_state(nn::layers::dense::LayerBackwardAdam<DEVICE, LS, PARAMETERS>& layer) {
        for(index_t i = 0; i < LS::OUTPUT_DIM; i++) {
            layer.d_biases_first_order_moment [i] = 0;
            layer.d_biases_second_order_moment[i] = 0;
            for(index_t j = 0; j < LS::INPUT_DIM; j++) {
                layer.d_weights_first_order_moment [i][j] = 0;
                layer.d_weights_second_order_moment[i][j] = 0;
            }
        }
    }
    template<typename DEVICE, typename LS, typename PARAMETERS>
    FUNCTION_PLACEMENT void gradient_descent(nn::layers::dense::LayerBackwardAdam<DEVICE, LS, PARAMETERS>& layer, typename LS::T first_order_moment_bias_correction, typename LS::T second_order_moment_bias_correction){
        for(index_t i = 0; i < LS::OUTPUT_DIM; i++) {
            typename LS::T bias_update = PARAMETERS::ALPHA * first_order_moment_bias_correction * layer.d_biases_first_order_moment[i] / (math::sqrt(typename DEVICE::SPEC::MATH(), layer.d_biases_second_order_moment[i] * second_order_moment_bias_correction) + PARAMETERS::EPSILON);
            layer.biases[i] -= bias_update;
            for(index_t j = 0; j < LS::INPUT_DIM; j++) {
                typename LS::T weight_update = PARAMETERS::ALPHA * first_order_moment_bias_correction * layer.d_weights_first_order_moment[i][j] / (math::sqrt(typename DEVICE::SPEC::MATH(), layer.d_weights_second_order_moment[i][j] * second_order_moment_bias_correction) + PARAMETERS::EPSILON);
                layer.weights[i][j] -= weight_update;
            }
        }
    }

    template<typename DEVICE, typename LS, typename PARAMETERS>
    FUNCTION_PLACEMENT void update_layer(nn::layers::dense::LayerBackwardAdam<DEVICE, LS, PARAMETERS>& layer, typename LS::T first_order_moment_bias_correction, typename LS::T second_order_moment_bias_correction) {
        // todo remove template params (auto inference)
        utils::polyak::update_matrix<DEVICE, typename LS::T, LS::OUTPUT_DIM, LS::INPUT_DIM>(layer.device, layer.d_weights_first_order_moment, layer.d_weights, PARAMETERS::BETA_1);
        utils::polyak::update       <DEVICE, typename LS::T, LS::OUTPUT_DIM>               (layer.device, layer. d_biases_first_order_moment, layer.d_biases , PARAMETERS::BETA_1);

        utils::polyak::update_squared_matrix<DEVICE, typename LS::T, LS::OUTPUT_DIM, LS::INPUT_DIM>(layer.device, layer.d_weights_second_order_moment, layer.d_weights, PARAMETERS::BETA_2);
        utils::polyak::update_squared       <DEVICE, typename LS::T, LS::OUTPUT_DIM>               (layer.device, layer. d_biases_second_order_moment, layer.d_biases , PARAMETERS::BETA_2);

        gradient_descent(layer, first_order_moment_bias_correction, second_order_moment_bias_correction);

    }
}

#endif