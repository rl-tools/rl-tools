#ifndef LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_GENERIC_H
#define LAYER_IN_C_NN_LAYERS_DENSE_OPERATIONS_GENERIC_H

#include <layer_in_c/containers.h>
#include <layer_in_c/nn/layers/dense/layer.h>
#include <layer_in_c/utils/generic/polyak.h>
#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif

namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer) {
        malloc(device, layer.weights);
        malloc(device, layer.biases);
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer) {
        free(device, layer.weights);
        free(device, layer.biases);
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer) {
        malloc((nn::layers::dense::Layer<SPEC>&) layer);
        malloc(device, layer.pre_activations);
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer) {
        free((nn::layers::dense::Layer<SPEC>&) layer);
        free(device, layer.pre_activations);
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer) {
        malloc((nn::layers::dense::LayerBackward<SPEC>&) layer);
        malloc(device, layer.output);
        malloc(device, layer.d_biases);
        malloc(device, layer.d_weights);
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer) {
        free((nn::layers::dense::LayerBackward<SPEC>&) layer);
        free(device, layer.output);
        free(device, layer.d_biases);
        free(device, layer.d_weights);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::dense::LayerBackwardAdam<SPEC, PARAMETERS>& layer) {
        malloc((nn::layers::dense::LayerBackwardGradient<SPEC>&) layer);
        malloc(device, layer.d_weights_first_order_moment);
        malloc(device, layer.d_weights_second_order_moment);
        malloc(device, layer.d_biases_first_order_moment);
        malloc(device, layer.d_biases_second_order_moment);
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::dense::LayerBackwardAdam<SPEC, PARAMETERS>& layer) {
        free((nn::layers::dense::LayerBackwardGradient<SPEC>&) layer);
        free(device, layer.d_weights_first_order_moment);
        free(device, layer.d_weights_second_order_moment);
        free(device, layer.d_biases_first_order_moment);
        free(device, layer.d_biases_second_order_moment);
    }

    template<typename DEVICE, typename SPEC, typename RNG>
    FUNCTION_PLACEMENT void init_kaiming(DEVICE& device, nn::layers::dense::Layer<SPEC>& layer, RNG& rng) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        T negative_slope = math::sqrt(typename DEVICE::SPEC::MATH(), (T)5);
        T gain = math::sqrt(typename DEVICE::SPEC::MATH(), (T)2.0 / (1 + negative_slope * negative_slope));
        T fan = SPEC::INPUT_DIM;
        T std = gain / math::sqrt(typename DEVICE::SPEC::MATH(), fan);
        T weight_bound = math::sqrt(typename DEVICE::SPEC::MATH(), (T)3.0) * std;
        T bias_bound = 1/math::sqrt(typename DEVICE::SPEC::MATH(), (T)SPEC::INPUT_DIM);
        for(TI i = 0; i < SPEC::OUTPUT_DIM; i++) {
            layer.biases.data[i] = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -bias_bound, bias_bound, rng);
            for(TI j = 0; j < SPEC::INPUT_DIM; j++) {
                layer.weights.data[i * SPEC::INPUT_DIM + j] = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -weight_bound, weight_bound, rng);
            }
        }
    }
    // evaluating a layer does not change its state (like pre_activations and outputs). Before using backward, to fill the state, use the forward method instead
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::dense::Layer<SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_DIM]) {
        // Warning do not use the same buffer for input and output!
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            output[i] = layer.biases.data[i];
            for(typename DEVICE::index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                output[i] += layer.weights.data[i * SPEC::INPUT_DIM + j] * input[j];
            }
            output[i] = activation<typename DEVICE::SPEC::MATH, typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(output[i]);
        }
    }

    template<typename DEVICE, typename SPEC, typename DEVICE::index_t BATCH_SIZE>
    FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::dense::Layer<SPEC>& layer, const Matrix<typename SPEC::T, typename DEVICE::index_t, BATCH_SIZE, SPEC::INPUT_DIM, RowMajor>& input, Matrix<typename SPEC::T, typename DEVICE::index_t, BATCH_SIZE, SPEC::OUTPUT_DIM, RowMajor>& output) {
        // Warning do not use the same buffer for input and output!
        using TI = typename DEVICE::index_t;
        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            for(TI output_i = 0; output_i < SPEC::OUTPUT_DIM; output_i++) {
                TI output_index = batch_i * SPEC::OUTPUT_DIM + output_i;
                output.data[output_index] = layer.biases.data[output_i];
                for(TI input_i = 0; input_i < SPEC::INPUT_DIM; input_i++) {
                    TI input_index = batch_i * SPEC::INPUT_DIM + input_i;
                    output.data[output_index] += layer.weights.data[output_i * SPEC::INPUT_DIM + input_i] * input.data[input_index];
                }
                output.data[output_index] = activation<typename DEVICE::SPEC::MATH, typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(output.data[output_index]);
            }
        }
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_DIM]) {
        // Warning do not use the same buffer for input and output!
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            layer.pre_activations[i] = layer.biases.data[i];
            for(typename DEVICE::index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                layer.pre_activations[i] += layer.weights.data[i * SPEC::INPUT_DIM + j] * input[j];
            }
            output[i] = activation<typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(layer.pre_activations[i]);
        }
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM]) {
        // Warning do not use the same buffer for input and output!
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            layer.pre_activations[i] = layer.biases.data[i];
            for(typename DEVICE::index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                layer.pre_activations[i] += layer.weights.data[i * SPEC::INPUT_DIM + j] * input[j];
            }
            layer.output[i] = activation<typename DEVICE::SPEC::MATH, typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(layer.pre_activations[i]);
        }
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], typename SPEC::T output[SPEC::OUTPUT_DIM]) {
        // compile time warning if used
        forward(device, layer, input);
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            output[i] = layer.output[i];
        }
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, const typename SPEC::T d_output[SPEC::OUTPUT_DIM], typename SPEC::T d_input[SPEC::INPUT_DIM]) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            typename SPEC::T d_pre_activation = d_activation_d_x<typename SPEC::T, SPEC::ACTIVATION_FUNCTION>(layer.pre_activations[i]) * d_output[i];
            for(typename DEVICE::index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                if(i == 0){
                    d_input[j] = 0;
                }
                d_input[j] += layer.weights.data[i * SPEC::INPUT_DIM + j] * d_pre_activation;
            }
        }
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::dense::LayerBackward<SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], const typename SPEC::T d_output[SPEC::OUTPUT_DIM], typename SPEC::T d_input[SPEC::INPUT_DIM]) {
        backward(layer, d_output, d_input);
    }

    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer, const typename SPEC::T input[SPEC::INPUT_DIM], const typename SPEC::T d_output[SPEC::OUTPUT_DIM], typename SPEC::T d_input[SPEC::INPUT_DIM]) {
        // todo: create sparate function that does not set d_input (to save cost on backward pass for the first layer)
        // todo: think about storing gradient in column major order to avoid iterating over the minor dimension
        constexpr auto INPUT_DIM = SPEC::INPUT_DIM;
        constexpr auto OUTPUT_DIM = SPEC::OUTPUT_DIM;
        using T = typename SPEC::T;
        for(typename DEVICE::index_t i = 0; i < OUTPUT_DIM; i++) {
            T d_pre_activation = d_activation_d_x<typename DEVICE::SPEC::MATH, T, SPEC::ACTIVATION_FUNCTION>(layer.pre_activations[i]) * d_output[i];
            layer.d_biases.data[i] += d_pre_activation;
            for(typename DEVICE::index_t j = 0; j < INPUT_DIM; j++) {
                if(i == 0){
                    d_input[j] = 0;
                }
                d_input[j] += layer.weights.data[i * SPEC::INPUT_DIM + j] * d_pre_activation;
                layer.d_weights.data[i * SPEC::INPUT_DIM + j] += d_pre_activation * input[j];
            }
        }
    }
    template<typename DEVICE, typename SPEC>
    FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::dense::LayerBackwardGradient<SPEC>& layer) {
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            layer.d_biases.data[i] = 0;
            for(typename DEVICE::index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                layer.d_weights.data[i * SPEC::INPUT_DIM + j] = 0;
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    FUNCTION_PLACEMENT void update_layer(DEVICE& device, nn::layers::dense::LayerBackwardSGD<SPEC, PARAMETERS>& layer){
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            layer.biases.data[i] -= PARAMETERS::ALPHA * layer.d_biases.data[i];
            for(typename DEVICE::index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                layer.weights.data[i * SPEC::INPUT_DIM + j] -= PARAMETERS::ALPHA * layer.d_weights.data[i * SPEC::INPUT_DIM + j];
            }
        }
    }

    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    FUNCTION_PLACEMENT void reset_optimizer_state(DEVICE& device, nn::layers::dense::LayerBackwardAdam<SPEC, PARAMETERS>& layer) {
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            layer.d_biases_first_order_moment [i] = 0;
            layer.d_biases_second_order_moment[i] = 0;
            for(typename DEVICE::index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                layer.d_weights_first_order_moment [i][j] = 0;
                layer.d_weights_second_order_moment->data[i * SPEC::INPUT_DIM + j] = 0;
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    FUNCTION_PLACEMENT void gradient_descent(DEVICE& device, nn::layers::dense::LayerBackwardAdam<SPEC, PARAMETERS>& layer, typename SPEC::T first_order_moment_bias_correction, typename SPEC::T second_order_moment_bias_correction){
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++) {
            typename SPEC::T bias_update = PARAMETERS::ALPHA * first_order_moment_bias_correction * layer.d_biases_first_order_moment[i] / (math::sqrt(typename DEVICE::SPEC::MATH(), layer.d_biases_second_order_moment[i] * second_order_moment_bias_correction) + PARAMETERS::EPSILON);
            layer.biases.data[i] -= bias_update;
            for(typename DEVICE::index_t j = 0; j < SPEC::INPUT_DIM; j++) {
                typename SPEC::T weight_update = PARAMETERS::ALPHA * first_order_moment_bias_correction * layer.d_weights_first_order_moment->data[i * SPEC::INPUT_DIM + j] / (math::sqrt(typename DEVICE::SPEC::MATH(), layer.d_weights_second_order_moment->data[i * SPEC::INPUT_DIM + j] * second_order_moment_bias_correction) + PARAMETERS::EPSILON);
                layer.weights.data[i * SPEC::INPUT_DIM + j] -= weight_update;
            }
        }
    }

    template<typename DEVICE, typename SPEC, typename PARAMETERS>
    FUNCTION_PLACEMENT void update_layer(DEVICE& device, nn::layers::dense::LayerBackwardAdam<SPEC, PARAMETERS>& layer, typename SPEC::T first_order_moment_bias_correction, typename SPEC::T second_order_moment_bias_correction) {
        // todo remove template params (auto inference)
        utils::polyak::update_matrix<DEVICE, typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(device, layer.d_weights_first_order_moment, layer.d_weights, PARAMETERS::BETA_1);
        utils::polyak::update       <DEVICE, typename SPEC::T, SPEC::OUTPUT_DIM>               (device, layer. d_biases_first_order_moment, layer.d_biases , PARAMETERS::BETA_1);

        utils::polyak::update_squared_matrix<DEVICE, typename SPEC::T, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(device, layer.d_weights_second_order_moment, layer.d_weights, PARAMETERS::BETA_2);
        utils::polyak::update_squared       <DEVICE, typename SPEC::T, SPEC::OUTPUT_DIM>               (device, layer. d_biases_second_order_moment, layer.d_biases , PARAMETERS::BETA_2);

        gradient_descent(device, layer, first_order_moment_bias_correction, second_order_moment_bias_correction);
    }

    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn::layers::dense::Layer<TARGET_SPEC>* target, const nn::layers::dense::Layer<SOURCE_SPEC>* source){
        static_assert(nn::layers::dense::check_spec_memory<TARGET_SPEC, SOURCE_SPEC>);
        for(typename TARGET_SPEC::TI i = 0; i < TARGET_SPEC::OUTPUT_DIM; i++) {
            target->biases.data[i] = source->biases.data[i];
            for(typename TARGET_SPEC::TI j = 0; j < TARGET_SPEC::INPUT_DIM; j++) {
                target->weights.data[i * TARGET_SPEC::INPUT_DIM + j] = source->weights.data[i * TARGET_SPEC::INPUT_DIM + j];
            }
        }
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn::layers::dense::Layer<TARGET_SPEC>& target, const nn::layers::dense::Layer<SOURCE_SPEC>& source){
        static_assert(nn::layers::dense::check_spec_memory<TARGET_SPEC, SOURCE_SPEC>);
        copy(&target, &source);
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn::layers::dense::LayerBackward<TARGET_SPEC>* target, const nn::layers::dense::LayerBackward<SOURCE_SPEC>* source){
        static_assert(nn::layers::dense::check_spec_memory<TARGET_SPEC, SOURCE_SPEC>);
        copy((nn::layers::dense::Layer<TARGET_SPEC>*) target, (nn::layers::dense::Layer<TARGET_SPEC>*) source);
        for(typename TARGET_SPEC::TI i = 0; i < TARGET_SPEC::OUTPUT_DIM; i++) {
            target->pre_activations[i] = source->pre_activations[i];
        }
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn::layers::dense::LayerBackward<TARGET_SPEC>& target, const nn::layers::dense::LayerBackward<SOURCE_SPEC>& source){
        static_assert(nn::layers::dense::check_spec_memory<TARGET_SPEC, SOURCE_SPEC>);
        copy(&target, &source);
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn::layers::dense::LayerBackwardGradient<TARGET_SPEC>* target, const nn::layers::dense::LayerBackwardGradient<SOURCE_SPEC>* source){
        static_assert(nn::layers::dense::check_spec_memory<TARGET_SPEC, SOURCE_SPEC>);
        copy((nn::layers::dense::LayerBackward<TARGET_SPEC>*)target, (nn::layers::dense::LayerBackward<SOURCE_SPEC>*)source);
        for(typename TARGET_SPEC::TI i = 0; i < TARGET_SPEC::OUTPUT_DIM; i++) {
            target->d_biases.data[i] = source->d_biases.data[i];
            target->output[i] = source->output[i];
            for(typename TARGET_SPEC::TI j = 0; j < TARGET_SPEC::INPUT_DIM; j++) {
                target->d_weights.data[i * TARGET_SPEC::INPUT_DIM + j] = source->d_weights.data[i * TARGET_SPEC::INPUT_DIM + j];
            }
        }
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC>
    FUNCTION_PLACEMENT void copy(nn::layers::dense::LayerBackwardGradient<TARGET_SPEC>& target, const nn::layers::dense::LayerBackwardGradient<SOURCE_SPEC>& source){
        static_assert(nn::layers::dense::check_spec_memory<TARGET_SPEC, SOURCE_SPEC>);
        copy(&target, &source);
    }

    template<typename TARGET_SPEC, typename SOURCE_SPEC, typename TARGET_PARAMETERS, typename SOURCE_PARAMETERS>
    FUNCTION_PLACEMENT void copy(nn::layers::dense::LayerBackwardAdam<TARGET_SPEC, TARGET_PARAMETERS>* target, const nn::layers::dense::LayerBackwardAdam<SOURCE_SPEC, SOURCE_PARAMETERS>* source){
        static_assert(nn::layers::dense::check_spec_memory<TARGET_SPEC, SOURCE_SPEC>);
        copy((nn::layers::dense::LayerBackwardGradient<TARGET_SPEC>*)target, (nn::layers::dense::LayerBackwardGradient<SOURCE_SPEC>*)source);
        for(typename TARGET_SPEC::TI i = 0; i < TARGET_SPEC::OUTPUT_DIM; i++) {
            target->d_biases_first_order_moment [i] = source->d_biases_first_order_moment [i];
            target->d_biases_second_order_moment[i] = source->d_biases_second_order_moment[i];
            for(typename TARGET_SPEC::TI j = 0; j < TARGET_SPEC::INPUT_DIM; j++) {
                target->d_weights_first_order_moment [i][j] = source->d_weights_first_order_moment [i][j];
                target->d_weights_second_order_moment->data[i * TARGET_SPEC::INPUT_DIM + j] = source->d_weights_second_order_moment->data[i * TARGET_SPEC::INPUT_DIM + j];
            }
        }
    }
    template<typename TARGET_SPEC, typename SOURCE_SPEC, typename TARGET_PARAMETERS, typename SOURCE_PARAMETERS>
    FUNCTION_PLACEMENT void copy(nn::layers::dense::LayerBackwardAdam<TARGET_SPEC, TARGET_PARAMETERS>& target, const nn::layers::dense::LayerBackwardAdam<SOURCE_SPEC, SOURCE_PARAMETERS>& source){
        static_assert(nn::layers::dense::check_spec_memory<TARGET_SPEC, SOURCE_SPEC>);
        copy(&target, &source);
    }

    namespace nn::layers::dense::helper{
        template <typename T, int N_ROWS, int N_COSPEC>
        T abs_diff_matrix(const T A[N_ROWS][N_COSPEC], const T B[N_ROWS][N_COSPEC]) {
            T acc = 0;
            for (int i = 0; i < N_ROWS; i++){
                for (int j = 0; j < N_COSPEC; j++){
                    acc += math::abs(A[i][j] - B[i][j]);
                }
            }
            return acc;
        }

        template <typename T, int N_ROWS>
        T abs_diff_vector(const T A[N_ROWS], const T B[N_ROWS]) {
            T acc = 0;
            for (int i = 0; i < N_ROWS; i++){
                acc += math::abs(A[i] - B[i]);
            }
            return acc;
        }
    }
    template <typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(const layer_in_c::nn::layers::dense::Layer<SPEC_1>* l1, const layer_in_c::nn::layers::dense::Layer<SPEC_2>* l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::T;
        T acc = 0;
        acc += nn::layers::dense::helper::abs_diff_matrix<T, SPEC_1::OUTPUT_DIM, SPEC_1::INPUT_DIM>(l1->weights, l2->weights);
        acc += nn::layers::dense::helper::abs_diff_vector<T, SPEC_1::OUTPUT_DIM>(l1->biases, l2->biases);
        return acc;
    }
    template <typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(const layer_in_c::nn::layers::dense::Layer<SPEC_1>& l1, const layer_in_c::nn::layers::dense::Layer<SPEC_2>& l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        return abs_diff(&l1, &l2);
    }
    template <typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(const layer_in_c::nn::layers::dense::LayerBackward<SPEC_1>* l1, const layer_in_c::nn::layers::dense::LayerBackward<SPEC_2>* l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::T;
        T acc = abs_diff((layer_in_c::nn::layers::dense::Layer<SPEC_1>*) l1, (layer_in_c::nn::layers::dense::Layer<SPEC_2>*) l2);
        acc += nn::layers::dense::helper::abs_diff_vector<T, SPEC_1::OUTPUT_DIM>(l1->pre_activations, l2->pre_activations);
        return acc;
    }
    template <typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(const layer_in_c::nn::layers::dense::LayerBackward<SPEC_1>& l1, const layer_in_c::nn::layers::dense::LayerBackward<SPEC_2>& l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        return abs_diff(&l1, &l2);
    }
    template <typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(const layer_in_c::nn::layers::dense::LayerBackwardGradient<SPEC_1>* l1, const layer_in_c::nn::layers::dense::LayerBackwardGradient<SPEC_2>* l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::T;
        T acc = abs_diff((layer_in_c::nn::layers::dense::LayerBackward<SPEC_1>*) l1, (layer_in_c::nn::layers::dense::LayerBackward<SPEC_2>*) l2);
        acc += nn::layers::dense::helper::abs_diff_vector<T, SPEC_1::OUTPUT_DIM>(l1->output, l2->output);
        acc += nn::layers::dense::helper::abs_diff_matrix<T, SPEC_1::OUTPUT_DIM, SPEC_1::INPUT_DIM>(l1->d_weights, l2->d_weights);
        acc += nn::layers::dense::helper::abs_diff_vector<T, SPEC_1::OUTPUT_DIM>(l1->d_biases, l2->d_biases);
        return acc;
    }
    template <typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(const layer_in_c::nn::layers::dense::LayerBackwardGradient<SPEC_1>& l1, const layer_in_c::nn::layers::dense::LayerBackwardGradient<SPEC_2>& l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        return abs_diff(&l1, &l2);
    }
    template <typename SPEC_1, typename SPEC_2, typename PARAMETERS_1, typename PARAMETERS_2>
    typename SPEC_1::T abs_diff(const layer_in_c::nn::layers::dense::LayerBackwardAdam<SPEC_1, PARAMETERS_1>* l1, const layer_in_c::nn::layers::dense::LayerBackwardAdam<SPEC_2, PARAMETERS_2>* l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::T;
        T acc = abs_diff((layer_in_c::nn::layers::dense::LayerBackwardGradient<SPEC_1>*) l1, (layer_in_c::nn::layers::dense::LayerBackwardGradient<SPEC_2>*) l2);
        acc += nn::layers::dense::helper::abs_diff_matrix<T, SPEC_1::OUTPUT_DIM, SPEC_1::INPUT_DIM>(l1->d_weights_first_order_moment, l2->d_weights_first_order_moment);
        acc += nn::layers::dense::helper::abs_diff_matrix<T, SPEC_1::OUTPUT_DIM, SPEC_1::INPUT_DIM>(l1->d_weights_second_order_moment, l2->d_weights_second_order_moment);
        acc += nn::layers::dense::helper::abs_diff_vector<T, SPEC_1::OUTPUT_DIM>(l1->d_biases_first_order_moment, l2->d_biases_first_order_moment);
        acc += nn::layers::dense::helper::abs_diff_vector<T, SPEC_1::OUTPUT_DIM>(l1->d_biases_second_order_moment, l2->d_biases_second_order_moment);
        return acc;
    }
    template <typename SPEC_1, typename SPEC_2, typename PARAMETERS_1, typename PARAMETERS_2>
    typename SPEC_1::T abs_diff(const layer_in_c::nn::layers::dense::LayerBackwardAdam<SPEC_1, PARAMETERS_1>& l1, const layer_in_c::nn::layers::dense::LayerBackwardAdam<SPEC_2, PARAMETERS_2>& l2) {
        static_assert(nn::layers::dense::check_spec_memory<SPEC_1, SPEC_2>);
        return abs_diff(&l1, &l2);
    }
    template <typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, layer_in_c::nn::layers::dense::LayerBackward<SPEC>* l) {
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++){
            l->pre_activations[i] = 0;
        }
    }
    template <typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, layer_in_c::nn::layers::dense::LayerBackward<SPEC>& l) {
        reset_forward_state(device, (layer_in_c::nn::layers::dense::Layer<SPEC>*) l);
    }
    template <typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, layer_in_c::nn::layers::dense::LayerBackwardGradient<SPEC>* l) {
        reset_forward_state(device, (layer_in_c::nn::layers::dense::LayerBackward<SPEC>*) l);
        for(typename DEVICE::index_t i = 0; i < SPEC::OUTPUT_DIM; i++){
            l->output[i] = 0;
        }
    }
    template <typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, layer_in_c::nn::layers::dense::LayerBackwardGradient<SPEC>& l) {
        reset_forward_state(device, &l);
    }
}

#endif