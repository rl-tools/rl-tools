#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_STANDARDIZE_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_STANDARDIZE_OPERATIONS_GENERIC_H
#include "../../../utils/generic/typing.h"
#include "../../../containers/matrix/matrix.h"

#include "../../../nn/nn.h"
#include "../../../nn/parameters/parameters.h"
#include "layer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::standardize::LayerForward<SPEC>& layer){
        malloc(device, layer.mean);
        malloc(device, layer.precision);
        malloc(device, layer.running_mean);
        malloc(device, layer.running_std);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::standardize::LayerForward<SPEC>& layer){
        free(device, layer.mean);
        free(device, layer.precision);
        free(device, layer.running_mean);
        free(device, layer.running_std);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::standardize::LayerGradient<SPEC>& layer){
        malloc(device, static_cast<nn::layers::standardize::LayerForward<SPEC>&>(layer));
        malloc(device, layer.output);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::standardize::LayerGradient<SPEC>& layer){
        free(device, static_cast<nn::layers::standardize::LayerForward<SPEC>&>(layer));
        free(device, layer.output);
    }
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::standardize::State& state) { } // no-op
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, nn::layers::standardize::State& source, nn::layers::standardize::State& target){}
    template<typename DEVICE, typename SPEC, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, const nn::layers::standardize::LayerForward<SPEC>& layer, nn::layers::standardize::State& state, RNG&, Mode<MODE> mode = Mode<mode::Default<>>{}) { } // no-op
    template<typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::standardize::State& state) { } // no-op
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::layers::standardize::Buffer& buffer){ }
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::layers::standardize::Buffer& buffer){ }
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init_weights(DEVICE& device, nn::layers::standardize::LayerForward<SPEC>& layer, RNG& rng) {
        set_all(device, layer.mean.parameters, 0);
        set_all(device, layer.precision.parameters, 1);
        layer.age = 0;
        set_all(device, layer.running_mean.parameters, 0);
        set_all(device, layer.running_std.parameters, 1);
    }
    namespace nn::layers::standardize{
        template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void _accumulate(DEVICE& device, nn::layers::standardize::LayerForward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& data) {
            static_assert(INPUT_SPEC::COLS == LAYER_SPEC::INPUT_DIM, "Data dimension must match layer input dimension");
            using T = typename LAYER_SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename DEVICE::index_t;
            constexpr TI DIM = LAYER_SPEC::INPUT_DIM;
            constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
            static_assert(BATCH_SIZE > 1, "Batch size must be greater than 1");
            layer.age++;
            for(TI dim_i = 0; dim_i < DIM; dim_i++){
                auto column = col(device, data, dim_i);
                T data_mean = mean(device, column);
                T data_std = std(device, column);
                T current_mean = get(device, layer.running_mean.parameters, dim_i);
                T new_mean = current_mean + (data_mean - current_mean) / layer.age;
                T current_std = get(device, layer.running_std.parameters, dim_i);
                T new_std = current_std + (data_std - current_std) / layer.age;
                set(device, layer.running_mean.parameters, new_mean, dim_i);
                set(device, layer.running_std.parameters, new_std, dim_i);
                set(device, layer.mean.parameters, new_mean, dim_i);
                T precision = new_std == 0 ? 0 : 1 / new_std;
                set(device, layer.precision.parameters, precision, dim_i);
            }
        }
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::standardize::LayerForward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::standardize::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        static_assert(nn::layers::standardize::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        using T = typename LAYER_SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            for(TI output_i = 0; output_i < LAYER_SPEC::OUTPUT_DIM; output_i++) {
                T mean = get(device, layer.mean.parameters, output_i);
                T precision = get(device, layer.precision.parameters, output_i);
                T input_value = get(input, batch_i, output_i);
                T output_value = (input_value - mean);
                if(precision != 0){
                    output_value *= precision;
                }
                set(output, batch_i, output_i, output_value);
            }
        }
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::standardize::LayerBackward<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::standardize::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::standardize::check_input_output<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>);
        if constexpr(mode::is<MODE, nn::layers::standardize::AccumulateMode>){
            nn::layers::standardize::_accumulate(device, static_cast<nn::layers::standardize::LayerForward<LAYER_SPEC>&>(layer), input);
        }
        evaluate(device, layer, input, output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::standardize::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, nn::layers::standardize::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::standardize::check_input_output<LAYER_SPEC, INPUT_SPEC, typename decltype(layer.output)::SPEC>);
        forward(device, layer, input, layer.output, buffer, rng, mode);
    }

    template<typename DEVICE, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, nn::layers::standardize::LayerBackward<LAYER_SPEC>& layer, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::standardize::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(nn::layers::standardize::check_input_output<LAYER_SPEC, D_INPUT_SPEC, D_OUTPUT_SPEC>);
        static_assert(LAYER_SPEC::INPUT_DIM == LAYER_SPEC::OUTPUT_DIM);
        using TI = typename DEVICE::index_t;
        constexpr TI OUTPUT_DIM = LAYER_SPEC::OUTPUT_DIM;
        constexpr TI BATCH_SIZE = D_OUTPUT_SPEC::ROWS;
        using T = typename LAYER_SPEC::TYPE_POLICY::DEFAULT;

        for(TI batch_i=0; batch_i < BATCH_SIZE; batch_i++){
            for(TI output_i = 0; output_i < OUTPUT_DIM; output_i++) {
                T d_output_value = get(d_output, batch_i, output_i);
                T precision = get(device, layer.precision.parameters, output_i);
                T d_input_value = d_output_value * precision;
                set(d_input, batch_i, output_i, d_input_value);
            }
        }
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::standardize::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn::layers::standardize::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        // this is a no-op as the standardize layer does not have trainable parameters
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn::layers::standardize::LayerGradient<LAYER_SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::standardize::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        backward_input(device, layer, d_output, d_input, buffer, mode);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void zero_gradient(DEVICE& device, nn::layers::standardize::LayerGradient<SPEC>& layer) {
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::layers::standardize::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer){
    }

    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::standardize::LayerBackward<SPEC>& l) { }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_forward_state(DEVICE& device, nn::layers::standardize::LayerGradient<SPEC>& l) {
        reset_forward_state(device, static_cast<nn::layers::standardize::LayerBackward<SPEC>&>(l));
        set_all(device, l.output, 0);
    }

    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::layers::standardize::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn::layers::standardize::LayerForward<SOURCE_SPEC>& source, nn::layers::standardize::LayerForward<TARGET_SPEC>& target){
        static_assert(nn::layers::standardize::check_compatibility<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, source.mean, target.mean);
        copy(source_device, target_device, source.precision, target.precision);
        target.age = source.age;
        copy(source_device, target_device, source.running_mean, target.running_mean);
        copy(source_device, target_device, source.running_std, target.running_std);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::layers::standardize::LayerGradient<SOURCE_SPEC>& source, nn::layers::standardize::LayerGradient<TARGET_SPEC>& target){
        static_assert(nn::layers::standardize::check_compatibility<SOURCE_SPEC, TARGET_SPEC>);
        copy(source_device, target_device, static_cast<const nn::layers::standardize::LayerForward<SOURCE_SPEC>&>(source), static_cast<nn::layers::standardize::LayerForward<TARGET_SPEC>&>(target));
        copy(source_device, target_device, source.output, target.output);
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const rl_tools::nn::layers::standardize::LayerForward<SPEC_1>& l1, const rl_tools::nn::layers::standardize::LayerForward<SPEC_2>& l2) {
        static_assert(nn::layers::standardize::check_compatibility<SPEC_1, SPEC_2>);
        using T = typename SPEC_1::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, l1.mean, l2.mean);
        acc += abs_diff(device, l1.precision, l2.precision);
        acc += math::abs(device.math, (T)l1.age - (T)l2.age);
        acc += abs_diff(device, l1.running_mean, l2.running_mean);
        acc += abs_diff(device, l1.running_std, l2.running_std);
        return acc;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const rl_tools::nn::layers::standardize::LayerGradient<SPEC_1>& l1, const rl_tools::nn::layers::standardize::LayerGradient<SPEC_2>& l2) {
        static_assert(nn::layers::standardize::check_compatibility<SPEC_1, SPEC_2>);
        return abs_diff(device, static_cast<const rl_tools::nn::layers::standardize::LayerForward<SPEC_1>&>(l1), static_cast<const rl_tools::nn::layers::standardize::LayerForward<SPEC_2>&>(l2));
    }
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT auto abs_diff(DEVICE& device, const rl_tools::nn::layers::standardize::State& s1, const rl_tools::nn::layers::standardize::State& s2) {
        return 0;
    }
    template<typename DEVICE, typename LAYER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto output(DEVICE& device, nn::layers::standardize::LayerGradient<LAYER_SPEC>& l){
        auto tensor_flat = to_tensor(device, l.output);
        auto tensor = view_memory<typename LAYER_SPEC::OUTPUT_SHAPE>(device, tensor_flat);
        return tensor;
    }
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const rl_tools::nn::layers::standardize::LayerForward<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        if constexpr(mode::is<MODE, nn::parameters::mode::ParametersOnly>){
            return false;
        }
        return is_nan(device, l.mean, mode) || is_nan(device, l.precision, mode) || is_nan(device, l.running_mean, mode) || is_nan(device, l.running_std, mode);
    }
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const rl_tools::nn::layers::standardize::LayerBackward<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        return is_nan(device, static_cast<const rl_tools::nn::layers::standardize::LayerForward<SPEC>&>(l), mode);
    }
    template <typename DEVICE, typename SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const rl_tools::nn::layers::standardize::LayerGradient<SPEC>& l, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        bool upstream_nan = is_nan(device, static_cast<const rl_tools::nn::layers::standardize::LayerBackward<SPEC>&>(l), mode);
        if constexpr(mode::is<MODE, nn::parameters::mode::ParametersOnly>){
            return upstream_nan;
        }
        return upstream_nan || is_nan(device, l.output, mode);
    }
    template<typename DEVICE, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, nn::layers::standardize::State& state, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        return false;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

// Tensor proxies
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const nn::layers::standardize::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::standardize::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        evaluate(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, const nn::layers::standardize::LayerForward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::standardize::State& state, Tensor<OUTPUT_SPEC>& output, nn::layers::standardize::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        evaluate(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::standardize::LayerBackward<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::standardize::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        forward(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::standardize::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn::layers::standardize::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        forward(device, layer, matrix_view_input, layer.output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void forward(DEVICE& device, nn::layers::standardize::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn::layers::standardize::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        forward(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_input(DEVICE& device, const nn::layers::standardize::LayerBackward<LAYER_SPEC>& layer, const Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::standardize::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        auto matrix_view_d_output = matrix_view(device, d_output);
        auto matrix_view_d_input = matrix_view(device, d_input);
        backward_input(device, layer, matrix_view_d_output, matrix_view_d_input, buffer, mode);
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward(DEVICE& device, nn::layers::standardize::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, nn::layers::standardize::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_d_output = matrix_view(device, d_output);
        backward(device, layer, matrix_view_input, matrix_view_d_output, buffer, mode);
    }

    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename MODE = mode::Default<>>
    RL_TOOLS_FUNCTION_PLACEMENT void backward_full(DEVICE& device, nn::layers::standardize::LayerGradient<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<D_OUTPUT_SPEC>& d_output, Tensor<D_INPUT_SPEC>& d_input, nn::layers::standardize::Buffer& buffer, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_d_output = matrix_view(device, d_output);
        auto matrix_view_d_input = matrix_view(device, d_input);
        backward_full(device, layer, matrix_view_input, matrix_view_d_output, matrix_view_d_input, buffer, mode);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT auto gradient_norm(DEVICE& device, const nn::layers::standardize::LayerGradient<SPEC>& layer) {
        return 0;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
