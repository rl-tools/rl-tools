#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_RANDOM_UNIFORM_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_RANDOM_UNIFORM_OPERATIONS_GENERIC_H
#include "model.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{

    template <typename DEVICE, typename SPEC>
    void malloc(const DEVICE& device, nn_models::RandomUniform<SPEC>){ }
    template <typename DEVICE, typename SPEC>
    void free(const DEVICE& device, nn_models::RandomUniform<SPEC>){ }
    template <typename DEVICE>
    void malloc(const DEVICE& device, nn_models::random_uniform::State){ }
    template <typename DEVICE>
    void free(const DEVICE& device, nn_models::random_uniform::State){ }
    template <typename DEVICE>
    void malloc(const DEVICE& device, nn_models::random_uniform::Buffer){ }
    template <typename DEVICE>
    void free(const DEVICE& device, nn_models::random_uniform::Buffer){ }

    template <typename DEVICE, typename SPEC, typename RNG, typename MODE = mode::Default<>>
    void reset(DEVICE& device, const nn_models::RandomUniform<SPEC>& model, nn_models::random_uniform::State& state, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){ }

    template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(const DEVICE& device, nn_models::RandomUniform<SPEC>, Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn_models::random_uniform::Buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        static_assert(SPEC::OUTPUT_DIM == OUTPUT_SPEC::COLS, "Output dimension mismatch");
        static_assert(SPEC::INPUT_DIM == INPUT_SPEC::COLS, "Input dimension mismatch");
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI row_i = 0; row_i < OUTPUT_SPEC::ROWS; row_i++){
            for(TI col_i = 0; col_i < OUTPUT_SPEC::COLS; col_i++){
                T value = 0;
                if(SPEC::RANGE == nn_models::random_uniform::Range::MINUS_ONE_TO_ONE){
                    value = random::uniform_real_distribution(device.random, (T)-1, (T)1, rng);
                }else if(SPEC::RANGE == nn_models::random_uniform::Range::ZERO_TO_ONE){
                    value = random::uniform_real_distribution(device.random, (T)0, (T)1, rng);
                }
                set(output, row_i, col_i, value);
            }
        }
    }
    template<typename DEVICE, typename MODULE_SPEC, typename MODE>
    bool is_nan(DEVICE& device, nn_models::RandomUniform<MODULE_SPEC>& model, const Mode<MODE>& mode = Mode<mode::Default<>>{}){
        return false;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

// Tensor proxies
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate(DEVICE& device, const nn_models::RandomUniform<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, nn_models::random_uniform::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        evaluate(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
    template<typename DEVICE, typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE = mode::Default<>>
    void evaluate_step(DEVICE& device, const nn_models::RandomUniform<LAYER_SPEC>& layer, const Tensor<INPUT_SPEC>& input, nn_models::random_uniform::State&, Tensor<OUTPUT_SPEC>& output, nn_models::random_uniform::Buffer& buffer, RNG& rng, const Mode<MODE>& mode = Mode<mode::Default<>>{}) {
        auto matrix_view_input = matrix_view(device, input);
        auto matrix_view_output = matrix_view(device, output);
        evaluate(device, layer, matrix_view_input, matrix_view_output, buffer, rng, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
