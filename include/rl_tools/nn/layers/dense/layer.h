#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_DENSE_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_DENSE_LAYER_H
#include "../../../nn/activation_functions.h"
#include "../../../utils/generic/typing.h"
#include "../../../containers.h"

#include "../../../nn/parameters/parameters.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::dense {
    template <typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output_f(){
        static_assert(INPUT_SPEC::COLS == LAYER_SPEC::INPUT_DIM);
        static_assert(INPUT_SPEC::ROWS == OUTPUT_SPEC::ROWS);
        //                INPUT_SPEC::ROWS <= OUTPUT_SPEC::ROWS && // todo: could be relaxed to not fill the full output
        static_assert(OUTPUT_SPEC::COLS == LAYER_SPEC::OUTPUT_DIM);
        static_assert(!LAYER_SPEC::ENFORCE_FLOATING_POINT_TYPE || utils::typing::is_same_v<typename LAYER_SPEC::T, typename INPUT_SPEC::T>);
        static_assert(!LAYER_SPEC::ENFORCE_FLOATING_POINT_TYPE || utils::typing::is_same_v<typename INPUT_SPEC::T, typename OUTPUT_SPEC::T>);
        return true;
    }
    template <typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output = check_input_output_f<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>();
    template<typename T_T, typename T_TI, T_TI T_INPUT_DIM, T_TI T_OUTPUT_DIM, nn::activation_functions::ActivationFunction T_ACTIVATION_FUNCTION, typename T_PARAMETER_TYPE, T_TI T_BATCH_SIZE=1, typename T_PARAMETER_GROUP=parameters::groups::Normal, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag, bool T_ENFORCE_FLOATING_POINT_TYPE=true, typename T_MEMORY_LAYOUT = matrix::layouts::RowMajorAlignmentOptimized<T_TI>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        static constexpr auto INPUT_DIM = T_INPUT_DIM;
        static constexpr auto OUTPUT_DIM = T_OUTPUT_DIM;
        static constexpr nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION = T_ACTIVATION_FUNCTION;
        using PARAMETER_TYPE = T_PARAMETER_TYPE;
        using PARAMETER_GROUP = T_PARAMETER_GROUP;
        static constexpr auto BATCH_SIZE = T_BATCH_SIZE;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        static constexpr bool ENFORCE_FLOATING_POINT_TYPE = T_ENFORCE_FLOATING_POINT_TYPE;
        using MEMORY_LAYOUT = T_MEMORY_LAYOUT;
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
        using CONTAINER_TYPE_TAG = typename SPEC::CONTAINER_TYPE_TAG;
        static constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
        static constexpr TI NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        using WEIGHTS_CONTAINER_SPEC = matrix::Specification<T, TI, OUTPUT_DIM, INPUT_DIM, typename SPEC::MEMORY_LAYOUT>;
        using WEIGHTS_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<WEIGHTS_CONTAINER_SPEC>;
        using WEIGHTS_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template spec<WEIGHTS_CONTAINER_TYPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights>;
        typename SPEC::PARAMETER_TYPE::template instance<WEIGHTS_PARAMETER_SPEC> weights;

        using BIASES_CONTAINER_SPEC = matrix::Specification<T, TI, 1, OUTPUT_DIM, typename SPEC::MEMORY_LAYOUT>;
        using BIASES_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<BIASES_CONTAINER_SPEC>;
        using BIASES_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template spec<BIASES_CONTAINER_TYPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Biases>;
        typename SPEC::PARAMETER_TYPE::template instance<BIASES_PARAMETER_SPEC> biases;
    };
    template<typename SPEC>
    struct LayerBackward : public Layer<SPEC> {
        static constexpr typename SPEC::TI BATCH_SIZE = SPEC::BATCH_SIZE;
        // This layer supports backpropagation wrt its input but not its weights (for this it stores the intermediate pre_activations)
        using PRE_ACTIVATIONS_CONTAINER_SPEC = matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::BATCH_SIZE, SPEC::OUTPUT_DIM>;
        using PRE_ACTIVATIONS_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<PRE_ACTIVATIONS_CONTAINER_SPEC>;
        PRE_ACTIVATIONS_CONTAINER_TYPE pre_activations;
    };
    template<typename SPEC>
    struct LayerBackwardGradient : public LayerBackward<SPEC> {
        // This layer supports backpropagation wrt its input but including its weights (for this it stores the intermediate outputs in addition to the pre_activations because they determine the gradient wrt the weights of the following layer)
        using OUTPUT_CONTAINER_SPEC = matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::BATCH_SIZE, SPEC::OUTPUT_DIM, typename SPEC::MEMORY_LAYOUT>;
        using OUTPUT_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<OUTPUT_CONTAINER_SPEC>;
        OUTPUT_CONTAINER_TYPE output;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif