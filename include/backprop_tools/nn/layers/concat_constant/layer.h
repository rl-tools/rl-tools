#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_CONCAT_CONSTANT_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_CONCAT_CONSTANT_LAYER_H
#include "../../../utils/generic/typing.h"
#include "../../../containers.h"


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::concat_constant {
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
    template<typename T_T, typename T_TI, T_TI T_INPUT_DIM, T_TI T_CONSTANT_DIM, typename T_PARAMETER_TYPE, T_TI T_BATCH_SIZE=1, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag, bool T_ENFORCE_FLOATING_POINT_TYPE=true, typename T_MEMORY_LAYOUT = matrix::layouts::RowMajorAlignmentOptimized<T_TI>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        static constexpr auto INPUT_DIM = T_INPUT_DIM;
        static constexpr auto CONSTANT_DIM = T_CONSTANT_DIM;
        static constexpr auto OUTPUT_DIM = INPUT_DIM + CONSTANT_DIM;
        using PARAMETER_TYPE = T_PARAMETER_TYPE;
        static constexpr auto BATCH_SIZE = T_BATCH_SIZE;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        static constexpr bool ENFORCE_FLOATING_POINT_TYPE = T_ENFORCE_FLOATING_POINT_TYPE;
        using MEMORY_LAYOUT = T_MEMORY_LAYOUT;
        // Summary
        static constexpr auto NUM_WEIGHTS = CONSTANT_DIM;
    };

    template<typename T_SPEC>
    struct Layer {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr TI CONSTANT_DIM = SPEC::CONSTANT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
        static constexpr TI NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        using WEIGHTS_CONTAINER_SPEC = matrix::Specification<T, TI, 1, CONSTANT_DIM, typename SPEC::MEMORY_LAYOUT>;
        using WEIGHTS_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<WEIGHTS_CONTAINER_SPEC>;
        typename SPEC::PARAMETER_TYPE::template instance<WEIGHTS_CONTAINER_TYPE> constants;
    };
    template<typename SPEC>
    struct LayerBackward : public Layer<SPEC> {
        static constexpr typename SPEC::TI BATCH_SIZE = SPEC::BATCH_SIZE;
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