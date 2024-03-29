#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_GRU_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_GRU_LAYER_H
#include "../../../nn/activation_functions.h"
#include "../../../nn/parameters/parameters.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::gru {
    template<typename T_T, typename T_TI, T_TI T_SEQUENCE_LENGTH, T_TI T_INPUT_DIM, T_TI T_HIDDEN_DIM, typename T_PARAMETER_TYPE, T_TI T_BATCH_SIZE=1, typename T_PARAMETER_GROUP=parameters::groups::Normal, typename T_CONTAINER_TYPE_TAG = TensorDynamicTag, bool T_ENFORCE_FLOATING_POINT_TYPE=true, typename T_MEMORY_LAYOUT = matrix::layouts::RowMajorAlignmentOptimized<T_TI>>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        static constexpr T_TI SEQUENCE_LENGTH = T_SEQUENCE_LENGTH;
        static constexpr T_TI INPUT_DIM = T_INPUT_DIM;
        static constexpr T_TI HIDDEN_DIM = T_HIDDEN_DIM;
        using PARAMETER_TYPE = T_PARAMETER_TYPE;
        using PARAMETER_GROUP = T_PARAMETER_GROUP;
        static constexpr auto BATCH_SIZE = T_BATCH_SIZE;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        static constexpr bool ENFORCE_FLOATING_POINT_TYPE = T_ENFORCE_FLOATING_POINT_TYPE;
        using MEMORY_LAYOUT = T_MEMORY_LAYOUT;
        // Summary
//        static constexpr auto NUM_WEIGHTS = HIDDEN_DIM * INPUT_DIM + HIDDEN_DIM; // todo
    };

    template<typename T_SPEC>
    struct Layer{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using CONTAINER_TYPE_TAG = typename SPEC::CONTAINER_TYPE_TAG;
        static constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr TI HIDDEN_DIM = SPEC::HIDDEN_DIM;
        using WEIGHTS_INPUT_CONTAINER_SHAPE = tensor::Shape<TI, 3*HIDDEN_DIM, INPUT_DIM>;
        using WEIGHTS_INPUT_CONTAINER_SPEC = tensor::Specification<T, TI, WEIGHTS_INPUT_CONTAINER_SHAPE>;
        using WEIGHTS_INPUT_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<WEIGHTS_INPUT_CONTAINER_SPEC>;
        using WEIGHTS_INPUT_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template spec<WEIGHTS_INPUT_CONTAINER_TYPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights>;
        typename SPEC::PARAMETER_TYPE::template instance<WEIGHTS_INPUT_PARAMETER_SPEC> weights_input;
        typename decltype(weights_input.parameters)::template VIEW_RANGE<tensor::ViewSpec<0, HIDDEN_DIM>> W_ir, W_iz, W_in;

        using BIASES_INPUT_CONTAINER_SHAPE = tensor::Shape<TI, 3*HIDDEN_DIM>;
        using BIASES_INPUT_CONTAINER_SPEC = tensor::Specification<T, TI, BIASES_INPUT_CONTAINER_SHAPE>;
        using BIASES_INPUT_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<BIASES_INPUT_CONTAINER_SPEC>;
        using BIASES_INPUT_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template spec<BIASES_INPUT_CONTAINER_TYPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Biases>;
        typename SPEC::PARAMETER_TYPE::template instance<BIASES_INPUT_PARAMETER_SPEC> biases_input;
        typename decltype(biases_input.parameters)::template VIEW_RANGE<tensor::ViewSpec<0, HIDDEN_DIM>> b_ir, b_iz, b_in;

        using WEIGHTS_HIDDEN_CONTAINER_SHAPE = tensor::Shape<TI, 3*HIDDEN_DIM, HIDDEN_DIM>;
        using WEIGHTS_HIDDEN_CONTAINER_SPEC = tensor::Specification<T, TI, WEIGHTS_HIDDEN_CONTAINER_SHAPE>;
        using WEIGHTS_HIDDEN_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<WEIGHTS_HIDDEN_CONTAINER_SPEC>;
        using WEIGHTS_HIDDEN_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template spec<WEIGHTS_HIDDEN_CONTAINER_TYPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Weights>;
        typename SPEC::PARAMETER_TYPE::template instance<WEIGHTS_HIDDEN_PARAMETER_SPEC> weights_hidden;
        typename decltype(weights_hidden.parameters)::template VIEW_RANGE<tensor::ViewSpec<0, HIDDEN_DIM>> W_hr, W_hz, W_hn;

        using BIASES_HIDDEN_CONTAINER_SHAPE = tensor::Shape<TI, 3*HIDDEN_DIM>;
        using BIASES_HIDDEN_CONTAINER_SPEC = tensor::Specification<T, TI, BIASES_HIDDEN_CONTAINER_SHAPE>;
        using BIASES_HIDDEN_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<BIASES_HIDDEN_CONTAINER_SPEC>;
        using BIASES_HIDDEN_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template spec<BIASES_HIDDEN_CONTAINER_TYPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Biases>;
        typename SPEC::PARAMETER_TYPE::template instance<BIASES_HIDDEN_PARAMETER_SPEC> biases_hidden;
        typename decltype(biases_hidden.parameters)::template VIEW_RANGE<tensor::ViewSpec<0, HIDDEN_DIM>> b_hr, b_hz, b_hn;

        using INITIAL_HIDDEN_STATE_CONTAINER_SHAPE = tensor::Shape<TI, HIDDEN_DIM>;
        using INITIAL_HIDDEN_STATE_CONTAINER_SPEC = tensor::Specification<T, TI, INITIAL_HIDDEN_STATE_CONTAINER_SHAPE>;
        using INITIAL_HIDDEN_STATE_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<INITIAL_HIDDEN_STATE_CONTAINER_SPEC>;
        using INITIAL_HIDDEN_STATE_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template spec<INITIAL_HIDDEN_STATE_CONTAINER_TYPE, typename SPEC::PARAMETER_GROUP, nn::parameters::categories::Biases>;
        typename SPEC::PARAMETER_TYPE::template instance<INITIAL_HIDDEN_STATE_PARAMETER_SPEC> initial_hidden_state;
    };

    template<typename T_SPEC>
    struct LayerBackward: Layer<T_SPEC>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using PRE_ACTIVATION_SPEC = tensor::Specification<T, TI, tensor::Shape<TI, SPEC::SEQUENCE_LENGTH, SPEC::BATCH_SIZE, 3*SPEC::HIDDEN_DIM>>;
        Tensor<PRE_ACTIVATION_SPEC> pre_activation;
        using OUTPUT_SPEC = tensor::Specification<T, TI, tensor::Shape<TI, SPEC::SEQUENCE_LENGTH, SPEC::BATCH_SIZE, 3*SPEC::HIDDEN_DIM>>;
        Tensor<OUTPUT_SPEC> output;
    };



    template <typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    bool constexpr check_input_output = length(typename INPUT_SPEC::SHAPE{}) == 3 && length(typename OUTPUT_SPEC::SHAPE{}) == 3 &&
            get<0>(typename INPUT_SPEC::SHAPE{}) == get<0>(typename OUTPUT_SPEC::SHAPE{}) &&
            get<1>(typename INPUT_SPEC::SHAPE{}) == get<1>(typename OUTPUT_SPEC::SHAPE{}) &&
            get<2>(typename INPUT_SPEC::SHAPE{}) == SPEC::INPUT_DIM && get<2>(typename OUTPUT_SPEC::SHAPE{}) == SPEC::HIDDEN_DIM;
}

#endif