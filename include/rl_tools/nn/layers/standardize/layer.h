#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_STANDARDIZE_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_STANDARDIZE_LAYER_H
#include "../../../utils/generic/typing.h"
#include "../../../containers.h"

#include "../../../nn/nn.h"
#include "../../../nn/parameters/parameters.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::standardize {
    template <typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output_f(){
        static_assert(INPUT_SPEC::COLS == LAYER_SPEC::INPUT_DIM);
        static_assert(OUTPUT_SPEC::COLS == LAYER_SPEC::OUTPUT_DIM);
        static_assert(LAYER_SPEC::INPUT_DIM == LAYER_SPEC::OUTPUT_DIM);
        static_assert(INPUT_SPEC::ROWS == OUTPUT_SPEC::ROWS);
        //                INPUT_SPEC::ROWS <= OUTPUT_SPEC::ROWS && // todo: could be relaxed to not fill the full output
        static_assert(!LAYER_SPEC::ENFORCE_FLOATING_POINT_TYPE || utils::typing::is_same_v<typename LAYER_SPEC::T, typename INPUT_SPEC::T>);
        static_assert(!LAYER_SPEC::ENFORCE_FLOATING_POINT_TYPE || utils::typing::is_same_v<typename INPUT_SPEC::T, typename OUTPUT_SPEC::T>);
        return true;
    }
    template <typename LAYER_SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    constexpr bool check_input_output = check_input_output_f<LAYER_SPEC, INPUT_SPEC, OUTPUT_SPEC>();

    template <typename LAYER_SPEC_1, typename LAYER_SPEC_2>
    constexpr bool check_compatibility_f(){
        static_assert(LAYER_SPEC_1::INPUT_DIM == LAYER_SPEC_2::INPUT_DIM);
        static_assert(LAYER_SPEC_1::OUTPUT_DIM == LAYER_SPEC_2::OUTPUT_DIM);
        static_assert(LAYER_SPEC_1::OUTPUT_DIM == LAYER_SPEC_1::OUTPUT_DIM);
        return true;
    }

    template <typename LAYER_SPEC_1, typename LAYER_SPEC_2>
    constexpr bool check_compatibility = check_compatibility_f<LAYER_SPEC_1, LAYER_SPEC_2>();
    template<typename T_T, typename T_TI, T_TI T_DIM, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag, bool T_ENFORCE_FLOATING_POINT_TYPE=true, typename T_MEMORY_LAYOUT = matrix::layouts::RowMajorAlignmentOptimized<T_TI>>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        static constexpr auto INPUT_DIM = T_DIM;
        static constexpr auto OUTPUT_DIM = T_DIM;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        static constexpr bool ENFORCE_FLOATING_POINT_TYPE = T_ENFORCE_FLOATING_POINT_TYPE;
        using MEMORY_LAYOUT = T_MEMORY_LAYOUT;
        // Summary
        static constexpr auto NUM_WEIGHTS = 0; //zero trainable parameters (the point is to not learn the mean and std by gradient descent, otherwise it would just be a normal layer)
    };
    template <typename T_CAPABILITY, typename T_SPEC>
    struct CapabilitySpecification: T_SPEC{
        using CAPABILITY = T_CAPABILITY;
        using PARAMETER_TYPE = typename CAPABILITY::PARAMETER_TYPE;
    };
    struct Buffer{};
    template<typename T_SPEC>
    struct LayerForward {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using CONTAINER_TYPE_TAG = typename SPEC::CONTAINER_TYPE_TAG;
        static constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
        static constexpr TI NUM_WEIGHTS = SPEC::NUM_WEIGHTS;
        using STATISTICS_CONTAINER_SPEC = matrix::Specification<T, TI, 1, INPUT_DIM, typename SPEC::MEMORY_LAYOUT>;
        using STATISTICS_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<STATISTICS_CONTAINER_SPEC>;
        using STATISTICS_PARAMETER_SPEC = nn::parameters::Plain::spec<STATISTICS_CONTAINER_TYPE, nn::parameters::groups::Normal, nn::parameters::categories::Constant>; // Constant from the view of a forward or backward pass
        typename nn::parameters::Plain::template instance<STATISTICS_PARAMETER_SPEC> mean, precision; // precision = 1/std
        template<TI BUFFER_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG = typename T_SPEC::CONTAINER_TYPE_TAG>
        using Buffer = standardize::Buffer;
    };
    template<typename SPEC>
    struct LayerBackward: public LayerForward<SPEC> {
    };
    template<typename SPEC>
    struct LayerGradient: public LayerBackward<SPEC> {
        // This layer supports backpropagation wrt its input but including its weights (for this it stores the intermediate outputs in addition to the pre_activations because they determine the gradient wrt the weights of the following layer)
        using OUTPUT_CONTAINER_SPEC = matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::CAPABILITY::BATCH_SIZE, SPEC::OUTPUT_DIM, typename SPEC::MEMORY_LAYOUT>;
        using OUTPUT_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<OUTPUT_CONTAINER_SPEC>;
        OUTPUT_CONTAINER_TYPE output;
    };
    template<typename CAPABILITY, typename SPEC>
    using Layer =
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward,
            LayerForward<CapabilitySpecification<CAPABILITY, SPEC>>,
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward,
            LayerBackward<CapabilitySpecification<CAPABILITY, SPEC>>,
        typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient,
            LayerGradient<CapabilitySpecification<CAPABILITY, SPEC>>, void>>>;

    template <typename T_SPEC>
    struct BindSpecification{
        template <typename CAPABILITY>
        using Layer = nn::layers::standardize::Layer<CAPABILITY, T_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
