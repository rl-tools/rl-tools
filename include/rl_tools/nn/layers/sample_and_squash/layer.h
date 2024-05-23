#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_SAMPLE_AND_SQUASH_LAYER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_SAMPLE_AND_SQUASH_LAYER_H
#include "../../../nn/activation_functions.h"
#include "../../../utils/generic/typing.h"
#include "../../../containers.h"
#include "../../../nn/parameters/parameters.h"
#include "../dense/layer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::sample_and_squash {
    template <typename T>
    struct DefaultParameters{
        static constexpr T LOG_STD_LOWER_BOUND = -20;
        static constexpr T LOG_STD_UPPER_BOUND = 2;
    };
    struct Buffer{};
    template<typename T_T, typename T_TI, T_TI T_DIM, typename T_PARAMETERS, nn::activation_functions::ActivationFunction T_ACTIVATION_FUNCTION=nn::activation_functions::ActivationFunction::TANH, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        static constexpr auto DIM = T_DIM;
        using PARAMETERS = T_PARAMETERS;
        static constexpr nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION = T_ACTIVATION_FUNCTION;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
    };
    template <typename SPEC>
    struct LayerForward{
        using CONTAINER_TYPE_TAG = typename SPEC::CONTAINER_TYPE_TAG;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI DIM = SPEC::DIM;
        static constexpr TI INPUT_DIM = 2*DIM; // mean and std
        static constexpr TI OUTPUT_DIM = DIM;
        template<TI BUFFER_BATCH_SIZE = SPEC::BATCH_SIZE, typename T_CONTAINER_TYPE_TAG = typename SPEC::CONTAINER_TYPE_TAG>
        using Buffer = sample_and_squash::Buffer;
    };
    template<typename SPEC>
    struct LayerBackward: public LayerForward<SPEC> {
        static constexpr typename SPEC::TI BATCH_SIZE = SPEC::BATCH_SIZE;
        // This layer supports backpropagation wrt its input but not its weights (for this it stores the intermediate pre_activations)
        using PRE_ACTIVATIONS_CONTAINER_SPEC = matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::BATCH_SIZE, SPEC::DIM>;
        using PRE_ACTIVATIONS_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<PRE_ACTIVATIONS_CONTAINER_SPEC>;
        PRE_ACTIVATIONS_CONTAINER_TYPE noise;
    };
    template<typename SPEC>
    struct LayerGradient: public LayerBackward<SPEC> {
        // This layer supports backpropagation wrt its input but including its weights (for this it stores the intermediate outputs in addition to the pre_activations because they determine the gradient wrt the weights of the following layer)
        using OUTPUT_CONTAINER_SPEC = matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::BATCH_SIZE, SPEC::DIM>;
        using OUTPUT_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<OUTPUT_CONTAINER_SPEC>;
        OUTPUT_CONTAINER_TYPE output;
    };
    template<typename CAPABILITY, typename SPEC>
    using Layer =
    typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward,
            LayerForward<layers::dense::CapabilitySpecification<CAPABILITY, SPEC>>,
    typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward,
            LayerBackward<layers::dense::CapabilitySpecification<CAPABILITY, SPEC>>,
    typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient,
            LayerGradient<layers::dense::CapabilitySpecification<CAPABILITY, SPEC>>, void>>>;

    template <typename T_SPEC>
    struct BindSpecification{
        template <typename CAPABILITY>
        using Layer = nn::layers::sample_and_squash::Layer<CAPABILITY, T_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif