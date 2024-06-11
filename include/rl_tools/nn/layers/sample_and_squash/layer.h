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
namespace rl_tools{
    namespace nn::layers::sample_and_squash{
        namespace mode{
            template <typename BASE_MODE>
            struct Sample{
                // this mode uses the noise from the Buffer for debugging / no-side-effect inference
            };
            template <typename BASE_MODE>
            struct ExternalNoise{
                // this mode uses the noise from the Buffer for debugging / no-side-effect inference
            };
        }
        template <typename T>
        struct DefaultParameters{
            static constexpr T LOG_STD_LOWER_BOUND = -20;
            static constexpr T LOG_STD_UPPER_BOUND = 2;
            static constexpr T LOG_PROBABILITY_EPSILON = 1e-6;
            static constexpr bool ADAPTIVE_ALPHA = true;
            static constexpr T ALPHA = 1.0;
            static constexpr T TARGET_ENTROPY = -1;
        };
        template <typename T_TI, T_TI T_BATCH_SIZE, typename T_SPEC>
        struct BufferSpecification {
            using TI = T_TI;
            static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
            using SPEC = T_SPEC;
        };

        template <typename BUFFER_SPEC>
        struct Buffer{
            using SPEC = typename BUFFER_SPEC::SPEC;
            using NOISE_CONTAINER_SPEC = matrix::Specification<typename SPEC::T, typename SPEC::TI, BUFFER_SPEC::BATCH_SIZE, SPEC::DIM>;
            using NOISE_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<NOISE_CONTAINER_SPEC>;
            NOISE_CONTAINER_TYPE noise;
        };
        template<typename T_T, typename T_TI, T_TI T_DIM, typename T_PARAMETERS = DefaultParameters<T_T>, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
        struct Specification {
            using T = T_T;
            using TI = T_TI;
            static constexpr auto DIM = T_DIM;
            using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
            using PARAMETERS = T_PARAMETERS;
        };
        template <typename SPEC>
        struct LayerForward{
            using CONTAINER_TYPE_TAG = typename SPEC::CONTAINER_TYPE_TAG;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            static constexpr TI DIM = SPEC::DIM;
            static constexpr TI INPUT_DIM = 2*DIM; // mean and std
            static constexpr TI OUTPUT_DIM = DIM;
            template<TI BUFFER_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG = typename SPEC::CONTAINER_TYPE_TAG>
            using Buffer = sample_and_squash::Buffer<sample_and_squash::BufferSpecification<TI, BUFFER_BATCH_SIZE, SPEC>>;
        };
        template<typename SPEC>
        struct LayerBackward: public LayerForward<SPEC> {
            static constexpr typename SPEC::TI BATCH_SIZE = SPEC::BATCH_SIZE;
            // This layer supports backpropagation wrt its input but not its weights (for this it stores the intermediate pre_activations)
            using PRE_ACTIVATIONS_CONTAINER_SPEC = matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::BATCH_SIZE, SPEC::DIM>;
            using PRE_ACTIVATIONS_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<PRE_ACTIVATIONS_CONTAINER_SPEC>;
            PRE_ACTIVATIONS_CONTAINER_TYPE pre_squashing, noise;
        };
        template<typename SPEC>
        struct LayerGradient: public LayerBackward<SPEC> {
            using LOG_PROBABILITIES_CONTAINER_SPEC = matrix::Specification<typename SPEC::T, typename SPEC::TI, 1, SPEC::BATCH_SIZE>;
            using LOG_PROBABILITIES_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<LOG_PROBABILITIES_CONTAINER_SPEC>;
            LOG_PROBABILITIES_CONTAINER_TYPE log_probabilities;
            using OUTPUT_CONTAINER_SPEC = matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::BATCH_SIZE, SPEC::DIM>;
            using OUTPUT_CONTAINER_TYPE = typename SPEC::CONTAINER_TYPE_TAG::template type<OUTPUT_CONTAINER_SPEC>;
            OUTPUT_CONTAINER_TYPE output;
            using ALPHA_CONTAINER = typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<typename SPEC::T, typename SPEC::TI, 1, 1>>;
            using ALPHA_PARAMETER_SPEC = typename SPEC::PARAMETER_TYPE::template spec<ALPHA_CONTAINER, nn::parameters::categories::Biases, nn::parameters::groups::Normal>;
            typename SPEC::PARAMETER_TYPE::template instance<ALPHA_PARAMETER_SPEC> log_alpha;
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
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif