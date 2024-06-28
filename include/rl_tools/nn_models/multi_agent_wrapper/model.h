#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_MODEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_MULTI_AGENT_WRAPPER_MODEL_H

#include "../../nn/nn.h"
#include "../../nn/parameters/parameters.h"
#include "../../nn/optimizers/sgd/sgd.h"
#include "../../nn/optimizers/adam/adam.h"
#include "../../utils/generic/typing.h"
#include "../../containers.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::multi_agent_wrapper {
    template <typename T_T, typename T_TI, T_TI T_N_AGENTS, typename T_MODULE, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using MODULE = T_MODULE;
        static constexpr TI N_AGENTS = T_N_AGENTS;
        static constexpr TI INPUT_DIM = MODULE::INPUT_DIM * N_AGENTS;
        static constexpr TI OUTPUT_DIM = MODULE::OUTPUT_DIM * N_AGENTS;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
    };
    template <typename T_CAPABILITY, typename T_SPEC>
    struct CapabilitySpecification: T_SPEC, T_CAPABILITY{
        using CAPABILITY = T_CAPABILITY;

        using PARAMETER_TYPE = typename CAPABILITY::PARAMETER_TYPE;
    };

    template<typename T_SPEC, typename T_SPEC::TI T_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
    struct ModuleBuffersSpecification{
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
    };

    template<typename T_BUFFER_SPEC>
    struct ModuleBuffer{
        using BUFFER_SPEC = T_BUFFER_SPEC;
        using SPEC = typename BUFFER_SPEC::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI BATCH_SIZE = T_BUFFER_SPEC::BATCH_SIZE;
        static constexpr TI INNER_BATCH_SIZE = BATCH_SIZE * SPEC::N_AGENTS;

        using INPUT_BUFFER_SPEC = matrix::Specification<T, TI, BATCH_SIZE, SPEC::INPUT_DIM>;
        using INPUT_BUFFER_TYPE = typename BUFFER_SPEC::CONTAINER_TYPE_TAG::template type<INPUT_BUFFER_SPEC>;
        using OUTPUT_BUFFER_SPEC = matrix::Specification<T, TI, BATCH_SIZE, SPEC::OUTPUT_DIM>;
        using OUTPUT_BUFFER_TYPE = typename BUFFER_SPEC::CONTAINER_TYPE_TAG::template type<OUTPUT_BUFFER_SPEC>;
        INPUT_BUFFER_TYPE input, d_input;
        OUTPUT_BUFFER_TYPE output;

        typename SPEC::MODULE::template Buffer<INNER_BATCH_SIZE> buffer;
    };

    template<typename T_SPEC>
    struct ModuleForward{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using MODULE = typename SPEC::MODULE;
        static constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;
        MODULE content;
        template<TI BUFFER_BATCH_SIZE, typename T_CONTAINER_TYPE_TAG = typename T_SPEC::CONTAINER_TYPE_TAG>
        using Buffer = ModuleBuffer<ModuleBuffersSpecification<SPEC, BUFFER_BATCH_SIZE, T_CONTAINER_TYPE_TAG>>;
    };

    template<typename SPEC>
    struct ModuleBackward: public ModuleForward<SPEC>{
//        static constexpr typename SPEC::TI BATCH_SIZE = SPEC::BATCH_SIZE;
    };
    template<typename T_SPEC>
    struct ModuleGradient: public ModuleBackward<T_SPEC>{
        using TI = typename T_SPEC::TI;
        static constexpr TI BATCH_SIZE = T_SPEC::CAPABILITY::BATCH_SIZE/T_SPEC::N_AGENTS;
    };

    template <typename T_CAPABILITY, auto N_AGENTS>
    struct UpgradeCapabilityBatchSize: T_CAPABILITY{
        static constexpr auto BATCH_SIZE = T_CAPABILITY::BATCH_SIZE * N_AGENTS;
    };

    template<typename CAPABILITY, typename SPEC>
    using _Module =
            typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Forward,
                    ModuleForward<CapabilitySpecification<CAPABILITY, SPEC>>,
                    typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Backward,
                            ModuleBackward<CapabilitySpecification<UpgradeCapabilityBatchSize<CAPABILITY, SPEC::N_AGENTS>, SPEC>>,
                            typename utils::typing::conditional_t<CAPABILITY::TAG == nn::LayerCapability::Gradient,
                                    ModuleGradient<CapabilitySpecification<UpgradeCapabilityBatchSize<CAPABILITY, SPEC::N_AGENTS>, SPEC>>, void>>>;

    template<typename T_CAPABILITY, typename T_SPEC>
    struct Module: _Module<T_CAPABILITY, T_SPEC>{
        template <typename TT_CAPABILITY>
        using CHANGE_CAPABILITY = Module<TT_CAPABILITY, T_SPEC>;
    };

    template <typename T_SPEC>
    struct BindSpecification{
        template <typename CAPABILITY>
        using NeuralNetwork = nn_models::multi_agent_wrapper::Module<CAPABILITY, T_SPEC>;
    };


}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
