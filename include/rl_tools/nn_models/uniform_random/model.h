#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_UNIFORM_RANDOM_MODEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_UNIFORM_RANDOM_MODEL_H
RL_TOOLS_NAMESPACE_WRAPPER_START
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::uniform_random{
    enum class Range{
        MINUS_ONE_TO_ONE,
        ZERO_TO_ONE
    };
    template <typename T_T, typename T_TI, T_TI T_INPUT_DIM, T_TI T_OUTPUT_DIM, Range T_RANGE>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        static constexpr T_TI INPUT_DIM = T_INPUT_DIM;
        static constexpr T_TI OUTPUT_DIM = T_OUTPUT_DIM;
        static constexpr Range RANGE = T_RANGE;
    };
    struct Buffer{};
}
RL_TOOLS_NAMESPACE_WRAPPER_END
namespace rl_tools::nn_models{
    template <typename T_SPEC>
    struct UniformRandom{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI INPUT_DIM = SPEC::INPUT_DIM;
        static constexpr TI OUTPUT_DIM = SPEC::OUTPUT_DIM;

        template <TI BATCH_SIZE=1>
        using Buffer = typename rl_tools::nn_models::uniform_random::Buffer;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
