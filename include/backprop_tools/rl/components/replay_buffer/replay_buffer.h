#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_REPLAY_BUFFER_REPLAY_BUFFER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_REPLAY_BUFFER_REPLAY_BUFFER_H
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components::replay_buffer{
    template<typename T_T, typename T_TI, T_TI T_OBSERVATION_DIM, T_TI T_OBSERVATION_DIM_PRIVILEGED, bool T_ASYMMETRIC_OBSERVATIONS, T_TI T_ACTION_DIM, T_TI T_CAPACITY, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI OBSERVATION_DIM = T_OBSERVATION_DIM;
        static constexpr bool ASYMMETRIC_OBSERVATIONS = T_ASYMMETRIC_OBSERVATIONS && T_OBSERVATION_DIM_PRIVILEGED > 0;
        static_assert(ASYMMETRIC_OBSERVATIONS == T_ASYMMETRIC_OBSERVATIONS, "ASYMMETRIC_OBSERVATIONS set but not available in the environment");
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ASYMMETRIC_OBSERVATIONS ? T_OBSERVATION_DIM_PRIVILEGED : OBSERVATION_DIM;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED_ACTUAL = ASYMMETRIC_OBSERVATIONS ? T_OBSERVATION_DIM_PRIVILEGED : 0;
        static constexpr TI ACTION_DIM = T_ACTION_DIM;
        static constexpr TI CAPACITY = T_CAPACITY;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
    };

    template<typename T_ENVIRONMENT, typename T_BASE_SPEC>
    struct SpecificationWithStates: T_BASE_SPEC{
        using BASE_SPEC = T_BASE_SPEC;
        using ENVIRONMENT = T_ENVIRONMENT;
    };

}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components {
    template <typename T_SPEC>
    struct ReplayBuffer {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI CAPACITY = SPEC::CAPACITY;
        static constexpr TI DATA_COLS = SPEC::OBSERVATION_DIM + SPEC::OBSERVATION_DIM_PRIVILEGED_ACTUAL + SPEC::ACTION_DIM + 1 + SPEC::OBSERVATION_DIM + SPEC::OBSERVATION_DIM_PRIVILEGED_ACTUAL + 1 + 1;

        // mem
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, SPEC::CAPACITY, DATA_COLS>> data;
        TI position = 0;
        bool full = false;

        // views
        template<typename SPEC::TI DIM>
        using DATA_VIEW = typename decltype(data)::template VIEW<CAPACITY, DIM>;

        DATA_VIEW<SPEC::OBSERVATION_DIM> observations;
        DATA_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED> observations_privileged;
        DATA_VIEW<SPEC::ACTION_DIM> actions;
        DATA_VIEW<1> rewards;
        DATA_VIEW<SPEC::OBSERVATION_DIM> next_observations;
        DATA_VIEW<SPEC::OBSERVATION_DIM_PRIVILEGED> next_observations_privileged;
        DATA_VIEW<1> terminated;
        DATA_VIEW<1> truncated;
    };

    template <typename T_SPEC>
    struct ReplayBufferWithStates: ReplayBuffer<typename T_SPEC::BASE_SPEC> {
        // mem
        using SPEC = typename T_SPEC::BASE_SPEC;
        using ENVIRONMENT = typename T_SPEC::ENVIRONMENT;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<typename ENVIRONMENT::State, TI, SPEC::CAPACITY, 1>> states;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<typename ENVIRONMENT::State, TI, SPEC::CAPACITY, 1>> next_states;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components::replay_buffer{
    template <typename T_SPEC, auto T_SIZE>
    struct SetSpecification{
        static constexpr auto SIZE = T_SIZE;
    };
    template <typename T_SET_SPEC>
    struct Set{
        using SET_SPEC = T_SET_SPEC;
        using SPEC = typename SET_SPEC::SPEC;
        ReplayBuffer<SPEC> replay_buffers[SET_SPEC::SIZE];
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif