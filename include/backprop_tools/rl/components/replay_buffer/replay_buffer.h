#ifndef BACKPROP_TOOLS_RL_ALGORITHMS_OFF_POLICY_RUNNER
#define BACKPROP_TOOLS_RL_ALGORITHMS_OFF_POLICY_RUNNER
namespace backprop_tools::rl::components::replay_buffer{
    template<typename T_T, typename T_TI, T_TI T_OBSERVATION_DIM, T_TI T_ACTION_DIM, T_TI T_CAPACITY, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI OBSERVATION_DIM = T_OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = T_ACTION_DIM;
        static constexpr TI CAPACITY = T_CAPACITY;
        using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
    };

}

namespace backprop_tools::rl::components {
    template <typename T_SPEC>
    struct ReplayBuffer {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI CAPACITY = SPEC::CAPACITY;
        static constexpr TI DATA_COLS = SPEC::OBSERVATION_DIM + SPEC::ACTION_DIM + 1 + SPEC::OBSERVATION_DIM + 1 + 1;

        // mem
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, SPEC::CAPACITY, DATA_COLS>> data;
        TI position = 0;
        bool full = false;

        // views
        template<typename SPEC::TI DIM>
        using DATA_VIEW = typename decltype(data)::template VIEW<CAPACITY, DIM>;

        DATA_VIEW<SPEC::OBSERVATION_DIM> observations;
        DATA_VIEW<SPEC::ACTION_DIM> actions;
        DATA_VIEW<1> rewards;
        DATA_VIEW<SPEC::OBSERVATION_DIM> next_observations;
        DATA_VIEW<1> terminated;
        DATA_VIEW<1> truncated;
    };
}
namespace backprop_tools::rl::components::replay_buffer{
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


#endif