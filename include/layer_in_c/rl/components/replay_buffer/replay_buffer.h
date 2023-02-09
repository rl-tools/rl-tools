#ifndef LAYER_IN_C_RL_ALGORITHMS_OFF_POLICY_RUNNER
#define LAYER_IN_C_RL_ALGORITHMS_OFF_POLICY_RUNNER
namespace layer_in_c::rl::components::replay_buffer{
    template<typename T_T, typename T_TI, T_TI T_OBSERVATION_DIM, T_TI T_ACTION_DIM, T_TI T_CAPACITY>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        static constexpr TI OBSERVATION_DIM = T_OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = T_ACTION_DIM;
        static constexpr TI CAPACITY = T_CAPACITY;
    };

}
namespace layer_in_c::rl::components {
    template <typename T_SPEC>
    struct ReplayBuffer {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr typename SPEC::TI CAPACITY = SPEC::CAPACITY;
        Matrix<matrix::Specification<T, TI, SPEC::CAPACITY, SPEC::OBSERVATION_DIM>> observations;
        Matrix<matrix::Specification<T, TI, SPEC::CAPACITY, SPEC::ACTION_DIM>> actions;
        Matrix<matrix::Specification<T, TI, 1, SPEC::CAPACITY>> rewards;
        Matrix<matrix::Specification<T, TI, SPEC::CAPACITY, SPEC::OBSERVATION_DIM>> next_observations;
        Matrix<matrix::Specification<bool, TI, 1, SPEC::CAPACITY>> terminated;
        Matrix<matrix::Specification<bool, TI, 1, SPEC::CAPACITY>> truncated;
        TI position = 0;
        bool full = false;
    };
}
#endif