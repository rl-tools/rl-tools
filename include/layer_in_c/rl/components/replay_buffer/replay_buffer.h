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
    template <typename SPEC>
    struct ReplayBuffer {
        using T = typename SPEC::T;
        static constexpr typename SPEC::TI CAPACITY = SPEC::CAPACITY;
        T observations[SPEC::CAPACITY][SPEC::OBSERVATION_DIM];
        T actions[SPEC::CAPACITY][SPEC::ACTION_DIM];
        T rewards[SPEC::CAPACITY];
        T next_observations[SPEC::CAPACITY][SPEC::OBSERVATION_DIM];
        bool terminated[SPEC::CAPACITY];
        bool truncated[SPEC::CAPACITY];
        typename SPEC::TI position = 0;
        bool full = false;
    };


}
#endif