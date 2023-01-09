#ifndef LAYER_IN_C_RL_ALGORITHMS_OFF_POLICY_RUNNER
#define LAYER_IN_C_RL_ALGORITHMS_OFF_POLICY_RUNNER
namespace layer_in_c::rl::components::replay_buffer{
    template<typename DEVICE, typename T_T, auto T_OBSERVATION_DIM, auto T_ACTION_DIM, auto T_CAPACITY>
    struct Specification{
        using T = T_T;
        static constexpr typename DEVICE::index_t OBSERVATION_DIM = T_OBSERVATION_DIM;
        static constexpr typename DEVICE::index_t ACTION_DIM = T_ACTION_DIM;
        static constexpr typename DEVICE::index_t CAPACITY = T_CAPACITY;
    };
}
namespace layer_in_c::rl::components {
    template <typename T_DEVICE, typename SPEC>
    struct ReplayBuffer {
        using T = typename SPEC::T;
        using DEVICE = T_DEVICE;
        static constexpr typename DEVICE::index_t CAPACITY = SPEC::CAPACITY;
        T observations[SPEC::CAPACITY][SPEC::OBSERVATION_DIM];
        T actions[SPEC::CAPACITY][SPEC::ACTION_DIM];
        T rewards[SPEC::CAPACITY];
        T next_observations[SPEC::CAPACITY][SPEC::OBSERVATION_DIM];
        bool terminated[SPEC::CAPACITY];
        bool truncated[SPEC::CAPACITY];
        typename DEVICE::index_t position = 0;
        bool full = false;
    };


}
#endif