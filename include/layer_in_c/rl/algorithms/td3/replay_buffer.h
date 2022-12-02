#ifndef LAYER_IN_C_RL_ALGORITHMS_OFF_POLICY_RUNNER
#define LAYER_IN_C_RL_ALGORITHMS_OFF_POLICY_RUNNER
namespace layer_in_c::rl::algorithms::td3 {
    template <typename T_T, int OBSERVATION_DIM, int ACTION_DIM, int CAPACITY>
    struct ReplayBuffer {
        using T = T_T;
        T observations[CAPACITY][OBSERVATION_DIM];
        T actions[CAPACITY][ACTION_DIM];
        T rewards[CAPACITY];
        T next_observations[CAPACITY][OBSERVATION_DIM];
        bool terminated[CAPACITY];
        bool truncated[CAPACITY];
        uint32_t position = 0;
        bool full = false;
    };


    template <typename T, int OBSERVATION_DIM, int ACTION_DIM, int CAPACITY>
    void add(ReplayBuffer<T, OBSERVATION_DIM, ACTION_DIM, CAPACITY>& buffer, const T observation[OBSERVATION_DIM], const T action[ACTION_DIM], const T reward, const T next_observation[OBSERVATION_DIM], const bool terminated, const bool truncated) {
        // todo: change to memcpy?
        for(int i = 0; i < OBSERVATION_DIM; i++) {
            buffer.observations[buffer.position][i] = observation[i];
            buffer.next_observations[buffer.position][i] = next_observation[i];
        }
        for(int i = 0; i < ACTION_DIM; i++) {
            buffer.actions[buffer.position][i] = action[i];
        }
        buffer.rewards[buffer.position] = reward;
        buffer.terminated[buffer.position] = terminated;
        buffer.truncated[buffer.position] = truncated;
        buffer.position = (buffer.position + 1) % CAPACITY;
        if(buffer.position == 0 && !buffer.full) {
            buffer.full = true;
        }
    }
}
#endif