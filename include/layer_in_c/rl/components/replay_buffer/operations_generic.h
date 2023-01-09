#ifndef LAYER_IN_C_RL_COMPONENTS_REPLAY_BUFFER_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_COMPONENTS_REPLAY_BUFFER_OPERATIONS_GENERIC_H

#include "replay_buffer.h"

namespace layer_in_c {
    template <typename T, typename DEVICE, auto OBSERVATION_DIM, auto ACTION_DIM, auto CAPACITY>
    void add(rl::components::ReplayBuffer<DEVICE, rl::components::replay_buffer::Spec<T, OBSERVATION_DIM, ACTION_DIM, CAPACITY>>& buffer, const T observation[OBSERVATION_DIM], const T action[ACTION_DIM], const T reward, const T next_observation[OBSERVATION_DIM], const bool terminated, const bool truncated) {
        // todo: change to memcpy?
        for(index_t i = 0; i < OBSERVATION_DIM; i++) {
            buffer.observations[buffer.position][i] = observation[i];
            buffer.next_observations[buffer.position][i] = next_observation[i];
        }
        for(index_t i = 0; i < ACTION_DIM; i++) {
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
