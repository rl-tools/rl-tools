#ifndef LAYER_IN_C_RL_COMPONENTS_REPLAY_BUFFER_OPERATIONS_CPU_H
#define LAYER_IN_C_RL_COMPONENTS_REPLAY_BUFFER_OPERATIONS_CPU_H

#include "replay_buffer.h"
#include "operations_generic.h"

namespace layer_in_c {
    template <typename T, size_t OBSERVATION_DIM, size_t ACTION_DIM, size_t CAPACITY>
    void add(rl::components::ReplayBuffer<devices::CPU, rl::components::replay_buffer::Spec<T, OBSERVATION_DIM, ACTION_DIM, CAPACITY>>& buffer, const T observation[OBSERVATION_DIM], const T action[ACTION_DIM], const T reward, const T next_observation[OBSERVATION_DIM], const bool terminated, const bool truncated) {
        add((rl::components::ReplayBuffer<devices::Generic, rl::components::replay_buffer::Spec<T, OBSERVATION_DIM, ACTION_DIM, CAPACITY>>&)buffer, observation, action, reward, next_observation, terminated, truncated);
    }
}

#endif
