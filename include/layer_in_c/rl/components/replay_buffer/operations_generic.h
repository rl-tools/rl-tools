#ifndef LAYER_IN_C_RL_COMPONENTS_REPLAY_BUFFER_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_COMPONENTS_REPLAY_BUFFER_OPERATIONS_GENERIC_H

#include "replay_buffer.h"

namespace layer_in_c {
    template <typename DEVICE, typename SPEC>
    void add(DEVICE& device, rl::components::ReplayBuffer<SPEC>& buffer, const typename SPEC::T observation[SPEC::OBSERVATION_DIM], const typename SPEC::T action[SPEC::ACTION_DIM], const typename SPEC::T reward, const typename SPEC::T next_observation[SPEC::OBSERVATION_DIM], const bool terminated, const bool truncated) {
        // todo: change to memcpy?
        for(typename DEVICE::index_t i = 0; i < SPEC::OBSERVATION_DIM; i++) {
            buffer.observations[buffer.position][i] = observation[i];
            buffer.next_observations[buffer.position][i] = next_observation[i];
        }
        for(typename DEVICE::index_t i = 0; i < SPEC::ACTION_DIM; i++) {
            buffer.actions[buffer.position][i] = action[i];
        }
        buffer.rewards[buffer.position] = reward;
        buffer.terminated[buffer.position] = terminated;
        buffer.truncated[buffer.position] = truncated;
        buffer.position = (buffer.position + 1) % SPEC::CAPACITY;
        if(buffer.position == 0 && !buffer.full) {
            buffer.full = true;
        }
    }
}
#endif
