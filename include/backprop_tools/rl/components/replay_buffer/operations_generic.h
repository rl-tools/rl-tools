#ifndef BACKPROP_TOOLS_RL_COMPONENTS_REPLAY_BUFFER_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_RL_COMPONENTS_REPLAY_BUFFER_OPERATIONS_GENERIC_H

#include "replay_buffer.h"
#include <backprop_tools/utils/generic/memcpy.h>

namespace backprop_tools {
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::ReplayBuffer<SPEC>& rb) {
        using DATA_SPEC = typename decltype(rb.data)::SPEC;
        malloc(device, rb.data);
        rb.observations      = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, SPEC::OBSERVATION_DIM>(device, rb.data, 0, 0);
        rb.actions           = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, SPEC::ACTION_DIM     >(device, rb.data, 0, SPEC::OBSERVATION_DIM);
        rb.rewards           = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, 1                    >(device, rb.data, 0, SPEC::OBSERVATION_DIM + SPEC::ACTION_DIM);
        rb.next_observations = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, SPEC::OBSERVATION_DIM>(device, rb.data, 0, SPEC::OBSERVATION_DIM + SPEC::ACTION_DIM + 1);
        rb.terminated        = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, 1                    >(device, rb.data, 0, SPEC::OBSERVATION_DIM + SPEC::ACTION_DIM + 1 + SPEC::OBSERVATION_DIM);
        rb.truncated         = view<DEVICE, DATA_SPEC, SPEC::CAPACITY, 1                    >(device, rb.data, 0, SPEC::OBSERVATION_DIM + SPEC::ACTION_DIM + 1 + SPEC::OBSERVATION_DIM + 1);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::ReplayBuffer<SPEC>& rb) {
        free(device, rb.data);
    }
    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::components::ReplayBuffer<SPEC>& rb) {
        rb.full = false;
        rb.position = 0;
    }
    template <typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename ACTION_SPEC, typename NEXT_OBSERVATION_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void add(DEVICE& device, rl::components::ReplayBuffer<SPEC>& buffer, const Matrix<OBSERVATION_SPEC>& observation, const Matrix<ACTION_SPEC>& action, const typename SPEC::T reward, const Matrix<NEXT_OBSERVATION_SPEC>& next_observation, const bool terminated, const bool truncated) {
        // todo: change to memcpy?
        for(typename DEVICE::index_t i = 0; i < SPEC::OBSERVATION_DIM; i++) {
            set(buffer.observations, buffer.position, i, get(observation, 0, i));
            set(buffer.next_observations, buffer.position, i, get(next_observation, 0, i));
        }
        for(typename DEVICE::index_t i = 0; i < SPEC::ACTION_DIM; i++) {
            set(buffer.actions, buffer.position, i, get(action, 0, i));
        }
        set(buffer.rewards, buffer.position, 0, reward);
        set(buffer.terminated, buffer.position, 0, terminated);
        set(buffer.truncated, buffer.position, 0, truncated);
        buffer.position = (buffer.position + 1) % SPEC::CAPACITY;
        if(buffer.position == 0 && !buffer.full) {
            buffer.full = true;
        }
//        add_scalar(device, device.logger, "replay_buffer/position", (typename SPEC::T)(buffer.full ? SPEC::CAPACITY : buffer.position), 1000);
    }
    template <typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, rl::components::ReplayBuffer<TARGET_SPEC>& target, rl::components::ReplayBuffer<SOURCE_SPEC>& source) {
        copy(target_device, source_device, target.data, source.data);
        target.full = source.full;
        target.position = source.position;
    }

    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    typename SPEC_1::T abs_diff(DEVICE& device, rl::components::ReplayBuffer<SPEC_1>& b1, rl::components::ReplayBuffer<SPEC_2>& b2) {
        typename SPEC_1::T acc = 0;
        acc += abs_diff(device, b1.observations, b2.observations);
        acc += abs_diff(device, b1.actions, b2.actions);
        acc += abs_diff(device, b1.rewards, b2.rewards);
        acc += abs_diff(device, b1.next_observations, b2.next_observations);
        acc += abs_diff(device, b1.terminated, b2.terminated);
        acc += abs_diff(device, b1.truncated, b2.truncated);
        return acc;
    }
}
#endif
