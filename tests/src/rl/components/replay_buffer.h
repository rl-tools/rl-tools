#ifndef LAYER_IN_C_TESTS_SRC_RL_COMPONENTS_REPLAY_BUFFER_H
#define LAYER_IN_C_TESTS_SRC_RL_COMPONENTS_REPLAY_BUFFER_H

#include <layer_in_c/rl/components/replay_buffer/replay_buffer.h>

namespace layer_in_c::test::rl::components::replay_buffer{
    template <typename DEVICE, typename SPEC, typename RNG>
    void sample(DEVICE& device, layer_in_c::rl::components::ReplayBuffer<SPEC>& rb, RNG& rng){
        using T = typename SPEC::T;
        randn(device, rb.observations, rng);
        randn(device, rb.actions, rng);
        randn(device, rb.next_observations, rng);
        randn(device, rb.rewards, rng);

        for (typename DEVICE::index_t row_i = 0; row_i < SPEC::CAPACITY; row_i++) {
            bool terminated = random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T) 0, (T) 1, rng) > 0.5;
            bool truncated = terminated || random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T) 0, (T) 1, rng) > 0.5;
            set(rb.terminated, row_i, 0, terminated);
            set(rb.truncated, row_i, 0, truncated);
        }
        rb.position = 0;
        rb.full = true;
    }
}

#endif
