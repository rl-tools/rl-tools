#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H
#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OFF_POLICY_RUNNER_H

// Please include the file containing the environments operations before including this file
#include <layer_in_c/rl/components/replay_buffer/replay_buffer.h>

namespace lic = layer_in_c;

namespace layer_in_c::rl::components::off_policy_runner {
    template<typename T>
    struct DefaultParameters{
        static constexpr T EXPLORATION_NOISE = 0.1;
    };
    template<typename T_T, typename T_ENVIRONMENT, auto T_REPLAY_BUFFER_CAPACITY, auto T_STEP_LIMIT, typename T_PARAMETERS>
    struct Spec{
        typedef T_T T;
        typedef T_ENVIRONMENT ENVIRONMENT;
        static constexpr index_t REPLAY_BUFFER_CAPACITY = T_REPLAY_BUFFER_CAPACITY;
        static constexpr index_t STEP_LIMIT = T_STEP_LIMIT;
        typedef T_PARAMETERS PARAMETERS;
    };

}

namespace layer_in_c::rl::components{
    template<typename DEVICE, typename SPEC>
    struct OffPolicyRunner {
        typedef replay_buffer::Spec<typename SPEC::T, SPEC::ENVIRONMENT::OBSERVATION_DIM, SPEC::ENVIRONMENT::ACTION_DIM, SPEC::REPLAY_BUFFER_CAPACITY> ReplayBufferSpec;

        typename SPEC::ENVIRONMENT env;
        ReplayBuffer<DEVICE, ReplayBufferSpec> replay_buffer;
        typename SPEC::ENVIRONMENT::State state;
        index_t episode_step = 0;
        typename SPEC::T episode_return = 0;

        DEVICE& device;
        explicit OffPolicyRunner(DEVICE& device) : device(device) {};
    };
}

#endif
