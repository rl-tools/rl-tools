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
    template<typename T_T, typename T_TI, typename T_ENVIRONMENT, T_TI T_REPLAY_BUFFER_CAPACITY, T_TI T_STEP_LIMIT, typename T_PARAMETERS>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        typedef T_ENVIRONMENT ENVIRONMENT;
        static constexpr TI REPLAY_BUFFER_CAPACITY = T_REPLAY_BUFFER_CAPACITY;
        static constexpr TI STEP_LIMIT = T_STEP_LIMIT;
        typedef T_PARAMETERS PARAMETERS;
    };

}

namespace layer_in_c::rl::components{
    template<typename SPEC>
    struct OffPolicyRunner {
        using ReplayBufferSpec = replay_buffer::Specification<typename SPEC::T, typename SPEC::TI, SPEC::ENVIRONMENT::OBSERVATION_DIM, SPEC::ENVIRONMENT::ACTION_DIM, SPEC::REPLAY_BUFFER_CAPACITY>;

        typename SPEC::ENVIRONMENT env;
        ReplayBuffer<ReplayBufferSpec> replay_buffer;
        typename SPEC::ENVIRONMENT::State state;
        typename SPEC::TI episode_step = 0;
        typename SPEC::T episode_return = 0;
    };
}

#endif
