#ifndef LAYER_IN_C_RL_ALGORITHMS_TD3_OFF_POLICY_RUNNER
#define LAYER_IN_C_RL_ALGORITHMS_TD3_OFF_POLICY_RUNNER
// Please include the file containing the environments operations before including this file
#include <layer_in_c/rl/components/replay_buffer/replay_buffer.h>
#include <layer_in_c/rl/environments/environments.h>

namespace lic = layer_in_c;

namespace layer_in_c::rl::algorithms::td3 {
    template<typename T, int T_CAPACITY, int T_STEP_LIMIT>
    struct DefaultOffPolicyRunnerParameters {
        static constexpr uint32_t CAPACITY = T_CAPACITY;
        static constexpr uint32_t STEP_LIMIT = T_STEP_LIMIT;
        static constexpr T EXPLORATION_NOISE = 0.1;
    };

    template<typename T, typename ENVIRONMENT, typename PARAMETERS>
    struct OffPolicyRunner {
        ReplayBuffer<T, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, PARAMETERS::CAPACITY> replay_buffer;
        typename ENVIRONMENT::State state;
        uint32_t episode_step = 0;
        T episode_return = 0;
    };
}
#endif