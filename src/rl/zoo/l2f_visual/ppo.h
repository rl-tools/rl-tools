#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_VISUAL_PPO_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_VISUAL_PPO_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f_visual {
    template <typename T, typename TI>
    struct PPO_PARAMETERS {
        static constexpr TI N_EPOCHS = 4;
        static constexpr TI BATCH_SIZE = 64;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;
        static constexpr T GAMMA = 0.99;
        static constexpr T LAMBDA = 0.95;
        static constexpr T EPSILON_CLIP = 0.2;
        static constexpr T INITIAL_ACTION_STD = 0.5;
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
        static constexpr T LEARNING_RATE = 3e-4;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
