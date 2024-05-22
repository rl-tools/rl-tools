#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_UTILS_EVALUATION_EVALUATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_UTILS_EVALUATION_EVALUATION_H
/*
 * This file relies on the environments methods hence it should be included after the operations of the environments that it will be used with
 */

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::utils::evaluation{
    template <typename T, typename TI, typename ENV_STATE>
    struct State{
        T episode_return = 0;
        TI episode_step = 0;

        ENV_STATE state;
    };
    template <auto T_N_EPISODES, auto T_STEP_LIMIT>
    struct Specification{
        constexpr static auto N_EPISODES = T_N_EPISODES;
        constexpr static auto STEP_LIMIT = T_STEP_LIMIT;
    };
    template <typename T, typename TI, auto T_N_EPISODES>
    struct Result{
        constexpr static auto N_EPISODES = T_N_EPISODES;
        T returns[N_EPISODES];
        T returns_mean;
        T returns_std;
        TI episode_length[N_EPISODES];
        T episode_length_mean;
        T episode_length_std;
    };
}
#endif