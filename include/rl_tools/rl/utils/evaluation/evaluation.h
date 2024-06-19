#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_UTILS_EVALUATION_EVALUATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_UTILS_EVALUATION_EVALUATION_H
/*
 * This file relies on the environments methods hence it should be included after the operations of the environments that it will be used with
 */
#include "../../../utils/generic/typing.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::utils::evaluation{
    template <typename T, typename TI, typename ENV_STATE>
    struct State{
        T episode_return = 0;
        TI episode_step = 0;

        ENV_STATE state;
    };
    template <typename T_T, typename T_TI, typename T_ENVIRONMENT, T_TI T_N_EPISODES, T_TI T_STEP_LIMIT>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using ENVIRONMENT = T_ENVIRONMENT;
        constexpr static TI N_EPISODES = T_N_EPISODES;
        constexpr static TI STEP_LIMIT = T_STEP_LIMIT;
    };
    template <typename T_SPEC>
    struct Data{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        typename ENVIRONMENT::Parameters parameters[SPEC::N_EPISODES];
        bool terminated[SPEC::N_EPISODES][SPEC::STEP_LIMIT];
        T rewards[SPEC::N_EPISODES][SPEC::STEP_LIMIT];
        typename ENVIRONMENT::State states[SPEC::N_EPISODES][SPEC::STEP_LIMIT];
        T actions[SPEC::N_EPISODES][SPEC::STEP_LIMIT][ENVIRONMENT::ACTION_DIM];
        T dt[SPEC::N_EPISODES][SPEC::STEP_LIMIT];
    };
    template <typename T_SPEC>
    struct NoData{};
    template <typename T_SPEC>
    struct Result{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        constexpr static auto N_EPISODES = SPEC::N_EPISODES;
        constexpr static auto STEP_LIMIT = SPEC::STEP_LIMIT;
        T returns[N_EPISODES];
        T returns_mean;
        T returns_std;
        TI episode_length[N_EPISODES];
        T episode_length_mean;
        T episode_length_std;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif