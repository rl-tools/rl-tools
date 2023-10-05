#include "../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_RL_UTILS_EVALUATION_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_RL_UTILS_EVALUATION_H

#include "../../math/operations_generic.h"

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::rl::utils::validation{
    template <typename T_T, typename T_TI, typename T_ENVIRONMENT>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using ENVIRONMENT = T_ENVIRONMENT;
    };

    template <typename T_SPEC, typename T_SPEC::TI SIZE>
    struct EpisodeBuffer{
        using SPEC = T_SPEC;
        MatrixDynamic<matrix::Specification<typename SPEC::ENVIRONMENT::State, typename SPEC::TI, SIZE, 1>> states;
        MatrixDynamic<matrix::Specification<typename SPEC::T, typename SPEC::TI, SIZE, SPEC::ENVIRONMENT::ACTION_DIM>> actions;
        MatrixDynamic<matrix::Specification<typename SPEC::ENVIRONMENT::State, typename SPEC::TI, SIZE, 1>> next_states;
        MatrixDynamic<matrix::Specification<typename SPEC::T, typename SPEC::TI, SIZE, 0>> rewards;
        MatrixDynamic<matrix::Specification<bool, typename SPEC::TI, SIZE, 1>> terminated;
    };

    template <typename T_SPEC, typename T_SPEC::TI T_N_EPISODES, typename T_SPEC::TI T_MAX_EPISODE_LENGTH>
    struct TaskSpecification{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr auto N_EPISODES = T_N_EPISODES;
        static constexpr auto MAX_EPISODE_LENGTH = T_MAX_EPISODE_LENGTH;
    };
    template <typename T_SPEC>
    struct Task{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::SPEC::ENVIRONMENT;
        TI step = 0;
        EpisodeBuffer<typename SPEC::SPEC, SPEC::MAX_EPISODE_LENGTH> episode_buffer[SPEC::N_EPISODES];
        MatrixDynamic<matrix::Specification<T, TI, SPEC::N_EPISODES, ENVIRONMENT::OBSERVATION_DIM>> observation_buffer;
        MatrixDynamic<matrix::Specification<T, TI, SPEC::N_EPISODES, ENVIRONMENT::ACTION_DIM>> action_buffer;
        TI episode_length[SPEC::N_EPISODES];
        bool terminated[SPEC::N_EPISODES];
        ENVIRONMENT environment[SPEC::N_EPISODES];
        typename ENVIRONMENT::State state[SPEC::N_EPISODES];
    };
}
namespace backprop_tools{
    template <typename DEVICE, typename SPEC, typename SPEC::TI SIZE>
    void malloc(DEVICE& device, rl::utils::validation::EpisodeBuffer<SPEC, SIZE>& eb){
        malloc(device, eb.states);
        malloc(device, eb.actions);
        malloc(device, eb.next_states);
        malloc(device, eb.rewards);
        malloc(device, eb.terminated);
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI SIZE>
    void free(DEVICE& device, rl::utils::validation::EpisodeBuffer<SPEC, SIZE>& eb){
        free(device, eb.states);
        free(device, eb.actions);
        free(device, eb.next_states);
        free(device, eb.rewards);
        free(device, eb.terminated);
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void reset(DEVICE& device, rl::utils::validation::Task<SPEC>& task, RNG& rng){
        using TI = typename DEVICE::index_t;
        task.step = 0;
        for(TI i = 0; i < SPEC::N_EPISODES; i++){
            sample_initial_state(device, task.environment[i], task.state[i], rng);
            task.episode_length[i] = 0;
            task.terminated[i] = false;
        }
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void init(DEVICE& device, rl::utils::validation::Task<SPEC>& task, typename SPEC::SPEC::ENVIRONMENT envs[SPEC::N_EPISODES], RNG& rng){
        using TI = typename SPEC::TI;
        for(TI i = 0; i < SPEC::N_EPISODES; i++){
            task.environment[i] = envs[i];
            malloc(device, task.episode_buffer[i]);
        }
        malloc(device, task.observation_buffer);
        malloc(device, task.action_buffer);
        reset(device, task, rng);
    }
    template <typename DEVICE, typename SPEC, typename POLICY, typename POLICY_BUFFERS, typename RNG>
    void step(DEVICE& device, rl::utils::validation::Task<SPEC>& task, POLICY& policy, POLICY_BUFFERS& buffers, RNG& rng){
        using TI = typename SPEC::TI;
        using T = typename SPEC::T;
        for(TI episode_i = 0; episode_i < SPEC::N_EPISODES; episode_i++){
            auto observation = row(device, task.observation_buffer, episode_i);
            observe(device, task.environment[episode_i], task.state[episode_i], observation, rng);
        }
        evaluate(device, policy, task.observation_buffer, task.action_buffer, buffers);

        for(TI episode_i = 0; episode_i < SPEC::N_EPISODES; episode_i++){
            auto& eb = task.episode_buffer[episode_i];
            bool terminated_flag = true;
            if(task.step == 0 || !get(eb.terminated, task.step - 1, 0)){
                auto action = row(device, task.action_buffer, episode_i);
                auto action_buffer = row(device, eb.actions, task.step);
                copy(device, device, action_buffer, action);
                set(eb.states, task.step, 0, task.state[episode_i]);
                step(device, task.environment[episode_i], task.state[episode_i], action, get(eb.next_states, task.step, 0), rng);
                T step_reward = reward(device, task.environment[episode_i], task.state[episode_i], action, get(eb.next_states, task.step, 0), rng);
                set(eb.rewards, task.step, 0, step_reward);

                task.state[episode_i] = get(eb.next_states, task.step, 0);
                terminated_flag = terminated(device, task.environment[episode_i], task.state[episode_i], rng);
                if(terminated_flag){
                    task.episode_length[episode_i] = task.step + 1;
                }
            }
            task.terminated[episode_i] = terminated_flag;
            set(eb.terminated, task.step, 0, terminated_flag);
        }
        if(task.step == SPEC::MAX_EPISODE_LENGTH - 1){
            task.step = 0;
        }
        else{
            task.step++;
        }
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void destroy(DEVICE& device, rl::utils::validation::Task<SPEC>& task){
        using TI = typename SPEC::TI;
        for(TI i = 0; i < SPEC::N_EPISODES; i++){
            free(device, task.episode_buffer[i]);
        }
        free(device, task.observation_buffer);
        free(device, task.action_buffer);
    }
};
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
