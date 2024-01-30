#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_OPERATIONS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_OPERATIONS_H

#include "../../../../../rl/algorithms/sac/operations_generic.h"
#include "../../../../../rl/algorithms/ppo/operations_generic.h"
#include "../../../../../rl/components/on_policy_runner/operations_generic.h"
#include "../../../../../rl/components/running_normalizer/operations_generic.h"
#include "../../../../../rl/utils/evaluation.h"

#include "config.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG>
    void malloc(DEVICE& device, rl::algorithms::ppo::loop::core::TrainingState<T_CONFIG>& ts){
        malloc(device, ts.ppo);
        malloc(device, ts.ppo_buffers);
        malloc(device, ts.on_policy_runner_dataset);
        malloc(device, ts.on_policy_runner);
        malloc(device, ts.actor_eval_buffers);
        malloc(device, ts.actor_deterministic_evaluation_buffers);
        malloc(device, ts.actor_buffers);
        malloc(device, ts.critic_buffers);
        malloc(device, ts.critic_buffers_gae);
        malloc(device, ts.observation_normalizer);
        malloc(device, ts.observations_mean);
        malloc(device, ts.observations_std);
        for(auto& env: ts.envs){
            malloc(device, env);
        }

    }
    template <typename DEVICE, typename T_CONFIG>
    void init(DEVICE& device, rl::algorithms::ppo::loop::core::TrainingState<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;

        ts.rng = random::default_engine(typename DEVICE::SPEC::RANDOM(), seed);

//        ts.actor_optimizer.parameters.alpha = 3e-4;
//        ts.critic_optimizer.parameters.alpha = 3e-4 * 2;

        init(device, ts.on_policy_runner, ts.envs, ts.rng);
        init(device, ts.observation_normalizer);
        set_all(device, ts.observations_mean, 0);
        set_all(device, ts.observations_std, 1);
        init(device, ts.ppo, ts.actor_optimizer, ts.critic_optimizer, ts.rng);

        init(device);
        init(device, device.logger);

        ts.step = 0;
    }

    template <typename DEVICE, typename T_CONFIG>
    void free(DEVICE& device, rl::algorithms::ppo::loop::core::TrainingState<T_CONFIG>& ts){
        free(device, ts.ppo);
        free(device, ts.ppo_buffers);
        free(device, ts.on_policy_runner_dataset);
        free(device, ts.on_policy_runner);
        free(device, ts.actor_eval_buffers);
        free(device, ts.actor_deterministic_eval_buffers);
        free(device, ts.actor_buffers);
        free(device, ts.critic_buffers);
        free(device, ts.critic_buffers_gae);
        free(device, ts.observation_normalizer);
        for(auto& env: ts.envs){
            free(device, env);
        }
    }

    template <typename T_CONFIG>
    auto& get_actor(rl::algorithms::ppo::loop::core::TrainingState<T_CONFIG>& ts){
        return ts.ppo.actor;
    }

    template <typename DEVICE, typename T_CONFIG>
    bool step(DEVICE& device, rl::algorithms::ppo::loop::core::TrainingState<T_CONFIG>& ts){
        using CONFIG = T_CONFIG;
        set_step(device, device.logger, ts.step);
        bool finished = false;
        collect(device, ts.on_policy_runner_dataset, ts.on_policy_runner, ts.ppo.actor, ts.actor_eval_buffers, ts.observation_normalizer.mean, ts.observation_normalizer.std, ts.rng);
        auto on_policy_runner_dataset_all_observations = CONFIG::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS ? ts.on_policy_runner_dataset.all_observations_normalized : ts.on_policy_runner_dataset.all_observations;
        evaluate(device, ts.ppo.critic, on_policy_runner_dataset_all_observations, ts.on_policy_runner_dataset.all_values, ts.critic_buffers_gae);
        estimate_generalized_advantages(device, ts.on_policy_runner_dataset, typename CONFIG::PPO_TYPE::SPEC::PARAMETERS{});
        train(device, ts.ppo, ts.on_policy_runner_dataset, ts.actor_optimizer, ts.critic_optimizer, ts.ppo_buffers, ts.actor_buffers, ts.critic_buffers, ts.rng);

        ts.step++;
        if(ts.step > CONFIG::PARAMETERS::STEP_LIMIT){
            return true;
        }
        else{
            return finished;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
