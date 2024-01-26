#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_TD3_LOOP_CORE_OPERATIONS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_TD3_LOOP_CORE_OPERATIONS_H

#include "../../../../../rl/algorithms/td3/operations_generic.h"
#include "../../../../../rl/components/off_policy_runner/operations_generic.h"
#include "../../../../../rl/utils/evaluation.h"

#include "config.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename T_CONFIG>
    void init(rl::algorithms::td3::loop::core::TrainingState<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;

        ts.rng = random::default_engine(typename CONFIG::DEVICE::SPEC::RANDOM(), seed);

        malloc(ts.device, ts.actor_critic);
        init(ts.device, ts.actor_critic, ts.rng);

        malloc(ts.device, ts.off_policy_runner);
        init(ts.device, ts.off_policy_runner, ts.envs);
        rl_tools::init(ts.device, ts.envs[0], ts.ui);

        malloc(ts.device, ts.critic_batch);
        malloc(ts.device, ts.critic_training_buffers);
        malloc(ts.device, ts.critic_buffers[0]);
        malloc(ts.device, ts.critic_buffers[1]);

        malloc(ts.device, ts.actor_batch);
        malloc(ts.device, ts.actor_training_buffers);
        malloc(ts.device, ts.actor_buffers_eval);
        malloc(ts.device, ts.actor_buffers[0]);
        malloc(ts.device, ts.actor_buffers[1]);

        malloc(ts.device, ts.observations_mean);
        malloc(ts.device, ts.observations_std);

        malloc(ts.device, ts.actor_deterministic_evaluation_buffers);

        set_all(ts.device, ts.observations_mean, 0);
        set_all(ts.device, ts.observations_std, 1);

        ts.off_policy_runner.parameters = rl::components::off_policy_runner::default_parameters<T>;

        init(ts.device);
        init(ts.device, ts.device.logger);

        ts.step = 0;
    }

    template <typename T_CONFIG>
    void destroy(rl::algorithms::td3::loop::core::TrainingState<T_CONFIG>& ts){
        free(ts.device, ts.critic_batch);
        free(ts.device, ts.critic_training_buffers);
        free(ts.device, ts.actor_batch);
        free(ts.device, ts.actor_training_buffers);
        free(ts.device, ts.off_policy_runner);
        free(ts.device, ts.actor_critic);
        free(ts.device, ts.observations_mean);
        free(ts.device, ts.observations_std);
    }

    template <typename T_CONFIG>
    auto& get_actor(rl::algorithms::td3::loop::core::TrainingState<T_CONFIG>& ts){
        return ts.actor_critic.actor;
    }

    template <typename T_CONFIG>
    bool step(rl::algorithms::td3::loop::core::TrainingState<T_CONFIG>& ts){
        bool finished = false;
        using CONFIG = T_CONFIG;
        step(ts.device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
        if(ts.step > CONFIG::PARAMETERS::N_WARMUP_STEPS){
            for(int critic_i = 0; critic_i < 2; critic_i++){
                gather_batch(ts.device, ts.off_policy_runner, ts.critic_batch, ts.rng);
                train_critic(ts.device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.actor_critic.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers);
            }
            if(ts.step % 1 == 0){
                {
                    gather_batch(ts.device, ts.off_policy_runner, ts.actor_batch, ts.rng);
                    train_actor(ts.device, ts.actor_critic, ts.actor_batch, ts.actor_critic.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers);
                }
                update_critic_targets(ts.device, ts.actor_critic);
            }
        }
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
