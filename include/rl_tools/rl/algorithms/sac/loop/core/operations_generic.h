#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_OPERATIONS_GENERIC_H

#include "../../../../../nn_models/operations_generic.h"
#include "../../../../../nn_models/uniform_random/operations_generic.h"
#include "../../../../../rl/algorithms/sac/operations_generic.h"
#include "../../../../../rl/components/off_policy_runner/operations_generic.h"
#include "../../../../../rl/utils/evaluation.h"

#include "config.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG>
    void malloc(DEVICE& device, rl::algorithms::sac::loop::core::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        malloc(device, ts.actor_critic);
        malloc(device, ts.off_policy_runner);
        malloc(device, ts.critic_batch[0]);
        malloc(device, ts.critic_batch[1]);
        malloc(device, ts.critic_training_buffers[0]);
        malloc(device, ts.critic_training_buffers[1]);
        malloc(device, ts.action_noise_critic[0]);
        malloc(device, ts.action_noise_critic[1]);
        malloc(device, ts.critic_buffers[0]);
        malloc(device, ts.critic_buffers[1]);
        malloc(device, ts.actor_batch);
        malloc(device, ts.actor_training_buffers);
        malloc(device, ts.action_noise_actor);
        malloc(device, ts.actor_buffers_eval);
        malloc(device, ts.actor_buffers[0]);
        malloc(device, ts.actor_buffers[1]);
        malloc(device, ts.observations_mean);
        malloc(device, ts.observations_std);
        malloc(device, ts.actor_deterministic_evaluation_buffers);
        for(auto& env: ts.envs){
            rl_tools::malloc(device, env);
        }
        ts.allocated = true;
    }
    template <typename DEVICE, typename T_CONFIG>
    void init(DEVICE& device, rl::algorithms::sac::loop::core::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        utils::assert_exit(device, ts.allocated, "State not allocated");
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;

        ts.rng = random::default_engine(typename DEVICE::SPEC::RANDOM{}, seed);

        init(device, ts.actor_critic, ts.rng);

        for(auto& env: ts.envs){
            rl_tools::init(device, env);
        }

        init(device, ts.off_policy_runner, ts.envs);


        set_all(device, ts.observations_mean, 0);
        set_all(device, ts.observations_std, 1);

        ts.off_policy_runner.parameters = rl::components::off_policy_runner::default_parameters<T>;

        init(device);
        init(device, device.logger);

        ts.step = 0;
    }
    template <typename DEVICE_SOURCE, typename DEVICE_TARGET, typename T_CONFIG_SOURCE, typename T_CONFIG_TARGET>
    void copy(DEVICE_SOURCE& device_source, DEVICE_TARGET& device_target, rl::algorithms::sac::loop::core::State<T_CONFIG_SOURCE>& source, rl::algorithms::sac::loop::core::State<T_CONFIG_TARGET>& target){
        copy(device_source, device_target, source.actor_critic, target.actor_critic);
        copy(device_source, device_target, source.off_policy_runner, target.off_policy_runner);
        copy(device_source, device_target, source.critic_batch[0], target.critic_batch[0]);
        copy(device_source, device_target, source.critic_batch[1], target.critic_batch[1]);
        copy(device_source, device_target, source.critic_training_buffers[0], target.critic_training_buffers[0]);
        copy(device_source, device_target, source.critic_training_buffers[1], target.critic_training_buffers[1]);
        copy(device_source, device_target, source.critic_buffers[0], target.critic_buffers[0]);
        copy(device_source, device_target, source.critic_buffers[1], target.critic_buffers[1]);
        copy(device_source, device_target, source.actor_batch, target.actor_batch);
        copy(device_source, device_target, source.actor_training_buffers, target.actor_training_buffers);
        copy(device_source, device_target, source.actor_buffers_eval, target.actor_buffers_eval);
        copy(device_source, device_target, source.actor_buffers[0], target.actor_buffers[0]);
        copy(device_source, device_target, source.actor_buffers[1], target.actor_buffers[1]);
        copy(device_source, device_target, source.observations_mean, target.observations_mean);
        copy(device_source, device_target, source.observations_std, target.observations_std);
        copy(device_source, device_target, source.actor_deterministic_evaluation_buffers, target.actor_deterministic_evaluation_buffers);
//        target.rng = source.rng;
        target.off_policy_runner.parameters = source.off_policy_runner.parameters;
        target.step = source.step;
    }

    template <typename DEVICE, typename T_CONFIG>
    void free(DEVICE& device, rl::algorithms::sac::loop::core::State<T_CONFIG>& ts){
        free(device, ts.actor_critic);
        free(device, ts.off_policy_runner);
        free(device, ts.critic_batch[0]);
        free(device, ts.critic_batch[1]);
        free(device, ts.critic_training_buffers[0]);
        free(device, ts.critic_training_buffers[1]);
        free(device, ts.action_noise_critic[0]);
        free(device, ts.action_noise_critic[1]);
        free(device, ts.critic_buffers[0]);
        free(device, ts.critic_buffers[1]);
        free(device, ts.actor_batch);
        free(device, ts.actor_training_buffers);
        free(device, ts.action_noise_actor);
        free(device, ts.actor_buffers_eval);
        free(device, ts.actor_buffers[0]);
        free(device, ts.actor_buffers[1]);
        free(device, ts.observations_mean);
        free(device, ts.observations_std);
        free(device, ts.actor_deterministic_evaluation_buffers);
        for(auto& env: ts.envs){
            rl_tools::free(device, env);
        }
    }

    template <typename DEVICE, typename T_CONFIG>
    bool step(DEVICE& device, rl::algorithms::sac::loop::core::State<T_CONFIG>& ts){
        using CONFIG = T_CONFIG;
        set_step(device, device.logger, ts.step);
        bool finished = false;
        if(ts.step >= CONFIG::CORE_PARAMETERS::N_WARMUP_STEPS){
            step(device, ts.off_policy_runner, get_actor(ts), ts.actor_buffers_eval, ts.rng);
        }
        else{
            typename CONFIG::EXPLORATION_POLICY exploration_policy;
            typename CONFIG::EXPLORATION_POLICY::template Buffer<> exploration_policy_buffer;
            step(device, ts.off_policy_runner, exploration_policy, exploration_policy_buffer, ts.rng);
        }
        if(ts.step >= CONFIG::CORE_PARAMETERS::N_WARMUP_STEPS){
            if(ts.step % CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0){
                for(int critic_i = 0; critic_i < 2; critic_i++){
                    gather_batch(device, ts.off_policy_runner, ts.critic_batch[critic_i], ts.rng);
                    randn(device, ts.action_noise_critic[critic_i], ts.rng);
                    train_critic(device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch[critic_i], ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers[critic_i], ts.action_noise_critic[critic_i], ts.rng);
                }
            }
            if(ts.step % CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0){
                gather_batch(device, ts.off_policy_runner, ts.actor_batch, ts.rng);
                randn(device, ts.action_noise_actor, ts.rng);
                train_actor(device, ts.actor_critic, ts.actor_batch, ts.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers, ts.action_noise_actor, ts.rng);
                update_critic_targets(device, ts.actor_critic);
            }
        }
        ts.step++;
        if(ts.step > CONFIG::CORE_PARAMETERS::STEP_LIMIT){
            return true;
        }
        else{
            return finished;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
