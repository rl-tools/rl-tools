#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_OPERATIONS_CPU_H

#include "operations_generic.h"
#include <thread>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC, typename T_CONFIG>
    bool step(devices::CPU<DEV_SPEC>& device, rl::algorithms::sac::loop::core::State<T_CONFIG>& ts){
        using DEVICE = devices::CPU<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using CONFIG = T_CONFIG;
        set_step(device, device.logger, ts.step);
        bool finished = false;
        step(device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
        if(ts.step > CONFIG::CORE_PARAMETERS::N_WARMUP_STEPS){
            if(ts.step % CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0){
                std::thread critic_threads[2];
                auto train_critic_i = [&](TI critic_i){
                    auto rng = random::split(device.random, critic_i, ts.rng);
                    gather_batch(device, ts.off_policy_runner, ts.critic_batch[critic_i], rng);
                    randn(device, ts.action_noise_critic[critic_i], rng);
                    train_critic(device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch[critic_i], ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers[critic_i], ts.action_noise_critic[critic_i], rng);
                };
                critic_threads[0] = std::thread([&](){ train_critic_i(0); });
                critic_threads[1] = std::thread([&](){ train_critic_i(1); });
                critic_threads[0].join();
                critic_threads[1].join();
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