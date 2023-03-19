#include <layer_in_c/operations/cpu_mux.h>
#include <layer_in_c/nn/operations_cpu_mux.h>
#include <layer_in_c/nn_models/operations_cpu.h>
namespace lic = layer_in_c;
#include "parameters_rl.h"
#include <layer_in_c/rl/components/on_policy_runner/operations_generic.h>
#include <layer_in_c/rl/algorithms/ppo/operations_generic.h>

#include <gtest/gtest.h>

namespace parameters = parameters_0;

using LOGGER = lic::devices::logging::CPU_TENSORBOARD;
using DEV_SPEC = lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, LOGGER>;

using DEVICE = DEVICE_FACTORY<DEV_SPEC>;
using T = float;
using TI = typename DEVICE::index_t;




TEST(LAYER_IN_C_RL_ALGORITHMS_PPO, TEST){
    using penv = parameters::environment<T, TI>;
    using prl = parameters::rl<T, TI, penv::ENVIRONMENT>;

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    prl::OPTIMIZER optimizer;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    prl::PPO_TYPE ppo;
    prl::PPO_BUFFERS_TYPE ppo_buffers;
    prl::ON_POLICY_RUNNER_TYPE on_policy_runner;
    prl::ON_POLICY_RUNNER_BUFFER_TYPE on_policy_runner_buffer;
    prl::ACTOR_EVAL_BUFFERS actor_eval_buffers;
    prl::ACTOR_BUFFERS actor_buffers;
    prl::CRITIC_BUFFERS critic_buffers;

    lic::malloc(device, ppo);
    lic::malloc(device, ppo_buffers);
    lic::malloc(device, on_policy_runner_buffer);
    lic::malloc(device, on_policy_runner);
    lic::malloc(device, actor_eval_buffers);
    lic::malloc(device, actor_buffers);
    lic::malloc(device, critic_buffers);

    penv::ENVIRONMENT envs[prl::N_ENVIRONMENTS];
    lic::init(device, on_policy_runner, envs, rng);
    lic::init(device, ppo, optimizer, rng);
    device.logger = &logger;
    lic::construct(device, device.logger);
    auto training_start = std::chrono::high_resolution_clock::now();
    for(TI ppo_step_i = 0; ppo_step_i < 100000; ppo_step_i++) {
        device.logger->step = on_policy_runner.step;

        if(ppo_step_i % 100 == 0){
            std::chrono::duration<T> training_elapsed = std::chrono::high_resolution_clock::now() - training_start;
            std::cout << "PPO step: " << ppo_step_i << " elapsed: " << training_elapsed.count() << "s" << std::endl;
            lic::add_scalar(device, device.logger, "ppo/step", ppo_step_i);
        }
        for (TI action_i = 0; action_i < penv::ENVIRONMENT::ACTION_DIM; action_i++) {
            T action_log_std = lic::get(ppo.actor.action_log_std.parameters, 0, action_i);
            std::stringstream topic;
            topic << "actor/action_std/" << action_i;
            lic::add_scalar(device, device.logger, topic.str(), lic::math::exp(DEVICE::SPEC::MATH(), action_log_std));
        }
        auto start = std::chrono::high_resolution_clock::now();
        {
            auto start = std::chrono::high_resolution_clock::now();
            lic::collect(device, on_policy_runner_buffer, on_policy_runner, ppo.actor, actor_eval_buffers, rng);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
//            std::cout << "Rollout: " << elapsed.count() << " s" << std::endl;
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            lic::estimate_generalized_advantages(device, ppo, on_policy_runner_buffer);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
//            std::cout << "GAE: " << elapsed.count() << " s" << std::endl;
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            lic::train(device, ppo, on_policy_runner_buffer, optimizer, ppo_buffers, actor_buffers, critic_buffers, rng);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
//            std::cout << "Train: " << elapsed.count() << " s" << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> elapsed = end - start;
//        std::cout << "Total: " << elapsed.count() << " s" << std::endl;
    }

    lic::free(device, ppo);
    lic::free(device, ppo_buffers);
    lic::free(device, on_policy_runner_buffer);
    lic::free(device, on_policy_runner);
    lic::free(device, actor_eval_buffers);
    lic::free(device, actor_buffers);
    lic::free(device, critic_buffers);

}