#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_H
#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_H

#include <thread>

#include "operations_generic_per_env.h"
namespace layer_in_c::rl::components::off_policy_runner{
    template<typename DEV_SPEC, typename SPEC, typename RNG>
    void prologue(devices::CPU<DEV_SPEC>& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng) {
        using DEVICE = devices::CPU<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        std::vector<std::thread> threads;

        constexpr TI NUM_THREADS = DEV_SPEC::EXECUTION_HINTS::RL_COMPONENTS_OFF_POLICY_RUNNER::NUM_THREADS;

        RNG rngs[SPEC::N_ENVIRONMENTS];
        auto base = random::uniform_int_distribution(typename DEV_SPEC::RANDOM(), 0, 1000000, rng);
        for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
            rngs[env_i] = layer_in_c::random::default_engine(typename DEV_SPEC::RANDOM(), base + env_i);
        }

        for (TI thread_i = 0; thread_i < NUM_THREADS; thread_i++) {
            threads.emplace_back([&device, thread_i, &runner, &rngs](){
                for (TI env_i = thread_i; env_i < SPEC::N_ENVIRONMENTS; env_i += NUM_THREADS) {
                    prologue_per_env(device, &runner, rngs[env_i], env_i);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    template<typename DEV_SPEC, typename SPEC, typename RNG>
    void epilogue(devices::CPU<DEV_SPEC>& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng) {
        using DEVICE = devices::CPU<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        std::vector<std::thread> threads;

        constexpr TI NUM_THREADS = DEV_SPEC::EXECUTION_HINTS::RL_COMPONENTS_OFF_POLICY_RUNNER::NUM_THREADS;

        RNG rngs[SPEC::N_ENVIRONMENTS];
        auto base = random::uniform_int_distribution(typename DEV_SPEC::RANDOM(), 0, 1000000, rng);
        for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
            rngs[env_i] = layer_in_c::random::default_engine(typename DEV_SPEC::RANDOM(), base + env_i);
        }

        for (TI thread_i = 0; thread_i < NUM_THREADS; thread_i++) {
            threads.emplace_back([&device, thread_i, &runner, &rngs](){
                for (TI env_i = thread_i; env_i < SPEC::N_ENVIRONMENTS; env_i += NUM_THREADS) {
                    epilogue_per_env(device, &runner, rngs[env_i], env_i);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }
}
#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_DELAY_OPERATIONS_GENERIC_INCLUDE
#include "operations_generic.h"
#endif

#endif
