#include "on_policy_runner.h"
#include "operations_generic_per_env.h"
namespace layer_in_c::rl::components::on_policy_runner{
    constexpr auto get_num_threads(devices::ExecutionHints hints) {
        return 1;
    }
    template<typename TI, TI NUM_THREADS>
    constexpr TI get_num_threads(rl::components::on_policy_runner::ExecutionHints<TI, NUM_THREADS> hints) {
        return NUM_THREADS;
    }
    template <typename DEV_SPEC, typename OBSERVATIONS_SPEC, typename SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
    void prologue(devices::CPU<DEV_SPEC>& device, Matrix<OBSERVATIONS_SPEC>& observations,  rl::components::OnPolicyRunner<SPEC>& runner, RNG& rng, const typename devices::CPU<DEV_SPEC>::index_t step_i){
        static_assert(OBSERVATIONS_SPEC::ROWS == SPEC::N_ENVIRONMENTS);
        static_assert(OBSERVATIONS_SPEC::COLS == SPEC::ENVIRONMENT::OBSERVATION_DIM);
        using DEVICE = devices::CPU<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
            per_env::prologue(device, observations, runner, rng, env_i);
        }
    }
    template <typename DEV_SPEC, typename BUFFER_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
    void epilogue(devices::CPU<DEV_SPEC>& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, RNG& rng, typename devices::CPU<DEV_SPEC>::index_t step_i){
        using SPEC = typename BUFFER_SPEC::SPEC;
        using DEVICE = devices::CPU<DEV_SPEC>;
        using TI = typename DEVICE::index_t;

        constexpr TI NUM_THREADS = get_num_threads(typename DEVICE::EXECUTION_HINTS());
        std::thread threads[NUM_THREADS];

        if(NUM_THREADS > 1){
            RNG rngs[SPEC::N_ENVIRONMENTS];
            auto base = random::uniform_int_distribution(typename DEV_SPEC::RANDOM(), 0, 1000000, rng);
            for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
                rngs[env_i] = layer_in_c::random::default_engine(typename DEV_SPEC::RANDOM(), base + env_i);
            }

            for (TI thread_i = 0; thread_i < NUM_THREADS; thread_i++) {
                threads[thread_i] = std::thread([&device, thread_i, &buffer, &runner, &actions, &action_log_std, &step_i, &rngs](){
                    for (TI env_i = thread_i; env_i < SPEC::N_ENVIRONMENTS; env_i += NUM_THREADS) {
                        TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                        per_env::epilogue(device, buffer, runner, actions, action_log_std, rngs[env_i], pos, env_i);
                    }
                });
            }

            for (auto& thread : threads) {
                thread.join();
            }
        }
        else{
            for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                per_env::epilogue(device, buffer, runner, actions, action_log_std, rng, pos, env_i);
            }
        }
    }
}
#ifndef LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CPU_DELAY_OPERATIONS_GENERIC_INCLUDE
#include "operations_generic.h"
#endif
