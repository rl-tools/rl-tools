#include "on_policy_runner.h"
#include "operations_generic_per_env.h"
namespace layer_in_c::rl::components::on_policy_runner{
    template <typename DEV_SPEC, typename BUFFER_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
    void prologue(devices::CPU<DEV_SPEC>& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, RNG& rng, typename devices::CPU<DEV_SPEC>::index_t step_i){
        using SPEC = typename BUFFER_SPEC::SPEC;
        using TI = typename SPEC::TI;
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
            rl::components::on_policy_runner::per_env::prologue(device, buffer, runner, rng, pos, env_i);
        }
    }
    template <typename DEV_SPEC, typename BUFFER_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
    void epilogue(devices::CPU<DEV_SPEC>& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, RNG& rng, typename devices::CPU<DEV_SPEC> step_i){
        using SPEC = typename BUFFER_SPEC::SPEC;
        using TI = typename SPEC::TI;
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
            rl::components::on_policy_runner::per_env::epilogue(device, buffer, runner, actions, action_log_std, rng, pos, env_i);
        }
    }
}
#ifndef LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CPU_DELAY_OPERATIONS_GENERIC_INCLUDE
#include "operations_generic.h"
#endif
