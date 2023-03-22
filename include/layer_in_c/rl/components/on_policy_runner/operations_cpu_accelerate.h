#ifndef LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CPU_ACCELERATE_H
#define LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CPU_ACCELERATE_H

#include "on_policy_runner.h"
#define LAYER_IN_C_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CPU_DELAY_OPERATIONS_GENERIC_INCLUDE
#include "operations_cpu.h"
namespace layer_in_c::rl::components::on_policy_runner{
    template <typename DEV_SPEC, typename BUFFER_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
    void prologue(devices::CPU_ACCELERATE<DEV_SPEC>& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, RNG& rng, typename devices::CPU<DEV_SPEC>::index_t step_i){
        prologue((devices::CPU<DEV_SPEC>&)device, buffer, runner, rng, step_i);
    }
    template <typename DEV_SPEC, typename BUFFER_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RNG> // todo: make this not PPO but general policy with output distribution
    void epilogue(devices::CPU_ACCELERATE<DEV_SPEC>& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, RNG& rng, typename devices::CPU<DEV_SPEC>::index_t step_i){
        epilogue((devices::CPU<DEV_SPEC>&)device, buffer, runner, actions, action_log_std, rng, step_i);
    }
}
#include "operations_generic.h"

#endif
