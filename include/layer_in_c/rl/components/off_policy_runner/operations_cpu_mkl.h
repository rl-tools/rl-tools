#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_MKL_H
#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_MKL_H

#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_DELAY_OPERATIONS_GENERIC_INCLUDE
#include "operations_cpu.h"
namespace layer_in_c::rl::components::off_policy_runner{
    template<typename DEV_SPEC, typename SPEC, typename RNG>
    void prologue(devices::CPU_MKL<DEV_SPEC>& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng) {
        prologue((devices::CPU<DEV_SPEC>&)device, runner, rng);
    }
    template<typename DEV_SPEC, typename SPEC, typename RNG>
    void epilogue(devices::CPU_MKL<DEV_SPEC>& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng) {
        epilogue((devices::CPU<DEV_SPEC>&)device, runner, rng);
    }
}
#include "operations_generic.h"
#endif
