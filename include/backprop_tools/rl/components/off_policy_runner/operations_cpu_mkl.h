#include "../../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_MKL_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_MKL_H

#define BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_CPU_DELAY_OPERATIONS_GENERIC_INCLUDE
#include "operations_cpu.h"
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::rl::components::off_policy_runner{
    template<typename DEV_SPEC, typename SPEC, typename RNG>
    void prologue(devices::CPU_MKL<DEV_SPEC>& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng) {
        prologue((devices::CPU<DEV_SPEC>&)device, runner, rng);
    }
    template<typename DEV_SPEC, typename SPEC, typename RNG>
    void epilogue(devices::CPU_MKL<DEV_SPEC>& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng) {
        epilogue((devices::CPU<DEV_SPEC>&)device, runner, rng);
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#include "operations_generic.h"
#endif
