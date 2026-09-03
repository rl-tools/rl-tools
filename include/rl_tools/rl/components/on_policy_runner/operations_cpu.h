#include "on_policy_runner.h"
#include "operations_generic_per_env.h"
#include "../episodes/operations_cpu.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components::on_policy_runner{
    constexpr auto get_num_threads(devices::ExecutionHints hints) {
        return 1;
    }
    template<typename TI, TI NUM_THREADS>
    constexpr TI get_num_threads(rl::components::on_policy_runner::ExecutionHints<TI, NUM_THREADS> hints) {
        return NUM_THREADS;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#ifndef RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CPU_DELAY_OPERATIONS_GENERIC_INCLUDE
#include "operations_generic.h"
#endif
