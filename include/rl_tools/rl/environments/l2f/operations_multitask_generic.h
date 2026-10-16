#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_MULTITASK_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_MULTITASK_GENERIC_H

#include "multirotor.h"

#include <rl_tools/utils/generic/vector_operations.h>
#include "quaternion_helper.h"

#include <rl_tools/utils/generic/typing.h>

#include <rl_tools/rl/environments/operations_generic.h>
#include "operations_cpu.h"

#include "../../../rl_tools.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename RNG, bool T_OVERWRITE>
    static void sample_initial_parameters(DEVICE& device, rl::environments::MultirotorMultiTask<SPEC>& env, typename rl::environments::MultirotorMultiTask<SPEC>::Parameters& parameters, RNG& rng){
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        using PARAMETERS = typename rl::environments::MultirotorMultiTask<SPEC>::Parameters;
        parameters = env.parameters;
        if constexpr(T_OVERWRITE || SPEC::SAMPLE_INITIAL_PARAMETERS){
            static_assert(SPEC::STATIC_PARAMETERS::N_DYNAMICS_VALUES >= 1);
            TI index = random::uniform_int_distribution(device.random, (TI)0, (TI)(SPEC::STATIC_PARAMETERS::N_DYNAMICS_VALUES - 1), rng);
            parameters.dynamics = SPEC::STATIC_PARAMETERS::DYNAMICS_VALUES[index];
            if constexpr(SPEC::STATIC_PARAMETERS::RANDOMIZE_THRUST_CURVES){
                for(TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
                    T factor = random::uniform_real_distribution(device.random, (T)1.0, (T)4.0, rng);
                    for(TI order_i = 0; order_i < 3; order_i++){
                        parameters.dynamics.rotor_thrust_coefficients[rotor_i][order_i] *= factor;
                    }
                }
            }
        }
    }

    namespace rl::environments::l2f::observations{
    }
    template <typename DEVICE, typename SPEC>
    auto get_description(DEVICE& device, rl::environments::MultirotorMultiTask<SPEC>& env){
        return "Multirotor Multitask: Samples random dynamics parameters from a distribution and provides the information as part of the observation to the agent.";
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
