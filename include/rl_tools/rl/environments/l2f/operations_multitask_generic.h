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

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename RNG>
    static void sample_initial_parameters(DEVICE& device, rl::environments::MultirotorMultiTask<SPEC>& env, typename rl::environments::MultirotorMultiTask<SPEC>::Parameters& parameters, RNG& rng, bool reset){
        using TI = typename DEVICE::index_t;
        parameters = env.parameters;
        static_assert(SPEC::STATIC_PARAMETERS::N_DYNAMICS_VALUES >= 1);
        TI index = random::uniform_int_distribution(device.random, (TI)0, (TI)(SPEC::STATIC_PARAMETERS::N_DYNAMICS_VALUES - 1), rng);
        parameters.dynamics = SPEC::STATIC_PARAMETERS::DYNAMICS_VALUES[index];
        // // hack
        // for(TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
        //     for(TI order_i = 0; order_i < 3; order_i++){
        //
        //         T normal = random::normal_distribution::sample(device.random, (T)0, (T)0.1, rng);
        //         parameters.dynamics.rotor_thrust_coefficients[rotor_i][order_i] *= normal;
        //     }
        // }
    }

    namespace rl::environments::l2f::observations{
        template<typename DEVICE, typename SPEC, typename OBSERVATION_SPEC, typename OBS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, const typename rl::environments::Multirotor<SPEC>::State& state, rl::environments::l2f::observation::ParametersMass<OBSERVATION_SPEC>, Matrix<OBS_SPEC>& observation, RNG& rng){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using OBSERVATION = rl::environments::l2f::observation::ParametersMass<OBSERVATION_SPEC>;
            static_assert(OBS_SPEC::COLS >= OBSERVATION::CURRENT_DIM);
            static_assert(OBS_SPEC::ROWS == 1);
            set(observation, 0, 0, parameters.dynamics.mass);
            auto next_observation = view(device, observation, matrix::ViewSpec<1, OBS_SPEC::COLS - OBSERVATION::CURRENT_DIM>{}, 0, OBSERVATION::CURRENT_DIM);
            observe(device, env, parameters, state, typename OBSERVATION::NEXT_COMPONENT{}, next_observation, rng);
        }
    }
    template <typename DEVICE, typename SPEC>
    auto get_description(DEVICE& device, rl::environments::MultirotorMultiTask<SPEC>& env){
        return "Multirotor Multitask: Samples random dynamics parameters from a distribution and provides the information as part of the observation to the agent.";
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
