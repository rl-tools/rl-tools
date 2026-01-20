#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_GET_DESIRED_STATE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_GET_DESIRED_STATE_H

#include "../multirotor.h"

#include <rl_tools/utils/generic/vector_operations.h>
#include "../quaternion_helper.h"

#include <rl_tools/utils/generic/typing.h>

#include <rl_tools/rl/environments/operations_generic.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void get_desired_state(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const rl::environments::l2f::StateBase<STATE_SPEC>& state, rl::environments::l2f::StateBase<STATE_SPEC>& desired_state, RNG& rng){
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        desired_state.position[0] = 0;
        desired_state.position[1] = 0;
        desired_state.position[2] = 0;
        desired_state.linear_velocity[0] = 0;
        desired_state.linear_velocity[1] = 0;
        desired_state.linear_velocity[2] = 0;
    }
    template<typename T_TI>
    struct TrajectoryIndexResult{
        T_TI index;
        bool forward;
    };
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT TrajectoryIndexResult<typename DEVICE::index_t> get_trajectory_index(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const rl::environments::l2f::StateTrajectory<STATE_SPEC>& state, typename DEVICE::index_t offset=0){
        using TI = typename DEVICE::index_t;
        using ENVIRONMENT = rl::environments::Multirotor<SPEC>;
        static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
        static_assert(decltype(parameters.trajectory)::LENGTH == EPISODE_STEP_LIMIT);
        TI full_step = state.trajectory_step + offset;
        TI interval = full_step / EPISODE_STEP_LIMIT;
        bool forward = interval % 2 == 0;
        TI progress = full_step % EPISODE_STEP_LIMIT;
        TI index = forward ? progress : (EPISODE_STEP_LIMIT - progress - 1);
        return {index, forward};
    }
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void get_desired_state(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, PARAMETERS& parameters, const rl::environments::l2f::StateTrajectory<STATE_SPEC>& state, rl::environments::l2f::StateTrajectory<STATE_SPEC>& desired_state, RNG& rng){
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        T traj_dt = (T)decltype(parameters.trajectory)::DT;
        utils::assert_exit(device, parameters.integration.dt == traj_dt, "Mismatch between environment and trajectory parameters integration dt");
        auto traj_idx = get_trajectory_index(device, env, parameters, state);
        T direction = traj_idx.forward ? (T)1 : (T)-1;
        for (TI axis_i=0; axis_i < 3; axis_i++){
            desired_state.position[axis_i]        = parameters.trajectory.steps[traj_idx.index].position[axis_i];
            desired_state.linear_velocity[axis_i] = parameters.trajectory.steps[traj_idx.index].linear_velocity[axis_i] * direction;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif


