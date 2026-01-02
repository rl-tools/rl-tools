#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_TRAJECTORIES_LISSAJOUS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_PARAMETERS_TRAJECTORIES_LISSAJOUS_H
#include "../../multirotor.h"

#include <rl_tools/math/operations_generic.h>

#include "trajectory.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::l2f::parameters::trajectories::lissajous{
        template <typename T>
        struct Parameters{
            T A = 0.5; // amplitude a
            T B = 1.0; // amplitude b
            T C = 0.0; // amplitude c
            T a = 2.0; // frequency a
            T b = 1.0; // frequency b
            T c = 1.0; // frequency c
            T duration = 6.5;
            T ramp_duration = 0.0;
        };

        template <typename DEVICE, typename T, typename PARAMETERS>
        rl::environments::l2f::parameters::trajectories::Step<T> evaluate(DEVICE& device, const PARAMETERS& params, T time){
            T time_velocity = (params.ramp_duration > 0)
                ? math::min(device.math, time, params.ramp_duration) / params.ramp_duration
                : (T)1.0;

            T ramp_time = time_velocity * math::min(device.math, time, params.ramp_duration) / 2.0;
            T progress = (ramp_time + math::max(device.math, (T)0.0, time - params.ramp_duration)) * 2.0 * math::PI<T> / params.duration;
            T d_progress = 2.0 * math::PI<T> * time_velocity / params.duration;

            rl::environments::l2f::parameters::trajectories::Step<T> step{};
            step.position[0] = params.A * math::sin(device.math, params.a * progress);
            step.position[1] = params.B * math::sin(device.math, params.b * progress);
            step.position[2] = params.C * math::sin(device.math, params.c * progress);
            step.yaw = 0;
            step.linear_velocity[0] = params.A * math::cos(device.math, params.a * progress) * params.a * d_progress;
            step.linear_velocity[1] = params.B * math::cos(device.math, params.b * progress) * params.b * d_progress;
            step.linear_velocity[2] = params.C * math::cos(device.math, params.c * progress) * params.c * d_progress;
            step.yaw_velocity = 0;

            return step;
        }
    }
    template <typename DEVICE, typename T, typename SPEC, typename RNG>
    void fill(DEVICE& device, rl::environments::l2f::parameters::trajectories::lissajous::Parameters<T>& params, rl_tools::rl::environments::l2f::parameters::trajectories::Trajectory<SPEC>& traj, RNG& rng){
        using TI = typename SPEC::TI;
        for(TI step_i = 0; step_i < SPEC::LENGTH; step_i++){
            T time = step_i * SPEC::DT;
            traj.steps[step_i] = rl::environments::l2f::parameters::trajectories::lissajous::evaluate(device, params, time);
        }
    }
    template <typename DEVICE, typename ENVIRONMENT, typename PARAMETERS, typename SPEC, typename RNG>
    void fill(DEVICE& device, ENVIRONMENT& env, PARAMETERS& params, rl_tools::rl::environments::l2f::parameters::trajectories::Trajectory<SPEC>& traj, RNG& rng){
        using T = typename SPEC::T;
        rl::environments::l2f::parameters::trajectories::lissajous::Parameters<T> default_parameters{};
        fill(device, default_parameters, traj, rng);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif