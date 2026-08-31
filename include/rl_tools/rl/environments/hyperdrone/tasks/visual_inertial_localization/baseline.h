#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_BASELINE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_BASELINE_H

#include "../../../l2f/quaternion_helper.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization {
    // IMU dead-reckoning reference estimator, operating in the dynamics frame from a known
    // initial pose/velocity. The simulated accelerometer is R_next^T((v_next - v_cur)/dt - g),
    // so the velocity update inverts it exactly (up to the gyro-integrated attitude error)
    template <typename T_T>
    struct DeadReckoningState {
        using T = T_T;
        T position[3];
        T orientation[4];
        T linear_velocity[3];
    };
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void dead_reckoning_step(DEVICE& device, DeadReckoningState<T>& state, const T accelerometer[3], const T gyroscope[3], const T gravity[3], T dt){
        using TI = typename DEVICE::index_t;
        T orientation_derivative[4];
        rl::environments::l2f::quaternion_derivative<DEVICE, T>(state.orientation, gyroscope, orientation_derivative);
        T orientation_norm = 0;
        for (TI dim_i = 0; dim_i < 4; dim_i++){
            state.orientation[dim_i] += orientation_derivative[dim_i] * dt;
            orientation_norm += state.orientation[dim_i] * state.orientation[dim_i];
        }
        orientation_norm = math::sqrt(device.math, orientation_norm);
        for (TI dim_i = 0; dim_i < 4; dim_i++){
            state.orientation[dim_i] /= orientation_norm;
        }
        T acceleration_world[3];
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, accelerometer, acceleration_world);
        for (TI dim_i = 0; dim_i < 3; dim_i++){
            T velocity_next = state.linear_velocity[dim_i] + (acceleration_world[dim_i] + gravity[dim_i]) * dt;
            state.position[dim_i] += (state.linear_velocity[dim_i] + velocity_next) * dt / (T)2;
            state.linear_velocity[dim_i] = velocity_next;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
