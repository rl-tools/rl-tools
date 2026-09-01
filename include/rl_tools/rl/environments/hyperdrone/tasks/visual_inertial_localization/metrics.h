#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_METRICS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_METRICS_H

#include "../../../l2f/quaternion_helper.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization {
    template <typename T_T, typename T_TI>
    struct TrajectoryMetricsAccumulator {
        using T = T_T;
        using TI = T_TI;
        TI count = 0;
        T position_squared_error_sum = 0;
        T rotation_error_sum = 0;
        T rotation_error_max = 0;
        T distance_traveled = 0;
        T previous_position[3] = {};
        bool has_previous = false;
    };
    template <typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, TrajectoryMetricsAccumulator<T, TI>& accumulator){
        accumulator.count = 0;
        accumulator.position_squared_error_sum = 0;
        accumulator.rotation_error_sum = 0;
        accumulator.rotation_error_max = 0;
        accumulator.distance_traveled = 0;
        accumulator.has_previous = false;
    }
    // T_origin^-1 * T: the benchmark's evaluation frame is the initial pose, so the estimator
    // needs no ground-truth anchor — it starts at identity by convention
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void relative_pose(DEVICE& device, const T origin_position[3], const T origin_orientation[4], const T position[3], const T orientation[4], T relative_position[3], T relative_orientation[4]){
        using TI = typename DEVICE::index_t;
        T conjugate_origin[4] = {origin_orientation[0], -origin_orientation[1], -origin_orientation[2], -origin_orientation[3]};
        T delta[3];
        for (TI dim_i = 0; dim_i < 3; dim_i++){
            delta[dim_i] = position[dim_i] - origin_position[dim_i];
        }
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(conjugate_origin, delta, relative_position);
        relative_orientation[0] = conjugate_origin[0] * orientation[0] - conjugate_origin[1] * orientation[1] - conjugate_origin[2] * orientation[2] - conjugate_origin[3] * orientation[3];
        relative_orientation[1] = conjugate_origin[0] * orientation[1] + conjugate_origin[1] * orientation[0] + conjugate_origin[2] * orientation[3] - conjugate_origin[3] * orientation[2];
        relative_orientation[2] = conjugate_origin[0] * orientation[2] - conjugate_origin[1] * orientation[3] + conjugate_origin[2] * orientation[0] + conjugate_origin[3] * orientation[1];
        relative_orientation[3] = conjugate_origin[0] * orientation[3] + conjugate_origin[1] * orientation[2] - conjugate_origin[2] * orientation[1] + conjugate_origin[3] * orientation[0];
    }
    template <typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T quaternion_geodesic_distance(DEVICE& device, const T a[4], const T b[4]){
        T dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        T dot_abs = math::abs(device.math, dot);
        dot_abs = dot_abs > (T)1 ? (T)1 : dot_abs;
        return (T)2 * math::acos(device.math, dot_abs);
    }
    template <typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void accumulate(DEVICE& device, TrajectoryMetricsAccumulator<T, TI>& accumulator, const T ground_truth_position[3], const T ground_truth_orientation[4], const T estimated_position[3], const T estimated_orientation[4]){
        T position_squared_error = 0;
        for (TI dim_i = 0; dim_i < 3; dim_i++){
            T delta = estimated_position[dim_i] - ground_truth_position[dim_i];
            position_squared_error += delta * delta;
        }
        accumulator.position_squared_error_sum += position_squared_error;
        T rotation_error = quaternion_geodesic_distance(device, ground_truth_orientation, estimated_orientation);
        accumulator.rotation_error_sum += rotation_error;
        if (rotation_error > accumulator.rotation_error_max){
            accumulator.rotation_error_max = rotation_error;
        }
        if (accumulator.has_previous){
            T step_distance_squared = 0;
            for (TI dim_i = 0; dim_i < 3; dim_i++){
                T delta = ground_truth_position[dim_i] - accumulator.previous_position[dim_i];
                step_distance_squared += delta * delta;
            }
            accumulator.distance_traveled += math::sqrt(device.math, step_distance_squared);
        }
        for (TI dim_i = 0; dim_i < 3; dim_i++){
            accumulator.previous_position[dim_i] = ground_truth_position[dim_i];
        }
        accumulator.has_previous = true;
        accumulator.count++;
    }
    template <typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT T ate_position_rmse(DEVICE& device, const TrajectoryMetricsAccumulator<T, TI>& accumulator){
        return accumulator.count == 0 ? (T)0 : math::sqrt(device.math, accumulator.position_squared_error_sum / (T)accumulator.count);
    }
    template <typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT T mean_rotation_error(DEVICE& device, const TrajectoryMetricsAccumulator<T, TI>& accumulator){
        return accumulator.count == 0 ? (T)0 : accumulator.rotation_error_sum / (T)accumulator.count;
    }
    template <typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT T max_rotation_error(DEVICE& device, const TrajectoryMetricsAccumulator<T, TI>& accumulator){
        return accumulator.rotation_error_max;
    }
    template <typename DEVICE, typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT T drift_per_distance(DEVICE& device, const TrajectoryMetricsAccumulator<T, TI>& accumulator){
        return accumulator.distance_traveled == 0 ? (T)0 : ate_position_rmse(device, accumulator) / accumulator.distance_traveled;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
