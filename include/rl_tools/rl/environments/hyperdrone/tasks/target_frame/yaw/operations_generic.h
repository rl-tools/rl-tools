#include "../../../../../../version.h"
#include "../../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_YAW_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_YAW_OPERATIONS_GENERIC_H

#include "../../../../../../containers/tensor/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename STATE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void yaw_target(DEVICE& device, const STATE& state, T& cosine, T& sine){
        const T qw = state.orientation[0];
        const T qx = state.orientation[1];
        const T qy = state.orientation[2];
        const T qz = state.orientation[3];
        cosine = static_cast<T>(1) - static_cast<T>(2) * (qy * qy + qz * qz);
        sine = static_cast<T>(2) * (qx * qy + qw * qz);
        const T norm = math::sqrt(device.math, cosine * cosine + sine * sine);
        if(norm > static_cast<T>(1e-6)){
            cosine /= norm;
            sine /= norm;
        } else {
            cosine = static_cast<T>(1);
            sine = static_cast<T>(0);
        }
    }

    template<typename DEVICE, typename STATE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void yaw_targets(DEVICE& device, const Tensor<STATE_SPEC>& states, Tensor<TARGET_SPEC>& targets){
        static_assert(length(typename STATE_SPEC::SHAPE{}) == 1);
        static_assert(length(typename TARGET_SPEC::SHAPE{}) == 2);
        static_assert(get<0>(typename STATE_SPEC::SHAPE{}) == get<0>(typename TARGET_SPEC::SHAPE{}));
        static_assert(get<1>(typename TARGET_SPEC::SHAPE{}) == 2);
        using TI = typename STATE_SPEC::TI;
        using T = typename STATE_SPEC::T::T;
        for(TI instance_i = 0; instance_i < get<0>(typename STATE_SPEC::SHAPE{}); instance_i++){
            T cosine, sine;
            yaw_target(device, get_ref(device, states, instance_i), cosine, sine);
            set(device, targets, static_cast<typename TARGET_SPEC::T>(cosine), instance_i, 0);
            set(device, targets, static_cast<typename TARGET_SPEC::T>(sine), instance_i, 1);
        }
    }

    template<typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T yaw_error(DEVICE& device, T predicted_cosine, T predicted_sine, T target_cosine, T target_sine){
        const T norm = math::sqrt(device.math, predicted_cosine * predicted_cosine + predicted_sine * predicted_sine);
        if(norm > static_cast<T>(1e-6)){
            predicted_cosine /= norm;
            predicted_sine /= norm;
        } else {
            predicted_cosine = static_cast<T>(1);
            predicted_sine = static_cast<T>(0);
        }
        return math::atan2(device.math, predicted_sine * target_cosine - predicted_cosine * target_sine, predicted_cosine * target_cosine + predicted_sine * target_sine);
    }

    template<typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T yaw_curriculum_threshold(DEVICE& device, T current_degrees, T ratio, T minimum_degrees, T maximum_degrees){
        return math::clamp(device.math, ratio * current_degrees, minimum_degrees, maximum_degrees);
    }

    template<typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T yaw_curriculum_advance(DEVICE& device, T current_degrees, T full_degrees, T step_degrees, T frontier_rmse_degrees, T threshold_degrees){
        if(current_degrees < full_degrees && math::is_finite(device.math, frontier_rmse_degrees) && frontier_rmse_degrees <= threshold_degrees){
            return math::min(device.math, current_degrees + step_degrees, full_degrees);
        }
        return current_degrees;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
