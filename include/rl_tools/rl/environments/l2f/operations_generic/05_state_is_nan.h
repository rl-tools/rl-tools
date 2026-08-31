#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_STATE_IS_NAN_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_STATE_IS_NAN_H

#include "../multirotor.h"

#include <rl_tools/utils/generic/vector_operations.h>
#include "../quaternion_helper.h"

#include <rl_tools/utils/generic/typing.h>

#include <rl_tools/rl/environments/operations_generic.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::l2f{
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StateBase<STATE_SPEC>& state){
            bool nan = false;
            for(typename DEVICE::index_t i = 0; i < 3; i++){
                nan = nan || math::is_nan(device.math, state.position[i]);
            }
            for(typename DEVICE::index_t i = 0; i < 4; i++){
                nan = nan || math::is_nan(device.math, state.orientation[i]);
            }
            for(typename DEVICE::index_t i = 0; i < 3; i++){
                nan = nan || math::is_nan(device.math, state.linear_velocity[i]);
            }
            for(typename DEVICE::index_t i = 0; i < 3; i++){
                nan = nan || math::is_nan(device.math, state.angular_velocity[i]);
            }
            return nan;
        }
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& state){
            is_nan(device, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state));
            bool nan = false;
            nan = nan || math::is_nan(device.math, state.position_integral[0]);
            nan = nan || math::is_nan(device.math, state.position_integral[1]);
            nan = nan || math::is_nan(device.math, state.position_integral[2]);
            nan = nan || math::is_nan(device.math, state.orientation_integral);
            return nan;
        }
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StateRandomForce<STATE_SPEC>& state){
            is_nan(device, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state));
            bool nan = false;
            nan = nan || math::is_nan(device.math, state.force[0]);
            nan = nan || math::is_nan(device.math, state.force[1]);
            nan = nan || math::is_nan(device.math, state.force[2]);
            nan = nan || math::is_nan(device.math, state.torque[0]);
            nan = nan || math::is_nan(device.math, state.torque[1]);
            nan = nan || math::is_nan(device.math, state.torque[2]);
            return nan;
        }
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StateRotors<STATE_SPEC>& state){
            is_nan(device, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state));
            bool nan = false;
            for(typename DEVICE::index_t i = 0; i < 4; i++){
                nan = nan || math::is_nan(device.math, state.rpm[i]);
            }
            return nan;
        }
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StateRotorsHistory<STATE_SPEC>& state){
            using STATE = rl::environments::l2f::StateRotorsHistory<STATE_SPEC>;
            using TI = typename DEVICE::index_t;
            is_nan(device, static_cast<typename STATE::NEXT_COMPONENT&>(state));
            bool nan = false;
            for(TI step_i = 0; step_i < STATE_SPEC::HISTORY_LENGTH; step_i++){
                for(TI action_i = 0; action_i < STATE::ACTION_DIM; action_i++){
                    nan = nan || math::is_nan(device.math, state.action_history[step_i][action_i]);
                }
            }
            return nan;
        }
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StateGyroBias<STATE_SPEC>& state){
            is_nan(device, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state));
            bool nan = false;
            for(typename DEVICE::index_t i = 0; i < 3; i++){
                nan = nan || math::is_nan(device.math, state.gyro_bias[i]);
            }
            return nan;
        }
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StateIMU<STATE_SPEC>& state){
            is_nan(device, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state));
            bool nan = false;
            for(typename DEVICE::index_t i = 0; i < 3; i++){
                nan = nan || math::is_nan(device.math, state.imu_accelerometer[i]);
                nan = nan || math::is_nan(device.math, state.imu_gyroscope[i]);
            }
            return nan;
        }
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StateMahony<STATE_SPEC>& state){
            is_nan(device, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state));
            bool nan = false;
            for(typename DEVICE::index_t i = 0; i < 4; i++){
                nan = nan || math::is_nan(device.math, state.q_estimate[i]);
            }
            for(typename DEVICE::index_t i = 0; i < 3; i++){
                nan = nan || math::is_nan(device.math, state.bias_estimate[i]);
            }
            return nan;
        }
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StateCTBRController<STATE_SPEC>& state){
            is_nan(device, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state));
            bool nan = false;
            for(typename DEVICE::index_t i = 0; i < 3; i++){
                nan = nan || math::is_nan(device.math, state.previous_angular_velocity[i]);
            }
            return nan;
        }
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StateLinearAccelerationHistory<STATE_SPEC>& state){
            using STATE = rl::environments::l2f::StateLinearAccelerationHistory<STATE_SPEC>;
            using TI = typename DEVICE::index_t;
            is_nan(device, static_cast<typename STATE::NEXT_COMPONENT&>(state));
            bool nan = false;
            for(TI step_i = 0; step_i < STATE::HISTORY_MEM_LENGTH; step_i++){
                for(TI dim_i = 0; dim_i < STATE::ACCELERATION_DIM; dim_i++){
                    nan = nan || math::is_nan(device.math, state.linear_acceleration_body_history[step_i][dim_i]);
                }
            }
            return nan;
        }
        template<typename DEVICE, typename STATE_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT static bool _is_nan(DEVICE& device, rl::environments::l2f::StateRenderRotorPhase<STATE_SPEC>& state){
            using STATE = rl::environments::l2f::StateRenderRotorPhase<STATE_SPEC>;
            using TI = typename DEVICE::index_t;
            is_nan(device, static_cast<typename STATE::NEXT_COMPONENT&>(state));
            bool nan = false;
            for(TI rotor_i = 0; rotor_i < STATE::ACTION_DIM; rotor_i++){
                nan = nan || math::is_nan(device.math, state.rotor_phase[rotor_i]);
            }
            return nan;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
