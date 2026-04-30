#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_SETPOINT_ATTITUDE_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_SETPOINT_ATTITUDE_CPU_H

#include "setpoint_attitude.h"

#ifdef RL_TOOLS_ENABLE_JSON
#include <nlohmann/json.hpp>
#endif

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename OBS_SPEC>
    std::string string(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::l2f::observation::AttitudeSetpoint<OBS_SPEC>& obs, bool first){
        using OBSERVATION = rl::environments::l2f::observation::AttitudeSetpoint<OBS_SPEC>;
        return std::string(first ? "" : ".") + "AttitudeSetpoint" + rl::environments::l2f::obs_helper::dispatch(device, env, typename OBSERVATION::NEXT_COMPONENT{}, false);
    }

    template <typename DEVICE, typename T_A, typename TI_A, typename T_B, typename TI_B>
    RL_TOOLS_FUNCTION_PLACEMENT T_A abs_diff(DEVICE& device, const rl::environments::l2f::parameters::AttitudeSetpointSampling<T_A, TI_A>& a, const rl::environments::l2f::parameters::AttitudeSetpointSampling<T_B, TI_B>& b) {
        T_A acc = 0;
        acc += math::abs(device.math, a.max_tilt_angle - b.max_tilt_angle);
        acc += math::abs(device.math, a.max_yaw_rate - b.max_yaw_rate);
        acc += math::abs(device.math, a.thrust_min_g - b.thrust_min_g);
        acc += math::abs(device.math, a.thrust_max_g - b.thrust_max_g);
        acc += math::abs(device.math, (T_A)a.hold_steps_min - (T_A)b.hold_steps_min);
        acc += math::abs(device.math, (T_A)a.hold_steps_max - (T_A)b.hold_steps_max);
        return acc;
    }
    template <typename DEVICE, typename SPEC_A, typename SPEC_B>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_A::T abs_diff(DEVICE& device, const rl::environments::l2f::ParametersAttitudeSetpoint<SPEC_A>& a, const rl::environments::l2f::ParametersAttitudeSetpoint<SPEC_B>& b) {
        typename SPEC_A::T acc = 0;
        acc += abs_diff(device, static_cast<const typename SPEC_A::NEXT_COMPONENT&>(a), static_cast<const typename SPEC_B::NEXT_COMPONENT&>(b));
        acc += abs_diff(device, a.attitude_setpoint_sampling, b.attitude_setpoint_sampling);
        return acc;
    }

    template <typename DEVICE, typename T, typename TI>
    std::string json(DEVICE& device, const rl::environments::l2f::parameters::AttitudeSetpointSampling<T, TI>& sampling){
        std::string json_string = "{";
        json_string += "\"max_tilt_angle\": " + std::to_string(sampling.max_tilt_angle) + ", ";
        json_string += "\"max_yaw_rate\": " + std::to_string(sampling.max_yaw_rate) + ", ";
        json_string += "\"thrust_min_g\": " + std::to_string(sampling.thrust_min_g) + ", ";
        json_string += "\"thrust_max_g\": " + std::to_string(sampling.thrust_max_g) + ", ";
        json_string += "\"hold_steps_min\": " + std::to_string(sampling.hold_steps_min) + ", ";
        json_string += "\"hold_steps_max\": " + std::to_string(sampling.hold_steps_max);
        json_string += "}";
        return json_string;
    }
    template <typename DEVICE, typename SPEC, typename PARAM_SPEC>
    std::string json(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::l2f::ParametersAttitudeSetpoint<PARAM_SPEC>& parameters, bool top_level=true){
        std::string json_string = top_level ? "{" : "";
        json_string += json(device, env, static_cast<const typename PARAM_SPEC::NEXT_COMPONENT&>(parameters), false);
        json_string += ", \"attitude_setpoint_sampling\": " + json(device, parameters.attitude_setpoint_sampling);
        json_string += top_level ? "}" : "";
        return json_string;
    }
    template <typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC>
    std::string json(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const PARAMETERS& parameters, const rl::environments::l2f::StateAttitudeSetpoint<STATE_SPEC>& state, bool top_level=true){
        std::string json_string = top_level ? "{" : "";
        json_string += json(device, env, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), false) + ", ";
        json_string += "\"attitude_setpoint\": {";
        json_string += "\"roll\": " + std::to_string(state.target_roll) + ", ";
        json_string += "\"pitch\": " + std::to_string(state.target_pitch) + ", ";
        json_string += "\"yaw_rate\": " + std::to_string(state.target_yaw_rate) + ", ";
        json_string += "\"thrust_g\": " + std::to_string(state.target_thrust_g) + ", ";
        json_string += "\"steps_remaining\": " + std::to_string(state.target_steps_remaining);
        json_string += "}";
        json_string += top_level ? "}" : "";
        return json_string;
    }

#ifdef RL_TOOLS_ENABLE_JSON
    template <typename DEVICE, typename SPEC, typename PARAM_SPEC>
    void from_json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, nlohmann::json json_object, rl::environments::l2f::ParametersAttitudeSetpoint<PARAM_SPEC>& parameters){
        from_json(device, env, json_object, static_cast<typename PARAM_SPEC::NEXT_COMPONENT&>(parameters));
        const auto& s = json_object["attitude_setpoint_sampling"];
        parameters.attitude_setpoint_sampling.max_tilt_angle = s["max_tilt_angle"];
        parameters.attitude_setpoint_sampling.max_yaw_rate = s["max_yaw_rate"];
        parameters.attitude_setpoint_sampling.thrust_min_g = s["thrust_min_g"];
        parameters.attitude_setpoint_sampling.thrust_max_g = s["thrust_max_g"];
        parameters.attitude_setpoint_sampling.hold_steps_min = s["hold_steps_min"];
        parameters.attitude_setpoint_sampling.hold_steps_max = s["hold_steps_max"];
    }
    template <typename DEVICE, typename SPEC, typename PARAM_SPEC>
    void from_json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, std::string json_string, rl::environments::l2f::ParametersAttitudeSetpoint<PARAM_SPEC>& parameters){
        nlohmann::json json_object = nlohmann::json::parse(json_string);
        from_json(device, env, json_object, parameters);
    }
    template <typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC>
    void from_json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, const PARAMETERS& parameters, nlohmann::json json_object, rl::environments::l2f::StateAttitudeSetpoint<STATE_SPEC>& state){
        from_json(device, env, parameters, json_object, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(state));
        const auto& s = json_object["attitude_setpoint"];
        state.target_roll = s["roll"];
        state.target_pitch = s["pitch"];
        state.target_yaw_rate = s["yaw_rate"];
        state.target_thrust_g = s["thrust_g"];
        state.target_steps_remaining = s["steps_remaining"];
    }
    template <typename DEVICE, typename SPEC, typename PARAMETERS, typename STATE_SPEC>
    void from_json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, const PARAMETERS& parameters, std::string json_string, rl::environments::l2f::StateAttitudeSetpoint<STATE_SPEC>& state){
        nlohmann::json json_object = nlohmann::json::parse(json_string);
        from_json(device, env, parameters, json_object, state);
    }
#endif
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
