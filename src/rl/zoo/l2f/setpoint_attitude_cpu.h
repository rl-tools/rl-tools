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
#endif
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
