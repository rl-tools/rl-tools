#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_CAR_OPERATIONS_JSON_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_CAR_OPERATIONS_JSON_H

#include "car.h"
#include <nlohmann/json.hpp>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::car{
    template <typename DEVICE, typename T>
    nlohmann::json json(const DEVICE& device, const rl::environments::car::Parameters<T>& parameters){
        nlohmann::json j;
        j["lf"]  = parameters.lf;
        j["lr"]  = parameters.lr;
        j["dt"]  = parameters.dt;
        return j;
    }
    template <typename DEVICE, typename T, typename TI, TI T_HEIGHT, TI T_WIDTH, TI T_TRACK_SCALE>
    nlohmann::json json(const DEVICE& device, const rl::environments::car::ParametersTrack<T, TI, T_HEIGHT, T_WIDTH, T_TRACK_SCALE>& parameters){
        nlohmann::json j = json(device, static_cast<const rl::environments::car::Parameters<T>&>(parameters));
        j["height"] = T_HEIGHT;
        j["width"] = T_WIDTH;
        j["track_scale"] = T_TRACK_SCALE/(T)1000.0;
        return j;
    }
    template <typename DEVICE, typename T, typename TI>
    nlohmann::json json(const DEVICE& device, const rl::environments::car::State<T, TI>& state){
        nlohmann::json j;
        j["x"] = state.x;
        j["y"] = state.y;
        j["mu"] = state.mu;
        j["vx"] = state.vx;
        j["vy"] = state.vy;
        j["omega"] = state.omega;
        return j;
    }
}

#endif