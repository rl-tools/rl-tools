#ifndef LEARNING_TO_FLY_SIMULATOR_OPERATIONS_CPU_H
#define LEARNING_TO_FLY_SIMULATOR_OPERATIONS_CPU_H
#include "operations_generic.h"

#include <random>
#ifdef RL_TOOLS_ENABLE_JSON
#include <nlohmann/json.hpp>
#endif
#include <string>
namespace rl_tools{
#ifdef RL_TOOLS_ENABLE_JSON
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters){
        return "{}";
    }
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::Parameters& parameters, typename rl::environments::Multirotor<SPEC>::State& state){
        return "{}";
    }
#endif
}

#endif