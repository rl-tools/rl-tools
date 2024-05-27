#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_EXTRACK_STATE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_EXTRACK_STATE_H


#include <chrono>
#include <filesystem>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::loop::steps::extrack{
    template<typename T_CONFIG, typename T_NEXT = typename T_CONFIG::NEXT::template State<typename T_CONFIG::NEXT>>
    struct State: T_NEXT {
        using CONFIG = T_CONFIG;
        using NEXT = T_NEXT;
        std::filesystem::path extrack_base_path = "experiments";
        std::string extrack_experiment;
        std::string extrack_name = "default";
        std::string extrack_population_variates = "default";
        std::string extrack_population_values = "default";
        std::filesystem::path extrack_experiment_path;
        std::filesystem::path extrack_setup_path;
        std::filesystem::path extrack_config_path;
        std::filesystem::path extrack_seed_path;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif




