#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_EXTRACK_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_EXTRACK_OPERATIONS_CPU_H

#include "config.h"
#include "./utils/operations_cpu.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>

#include <cstdlib>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG>
    void init(DEVICE& device, rl::loop::steps::extrack::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using STATE = rl::loop::steps::extrack::State<T_CONFIG>;
        init(device, static_cast<typename STATE::NEXT&>(ts), seed);
        if(ts.extrack_experiment_path.empty()){
            utils::assert_exit(device, !ts.extrack_base_path.empty(), "Extrack base path (-e,--extrack) must be set if the Extrack experiment path (--ee,--extrack-experiment) is not set.");
            if(ts.extrack_experiment.empty()){

                auto environment_extrack_experiment = std::getenv("RL_TOOLS_EXTRACK_EXPERIMENT");
                if(environment_extrack_experiment != nullptr){
                    ts.extrack_experiment = std::string(environment_extrack_experiment);
                } else {
                    ts.extrack_experiment = rl::loop::steps::extrack::utils::get_timestamp_string();
                }
            }
            ts.extrack_experiment_path = ts.extrack_base_path / ts.extrack_experiment;
        }
        {
            std::string commit_hash = RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH);
            std::string setup_name = commit_hash.substr(0, 7) + "_" + ts.extrack_name;
            if(ts.extrack_population_variates != ""){
                setup_name = setup_name + "_" + ts.extrack_population_variates;
            }
            ts.extrack_setup_path = ts.extrack_experiment_path / setup_name;
            ts.extrack_config_path = ts.extrack_setup_path / ts.extrack_population_values;
        }
        {
            std::stringstream padded_seed_ss;
            padded_seed_ss << std::setw(4) << std::setfill('0') << seed;
            ts.extrack_seed_path = ts.extrack_config_path / padded_seed_ss.str();
        }
        std::cerr << "Seed: " << seed << std::endl;
        std::cerr << "Extrack Experiment: " << ts.extrack_seed_path << std::endl;
    }


    template <typename DEVICE, typename T_CONFIG>
    void free(DEVICE& device, rl::loop::steps::extrack::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::extrack::State<T_CONFIG>;
        free(device, static_cast<typename STATE::NEXT&>(ts));
    }

    template <typename DEVICE, typename CONFIG>
    bool step(DEVICE& device, rl::loop::steps::extrack::State<CONFIG>& ts){
        using TI = typename CONFIG::TI;
        using STATE = rl::loop::steps::extrack::State<CONFIG>;
        bool finished = step(device, static_cast<typename STATE::NEXT&>(ts));
        return finished;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
