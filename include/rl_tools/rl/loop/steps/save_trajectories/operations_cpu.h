#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_SAVE_TRAJECTORIES_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_SAVE_TRAJECTORIES_OPERATIONS_CPU_H

#include "../../../../rl/algorithms/sac/operations_generic.h"
#include "../../../../rl/components/off_policy_runner/operations_generic.h"

#include "../../../../rl/environments/operations_generic.h"


#include "../../../../rl/utils/evaluation/operations_generic.h"

#include "../../../../utils/zlib/operations_cpu.h"

#include "config.h"
#include <string>
#include <sstream>
#include <fstream>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG>
    void malloc(DEVICE& device, rl::loop::steps::save_trajectories::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::save_trajectories::State<T_CONFIG>;
        malloc(device, ts.actor_deterministic_save_trajectories_buffers);
        ts.save_trajectories_buffer = new typename STATE::template DATA_TYPE<typename T_CONFIG::SAVE_TRAJECTORIES_SPEC>;
        malloc(device, static_cast<typename STATE::NEXT&>(ts));
    }
    template <typename DEVICE, typename T_CONFIG>
    void init(DEVICE& device, rl::loop::steps::save_trajectories::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using STATE = rl::loop::steps::save_trajectories::State<T_CONFIG>;
        init(device, static_cast<typename STATE::NEXT&>(ts), seed);
        ts.rng_save_trajectories = random::default_engine(typename DEVICE::SPEC::RANDOM{}, seed);
        ts.save_trajectories_ui_written = false;
    }

    template <typename DEVICE, typename T_CONFIG>
    void free(DEVICE& device, rl::loop::steps::save_trajectories::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::save_trajectories::State<T_CONFIG>;
        delete ts.save_trajectories_buffer;
        free(device, ts.actor_deterministic_save_trajectories_buffers);
        free(device, ts.actor_deterministic_evaluation_buffers);
        free(device, static_cast<typename STATE::NEXT&>(ts));
    }

    namespace rl::loop::steps::save_trajectories{
        template <typename DEVICE, typename ENVIRONMENT, typename SPEC>
        std::string to_string(DEVICE& device, ENVIRONMENT& env, rl::utils::evaluation::Data<SPEC>& data){
            using TI = typename DEVICE::index_t;
            std::string episodes_json = "[";
            for(TI episode_i = 0; episode_i < SPEC::N_EPISODES; episode_i++){
                std::string episode_json = "{";
                episode_json += "\"parameters\": " + std::string(json(device, env, data.parameters[episode_i])) + ",";
                std::string trajectory_json = "[";
                for(TI step_i = 0; step_i < SPEC::STEP_LIMIT; step_i++){
                    std::string step_json = "{";
                    step_json += "\"state\":" + std::string(json(device, env, data.parameters[episode_i], data.states[episode_i][step_i])) + ",";
                    std::string action_json = "\"action\":[";
                    for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                        action_json += std::to_string(data.actions[episode_i][step_i][action_i]) + ",";
                    }
                    action_json.pop_back();
                    action_json += "]";
                    step_json += action_json + ",";
                    step_json += "\"dt\":" + std::to_string(data.dt[episode_i][step_i]) + ",";
                    step_json += "\"reward\":" + std::to_string(data.rewards[episode_i][step_i]) + ",";
                    step_json += "\"terminated\":" + (data.terminated[episode_i][step_i] ? std::string("true") : std::string("false"));;
                    step_json += "}";
                    trajectory_json += step_json + ",";
                }
                trajectory_json.pop_back();
                trajectory_json += "]";
                episode_json += "\"trajectory\":" + trajectory_json + "}";
                episodes_json += episode_json + ",";
            }
            episodes_json.pop_back();
            episodes_json += "]";
            return episodes_json;
        }

    }

    template <typename DEVICE, typename CONFIG>
    bool step(DEVICE& device, rl::loop::steps::save_trajectories::State<CONFIG>& ts){
        using TS = rl::loop::steps::save_trajectories::State<CONFIG>;
        using TI = typename CONFIG::TI;
        using PARAMETERS = typename CONFIG::SAVE_TRAJECTORIES_PARAMETERS;
        using STATE = rl::loop::steps::save_trajectories::State<CONFIG>;
        if constexpr(PARAMETERS::SAVE_TRAJECTORIES == true){
            if(ts.step % PARAMETERS::INTERVAL == 0){
                if(!ts.save_trajectories_ui_written){
                    ts.save_trajectories_ui_written = true;
                    std::string ui = get_ui(device, ts.env_eval);
                    if(!ui.empty()){
                        std::filesystem::create_directories(ts.extrack_seed_path);
                        std::ofstream ui_jsmf(ts.extrack_seed_path / "ui.esm.js");
                        ui_jsmf << ui;
                        std::cout << "UI written to: " << ts.extrack_seed_path / "ui.esm.js" << std::endl;
                    }
                }
                typename TS::SAVE_TRAJECTORIES_ACTOR_TYPE evaluation_actor;
                malloc(device, evaluation_actor);
                auto actor = get_actor(ts);
                copy(device, device, actor, evaluation_actor);
                evaluate(device, ts.env_eval, ts.env_eval_parameters, ts.ui, evaluation_actor, ts.save_trajectories_result, *ts.save_trajectories_buffer, ts.actor_deterministic_save_trajectories_buffers, ts.rng_save_trajectories, ts.evaluation_mode, false, CONFIG::EVALUATION_PARAMETERS::SAMPLE_ENVIRONMENT_PARAMETERS);
                free(device, evaluation_actor);

                using PARAMS = typename CONFIG::SAVE_TRAJECTORIES_PARAMETERS;

                std::string trajectories_json = rl::loop::steps::save_trajectories::to_string(device, ts.env_eval, *ts.save_trajectories_buffer);
#ifndef RL_TOOLS_ENABLE_ZLIB
                std::string file_extension = "json";
                std::string trajectories_output = trajectories_json;
#else
                std::string file_extension = "json.gz";
                std::vector<uint8_t> trajectories_output;
                if(!compress_zlib(trajectories_json, trajectories_output)){
                    std::cerr << "Error while compressing trajectories." << std::endl;
                    return true;
                }
#endif


                {
                    std::stringstream step_ss;
                    step_ss << std::setw(15) << std::setfill('0') << ts.step;
                    std::filesystem::path step_folder = ts.extrack_seed_path / "steps" / step_ss.str();
                    std::filesystem::create_directories(step_folder);
                    std::filesystem::path trajectories_path = step_folder / ("trajectories." + file_extension);
                    std::cerr << "Saving Trajectories at step: " << ts.step << " to: " << trajectories_path << std::endl;
#ifndef RL_TOOLS_ENABLE_ZLIB
                    std::ofstream trajectories_file(trajectories_path);
                    trajectories_file << trajectories_output;
#else
                    std::ofstream trajectories_file(trajectories_path, std::ios::binary);
                    trajectories_file.write(reinterpret_cast<const char*>(trajectories_output.data()), trajectories_output.size());
#endif
                }
            }
        }
        bool finished = step(device, static_cast<typename STATE::NEXT&>(ts));
        return finished;
    }
    // to log the configuration
    template <typename DEVICE, typename PARAMETERS, typename utils::typing::enable_if<utils::typing::is_same_v<typename PARAMETERS::TAG, rl::loop::steps::save_trajectories::ParametersTag>>::type* = nullptr>
    void log(DEVICE& device, PARAMETERS){}
    template <typename DEVICE, typename CONFIG, typename utils::typing::enable_if<utils::typing::is_same_v<typename CONFIG::TAG, rl::loop::steps::save_trajectories::ConfigTag>>::type* = nullptr>
    void log(DEVICE& device, CONFIG){
        log(device, typename CONFIG::NEXT{});
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
