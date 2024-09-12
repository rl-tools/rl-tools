#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/sequential_v2/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>

#include <rl_tools/containers/matrix/persist.h>
#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn_models/sequential_v2/persist.h>

#include <rl_tools/ui_server/client/operations_websocket.h>

namespace rlt = rl_tools;

//#include <CLI/CLI.hpp>
#include <regex>
#include <iostream>
#include <filesystem>


#include "approximators.h"


constexpr bool ORIGINAL_CONDITIONS = true;
constexpr bool AUTOMATIC_RESET = false;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

#include "parameters.h"



std::filesystem::path find_latest_checkpoint(std::filesystem::path experiments_path){
    std::vector<std::string> experiments;
    for (const auto & entry : std::filesystem::directory_iterator(experiments_path)){
        if (entry.is_directory()){
            experiments.push_back(entry.path().filename());
        }
    }
    std::sort(experiments.begin(), experiments.end());
    while(true){ // finding experiment
        if(experiments.empty()){
            std::cout << "No experiments found" << std::endl;
            return std::filesystem::path{};
        }
        std::filesystem::path latest_experiment = experiments_path / experiments.back();
        std::cout << "Latest experiment: " << latest_experiment << std::endl;
        std::vector<std::string> populations;
        for (const auto & entry : std::filesystem::directory_iterator(latest_experiment)){
            if (entry.is_directory()){
                populations.push_back(entry.path().filename());
            }
        }
        std::sort(populations.begin(), populations.end());
        while(true){ // finding population
            if(populations.empty()){
                std::cout << "No populations found" << std::endl;
                experiments.pop_back();
                break;
            }
            std::filesystem::path latest_population = latest_experiment / populations.back();
            std::cout << "Latest population: " << latest_population << std::endl;
            std::vector<std::string> configurations;
            for (const auto & entry : std::filesystem::directory_iterator(latest_population)){
                if (entry.is_directory()){
                    configurations.push_back(entry.path().filename());
                }
            }
            std::sort(configurations.begin(), configurations.end());
            while(true){ // finding configuration
                if(configurations.empty()){
                    std::cout << "No configurations found" << std::endl;
                    populations.pop_back();
                    break;
                }
                std::filesystem::path latest_configuration = latest_population / configurations.back();
                std::cout << "Latest configuration: " << latest_configuration << std::endl;
                std::vector<std::string> seeds;
                for (const auto & entry : std::filesystem::directory_iterator(latest_configuration)){
                    if (entry.is_directory()){
                        seeds.push_back(entry.path().filename());
                    }
                }
                std::sort(seeds.begin(), seeds.end());
                while(true){
                    if(seeds.empty()){
                        std::cout << "No seeds found" << std::endl;
                        configurations.pop_back();
                        break;
                    }
                    std::filesystem::path latest_seed = latest_configuration / seeds.back();
                    std::cout << "Latest seed: " << latest_seed << std::endl;
                    auto steps_directory = latest_seed / "steps";
                    if(!std::filesystem::exists(steps_directory)){
                        std::cout << "No steps directory found" << std::endl;
                        seeds.pop_back();
                        continue;
                    }
                    std::vector<std::string> steps;
                    for (const auto & entry : std::filesystem::directory_iterator(steps_directory)){
                        if (entry.is_directory()){
                            steps.push_back(entry.path().filename());
                        }
                    }
                    std::sort(steps.begin(), steps.end());
                    while(true){
                        if(steps.empty()){
                            std::cout << "No steps found" << std::endl;
                            seeds.pop_back();
                            break;
                        }
                        std::filesystem::path latest_step = steps_directory / steps.back();
                        std::cout << "Latest step: " << latest_step << std::endl;
                        auto hdf5_checkpoint = latest_step / "checkpoint.h5";
                        if(!std::filesystem::exists(hdf5_checkpoint)){
                            std::cout << "No checkpoint.h5 found" << std::endl;
                            steps.pop_back();
                            continue;
                        }
                        std::cout << "Found checkpoint.h5: " << hdf5_checkpoint << std::endl;
                        return hdf5_checkpoint;
                    }
                }
            }
        }
    }
}

int main(){
    std::filesystem::path experiments_path = "experiments";
    std::filesystem::path latest_checkpoint = find_latest_checkpoint(experiments_path);
    if(latest_checkpoint.empty()){
        std::cout << "No checkpoints found" << std::endl;
        return 1;
    }
    std::cout << "Latest checkpoint: " << latest_checkpoint << std::endl;

    using ACTOR = LOOP_CORE_CONFIG::ACTOR_CRITIC_TYPE::SPEC::ACTOR_NETWORK_TYPE::CHANGE_CAPABILITY<rlt::nn::layer_capability::Forward>;
    ACTOR actor;
    ACTOR::Buffer<1> buffer;


    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);

    rlt::malloc(device, actor);
    rlt::malloc(device, buffer);

    HighFive::File file(latest_checkpoint, HighFive::File::ReadOnly);

    rlt::load(device, actor, file.getGroup("actor"));

    using ENVIRONMENT_UI = rlt::ui_server::client::UIWebSocket<ENVIRONMENT>;
    ENVIRONMENT_UI ui;

    ENVIRONMENT env;
    ENVIRONMENT::Parameters parameters;
    ENVIRONMENT::State state, next_state;


    rlt::malloc(device, env);
    rlt::sample_initial_parameters(device, env, parameters, rng);

    using STEP_BY_STEP_MODE = rlt::nn::layers::gru::StepByStepMode<rlt::mode::Default<>, rlt::nn::layers::gru::StepByStepModeSpecification<TI, ORIGINAL_CONDITIONS || AUTOMATIC_RESET>>;
    rlt::Mode<STEP_BY_STEP_MODE> mode;

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENVIRONMENT::Observation::DIM>>> observation;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ENVIRONMENT::ACTION_DIM>>> action;
    rlt::malloc(device, observation);
    rlt::malloc(device, action);

    auto observation_matrix_view = rlt::matrix_view(device, observation);
    auto action_matrix_view = rlt::matrix_view(device, action);




    rlt::init(device, env, parameters, ui);
    TI step = 0;
    mode.reset = true;
    bool truncated = true;
    T cumulative_rewards = 0;
    while(true){
        if(truncated){
            rlt::sample_initial_state(device, env, parameters, state, rng);
        }
        mode.step = step;

        rlt::observe(device, env, parameters, state, ENVIRONMENT::Observation{}, observation_matrix_view, rng);
        rlt::evaluate(device, actor, observation, action, buffer, rng, mode);
        T dt = rlt::step(device, env, parameters, state, action_matrix_view, next_state, rng);
        state = next_state;
        T reward = rlt::reward(device, env, parameters, state, action_matrix_view, next_state, rng);
        cumulative_rewards += reward;

        rlt::set_state(device, env, parameters, ui, state);
        std::this_thread::sleep_for(std::chrono::milliseconds((TI)(1000 * dt)));
        bool terminated = rlt::terminated(device, env, parameters, state, rng);
        truncated = terminated || step >= 500;
        mode.reset = truncated;
        if(truncated){
            std::cout << "Episode terminated after " << step << " steps with cumulative rewards: " << cumulative_rewards << std::endl;
            step = 0;
            cumulative_rewards = 0;
        }
        step++;
    }




    return 0;
}