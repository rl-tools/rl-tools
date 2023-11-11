#include <rl_tools/operations/cpu.h>

#include <rl_tools/rl/environments/mujoco/ant/operations_cpu.h>
#include <rl_tools/rl/environments/mujoco/ant/ui.h>
#include <rl_tools/nn_models/operations_cpu.h>
#include <rl_tools/nn_models/persist.h>
#include <rl_tools/rl/components/running_normalizer/operations_generic.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#ifdef RL_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
#include "ppo/parameters.h"
#else
#include "td3/parameters.h"
#endif

#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <highfive/H5File.hpp>
#include <CLI/CLI.hpp>

namespace TEST_DEFINITIONS{
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    namespace parameter_set = parameters_0;

    using parameters_environment = parameter_set::environment<T, TI>;
    struct ENVIRONMENT_EVALUATION_PARAMETERS: parameters_environment::ENVIRONMENT_SPEC::PARAMETERS{
        constexpr static TI FRAME_SKIP = 5; // for smoother playback
    };
    using ENVIRONMENT_EVALUATION_SPEC = rlt::rl::environments::mujoco::ant::Specification<T, TI, ENVIRONMENT_EVALUATION_PARAMETERS>;
    using ENVIRONMENT = rlt::rl::environments::mujoco::Ant<ENVIRONMENT_EVALUATION_SPEC>;
    using UI = rlt::rl::environments::mujoco::ant::UI<ENVIRONMENT>;

    using parameters_rl = parameter_set::rl<T, TI, ENVIRONMENT>;
    constexpr TI MAX_EPISODE_LENGTH = 1000;
}


int main(int argc, char** argv) {
    using namespace TEST_DEFINITIONS;
    CLI::App app;
    std::string run = "", checkpoint = "";
    DEVICE::index_t startup_timeout = 0;
    app.add_option("--run", run, "path to the run's directory");
    app.add_option("--checkpoint", checkpoint, "path to the checkpoint");
    app.add_option("--timeout", startup_timeout, "time to wait after first render");

    CLI11_PARSE(app, argc, argv);
    DEVICE dev;
    ENVIRONMENT env;
    UI ui;
    parameters_rl::ACTOR_TYPE actor;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    typename ENVIRONMENT::State state, next_state;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    rlt::rl::components::RunningNormalizer<rlt::rl::components::running_normalizer::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM>> observation_normalizer;

    rlt::malloc(dev, env);
    rlt::malloc(dev, actor);
    rlt::malloc(dev, action);
    rlt::malloc(dev, observation);
    rlt::malloc(dev, observation_normalizer);

    rlt::init(dev, env, ui);
    rlt::init(dev, observation_normalizer);
    DEVICE::index_t episode_i = 0;
    while(true){
        std::filesystem::path actor_run;
        if(run == "" && checkpoint == ""){
#ifdef RL_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
            std::filesystem::path actor_checkpoints_dir = std::filesystem::path("checkpoints") / "ppo_ant";
#else
            std::filesystem::path actor_checkpoints_dir = std::filesystem::path("checkpoints") / "td3_ant";
#endif
            std::vector<std::filesystem::path> actor_runs;

            for (const auto& run : std::filesystem::directory_iterator(actor_checkpoints_dir)) {
                if (run.is_directory()) {
                    actor_runs.push_back(run.path());
                }
            }
            std::sort(actor_runs.begin(), actor_runs.end());
            actor_run = actor_runs.back();
        }
        else{
            actor_run = run;
        }
        if(checkpoint == ""){
            std::vector<std::filesystem::path> actor_checkpoints;
            for (const auto& checkpoint : std::filesystem::directory_iterator(actor_run)) {
                if (checkpoint.is_regular_file()) {
                    actor_checkpoints.push_back(checkpoint.path());
                }
            }
            std::sort(actor_checkpoints.begin(), actor_checkpoints.end());
            checkpoint = actor_checkpoints.back().string();
        }

        std::cout << "Loading actor from " << checkpoint << std::endl;
        {
            try{
                auto data_file = HighFive::File(checkpoint, HighFive::File::ReadOnly);
                rlt::load(dev, actor, data_file.getGroup("actor"));
#ifdef RL_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
                rlt::load(dev, observation_normalizer.mean, data_file.getGroup("observation_normalizer"), "mean");
                rlt::load(dev, observation_normalizer.std, data_file.getGroup("observation_normalizer"), "std");
#endif
            }
            catch(HighFive::FileException& e){
                std::cout << "Failed to load actor from " << checkpoint << std::endl;
                std::cout << "Error: " << e.what() << std::endl;
                continue;
            }
        }

        rlt::sample_initial_state(dev, env, state, rng);
        T reward_acc = 0;
        for(int step_i = 0; step_i < MAX_EPISODE_LENGTH; step_i++){
            auto start = std::chrono::high_resolution_clock::now();
            rlt::observe(dev, env, state, observation, rng);
            rlt::normalize(dev, observation_normalizer.mean, observation_normalizer.std, observation);
            rlt::evaluate(dev, actor, observation, action);
            T dt = rlt::step(dev, env, state, action, next_state, rng);
            bool terminated_flag = rlt::terminated(dev, env, next_state, rng);
            reward_acc += rlt::reward(dev, env, state, action, next_state, rng);
            rlt::set_state(dev, ui, state);
            state = next_state;
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end-start;
            if(startup_timeout > 0 && episode_i == 0 && step_i == 0){
                for(int timeout_step_i = 0; timeout_step_i < startup_timeout; timeout_step_i++){
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    if(timeout_step_i % 100 == 0){
                        rlt::set_state(dev, ui, state);
                    }
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds((int)((dt - diff.count())*1000)));
            if(terminated_flag || step_i == (MAX_EPISODE_LENGTH - 1)){
                std::cout << "Episode terminated after " << step_i << " steps with reward " << reward_acc << std::endl;
                break;
            }
        }
        episode_i++;
    }
}

