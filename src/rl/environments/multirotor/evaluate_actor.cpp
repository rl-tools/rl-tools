#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>
#include <backprop_tools/rl/environments/multirotor/ui.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/nn_models/persist.h>

namespace bpt = backprop_tools;

#ifdef BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
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
    using DEVICE = bpt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    namespace parameter_set = parameters_0;

    using penv = parameter_set::environment<T, TI>;
    using ENVIRONMENT = penv::ENVIRONMENT;
    using UI = bpt::rl::environments::multirotor::UI<ENVIRONMENT>;

    using prl = parameter_set::rl<T, TI, penv::ENVIRONMENT>;
    constexpr TI MAX_EPISODE_LENGTH = 1000;
}


int main(int argc, char** argv) {
    using namespace TEST_DEFINITIONS;
    CLI::App app;
    std::string arg_run = "", arg_checkpoint = "";
    DEVICE::index_t startup_timeout = 0;
    app.add_option("--run", arg_run, "path to the run's directory");
    app.add_option("--checkpoint", arg_checkpoint, "path to the checkpoint");
    app.add_option("--timeout", startup_timeout, "time to wait after first render");

    CLI11_PARSE(app, argc, argv);
    DEVICE dev;
    ENVIRONMENT env;
    env.parameters = penv::parameters;
    UI ui;
    typename prl::ACTOR_TYPE actor;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    typename ENVIRONMENT::State state, next_state;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    bpt::malloc(dev, env);
    bpt::malloc(dev, actor);
    bpt::malloc(dev, action);
    bpt::malloc(dev, observation);

    ui.host = "localhost";
    ui.port = "8080";
    bpt::init(dev, env, ui);
    DEVICE::index_t episode_i = 0;
    std::string run = arg_run;
    std::string checkpoint = arg_checkpoint;
    while(true){
        std::filesystem::path actor_run;
        if(run == "" && checkpoint == ""){
#ifdef BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
            std::filesystem::path actor_checkpoints_dir = std::filesystem::path("checkpoints") / "multirotor_ppo";
#else
            std::filesystem::path actor_checkpoints_dir = std::filesystem::path("checkpoints") / "multirotor_td3";
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
                    if(checkpoint.path().extension() == ".h5" || checkpoint.path().extension() == ".hdf5"){
                        actor_checkpoints.push_back(checkpoint.path());
                    }
                }
            }
            std::sort(actor_checkpoints.begin(), actor_checkpoints.end());
            checkpoint = actor_checkpoints.back().string();
        }

        std::cout << "Loading actor from " << checkpoint << std::endl;
        {
            try{
                auto data_file = HighFive::File(checkpoint, HighFive::File::ReadOnly);
                bpt::load(dev, actor, data_file.getGroup("actor"));
#ifdef BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
                bpt::load(dev, observation_normalizer.mean, data_file.getGroup("observation_normalizer"), "mean");
                bpt::load(dev, observation_normalizer.std, data_file.getGroup("observation_normalizer"), "std");
#endif
            }
            catch(HighFive::FileException& e){
                std::cout << "Failed to load actor from " << checkpoint << std::endl;
                std::cout << "Error: " << e.what() << std::endl;
                continue;
            }
        }
        if(arg_checkpoint == ""){
            checkpoint = "";
        }
        if(arg_run == ""){
            run = "";
        }

        bpt::sample_initial_state(dev, env, state, rng);
        T reward_acc = 0;
        for(int step_i = 0; step_i < MAX_EPISODE_LENGTH; step_i++){
            auto start = std::chrono::high_resolution_clock::now();
            bpt::observe(dev, env, state, observation, rng);
            bpt::evaluate(dev, actor, observation, action);
            T dt = bpt::step(dev, env, state, action, next_state);
            bool terminated_flag = bpt::terminated(dev, env, next_state, rng);
            reward_acc += bpt::reward(dev, env, state, action, next_state);
            bpt::set_state(dev, ui, state);
            state = next_state;
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end-start;
            if(startup_timeout > 0 && episode_i == 0 && step_i == 0){
                for(int timeout_step_i = 0; timeout_step_i < startup_timeout; timeout_step_i++){
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    if(timeout_step_i % 100 == 0){
                        bpt::set_state(dev, ui, state);
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
