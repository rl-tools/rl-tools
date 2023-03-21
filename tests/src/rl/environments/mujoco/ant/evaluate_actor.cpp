
#include <layer_in_c/operations/cpu_tensorboard.h>

#include <layer_in_c/rl/environments/mujoco/ant/operations_cpu.h>
#include <layer_in_c/rl/environments/mujoco/ant/ui.h>
#include <layer_in_c/nn_models/operations_cpu.h>
#include <layer_in_c/nn_models/persist.h>

namespace lic = layer_in_c;

#ifdef LAYER_IN_C_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
#include "parameters_ppo.h"
#else
#include "parameters_td3.h"
#endif

#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <highfive/H5File.hpp>
#include <CLI/CLI.hpp>

namespace TEST_DEFINITIONS{
    using DEVICE = lic::devices::DefaultCPU_TENSORBOARD;
    using T = double;
    using TI = typename DEVICE::index_t;
    namespace parameter_set = parameters_0;

    using parameters_environment = parameter_set::environment<T, TI>;
    struct ENVIRONMENT_EVALUATION_PARAMETERS: parameters_environment::ENVIRONMENT_SPEC::PARAMETERS{
        constexpr static TI FRAME_SKIP = 5; // for smoother playback
    };
    using ENVIRONMENT_EVALUATION_SPEC = lic::rl::environments::mujoco::ant::Specification<T, TI, ENVIRONMENT_EVALUATION_PARAMETERS>;
    using ENVIRONMENT = lic::rl::environments::mujoco::Ant<ENVIRONMENT_EVALUATION_SPEC>;
    using UI = lic::rl::environments::mujoco::ant::UI<ENVIRONMENT>;

    using parameters_rl = parameter_set::rl<T, TI, ENVIRONMENT>;
    constexpr TI MAX_EPISODE_LENGTH = 1000;
}


int main(int argc, char** argv) {
    using namespace TEST_DEFINITIONS;
    CLI::App app;
    std::string run = "";
    app.add_option("--run", run, "path to the run's directory");
    CLI11_PARSE(app, argc, argv);
    DEVICE dev;
    ENVIRONMENT env;
    UI ui;
    parameters_rl::ACTOR_TYPE actor;
    lic::Matrix<lic::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    lic::Matrix<lic::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    typename ENVIRONMENT::State state, next_state;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    lic::malloc(dev, env);
    lic::malloc(dev, actor);
    lic::malloc(dev, action);
    lic::malloc(dev, observation);

    lic::init(dev, env, ui);
    while(true){
        std::filesystem::path actor_run;
        if(run == ""){
            std::filesystem::path actor_checkpoints_dir = "actor_checkpoints";
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
        std::vector<std::filesystem::path> actor_checkpoints;
        for (const auto& checkpoint : std::filesystem::directory_iterator(actor_run)) {
            if (checkpoint.is_regular_file()) {
                actor_checkpoints.push_back(checkpoint.path());
            }
        }
        std::sort(actor_checkpoints.begin(), actor_checkpoints.end());

        std::cout << "Loading actor from " << actor_checkpoints.back() << std::endl;
        auto data_file = HighFive::File(actor_checkpoints.back(), HighFive::File::ReadOnly);
        lic::load(dev, actor, data_file.getGroup("actor"));

        lic::sample_initial_state(dev, env, state, rng);
        T reward_acc = 0;
        for(int step_i = 0; step_i < MAX_EPISODE_LENGTH; step_i++){
            auto start = std::chrono::high_resolution_clock::now();
            lic::observe(dev, env, state, observation);
            lic::evaluate(dev, actor, observation, action);
            T dt = lic::step(dev, env, state, action, next_state);
            bool terminated_flag = lic::terminated(dev, env, next_state, rng);
            reward_acc += lic::reward(dev, env, state, action, next_state);
            lic::set_state(dev, ui, state);
            state = next_state;
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end-start;
            std::this_thread::sleep_for(std::chrono::milliseconds((int)((dt - diff.count())*1000)));
            if(terminated_flag || step_i == (MAX_EPISODE_LENGTH - 1)){
                std::cout << "Episode terminated after " << step_i << " steps with reward " << reward_acc << std::endl;
                break;
            }
        }
    }
}

