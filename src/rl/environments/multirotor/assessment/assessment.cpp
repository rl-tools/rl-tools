#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>
#include <backprop_tools/rl/environments/multirotor/ui.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/nn_models/persist.h>

namespace bpt = BACKPROP_TOOLS_NAMESPACE_WRAPPER ::backprop_tools;

#ifdef BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
#include "ppo/parameters.h"
#else
#include "../td3/parameters.h"
#endif

#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <highfive/H5File.hpp>
#include <CLI/CLI.hpp>
#include <tuple>

#include "full_assessment.h"


namespace TEST_DEFINITIONS{
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    namespace parameter_set = parameters_0;

    using penv = parameter_set::environment<T, TI, parameters::DefaultAblationSpec>;
    using ENVIRONMENT = penv::ENVIRONMENT;
    using UI = bpt::rl::environments::multirotor::UI<ENVIRONMENT>;

    using prl = parameter_set::rl<T, TI, penv::ENVIRONMENT>;
    constexpr TI MAX_EPISODE_LENGTH = 1000;
    constexpr bool RANDOMIZE_DOMAIN_PARAMETERS = true;
    constexpr bool INIT_SIMPLE = true;
    constexpr bool DEACTIVATE_OBSERVATION_NOISE = true;
}
template <typename DEVICE, typename ACTOR_TYPE>
void load_actor(DEVICE& device, std::string arg_run, std::string arg_checkpoint, ACTOR_TYPE& actor){

    std::string run = arg_run;
    std::string checkpoint = arg_checkpoint;

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
        auto data_file = HighFive::File(checkpoint, HighFive::File::ReadOnly);
        bpt::load(device, actor, data_file.getGroup("actor"));
#ifdef BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
        bpt::load(device, observation_normalizer.mean, data_file.getGroup("observation_normalizer"), "mean");
        bpt::load(device, observation_normalizer.std, data_file.getGroup("observation_normalizer"), "std");
#endif
    }
}
int main(int argc, char** argv) {
    using DEVICE = TEST_DEFINITIONS::DEVICE;


    CLI::App app;
    std::string arg_run = "", arg_checkpoint = "";
    app.add_option("--run", arg_run, "path to the run's directory");
    app.add_option("--checkpoint", arg_checkpoint, "path to the checkpoint");

    CLI11_PARSE(app, argc, argv);

    DEVICE device;
    typename TEST_DEFINITIONS::prl::ACTOR_TYPE actor;
    bpt::malloc(device, actor);
    load_actor<DEVICE, typename TEST_DEFINITIONS::prl::ACTOR_TYPE>(device, arg_run, arg_checkpoint, actor);

    full_assessment<DEVICE, TEST_DEFINITIONS::ENVIRONMENT, typename TEST_DEFINITIONS::prl::ACTOR_TYPE>(device, actor, TEST_DEFINITIONS::penv::parameters);

    bpt::free(device, actor);
}

