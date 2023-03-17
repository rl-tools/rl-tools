
#include <layer_in_c/operations/cpu_tensorboard.h>

#include <layer_in_c/rl/environments/mujoco/ant/operations_cpu.h>
#include <layer_in_c/rl/environments/mujoco/ant/ui.h>
#include <layer_in_c/nn_models/operations_cpu.h>
#include <layer_in_c/nn_models/persist.h>

namespace lic = layer_in_c;

#include "parameters.h"

#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <highfive/H5File.hpp>

namespace TEST_DEFINITIONS{
    using DEVICE = lic::devices::DefaultCPU_TENSORBOARD;
    using T = double;
    using TI = typename DEVICE::index_t;
    namespace parameter_set = parameters_0;

    using parameters_environment = parameter_set::environment<DEVICE, T>;
    struct ENVIRONMENT_EVALUATION_PARAMETERS: parameters_environment::ENVIRONMENT_SPEC::PARAMETERS{
        constexpr static TI FRAME_SKIP = 1; // for smoother playback
    };
    using ENVIRONMENT_EVALUATION_SPEC = lic::rl::environments::mujoco::ant::Specification<T, TI, ENVIRONMENT_EVALUATION_PARAMETERS>;
    using ENVIRONMENT = lic::rl::environments::mujoco::Ant<ENVIRONMENT_EVALUATION_SPEC>;
    using UI = lic::rl::environments::mujoco::ant::UI<ENVIRONMENT>;

    using parameters_rl = parameter_set::rl<DEVICE, T, ENVIRONMENT>;
}


int main(int argc, char** argv) {
    while(true){
        using namespace TEST_DEFINITIONS;

        std::filesystem::path actor_checkpoints_dir = "actor_checkpoints";
        std::vector<std::filesystem::path> actor_runs;

        for (const auto& run : std::filesystem::directory_iterator(actor_checkpoints_dir)) {
            if (run.is_directory()) {
                actor_runs.push_back(run.path());
            }
        }
        std::sort(actor_runs.begin(), actor_runs.end());
        std::vector<std::filesystem::path> actor_checkpoints;
        for (const auto& checkpoint : std::filesystem::directory_iterator(actor_runs.back())) {
            if (checkpoint.is_regular_file()) {
                actor_checkpoints.push_back(checkpoint.path());
            }
        }
        std::sort(actor_checkpoints.begin(), actor_checkpoints.end());

        DEVICE dev;
        ENVIRONMENT env;
        UI ui;
        parameters_rl::ACTOR_NETWORK_TYPE actor;
        lic::Matrix<lic::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
        lic::Matrix<lic::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
        typename ENVIRONMENT::State state, next_state;
        auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

        lic::malloc(dev, env);
        lic::malloc(dev, actor);
        lic::malloc(dev, action);
        lic::malloc(dev, observation);

        lic::init(dev, env, ui);
        std::cout << "Loading actor from " << actor_checkpoints.back() << std::endl;
        auto data_file = HighFive::File(actor_checkpoints.back(), HighFive::File::ReadOnly);
        lic::load(dev, actor, data_file.getGroup("actor"));

        lic::sample_initial_state(dev, env, state, rng);
        for(int i = 0; i < 5*1000; i++){
            auto start = std::chrono::high_resolution_clock::now();
            lic::observe(dev, env, state, observation);
            lic::evaluate(dev, actor, observation, action);
            T dt = lic::step(dev, env, state, action, next_state);
            lic::set_state(dev, ui, state);
            state = next_state;
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end-start;
            std::this_thread::sleep_for(std::chrono::milliseconds((int)((dt - diff.count())*1000)));
        }
    }
}

