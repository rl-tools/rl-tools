#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>

#include "td3/pendulum-v1.h"
#include "sac/pendulum-v1.h"

#include <rl_tools/rl/algorithms/td3/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

#ifdef RL_TOOLS_ENABLE_CLI11
#include <CLI/CLI.hpp>
#endif


using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

// These options should be coherent ------------------------------------------------------
//using LOOP_CONFIG = rlt::rl::zoo::td3::PendulumV1<DEVICE, T, TI, RNG>::LOOP_CONFIG;
//std::string algorithm = "td3";
using LOOP_CORE_CONFIG = rlt::rl::zoo::sac::PendulumV1<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;

constexpr TI NUM_CHECKPOINTS = 10;
constexpr TI NUM_EVALUATIONS = 100;
constexpr TI NUM_SAVE_TRAJECTORIES = 10;
using LOOP_EXTRACK_CONFIG = rlt::rl::loop::steps::extrack::Config<LOOP_CORE_CONFIG>;
struct LOOP_CHECKPOINT_PARAMETERS: rlt::rl::loop::steps::checkpoint::Parameters<T, TI>{
    static constexpr TI CHECKPOINT_INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_CHECKPOINTS;
    static constexpr TI CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL_TEMP == 0 ? 1 : CHECKPOINT_INTERVAL_TEMP;
};
using LOOP_CHECKPOINT_CONFIG = rlt::rl::loop::steps::checkpoint::Config<LOOP_EXTRACK_CONFIG, LOOP_CHECKPOINT_PARAMETERS>;
struct LOOP_EVALUATION_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>{
    static constexpr TI EVALUATION_INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_EVALUATIONS;
    static constexpr TI EVALUATION_INTERVAL = EVALUATION_INTERVAL_TEMP == 0 ? 1 : EVALUATION_INTERVAL_TEMP;
    static constexpr TI NUM_EVALUATION_EPISODES = 100;
    static constexpr TI N_EVALUATIONS = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using LOOP_EVALUATION_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CHECKPOINT_CONFIG, LOOP_EVALUATION_PARAMETERS>;
struct LOOP_SAVE_TRAJECTORIES_PARAMETERS: rlt::rl::loop::steps::save_trajectories::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>{
    static constexpr TI INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_SAVE_TRAJECTORIES;
    static constexpr TI INTERVAL = INTERVAL_TEMP == 0 ? 1 : INTERVAL_TEMP;
    static constexpr TI NUM_EPISODES = 10;
};
using LOOP_SAVE_TRAJECTORIES_CONFIG = rlt::rl::loop::steps::save_trajectories::Config<LOOP_EVALUATION_CONFIG, LOOP_SAVE_TRAJECTORIES_PARAMETERS>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_SAVE_TRAJECTORIES_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
std::string algorithm = "sac";
std::string environment = "pendulum-v1";
// ---------------------------------------------------------------------------------------

int main(int argc, char** argv){
    using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;
    DEVICE device;
    TI seed = 0;
    LOOP_STATE ts;
#ifdef RL_TOOLS_ENABLE_CLI11
    CLI::App app{"rl_zoo"};
    app.add_option("-s,--seed", seed, "seed");
    app.add_option("-e,--extrack", ts.extrack_base_path, "extrack");
    app.add_option("--ee,--extrack-experiment", ts.extrack_experiment_path, "extrack-experiment");
    CLI11_PARSE(app, argc, argv);
#endif
    ts.config_name = algorithm + "_" + environment;
    rlt::malloc(device);
    rlt::init(device);
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
#ifdef RL_TOOLS_ENABLE_HDF5
    rlt::init(device, device.logger, ts.extrack_seed_path);
#endif
    std::cout << "Checkpoint Interval: " << LOOP_CHECKPOINT_CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL << std::endl;
    std::cout << "Evaluation Interval: " << LOOP_EVALUATION_CONFIG::EVALUATION_PARAMETERS::EVALUATION_INTERVAL << std::endl;
    while(!rlt::step(device, ts)){
        if(ts.step == 5000){
            std::cout << "steppin yourself > callbacks 'n' hooks: " << ts.step << std::endl;
        }
    }
    std::filesystem::create_directories(ts.extrack_seed_path);
    std::ofstream return_file(ts.extrack_seed_path / "return.json");
    return_file << "[";
    for(TI evaluation_i = 0; evaluation_i < LOOP_CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS; evaluation_i++){
        return_file << "{";
        return_file << "\"step\": " << LOOP_CONFIG::EVALUATION_PARAMETERS::EVALUATION_INTERVAL * evaluation_i << ", ";
        return_file << "\"returns_mean\": " << ts.evaluation_results[evaluation_i].returns_mean << ", ";
        return_file << "\"returns_std\": " << ts.evaluation_results[evaluation_i].returns_std << ", ";
        return_file << "\"episode_length_mean\": " << ts.evaluation_results[evaluation_i].episode_length_mean << ", ";
        return_file << "\"episode_length_std\": " << ts.evaluation_results[evaluation_i].episode_length_std << ", ";
        return_file << "\"returns\": [";
        for(TI episode_i = 0; episode_i < LOOP_CONFIG::EVALUATION_RESULT_SPEC::N_EPISODES; episode_i++){
            return_file << ts.evaluation_results[evaluation_i].returns[episode_i];
            if(episode_i < LOOP_CONFIG::EVALUATION_RESULT_SPEC::N_EPISODES - 1){
                return_file << ", ";
            }
        }
        return_file << "]";
        return_file << "}";
        if(evaluation_i < LOOP_CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS - 1){
            return_file << ", ";
        }
    }
    return_file << "]";

#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    rlt::free(device, device.logger);
#endif
    rlt::free(device);
    return 0;
}
