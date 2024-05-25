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
using LOOP_CONFIG = rlt::rl::zoo::sac::PendulumV1<DEVICE, T, TI, RNG>::LOOP_CONFIG;
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
    while(!rlt::step(device, ts)){
        if(ts.step == 5000){
            std::cout << "steppin yourself > callbacks 'n' hooks: " << ts.step << std::endl;
        }
    }
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    rlt::free(device, device.logger);
#endif
    rlt::free(device);
    return 0;
}
