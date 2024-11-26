#ifdef BENCHMARK
#undef RL_TOOLS_ENABLE_TENSORBOARD
#endif

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/td3_sampling/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
#endif

#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn/layers/td3_sampling/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist_code.h>

// Environment Configurations
#include "pendulum-v1/sac.h"
#include "pendulum-v1/td3.h"
#include "pendulum-v1/ppo.h"
#include "acrobot-swingup-v0/sac.h"
#include "bottleneck-v0/ppo.h"
#include "l2f/sac.h"
#include "l2f/td3.h"
#include "l2f/ppo.h"
#ifdef RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4
#include "ant-v4/ppo.h"
#include "ant-v4/td3.h"
#endif

// Algorithm Loops
#include <rl_tools/rl/algorithms/td3/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#if defined(RL_TOOLS_BACKEND_ENABLE_MKL) && !defined(RL_TOOLS_BACKEND_DISABLE_BLAS)
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mkl.h>
#else
#if defined(RL_TOOLS_BACKEND_ENABLE_ACCELERATE) && !defined(RL_TOOLS_BACKEND_DISABLE_BLAS)
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>

// Additional Loop steps
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>

#include <rl_tools/rl/utils/evaluation/operations_cpu.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using SUPER_DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = typename SUPER_DEVICE::index_t;

namespace execution_hints{
    struct HINTS: rlt::rl::components::on_policy_runner::ExecutionHints<TI, 16>{};
}
struct DEV_SPEC: rlt::devices::DEVICE_FACTORY<>::SPEC{
    using EXECUTION_HINTS = execution_hints::HINTS;
};

using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
constexpr TI BASE_SEED = 0;

#if defined(RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
using LOOP_CORE_CONFIG = rlt::rl::zoo::pendulum_v1::sac::FACTORY<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op, this allows to have a different EPISODE_STEP_LIMIT for training and evaluation (on a per algorithm&environment baseis)
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_ACROBOT_SWINGUP_V0)
using LOOP_CORE_CONFIG = rlt::rl::zoo::acrobot_swingup_v0::sac::FACTORY<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
using LOOP_EVALUATION_PARAMETER_OVERWRITES = rlt::rl::zoo::acrobot_swingup_v0::sac::FACTORY<DEVICE, T, TI, RNG>::LOOP_EVALUATION_PARAMETER_OVERWRITES<BASE>;
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)
using LOOP_CORE_CONFIG = rlt::rl::zoo::l2f::sac::FACTORY<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{};
#else
#error "RLtools Zoo SAC: Environment not defined"
#endif
#elif defined(RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
using LOOP_CORE_CONFIG = rlt::rl::zoo::pendulum_v1::td3::FACTORY<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)
using LOOP_CORE_CONFIG = rlt::rl::zoo::l2f::td3::FACTORY<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4)
using LOOP_CORE_CONFIG = rlt::rl::zoo::ant_v4::td3::FACTORY<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op
#else
#error "RLtools Zoo TD3: Environment not defined"
#endif
#elif defined(RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
using LOOP_CORE_CONFIG = rlt::rl::zoo::pendulum_v1::ppo::FACTORY<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_BOTTLENECK_V0)
using LOOP_CORE_CONFIG = rlt::rl::zoo::bottleneck_v0::ppo::FACTORY<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
using LOOP_EVALUATION_PARAMETER_OVERWRITES = rlt::rl::zoo::bottleneck_v0::ppo::FACTORY<DEVICE, T, TI, RNG>::LOOP_EVALUATION_PARAMETER_OVERWRITES<BASE>;
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4)
using LOOP_CORE_CONFIG = rlt::rl::zoo::ant_v4::ppo::FACTORY<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)
using LOOP_CORE_CONFIG = rlt::rl::zoo::l2f::ppo::FACTORY<DEVICE, T, TI, RNG>::LOOP_CORE_CONFIG;
template <typename BASE>
struct LOOP_EVALUATION_PARAMETER_OVERWRITES: BASE{}; // no-op
#else
#error "RLtools Zoo PPO: Environment not defined"
#endif
#else
#error "RLtools Zoo: Algorithm not defined"
#endif

constexpr TI NUM_CHECKPOINTS = 10;
constexpr TI NUM_EVALUATIONS = 100;
constexpr TI NUM_SAVE_TRAJECTORIES = 10;
#ifdef RL_TOOLS_RL_ZOO_ALGORITHM_PPO
static constexpr TI TIMING_INTERVAL = 10;
#else
static constexpr TI TIMING_INTERVAL = 10000;
#endif
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_CORE_CONFIG, rlt::rl::loop::steps::timing::Parameters<TI, TIMING_INTERVAL>>;
#ifndef BENCHMARK
using LOOP_EXTRACK_CONFIG = rlt::rl::loop::steps::extrack::Config<LOOP_TIMING_CONFIG>;
struct LOOP_CHECKPOINT_PARAMETERS: rlt::rl::loop::steps::checkpoint::Parameters<T, TI>{
    static constexpr TI CHECKPOINT_INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_CHECKPOINTS;
    static constexpr TI CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL_TEMP == 0 ? 1 : CHECKPOINT_INTERVAL_TEMP;
};
using LOOP_CHECKPOINT_CONFIG = rlt::rl::loop::steps::checkpoint::Config<LOOP_EXTRACK_CONFIG, LOOP_CHECKPOINT_PARAMETERS>;
struct LOOP_EVALUATION_PARAMETERS: LOOP_EVALUATION_PARAMETER_OVERWRITES<rlt::rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>>{
    static constexpr TI EVALUATION_INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_EVALUATIONS;
    static constexpr TI EVALUATION_INTERVAL = EVALUATION_INTERVAL_TEMP == 0 ? 1 : EVALUATION_INTERVAL_TEMP;
    static constexpr TI NUM_EVALUATION_EPISODES = 100;
    static constexpr TI N_EVALUATIONS = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using LOOP_EVALUATION_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CHECKPOINT_CONFIG, LOOP_EVALUATION_PARAMETERS>;
struct LOOP_SAVE_TRAJECTORIES_PARAMETERS: LOOP_EVALUATION_PARAMETER_OVERWRITES<rlt::rl::loop::steps::save_trajectories::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>>{
    static constexpr TI INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_SAVE_TRAJECTORIES;
    static constexpr TI INTERVAL = INTERVAL_TEMP == 0 ? 1 : INTERVAL_TEMP;
    static constexpr TI NUM_EPISODES = 100;
};
using LOOP_SAVE_TRAJECTORIES_CONFIG = rlt::rl::loop::steps::save_trajectories::Config<LOOP_EVALUATION_CONFIG, LOOP_SAVE_TRAJECTORIES_PARAMETERS>;
using LOOP_CONFIG = LOOP_SAVE_TRAJECTORIES_CONFIG;
#else
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
#endif

#if defined(RL_TOOLS_RL_ZOO_ALGORITHM_SAC)
std::string algorithm = "sac";
#elif defined(RL_TOOLS_RL_ZOO_ALGORITHM_TD3)
std::string algorithm = "td3";
#elif defined(RL_TOOLS_RL_ZOO_ALGORITHM_PPO)
std::string algorithm = "ppo";
#else
#error "RLtools Zoo: Algorithm not defined"
#endif
#if defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_PENDULUM_V1)
std::string environment = "pendulum-v1";
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_ACROBOT_SWINGUP_V0)
std::string environment = "acrobot-swingup-v0";
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_BOTTLENECK_V0)
std::string environment = "bottleneck-v0";
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4)
std::string environment = "ant-v4";
#elif defined(RL_TOOLS_RL_ZOO_ENVIRONMENT_L2F)
std::string environment = "l2f";
#else
#error "RLtools Zoo: Environment not defined"
#endif
// ---------------------------------------------------------------------------------------

int zoo(int initial_seed, int num_seeds, std::string extrack_base_path, std::string extrack_experiment, std::string extrack_experiment_path, std::string config_path){
    using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;
    DEVICE device;
    static_assert(sizeof(LOOP_STATE) < 100000000);
    LOOP_STATE test_state;
//    rlt::utils::assert_exit(device, num_seeds > 0, "Number of seeds must be greater than 0.");
    for(TI seed = initial_seed; seed < (TI)num_seeds; seed++){
        LOOP_STATE ts;
#ifndef BENCHMARK
        ts.extrack_name = "zoo";
        if(extrack_base_path != ""){
            ts.extrack_base_path = extrack_base_path;
        }
        if(extrack_experiment != ""){
            ts.extrack_experiment = extrack_experiment;
        }
        ts.extrack_population_variates = "environment_algorithm";
        ts.extrack_population_values = environment + "_" + algorithm;
        if(extrack_experiment_path != ""){
            ts.extrack_experiment = extrack_experiment_path;
        }
#endif
        rlt::malloc(device);
        rlt::init(device);
        rlt::malloc(device, ts);
        rlt::init(device, ts, seed);
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
        rlt::init(device, device.logger, ts.extrack_seed_path);
#endif
#ifndef BENCHMARK
        std::cout << "Checkpoint Interval: " << LOOP_CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL << std::endl;
        std::cout << "Evaluation Interval: " << LOOP_CONFIG::EVALUATION_PARAMETERS::EVALUATION_INTERVAL << std::endl;
        std::cout << "Save Trajectories Interval: " << LOOP_CONFIG::SAVE_TRAJECTORIES_PARAMETERS::INTERVAL << std::endl;
#endif
        while(!rlt::step(device, ts)){
        }
#ifndef BENCHMARK
        std::filesystem::create_directories(ts.extrack_seed_path);
        std::ofstream return_file(ts.extrack_seed_path / "return.json");
        return_file << "[";
        for(TI evaluation_i = 0; evaluation_i < LOOP_CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS; evaluation_i++){
            auto& result = get(ts.evaluation_results, 0, evaluation_i);
            return_file << rlt::json(device, result, LOOP_CONFIG::EVALUATION_PARAMETERS::EVALUATION_INTERVAL * LOOP_CONFIG::ENVIRONMENT_STEPS_PER_LOOP_STEP * evaluation_i);
            if(evaluation_i < LOOP_CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS - 1){
                return_file << ", ";
            }
        }
        return_file << "]";
        std::ofstream return_file_confirmation(ts.extrack_seed_path / "return.json.set");
        return_file_confirmation.close();
#endif
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
        rlt::free(device, device.logger);
#endif
        rlt::free(device);
    }
    return 0;
}
