//#define RL_TOOLS_NAMESPACE_WRAPPER zoo
#define RL_TOOLS_RL_ENVIRONMENTS_MULTI_AGENT_BOTTLENECK_CHECK_NAN

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
#endif

#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist_code.h>

#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/environments/acrobot/operations_cpu.h>
#include <rl_tools/rl/environments/multi_agent/bottleneck/operations_cpu.h>
#ifdef RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4
#include <rl_tools/rl/environments/mujoco/ant/operations_cpu.h>
#endif

#include "sac/pendulum-v1.h"
#include "sac/acrobot-swingup-v0.h"
#include "sac/l2f.h"
#include "td3/pendulum-v1.h"
#include "td3/l2f.h"
#include "ppo/pendulum-v1.h"
#include "ppo/bottleneck-v0.h"
#ifdef RL_TOOLS_RL_ZOO_ENVIRONMENT_ANT_V4
#include "ppo/ant-v4.h"
#endif

#include <rl_tools/rl/algorithms/td3/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;
constexpr TI BASE_SEED = 0;

#include "config.h"

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
    rlt::utils::assert_exit(device, num_seeds > 0, "Number of seeds must be greater than 0.");
    for(TI seed = initial_seed; seed < (TI)num_seeds; seed++){
        LOOP_STATE ts;
        ts.extrack_name = "zoo";
        if(extrack_base_path != ""){
            ts.extrack_base_path = extrack_base_path;
        }
        if(extrack_experiment != ""){
            ts.extrack_experiment = extrack_experiment;
        }
        ts.extrack_population_variates = "algorithm_environment";
        ts.extrack_population_values = algorithm + "_" + environment;
        if(extrack_experiment_path != ""){
            ts.extrack_experiment = extrack_experiment_path;
        }
        rlt::malloc(device);
        rlt::init(device);
        rlt::malloc(device, ts);
        rlt::init(device, ts, seed);
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
        rlt::init(device, device.logger, ts.extrack_seed_path);
#endif
        std::cout << "Checkpoint Interval: " << LOOP_CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL << std::endl;
        std::cout << "Evaluation Interval: " << LOOP_CONFIG::EVALUATION_PARAMETERS::EVALUATION_INTERVAL << std::endl;
        std::cout << "Save Trajectories Interval: " << LOOP_CONFIG::SAVE_TRAJECTORIES_PARAMETERS::INTERVAL << std::endl;
        while(!rlt::step(device, ts)){
        }
        std::filesystem::create_directories(ts.extrack_seed_path);
        std::ofstream return_file(ts.extrack_seed_path / "return.json");
        return_file << "[";
        for(TI evaluation_i = 0; evaluation_i < LOOP_CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS; evaluation_i++){
            return_file << "{";
            return_file << "\"step\": " << LOOP_CONFIG::EVALUATION_PARAMETERS::EVALUATION_INTERVAL *  LOOP_CONFIG::ENVIRONMENT_STEPS_PER_LOOP_STEP * evaluation_i << ", ";
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
        std::ofstream return_file_confirmation(ts.extrack_seed_path / "return.json.set");
        return_file_confirmation.close();

#ifdef RL_TOOLS_ENABLE_TENSORBOARD
        rlt::free(device, device.logger);
#endif
        rlt::free(device);
    }
    return 0;
}
