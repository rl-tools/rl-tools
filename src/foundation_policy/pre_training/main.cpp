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

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/td3_sampling/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
#include <rl_tools/rl/components/replay_buffer/persist.h>

#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/gru/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn/layers/td3_sampling/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist_code.h>

#include "environment.h"

#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>

#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/nn_analytics/operations_cpu.h>

#include <rl_tools/rl/utils/evaluation/operations_cpu.h>

#include "config.h"
#include "options.h"

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;
using T = float;
constexpr bool DYNAMIC_ALLOCATION = true;

using OPTIONS = OPTIONS_PRE_TRAINING;

using FACTORY = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS, DYNAMIC_ALLOCATION>;
using LOOP_CORE_CONFIG = FACTORY::LOOP_CORE_CONFIG;
using LOOP_CONFIG = builder::LOOP_ASSEMBLY<LOOP_CORE_CONFIG>::LOOP_CONFIG;
// note: make sure that the rng_params is invoked in the exact same way in pre- as in post-training, to make sure the params used to sample parameters to generate data from the trained policy are matching the ones seen by the particular policy for the seed during pretraining

int main(int argc, char** argv){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    TI seed = 0;
    rlt::init(device, rng, seed);

    std::vector<std::string> file_paths;
    if (argc > 1){
        std::string file_path = argv[1];
        file_paths.push_back(file_path);
    }
    else{
        // iterate dynamics_parameters directory
        std::filesystem::path dynamics_parameters_path = "./src/foundation_policy/dynamics_parameters/";
        if (!std::filesystem::exists(dynamics_parameters_path)){
            std::cerr << "Dynamics parameters path does not exist: " << dynamics_parameters_path << std::endl;
            return 1;
        }
        for (const auto& entry : std::filesystem::directory_iterator(dynamics_parameters_path)){
            file_paths.push_back(entry.path().string());
        }
        std::sort(file_paths.begin(), file_paths.end());
    }
    for (const auto& file_path_string : file_paths){
        typename LOOP_CONFIG::template State <LOOP_CONFIG> ts;
        rlt::malloc(device, ts);
        ts.extrack_config.name = "foundation-policy-pre-training";

        auto& base_env = rlt::get(ts.off_policy_runner.envs, 0, 0);
        std::filesystem::path file_path = file_path_string;
        if (file_path.filename().string()[0] == '.'){
            continue;
        }
        if (!std::filesystem::exists(file_path)){
            std::cerr << "Dynamics parameters path does not exist: " << file_path << std::endl;
            break;
        }
        std::ifstream file(file_path, std::ios::in | std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + file_path.string());
        }
        std::cout << "Loading dynamics parameters from: " << file_path.string() << std::endl;
        std::ostringstream buffer;
        buffer << file.rdbuf();
        decltype(base_env.parameters) new_params;
        rlt::from_json(device, base_env, buffer.str(), new_params);
        ts.extrack_config.population_variates = "dynamics-id";
        ts.extrack_config.population_values = file_path.filename().stem().string();
        rlt::init(device, ts, seed);
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
        rlt::init(device, device.logger, ts.extrack_paths.seed);
#endif
        // auto old_init = base_env.parameters.mdp.init;
        base_env.parameters = new_params;
        // base_env.parameters.mdp.init = old_init;
        for (TI env_i = 1; env_i < LOOP_CONFIG::CORE_PARAMETERS::N_ENVIRONMENTS; env_i++){
            auto& env = rlt::get(ts.off_policy_runner.envs, 0, env_i);
            env.parameters = base_env.parameters;
        }
        ts.env_eval.parameters = base_env.parameters;


        bool finished = false;
        while(!finished){
            // if (ts.step % LOOP_CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL == 0){
            //     auto step_folder = rlt::get_step_folder(device, ts.extrack_config, ts.extrack_paths, ts.step);
            //     std::filesystem::path checkpoint_path = std::filesystem::path(step_folder) / "critic_checkpoint.h5";
            //     std::cerr << "Checkpointing critic to: " << checkpoint_path.string() << std::endl;
            //     auto file = HighFive::File(checkpoint_path.string(), HighFive::File::Overwrite);
            //     auto group_0 = file.createGroup("critic_0");
            //     auto group_1 = file.createGroup("critic_1");
            //     rl_tools::save(device, ts.actor_critic.critics[0], group_0);
            //     rl_tools::save(device, ts.actor_critic.critics[1], group_1);
            // }
            finished = rlt::step(device, ts);
        }
        std::filesystem::create_directories(ts.extrack_paths.seed);
        std::ofstream return_file(ts.extrack_paths.seed / "return.json");
        return_file << "[";
        for(TI evaluation_i = 0; evaluation_i < LOOP_CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS; evaluation_i++){
            auto& result = get(ts.evaluation_results, 0, evaluation_i);
            return_file << rlt::json(device, result, LOOP_CONFIG::EVALUATION_PARAMETERS::EVALUATION_INTERVAL * LOOP_CONFIG::ENVIRONMENT_STEPS_PER_LOOP_STEP * evaluation_i);
            if(evaluation_i < LOOP_CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS - 1){
                return_file << ", ";
            }
        }
        return_file << "]";
        std::ofstream return_file_confirmation(ts.extrack_paths.seed / "return.json.set");
        return_file_confirmation.close();
        rlt::free(device, ts);
    }
    rlt::free(device, rng);
    return 0;
}