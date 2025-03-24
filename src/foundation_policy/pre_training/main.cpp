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
using RNG_PARAMS_DEVICE = rlt::devices::random::Generic<DEVICE::SPEC::MATH>;
using RNG_PARAMS = RNG_PARAMS_DEVICE::ENGINE<>;
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
    RNG_PARAMS rng_params;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::malloc(device, rng_params);
    TI seed = argc >= 2 ? std::stoi(argv[1]) : 0;
    rlt::init(device, rng, seed);
    rlt::init(device, rng_params, seed);
    // warmup
    for(TI i=0; i < FACTORY::RNG_PARAMS_WARMUP_STEPS; i++){
        rlt::random::uniform_int_distribution(RNG_PARAMS_DEVICE{}, 0, 1, rng_params);
    }
    typename LOOP_CONFIG::template State <LOOP_CONFIG> ts;
    rlt::malloc(device, ts);
    ts.extrack_config.name = "foundation-policy-pre-training";
    ts.extrack_config.population_variates = "motor-mapping_thrust-curves_motor-delay_rng-warmup";
    ts.extrack_config.population_values = std::string(OPTIONS::RANDOMIZE_MOTOR_MAPPING ? "true" : "false") + "_";
    ts.extrack_config.population_values += std::string(OPTIONS::RANDOMIZE_THRUST_CURVES ? "true" : "false") + "_";
    ts.extrack_config.population_values += std::string(OPTIONS::MOTOR_DELAY ? "true" : "false") + "_";
    ts.extrack_config.population_values += std::to_string(FACTORY::RNG_PARAMS_WARMUP_STEPS);
    rlt::init(device, ts, seed);
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    rlt::init(device, device.logger, ts.extrack_paths.seed);
#endif

    auto& base_env = rlt::get(ts.off_policy_runner.envs, 0, 0);
    if (argc > 2){
        std::filesystem::path dynamics_parameters_path = argv[2];
        if (!std::filesystem::exists(dynamics_parameters_path)){
            std::cerr << "Dynamics parameters path does not exist: " << dynamics_parameters_path << std::endl;
            return 1;
        }
        std::ifstream file(dynamics_parameters_path, std::ios::in | std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file: " + dynamics_parameters_path.string());
        }

        std::ostringstream buffer;
        buffer << file.rdbuf();
        decltype(base_env.parameters) new_params;
        rlt::from_json(device, base_env, buffer.str(), new_params);
        base_env.parameters.dynamics = new_params.dynamics;
    }
    else{
        rlt::sample_initial_parameters(device, base_env, base_env.parameters, rng_params);
    }

    for (TI env_i = 1; env_i < LOOP_CONFIG::CORE_PARAMETERS::N_ENVIRONMENTS; env_i++){
        auto& env = rlt::get(ts.off_policy_runner.envs, 0, env_i);
        env.parameters = base_env.parameters;
    }
    ts.env_eval.parameters = base_env.parameters;

    bool finished = false;
    while(!finished){
        if (ts.step % LOOP_CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL == 0){
            auto step_folder = rlt::get_step_folder(device, ts.extrack_config, ts.extrack_paths, ts.step);
            std::filesystem::path checkpoint_path = std::filesystem::path(step_folder) / "critic_checkpoint.h5";
            std::cerr << "Checkpointing critic to: " << checkpoint_path.string() << std::endl;
            auto file = HighFive::File(checkpoint_path.string(), HighFive::File::Overwrite);
            auto group_0 = file.createGroup("critic_0");
            auto group_1 = file.createGroup("critic_1");
            rl_tools::save(device, ts.actor_critic.critics[0], group_0);
            rl_tools::save(device, ts.actor_critic.critics[1], group_1);
        }
        finished = rlt::step(device, ts);
    }
    return 0;
}