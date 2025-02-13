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

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_PARAMS_DEVICE = rlt::devices::random::Generic<DEVICE::SPEC::MATH>;
using RNG_PARAMS = RNG_PARAMS_DEVICE::ENGINE<>;
using TI = typename DEVICE::index_t;
using T = float;
constexpr bool DYNAMIC_ALLOCATION = true;

struct OPTIONS{
    static constexpr bool SEQUENTIAL_MODEL = false;
    static constexpr bool MOTOR_DELAY = false;
    static constexpr bool RANDOMIZE_MOTOR_MAPPING = true;
    static constexpr bool RANDOMIZE_THRUST_CURVES = false;
    static constexpr bool OBSERVE_THRASH_MARKOV = false;
    static constexpr bool SAMPLE_INITIAL_PARAMETERS = false;
};
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
    typename LOOP_CONFIG::template State <LOOP_CONFIG> ts;
    rlt::malloc(device, ts);
    ts.extrack_config.name = "foundation-policy-pre-training";
    ts.extrack_config.population_variates = "motor-mapping_thrust-curves_rng-warmup";
    ts.extrack_config.population_values = std::string(OPTIONS::RANDOMIZE_MOTOR_MAPPING ? "true" : "false") + "_" + (OPTIONS::RANDOMIZE_THRUST_CURVES ? "true" : "false") + "_" + std::to_string(FACTORY::RNG_PARAMS_WARMUP_STEPS);
    rlt::init(device, ts, seed);
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    rlt::init(device, device.logger, ts.extrack_paths.seed);
#endif
    // warmup
    for(TI i=0; i < FACTORY::RNG_PARAMS_WARMUP_STEPS; i++){
        rlt::random::uniform_int_distribution(RNG_PARAMS_DEVICE{}, 0, 1, rng_params);
    }

    auto& base_env = rlt::get(ts.off_policy_runner.envs, 0, 0);
    rlt::sample_initial_parameters<DEVICE, LOOP_CONFIG::ENVIRONMENT::SPEC, RNG_PARAMS, true>(device, base_env, base_env.parameters, rng_params);
    for (TI env_i = 1; env_i < LOOP_CONFIG::CORE_PARAMETERS::N_ENVIRONMENTS; env_i++){
        auto& env = rlt::get(ts.off_policy_runner.envs, 0, env_i);
        env.parameters = base_env.parameters;
    }
    ts.env_eval.parameters = base_env.parameters;

    while(!rlt::step(device, ts)){
    }
    return 0;
}