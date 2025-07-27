#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>

// #include <rl_tools/containers/tensor/persist.h>
// #include <rl_tools/nn/layers/sample_and_squash/persist.h>
// #include <rl_tools/nn/layers/dense/persist.h>
// #include <rl_tools/nn/layers/standardize/persist.h>
// #include <rl_tools/nn/layers/gru/persist.h>
// #include <rl_tools/nn/layers/td3_sampling/persist.h>
// #include <rl_tools/nn_models/mlp/persist.h>
// #include <rl_tools/nn_models/sequential/persist.h>
// #include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
// #include <rl_tools/rl/components/replay_buffer/persist.h>

#include <rl_tools/rl/environments/l2f/operations_generic.h>

#include <rl_tools/rl/utils/evaluation/operations_cpu.h>

#include "rl_tools/rl/environments/multi_agent/environments.h"

namespace rlt = rl_tools;

#include "../blob/checkpoint.h"
#include "../post_training/environment.h"


using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_PARAMS_DEVICE = rlt::devices::random::Generic<DEVICE::SPEC::MATH>;
using RNG_PARAMS = RNG_PARAMS_DEVICE::ENGINE<>;
using TI = DEVICE::index_t;
using T = float;

struct ENV_OPTIONS {
    static constexpr bool RANDOMIZE_MOTOR_MAPPING = false;
    static constexpr bool RANDOMIZE_THRUST_CURVES = false;
    static constexpr bool OBSERVE_THRASH_MARKOV = false;
    static constexpr bool MOTOR_DELAY = true;
    static constexpr bool ACTION_HISTORY = true;
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr bool OBSERVATION_NOISE = true;
};

using ENVIRONMENT = typename builder::ENVIRONMENT_FACTORY_POST_TRAINING<DEVICE, T, TI, ENV_OPTIONS>::ENVIRONMENT;
static_assert(ENVIRONMENT::EPISODE_STEP_LIMIT == 500);

constexpr TI NUM_EPISODES_EVAL = 100;


using EVAL_ACTOR = rlt::checkpoint::actor::TYPE;

int main(int argc, char** argv){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    TI seed = 0;
    rlt::init(device, rng, seed);
    ENVIRONMENT env;
    ENVIRONMENT::Parameters params;
    rlt::malloc(device, env);
    rlt::init(device, env);

    using EVAL_SPEC = rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>;
    rlt::rl::utils::evaluation::Result<EVAL_SPEC> result;
    rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<EVAL_SPEC>> data;
    RNG rng_copy = rng;
    rlt::malloc(device, data);
    rlt::Mode<rlt::mode::Evaluation<rlt::nn::layers::sample_and_squash::mode::DisableEntropy<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Final>>>> mode;
    rlt::rl::environments::DummyUI ui;
    {
        static constexpr bool DYNAMIC_ALLOCATION = true;
        using ADJUSTED_POLICY = typename EVAL_ACTOR::template CHANGE_BATCH_SIZE<TI, EVAL_SPEC::N_EPISODES>;
        ADJUSTED_POLICY::template State<DYNAMIC_ALLOCATION> policy_state;
        ADJUSTED_POLICY::template Buffer<DYNAMIC_ALLOCATION> policy_evaluation_buffers;
        rlt::rl::utils::evaluation::Buffer<rlt::rl::utils::evaluation::BufferSpecification<EVAL_SPEC, DYNAMIC_ALLOCATION>> evaluation_buffers;
        rlt::malloc(device, policy_state);
        rlt::malloc(device, policy_evaluation_buffers);
        rlt::malloc(device, evaluation_buffers);
        rlt::evaluate(device, env, ui, rlt::checkpoint::actor::module, policy_state, policy_evaluation_buffers, evaluation_buffers, result, data, rng, mode);
        for (TI episode_i = 0; episode_i < EVAL_SPEC::N_EPISODES; ++episode_i) {
            if (result.episode_length[episode_i] == EVAL_SPEC::STEP_LIMIT) {
                std::cout << "Episode " << episode_i << ": Return = " << result.returns[episode_i] << " Length: " << result.episode_length[episode_i] << std::endl;
            }
        }
        rlt::free(device, policy_state);
        rlt::free(device, policy_evaluation_buffers);
        rlt::free(device, evaluation_buffers);
    }
}
