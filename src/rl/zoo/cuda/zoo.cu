#ifdef RL_TOOLS_DEBUG
#define RL_TOOLS_DEBUG_DEVICE_CUDA_SYNCHRONIZE_STATUS_CHECK
#endif
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_cuda.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>

#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>

#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cuda.h>
#include <rl_tools/rl/algorithms/ppo/operations_cuda.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_cuda.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

#include <rl_tools/rl/utils/evaluation/operations_cpu.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include "../l2f/ppo_cuda.h"

using DEVICE = rlt::devices::DEVICE_FACTORY_CUDA<>;
using DEVICE_EVALUATION = rlt::devices::DEVICE_FACTORY<>;

using TI = typename DEVICE::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<float>;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<rlt::devices::random::CUDA::Specification<TI, 1024>>;

constexpr bool DYNAMIC_ALLOCATION = true;

using FACTORY = rlt::rl::zoo::l2f::ppo::FACTORY<DEVICE, TYPE_POLICY, TI, RNG, DYNAMIC_ALLOCATION>;
using LOOP_CORE_CONFIG = FACTORY::LOOP_CORE_CONFIG;

static constexpr TI TIMING_INTERVAL = 10;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_CORE_CONFIG, rlt::rl::loop::steps::timing::Parameters<TI, TIMING_INTERVAL>>;

using LOOP_CONFIG = LOOP_TIMING_CONFIG;
using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

int main(int argc, char** argv){
    DEVICE device;
    DEVICE_EVALUATION device_evaluation;
    rlt::init(device);

    LOOP_STATE ts;
    rlt::malloc(device, ts);
    TI seed = argc > 1 ? std::atoi(argv[1]) : 0;
    rlt::init(device, ts, seed);

    std::cout << "Zoo CUDA L2F PPO Training" << std::endl;
    std::cout << "Step limit: " << LOOP_CONFIG::CORE_PARAMETERS::STEP_LIMIT << std::endl;

    // Evaluation setup on CPU
    constexpr TI NUM_EVALUATION_EPISODES = 10;
    constexpr TI EVALUATION_INTERVAL = LOOP_CONFIG::CORE_PARAMETERS::STEP_LIMIT / 10;
    using ACTOR_TYPE_ORIG = rlt::utils::typing::remove_reference_t<decltype(rlt::get_actor(ts))>;
    using ACTOR_TYPE_INFERENCE = typename ACTOR_TYPE_ORIG::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;
    using ACTOR_TYPE_EVALUATION = typename ACTOR_TYPE_INFERENCE::template CHANGE_BATCH_SIZE<TI, NUM_EVALUATION_EPISODES>;
    ACTOR_TYPE_EVALUATION actor_evaluation;
    typename ACTOR_TYPE_EVALUATION::template Buffer<> actor_buffers_evaluation;
    typename ACTOR_TYPE_EVALUATION::State<> actor_state_evaluation;
    using RESULT_SPEC = rlt::rl::utils::evaluation::Specification<TYPE_POLICY, TI, typename LOOP_CONFIG::ENVIRONMENT_EVALUATION, NUM_EVALUATION_EPISODES, LOOP_CONFIG::CORE_PARAMETERS::EPISODE_STEP_LIMIT>;
    rlt::rl::utils::evaluation::Result<RESULT_SPEC> result;
    rlt::rl::utils::evaluation::Buffer<rlt::rl::utils::evaluation::BufferSpecification<RESULT_SPEC>> eval_buffer;
    typename LOOP_CONFIG::ENVIRONMENT_EVALUATION env_eval;
    typename LOOP_CONFIG::ENVIRONMENT_EVALUATION::Parameters env_eval_parameters;
    rlt::rl::environments::DummyUI ui;
    typename DEVICE_EVALUATION::SPEC::RANDOM::ENGINE<> rng_eval;
    rlt::malloc(device_evaluation, actor_evaluation);
    rlt::malloc(device_evaluation, actor_buffers_evaluation);
    rlt::malloc(device_evaluation, actor_state_evaluation);
    rlt::malloc(device_evaluation, eval_buffer);
    rlt::init(device_evaluation, rng_eval, seed);
    rlt::init(device_evaluation, env_eval);
    rlt::initial_parameters(device_evaluation, env_eval, env_eval_parameters);

    bool finished = false;
    while(!finished){
        if(ts.step % EVALUATION_INTERVAL == 0){
            rlt::copy(device, device_evaluation, rlt::get_actor(ts), actor_evaluation);
            cudaStreamSynchronize(device.stream);
            rlt::evaluate(device_evaluation, env_eval, ui, actor_evaluation, actor_state_evaluation, actor_buffers_evaluation, eval_buffer, result, rng_eval, rlt::Mode<rlt::mode::Evaluation<>>{});
            std::cout << "Step: " << ts.step << "/" << LOOP_CONFIG::CORE_PARAMETERS::STEP_LIMIT << " Mean return: " << result.returns_mean << " Mean episode length: " << result.episode_length_mean << std::endl;
        }
        finished = rlt::step(device, ts);
    }

    // Final evaluation
    rlt::copy(device, device_evaluation, rlt::get_actor(ts), actor_evaluation);
    cudaStreamSynchronize(device.stream);
    rlt::evaluate(device_evaluation, env_eval, ui, actor_evaluation, actor_state_evaluation, actor_buffers_evaluation, eval_buffer, result, rng_eval, rlt::Mode<rlt::mode::Evaluation<>>{});
    std::cout << "Final - Step: " << ts.step << " Mean return: " << result.returns_mean << " Mean episode length: " << result.episode_length_mean << std::endl;

    rlt::free(device_evaluation, actor_evaluation);
    rlt::free(device_evaluation, actor_buffers_evaluation);
    rlt::free(device_evaluation, actor_state_evaluation);
    rlt::free(device_evaluation, eval_buffer);
    rlt::free(device, ts);
    return 0;
}
