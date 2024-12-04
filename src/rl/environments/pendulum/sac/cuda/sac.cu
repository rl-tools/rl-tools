// /usr/local/cuda/bin/nvcc -I include -DRL_TOOLS_BACKEND_ENABLE_CUDA -lcublas src/rl/environments/pendulum/sac/cuda/sac.cu

#ifdef RL_TOOLS_DEBUG
#define RL_TOOLS_DEBUG_DEVICE_CUDA_SYNCHRONIZE_STATUS_CHECK
#endif
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_cuda.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/rl/components/off_policy_runner/operations_cuda.h>

#include <rl_tools/rl/algorithms/sac/operations_cuda.h>
#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY_CUDA<>;
#ifndef _MSC_VER
using DEVICE_EVALUATION = rlt::devices::DEVICE_FACTORY<>;
#else
using DEVICE_INIT = rlt::devices::DefaultCPU; // for some reason MKL makes problems in this case (this example seems cursed)
#endif
DEVICE dummy_device; // this is needed because default_engine can not take a const device
using RNG = decltype(rlt::random::default_engine(dummy_device));
using TI = typename DEVICE::index_t;
using T = float;


using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<T, TI, ENVIRONMENT::ACTION_DIM>{
        static constexpr TI ACTOR_BATCH_SIZE = 100;
        static constexpr TI CRITIC_BATCH_SIZE = 100;
    };
    static constexpr TI STEP_LIMIT = 10000;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr bool COLLECT_EPISODE_STATS = false;
    static constexpr TI EPISODE_STATS_BUFFER_SIZE = 0;
};
template <typename RNG>
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;

struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CORE_CONFIG<RNG>>{
    static constexpr TI NUM_EVALUATION_EPISODES = 100;
};
template <typename RNG>
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG<RNG>, LOOP_EVAL_PARAMETERS>;
template <typename RNG>
using LOOP_CONFIG = LOOP_EVAL_CONFIG<RNG>;

using LOOP_STATE = typename LOOP_CONFIG<RNG>::template State<LOOP_CONFIG<RNG>>;


int main() {
    TI seed = 0;
    DEVICE device;
    DEVICE_EVALUATION device_evaluation;
    LOOP_STATE ts;
    using CONFIG = typename decltype(ts)::CONFIG;
    using CORE_PARAMETERS = typename CONFIG::CORE_PARAMETERS;
    using EVAL_PARAMETERS = typename CONFIG::EVALUATION_PARAMETERS;
    auto rng_evaluation = rlt::random::default_engine(device_evaluation, seed);
    using ACTOR_TYPE_ORIG = rlt::utils::typing::remove_reference_t<decltype(rlt::get_actor(ts))>;
    using ACTOR_TYPE_INFERENCE = ACTOR_TYPE_ORIG:: template CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;
    using ACTOR_TYPE_EVALUATION = ACTOR_TYPE_INFERENCE:: template CHANGE_BATCH_SIZE<TI, EVAL_PARAMETERS::NUM_EVALUATION_EPISODES>;
    ACTOR_TYPE_EVALUATION actor_evaluation;
    ACTOR_TYPE_EVALUATION::Buffer<> actor_buffers_evaluation;
    ENVIRONMENT env_evaluation;
    ENVIRONMENT::Parameters env_evaluation_parameters;
    rlt::rl::environments::DummyUI ui;
    rlt::init(device);
    rlt::malloc(device, ts);
    rlt::malloc(device_evaluation, actor_evaluation);
    rlt::malloc(device_evaluation, actor_buffers_evaluation);
    rlt::init(device, ts, 1);
    TI step = 0;
    bool finished = false;
    while(!finished){
        // Evaluation
        if(step % 1000 == 0){
            rlt::copy(device, device_evaluation, rlt::get_actor(ts), actor_evaluation);
            using RESULT_SPEC = rlt::rl::utils::evaluation::Specification<T, TI, typename decltype(ts)::CONFIG::ENVIRONMENT_EVALUATION, EVAL_PARAMETERS::NUM_EVALUATION_EPISODES, CORE_PARAMETERS::EPISODE_STEP_LIMIT>;
            rlt::rl::utils::evaluation::Result<RESULT_SPEC> result;
            rlt::evaluate(device_evaluation, env_evaluation, env_evaluation_parameters, ui, actor_evaluation, result, actor_buffers_evaluation, rng_evaluation, rlt::Mode<rlt::mode::Evaluation<>>{});
            rlt::log(device_evaluation, device_evaluation.logger, "Step: ", step, " Mean return: ", result.returns_mean);
        }

        // Training
        rlt::set_step(device, device.logger, step);
        rlt::step<1>(device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
        if(step > CONFIG::CORE_PARAMETERS::N_WARMUP_STEPS){
            if(step % CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0) {
                for(int critic_i = 0; critic_i < 2; critic_i++){
                    rlt::gather_batch(device, ts.off_policy_runner, ts.critic_batch, ts.rng);
                    rlt::randn(device, ts.action_noise_critic, ts.rng);
                    rlt::train_critic(device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers[critic_i], ts.action_noise_critic, ts.rng);
                }
            }
            if(step % CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0) {
                {
                    rlt::gather_batch(device, ts.off_policy_runner, ts.actor_batch, ts.rng);
                    rlt::randn(device, ts.action_noise_actor, ts.rng);
                    rlt::train_actor(device, ts.actor_critic, ts.actor_batch, ts.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers, ts.action_noise_actor, ts.rng);
                }
                rlt::update_critic_targets(device, ts.actor_critic);
            }
        }
        step++;
        finished = step > CORE_PARAMETERS::STEP_LIMIT;
     }
    rlt::malloc(device, ts);
    return 0;
}

// benchmark training should take < 2s on P1
