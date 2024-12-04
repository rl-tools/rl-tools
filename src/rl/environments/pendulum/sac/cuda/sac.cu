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
using DEVICE_INIT = rlt::devices::DEVICE_FACTORY<>;
#else
using DEVICE_INIT = rlt::devices::DefaultCPU; // for some reason MKL makes problems in this case (this example seems cursed)
#endif
DEVICE dummy_device; // this is needed because default_engine can not take a const device
using RNG = decltype(rlt::random::default_engine(dummy_device));
using RNG_INIT = decltype(rlt::random::default_engine(DEVICE_INIT{}));
using T = float;
using TI = typename DEVICE::index_t;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#pragma message "RL_TOOLS_FUNCTION_PLACEMENT: " TOSTRING(RL_TOOLS_FUNCTION_PLACEMENT)

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<T, TI, ENVIRONMENT::ACTION_DIM>{
        static constexpr TI ACTOR_BATCH_SIZE = 100;
        static constexpr TI CRITIC_BATCH_SIZE = 100;
    };
    static constexpr TI STEP_LIMIT = 20000;
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
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG<RNG>>;
template <typename RNG>
using LOOP_CONFIG = LOOP_TIMING_CONFIG<RNG>;

using LOOP_STATE = LOOP_CONFIG<RNG>::template State<LOOP_CONFIG<RNG>>;
using LOOP_STATE_INIT = LOOP_CONFIG<RNG_INIT>::template State<LOOP_CONFIG<RNG_INIT>>;


int main(){
    TI seed = 0;
    DEVICE device;
    DEVICE_INIT device_init;
    LOOP_STATE ts;
    LOOP_STATE_INIT ts_init;
    using CONFIG = decltype(ts)::CONFIG;
    using CORE_PARAMETERS = CONFIG::CORE_PARAMETERS;
    using EVAL_PARAMETERS = CONFIG::EVALUATION_PARAMETERS;
    rlt::init(device);
    rlt::malloc(device, ts);
    rlt::malloc(device_init, ts_init);
    rlt::init(device, ts, 1);
    rlt::check_status(device);
    rlt::init(device_init, ts_init, 1);
    // auto& episode_stats_source = ts_init.off_policy_runner.episode_stats;
    // rlt::utils::typing::remove_reference_t<decltype(episode_stats_source)> episode_stats_target;
    // episode_stats_target = episode_stats_source;
    rlt::copy(device, device_init, ts, ts_init);
    //    rlt::copy(device_init, device, ts_init.off_policy_runner, ts.off_policy_runner);

#ifdef _MSC_VER
    CONFIG::ENVIRONMENT env_eval;
    RNG_INIT rng_eval;
    rlt::rl::environments::DummyUI ui;
#endif

    // decltype(ts.off_policy_runner)* off_policy_runner_pointer;
    // cudaMalloc(&off_policy_runner_pointer, sizeof(decltype(ts.off_policy_runner)));
    // cudaMemcpy(off_policy_runner_pointer, &ts.off_policy_runner, sizeof(decltype(ts.off_policy_runner)), cudaMemcpyHostToDevice);
    // rlt::check_status(device);

    TI step = 0;
    // ts.rng = rlt::random::default_engine(device);
    bool finished = false;
    auto start_time = std::chrono::high_resolution_clock::now();
    while(!finished){
        rlt::set_step(device, device.logger, step);
        // rlt::step<1>(device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
        rlt::step<1>(device_init, ts_init.off_policy_runner, ts_init.actor_critic.actor, ts_init.actor_buffers_eval, ts_init.rng);
        if(step > CONFIG::CORE_PARAMETERS::N_WARMUP_STEPS){
            if(step % CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0) {
                for(int critic_i = 0; critic_i < 2; critic_i++){
                    // rlt::gather_batch(device, ts.off_policy_runner, ts.critic_batch, ts.rng);
                    rlt::gather_batch(device_init, ts_init.off_policy_runner, ts_init.critic_batch, ts_init.rng);
                    // rlt::randn(device, ts.action_noise_critic, ts.rng);
                    rlt::randn(device_init, ts_init.action_noise_critic, ts_init.rng);
                    // rlt::train_critic(device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers[critic_i], ts.action_noise_critic, ts.rng);
                    rlt::train_critic(device_init, ts_init.actor_critic, critic_i == 0 ? ts_init.actor_critic.critic_1 : ts_init.actor_critic.critic_2, ts_init.critic_batch, ts_init.critic_optimizers[critic_i], ts_init.actor_buffers[critic_i], ts_init.critic_buffers[critic_i], ts_init.critic_training_buffers[critic_i], ts_init.action_noise_critic, ts_init.rng);
                }
                device.stream = 0;
            }
            if(step % CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0) {
                {
                    // rlt::gather_batch(device, ts.off_policy_runner, ts.actor_batch, ts.rng);
                    rlt::gather_batch(device_init, ts_init.off_policy_runner, ts_init.actor_batch, ts_init.rng);
                    // rlt::randn(device, ts.action_noise_actor, ts.rng);
                    rlt::randn(device_init, ts_init.action_noise_actor, ts_init.rng);
                    // rlt::train_actor(device, ts.actor_critic, ts.actor_batch, ts.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers, ts.action_noise_actor, ts.rng);
                    rlt::train_actor(device_init, ts_init.actor_critic, ts_init.actor_batch, ts_init.actor_optimizer, ts_init.actor_buffers[0], ts_init.critic_buffers[0], ts_init.actor_training_buffers, ts_init.action_noise_actor, ts_init.rng);
                }
                // rlt::update_critic_targets(device, ts.actor_critic);
                rlt::update_critic_targets(device_init, ts_init.actor_critic);
            }
        }
// #ifndef BENCHMARK
//         if(step % 1000 == 0){
//             rlt::copy(device, device_init, ts.actor_critic.actor, ts_init.actor_critic.actor);
// #ifdef _MSC_VER
//             using RESULT_SPEC = rlt::rl::utils::evaluation::Specification<T, TI, typename LOOP_STATE::CONFIG::ENVIRONMENT_EVALUATION, EVAL_PARAMETERS::NUM_EVALUATION_EPISODES, CORE_PARAMETERS::EPISODE_STEP_LIMIT>;
//             rlt::rl::utils::evaluation::Result<RESULT_SPEC> result;
//             rlt::evaluate(device_init, env_eval, ui, ts_init.actor_critic.actor, result, ts_init.actor_deterministic_evaluation_buffers, rng_eval, false);
// //            auto result = rlt::evaluate(device_init, env_eval, ui, ts_init.actor_critic.actor, rlt::rl::utils::evaluation::Specification<EVAL_PARAMETERS::NUM_EVALUATION_EPISODES, CORE_PARAMETERS::EPISODE_STEP_LIMIT>(), ts_init.actor_deterministic_evaluation_buffers, rng_eval, false);
// #else
//             using RESULT_SPEC = rlt::rl::utils::evaluation::Specification<T, TI, typename LOOP_STATE::CONFIG::ENVIRONMENT_EVALUATION, EVAL_PARAMETERS::NUM_EVALUATION_EPISODES, CORE_PARAMETERS::EPISODE_STEP_LIMIT>;
//             rlt::rl::utils::evaluation::Result<RESULT_SPEC> result;
//             rlt::evaluate(device_init, ts_init.env_eval, ts_init.env_eval_parameters, ts_init.ui, ts_init.actor_critic.actor, result, ts_init.actor_deterministic_evaluation_buffers, ts_init.rng_eval, rlt::Mode<rlt::mode::Evaluation<>>{});
// #endif
//             rlt::log(device_init, device_init.logger, "Step: ", step, " Mean return: ", result.returns_mean);
// //            add_scalar(device, device.logger, "evaluation/return/mean", result.returns_mean);
// //            add_scalar(device, device.logger, "evaluation/return/std", result.returns_std);
//         }
// #endif
//         step++;
//         finished = step > CORE_PARAMETERS::STEP_LIMIT;
        step++;
     }
//     auto end_time = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed_seconds = end_time - start_time;
//     std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
//     rlt::free(device, ts);
//     rlt::free(device_init, ts_init);
}

// benchmark training should take < 2s on P1
