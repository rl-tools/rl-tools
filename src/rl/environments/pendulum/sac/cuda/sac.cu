#define RL_TOOLS_DEBUG_DEVICE_CUDA_SYNCHRONIZE_STATUS_CHECK
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>

#include <rl_tools/nn/optimizers/adam/operations_cuda.h>
#include <rl_tools/rl/algorithms/sac/operations_cuda.h>

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations.h>
#include <rl_tools/rl/loop/steps/evaluation/operations.h>
#include <rl_tools/rl/loop/steps/timing/operations.h>

#include <rl_tools/rl/components/off_policy_runner/operations_cuda.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY_CUDA<>;
using DEVICE_INIT = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    static constexpr TI STEP_LIMIT = 10000;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr bool COLLECT_EPISODE_STATS = false;
    static constexpr TI EPISODE_STATS_BUFFER_SIZE = 0;
};
#ifdef BENCHMARK
template <typename DEVICE>
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::DefaultConfig<DEVICE, T, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
template <typename DEVICE>
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::DefaultConfig<LOOP_CORE_CONFIG<DEVICE>>;
template <typename DEVICE>
using LOOP_CONFIG = LOOP_TIMING_CONFIG<DEVICE>;
#else
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using RNG_INIT = decltype(rlt::random::default_engine(typename DEVICE_INIT::SPEC::RANDOM{}));
template <typename RNG>
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::DefaultConfig<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::DefaultConfigApproximatorsMLP>;
template <typename RNG>
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::DefaultConfig<LOOP_CORE_CONFIG<RNG>>;
template <typename RNG>
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::DefaultConfig<LOOP_EVAL_CONFIG<RNG>>;
template <typename RNG>
using LOOP_CONFIG = LOOP_TIMING_CONFIG<RNG>;
#endif

using LOOP_STATE = LOOP_CONFIG<RNG>::template State<LOOP_CONFIG<RNG>>;
using LOOP_STATE_INIT = LOOP_CONFIG<RNG_INIT>::template State<LOOP_CONFIG<RNG_INIT>>;


int main(){
    DEVICE device;
    DEVICE_INIT device_init;
    LOOP_STATE ts;
    LOOP_STATE_INIT ts_init;
    using CORE_PARAMETERS = decltype(ts)::CONFIG::NEXT::NEXT::PARAMETERS;
    using EVAL_PARAMETERS = decltype(ts)::CONFIG::NEXT::PARAMETERS;
    rlt::init(device);
    rlt::malloc(device, ts);
    rlt::malloc(device_init, ts_init);
    rlt::init(device_init, ts_init, 0);
    rlt::copy(device_init, device, ts_init, ts);
//    rlt::copy(device_init, device, ts_init.off_policy_runner, ts.off_policy_runner);


    decltype(ts.off_policy_runner)* off_policy_runner_pointer;
    cudaMalloc(&off_policy_runner_pointer, sizeof(decltype(ts.off_policy_runner)));
    cudaMemcpy(off_policy_runner_pointer, &ts.off_policy_runner, sizeof(decltype(ts.off_policy_runner)), cudaMemcpyHostToDevice);
    rlt::check_status(device);

    TI step = 0;
    bool finished = false;
    while(!finished){
        rlt::set_step(device, device.logger, step);
//        evaluate(device, ts.actor_critic.actor, ts.off_policy_runner.buffers.observations, ts.off_policy_runner.buffers.actions, ts.actor_buffers_eval);
        rlt::step(device, ts.off_policy_runner, off_policy_runner_pointer, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
//        {
//            ts.rng = rlt::random::next(typename DEVICE::SPEC::RANDOM{}, ts.rng);
//            rlt::rl::components::off_policy_runner::prologue(device, off_policy_runner_pointer, ts.rng);
//            rlt::check_status(device);
//            rlt::rl::components::off_policy_runner::interlude(device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval);
//            ts.rng = rlt::random::next(typename DEVICE::SPEC::RANDOM{}, ts.rng);
//            rlt::rl::components::off_policy_runner::epilogue(device, off_policy_runner_pointer, ts.rng);
//        }
        if(step > CORE_PARAMETERS::N_WARMUP_STEPS){
            for(int critic_i = 0; critic_i < 2; critic_i++){
                rlt::gather_batch(device, off_policy_runner_pointer, ts.critic_batch, ts.rng);
                rlt::randn(device, ts.action_noise_critic[critic_i], ts.rng);
                rlt::train_critic(device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers, ts.action_noise_critic[critic_i]);
            }
            if(step % 1 == 0){
                {
                    rlt::gather_batch(device, off_policy_runner_pointer, ts.actor_batch, ts.rng);
                    rlt::randn(device, ts.action_noise_actor, ts.rng);
                    rlt::train_actor(device, ts.actor_critic, ts.actor_batch, ts.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers, ts.action_noise_actor);
                }
                rlt::update_critic_targets(device, ts.actor_critic);
            }
        }
        if(step % 1000 == 0){
            rlt::copy(device, device_init, ts.actor_critic.actor, ts_init.actor_critic.actor);
            auto result = rlt::evaluate(device_init, ts_init.env_eval, ts_init.ui, ts_init.actor_critic.actor, rlt::rl::utils::evaluation::Specification<EVAL_PARAMETERS::NUM_EVALUATION_EPISODES, CORE_PARAMETERS::ENVIRONMENT_STEP_LIMIT>(), ts_init.observations_mean, ts_init.observations_std, ts_init.actor_deterministic_evaluation_buffers, ts_init.rng_eval, false);
            rlt::log(device_init, device_init.logger, "Step: ", step, " Mean return: ", result.returns_mean);
//            add_scalar(device, device.logger, "evaluation/return/mean", result.returns_mean);
//            add_scalar(device, device.logger, "evaluation/return/std", result.returns_std);
        }
        step++;
        finished = step > CORE_PARAMETERS::STEP_LIMIT;
    }
    rlt::free(device, ts);
    rlt::free(device_init, ts_init);
}


// benchmark training should take < 2s on P1, < 0.75 on M3
