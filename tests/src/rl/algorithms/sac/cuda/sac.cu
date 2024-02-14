#define RL_TOOLS_DEBUG_DEVICE_CUDA_SYNCHRONIZE_STATUS_CHECK

// ------------ Groups 1 ------------
#include <rl_tools/operations/cuda/group_1.h>
#include <rl_tools/operations/cpu_mkl/group_1.h>
#include <rl_tools/operations/cpu_tensorboard/group_1.h>
// ------------ Groups 2 ------------
#include <rl_tools/operations/cuda/group_2.h>
#include <rl_tools/operations/cpu_mkl/group_2.h>
#include <rl_tools/operations/cpu_tensorboard/group_2.h>
// ------------ Groups 3 ------------
#include <rl_tools/operations/cuda/group_3.h>
#include <rl_tools/operations/cpu_mkl/group_3.h>
#include <rl_tools/operations/cpu_tensorboard/group_3.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;

#include <rl_tools/nn/optimizers/adam/operations_cuda.h>
#include <rl_tools/utils/polyak/operations_cuda.h>
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mkl.h>
using DEV_SPEC_INIT = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, rlt::devices::logging::CPU_TENSORBOARD<>>;
using DEVICE_INIT = rlt::devices::CPU_MKL<DEV_SPEC_INIT>;
//using DEVICE = rlt::devices::CPU_MKL<DEV_SPEC_INIT>;
using DEVICE = rlt::devices::DefaultCUDA;
using DEV_SPEC = DEVICE::SPEC;

#include "parameters.h"

#include <rl_tools/nn_models/operations_generic.h>
#include <rl_tools/rl/components/off_policy_runner/operations_cuda.h>
#include <rl_tools/rl/algorithms/sac/operations_cuda.h>
#include <rl_tools/rl/algorithms/sac/operations_generic.h>

#include <rl_tools/rl/utils/evaluation.h>


#include <gtest/gtest.h>
#include <filesystem>

using DTYPE = double;


using p = parameters_pendulum_0<DEVICE, DTYPE>;
using rlp = p::rl<p::env::ENVIRONMENT>;

static_assert(rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);


static constexpr bool check_diff = true;

TEST(RL_TOOLS_RL_ALGORITHMS_SAC_CUDA, TEST_FULL_TRAINING) {
    DEVICE_INIT::SPEC::LOGGING logger;
    DEVICE device;
    DEVICE_INIT device_init;
    using T = DTYPE;
    using TI = DEVICE::index_t;

    rlp::ACTOR_CRITIC_TYPE actor_critic_init, actor_critic_init2;
    rlp::ACTOR_CRITIC_TYPE actor_critic;
    rlp::OFF_POLICY_RUNNER_TYPE off_policy_runner_init, off_policy_runner;
    rlp::OFF_POLICY_RUNNER_TYPE* off_policy_runner_pointer;

    rlp::CRITIC_BATCH_TYPE critic_batch, critic_batch_init;
    rlp::CRITIC_BATCH_TYPE* critic_batch_pointer;
    rlp::CRITIC_TRAINING_BUFFERS_TYPE critic_training_buffers, critic_training_buffers_init, critic_training_buffers_init2;
    rlp::CRITIC_NETWORK_TYPE::Buffer<rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2], critic_buffers_init[2];
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE, p::env::ENVIRONMENT::ACTION_DIM>> action_noise_critic_init[2], action_noise_critic[2];

    rlp::ACTOR_BATCH_TYPE actor_batch, actor_batch_init;
    rlp::ACTOR_BATCH_TYPE* actor_batch_pointer;
    rlp::ACTOR_TRAINING_BUFFERS_TYPE actor_training_buffers, actor_training_buffers_init, actor_training_buffers_init2;
    rlp::ACTOR_NETWORK_TYPE::Buffer<rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2], actor_buffers_init[2];
    rlp::ACTOR_NETWORK_TYPE::Buffer<rlp::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    rlp::ACTOR_NETWORK_TYPE::Buffer<rlp::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval_init;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE, p::env::ENVIRONMENT::ACTION_DIM>> action_noise_actor_init, action_noise_actor;

    rlt::init(device);
    rlt::construct(device_init, device_init.logger);
    auto rng_init = rlt::random::default_engine(DEVICE_INIT::SPEC::RANDOM());
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM());
    p::env::ENVIRONMENT envs[decltype(off_policy_runner_init)::N_ENVIRONMENTS];
    rlt::rl::environments::DummyUI ui;


    rlt::malloc(device_init, actor_critic_init);
    rlt::malloc(device_init, actor_critic_init2);
    rlt::malloc(device, actor_critic);
    rlt::malloc(device_init, off_policy_runner_init);
    rlt::malloc(device, off_policy_runner);
    cudaMalloc(&off_policy_runner_pointer, sizeof(rlp::OFF_POLICY_RUNNER_TYPE));
    rlt::check_status(device);

    rlt::malloc(device_init, critic_batch_init);
    rlt::malloc(device, critic_batch);
    cudaMalloc(&critic_batch_pointer, sizeof(rlp::CRITIC_BATCH_TYPE));
    rlt::check_status(device);
    rlt::malloc(device_init, critic_training_buffers_init);
    rlt::malloc(device_init, critic_training_buffers_init2);
    rlt::malloc(device, critic_training_buffers);
    rlt::malloc(device_init, action_noise_critic_init[0]);
    rlt::malloc(device_init, action_noise_critic_init[1]);
    rlt::malloc(device, action_noise_critic[0]);
    rlt::malloc(device, action_noise_critic[1]);
    rlt::malloc(device_init, critic_buffers_init[0]);
    rlt::malloc(device_init, critic_buffers_init[1]);
    rlt::malloc(device, critic_buffers[0]);
    rlt::malloc(device, critic_buffers[1]);

    rlt::malloc(device_init, actor_batch_init);
    rlt::malloc(device, actor_batch);
    cudaMalloc(&actor_batch_pointer, sizeof(rlp::ACTOR_BATCH_TYPE));
    rlt::check_status(device);
    rlt::malloc(device_init, actor_training_buffers_init);
    rlt::malloc(device_init, actor_training_buffers_init2);
    rlt::malloc(device, actor_training_buffers);
    rlt::malloc(device_init, action_noise_actor_init);
    rlt::malloc(device, action_noise_actor);
    rlt::malloc(device, actor_buffers_eval);
    rlt::malloc(device_init, actor_buffers_eval_init);
    rlt::malloc(device_init, actor_buffers_init[0]);
    rlt::malloc(device_init, actor_buffers_init[1]);
    rlt::malloc(device, actor_buffers[0]);
    rlt::malloc(device, actor_buffers[1]);

    rlt::init(device_init, actor_critic_init, rng_init);
    rlt::zero_gradient(device_init, actor_critic_init.critic_1);
    rlt::zero_gradient(device_init, actor_critic_init.critic_2);
    rlt::zero_gradient(device_init, actor_critic_init.actor);
    rlt::reset_forward_state(device_init, actor_critic_init.critic_1);
    rlt::reset_forward_state(device_init, actor_critic_init.critic_2);
    rlt::reset_forward_state(device_init, actor_critic_init.actor);
//    rlt::reset_optimizer_state(device_init, optimizer, actor_critic_init.critic_1);
//    rlt::reset_optimizer_state(device_init, optimizer, actor_critic_init.critic_2);
//    rlt::reset_optimizer_state(device_init, optimizer, actor_critic_init.actor);
    rlt::copy(device_init, device, actor_critic_init, actor_critic);
//    for(int i = 0; i < decltype(off_policy_runner_init)::N_ENVIRONMENTS; i += 1){
//        auto parameters = p::env::parameters;
//        envs[i].parameters = parameters;
//    }
    rlt::init(device_init, off_policy_runner_init, envs);
    rlt::copy(device_init, device, off_policy_runner_init, off_policy_runner);
    cudaMemcpy(off_policy_runner_pointer, &off_policy_runner, sizeof(rlp::OFF_POLICY_RUNNER_TYPE), cudaMemcpyHostToDevice);
    rlt::check_status(device);
    cudaMemcpy(actor_batch_pointer, &actor_batch, sizeof(rlp::ACTOR_BATCH_TYPE), cudaMemcpyHostToDevice);
    rlt::check_status(device);
    cudaMemcpy(critic_batch_pointer, &critic_batch, sizeof(rlp::CRITIC_BATCH_TYPE), cudaMemcpyHostToDevice);
    rlt::check_status(device);

    auto start_time = std::chrono::high_resolution_clock::now();

    constexpr DEVICE::index_t step_limit = 20000;
    T epsilon = 1e-13;
    T epsilon_decay = 1;
    T epsilon_decay_rate = 1.02;
    T returns_acc = 0;
    T returns_acc_count = 0;
    for(int step_i = 0; step_i < step_limit; step_i += 1){
        bool check_diff_now = check_diff && (step_i < rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_CRITIC + 100);
        if(step_i % 1000 == 0){
            {
                auto rng_init_copy = rng_init;
                rlt::copy(device, device_init, actor_critic, actor_critic_init2);
                auto results = rlt::evaluate(device_init, envs[0], ui, actor_critic_init2.actor, rlt::rl::utils::evaluation::Specification<100, rlp::EPISODE_STEP_LIMIT>(), actor_buffers_eval_init, rng_init_copy);
                std::cout << "Mean return (GPU): " << results.returns_mean << std::endl;
                if(step_i > 10000){
                    ASSERT_GT(results.returns_mean, -400);
                    returns_acc += results.returns_mean;
                    returns_acc_count += 1;
                }
            }
            {
                auto rng_init_copy = rng_init;
                auto results = rlt::evaluate(device_init, envs[0], ui, actor_critic_init.actor, rlt::rl::utils::evaluation::Specification<100, rlp::EPISODE_STEP_LIMIT>(), actor_buffers_eval_init, rng_init_copy);
                std::cout << "Mean return (CPU): " << results.returns_mean << std::endl;
                if(step_i > 10000){
                    ASSERT_GT(results.returns_mean, -400);
                }
            }
        }
        rlt::step(device, off_policy_runner, off_policy_runner_pointer, actor_critic.actor, actor_buffers_eval, rng);
//        rng = rlt::random::next(DEVICE::SPEC::RANDOM(), rng);
//        rlt::rl::components::off_policy_runner::prologue(device, *off_policy_runner_pointer, rng);
//        rng = rlt::random::next(DEVICE::SPEC::RANDOM(), rng);
//        rlt::rl::components::off_policy_runner::interlude(device, off_policy_runner, actor_critic.actor, actor_buffers_eval);
//        rng = rlt::random::next(DEVICE::SPEC::RANDOM(), rng);
//        rlt::rl::components::off_policy_runner::epilogue(device, *off_policy_runner_pointer, rng);

        if(step_i % 1000 == 0){
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = current_time - start_time;
            std::cout << "step_i: " << step_i << " " << elapsed_seconds.count() << "s" << std::endl;
        }

        if(step_i > rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_CRITIC && step_i % rlp::ACTOR_CRITIC_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0) {
            for (int critic_i = 0; critic_i < 2; critic_i++) {
                rng = rlt::random::next(DEVICE::SPEC::RANDOM(), rng);
//                cudaDeviceSynchronize();
//                auto start = std::chrono::high_resolution_clock::now();
                rng = rlt::random::next(DEVICE::SPEC::RANDOM(), rng);
                rlt::gather_batch(device, off_policy_runner_pointer, critic_batch, rng);
                rlt::copy(device, device_init, critic_batch, critic_batch_init);
                if(check_diff_now){
                    rlt::copy(device, device_init, actor_critic, actor_critic_init2);
                    DTYPE diff_before = rlt::abs_diff(device_init, actor_critic_init.critic_1, actor_critic_init2.critic_1);
                    std::cout << "step: " << step_i << " " << "diff before: " << diff_before << std::endl;
                }
                rlt::randn(device_init, action_noise_critic_init[critic_i], rng_init);
                rlt::copy(device_init, device, action_noise_critic_init[critic_i], action_noise_critic[critic_i]);
                rlt::train_critic(device     , actor_critic     , critic_i == 0 ? actor_critic.critic_1      : actor_critic.critic_2     , critic_batch     , actor_critic.critic_optimizers[critic_i]     , actor_buffers[critic_i]     , critic_buffers[critic_i]     , critic_training_buffers     , action_noise_critic[critic_i]);
                rlt::train_critic(device_init, actor_critic_init, critic_i == 0 ? actor_critic_init.critic_1 : actor_critic_init.critic_2, critic_batch_init, actor_critic_init.critic_optimizers[critic_i], actor_buffers_init[critic_i], critic_buffers_init[critic_i], critic_training_buffers_init, action_noise_critic_init[critic_i]);
                if(check_diff_now){
                    rlt::copy(device, device_init, critic_training_buffers, critic_training_buffers_init2);
//                rlt::copy(device, device_init, critic_training_buffers.next_actions_mean, critic_training_buffers_init2.next_actions_mean);
                    T next_action_log_std_diff = rlt::abs_diff(device_init, critic_training_buffers_init.next_actions_log_std, critic_training_buffers_init2.next_actions_log_std);
                    std::cout << "step: " << step_i << " " << "next_action_log_std_diff: " << next_action_log_std_diff << std::endl;
                    ASSERT_LT(next_action_log_std_diff, epsilon*epsilon_decay);
                    T next_actions_mean_diff = rlt::abs_diff(device_init, critic_training_buffers_init.next_actions_mean, critic_training_buffers_init2.next_actions_mean);
                    std::cout << "step: " << step_i << " " << "next_actions_mean_diff: " << next_actions_mean_diff << std::endl;
                    ASSERT_LT(next_actions_mean_diff, epsilon*epsilon_decay);
                    T next_state_action_value_input_diff = rlt::abs_diff(device_init, critic_training_buffers_init.next_state_action_value_input, critic_training_buffers_init2.next_state_action_value_input);
                    std::cout << "step: " << step_i << " " << "next_state_action_value_input_diff: " << next_state_action_value_input_diff << std::endl;
                    ASSERT_LT(next_state_action_value_input_diff, epsilon*epsilon_decay);
                    T next_state_action_value_critic_1_diff = rlt::abs_diff(device_init, critic_training_buffers_init.next_state_action_value_critic_1, critic_training_buffers_init2.next_state_action_value_critic_1);
                    std::cout << "step: " << step_i << " " << "next_state_action_value_critic_1_diff: " << next_state_action_value_critic_1_diff << std::endl;
                    ASSERT_LT(next_state_action_value_critic_1_diff, epsilon*epsilon_decay);
                    T next_state_action_value_critic_2_diff = rlt::abs_diff(device_init, critic_training_buffers_init.next_state_action_value_critic_2, critic_training_buffers_init2.next_state_action_value_critic_2);
                    std::cout << "step: " << step_i << " " << "next_state_action_value_critic_2_diff: " << next_state_action_value_critic_2_diff << std::endl;
                    ASSERT_LT(next_state_action_value_critic_2_diff, epsilon*epsilon_decay);
                    T target_action_value_diff = rlt::abs_diff(device_init, critic_training_buffers_init.target_action_value, critic_training_buffers_init2.target_action_value);
                    std::cout << "step: " << step_i << " " << "target_action_value_diff: " << target_action_value_diff << std::endl;
                    ASSERT_LT(target_action_value_diff, epsilon*epsilon_decay);
                    T d_output_diff = rlt::abs_diff(device_init, critic_training_buffers_init.d_output, critic_training_buffers_init2.d_output);
                    std::cout << "step: " << step_i << " " << "d_output_diff: " << d_output_diff << std::endl;
                    ASSERT_LT(d_output_diff, epsilon*epsilon_decay);

                    rlt::copy(device, device_init, actor_critic, actor_critic_init2);
                    T diff_gradient_first_layer = rlt::abs_diff(device_init, actor_critic_init.critic_1.input_layer.weights.gradient, actor_critic_init2.critic_1.input_layer.weights.gradient);
                    std::cout << "step: " << step_i << " " << "diff_gradient_first_layer: " << diff_gradient_first_layer << std::endl;
                    ASSERT_LT(diff_gradient_first_layer, epsilon*epsilon_decay);
                    T diff_weights_first_layer = rlt::abs_diff(device_init, actor_critic_init.critic_1.input_layer.weights.parameters, actor_critic_init2.critic_1.input_layer.weights.parameters);
                    std::cout << "step: " << step_i << " " << "diff_weights_first_layer: " << diff_weights_first_layer << std::endl;
                    ASSERT_LT(diff_weights_first_layer, epsilon*epsilon_decay);
                    DTYPE diff_after = rlt::abs_diff(device_init, actor_critic_init.critic_1, actor_critic_init2.critic_1);
                    std::cout << "step: " << step_i << " " << "diff after: " << diff_after << std::endl;
                    ASSERT_LT(diff_after, 1e-10*epsilon_decay);

                }
//                cudaDeviceSynchronize();
//                auto end = std::chrono::high_resolution_clock::now();
//                auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//                std::cout << "critic_i: " << critic_i << " " << duration_microseconds << "us" << std::endl;
            }
        }

        if(step_i > rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_ACTOR && step_i % rlp::ACTOR_CRITIC_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0) {
            cudaDeviceSynchronize();
//            auto start = std::chrono::high_resolution_clock::now();
            if(check_diff_now){
                rlt::copy(device, device_init, actor_critic, actor_critic_init2);
                T diff_before = rlt::abs_diff(device_init, actor_critic_init.actor, actor_critic_init2.actor);
                std::cout << "step: " << step_i << " " << "actor diff before: " << diff_before << std::endl;
            }
            rng = rlt::random::next(DEVICE::SPEC::RANDOM(), rng);
            rlt::gather_batch(device, off_policy_runner_pointer, actor_batch, rng);
            rlt::copy(device, device_init, actor_batch, actor_batch_init);
            rlt::randn(device_init, action_noise_actor_init, rng_init);
            rlt::copy(device_init, device, action_noise_actor_init, action_noise_actor);
            rlt::train_actor(device     , actor_critic     , actor_batch     , actor_critic.actor_optimizer     , actor_buffers[0]     , critic_buffers[0]     , actor_training_buffers     , action_noise_actor);
            rlt::train_actor(device_init, actor_critic_init, actor_batch_init, actor_critic_init.actor_optimizer, actor_buffers_init[0], critic_buffers_init[0], actor_training_buffers_init, action_noise_actor_init);
            if(check_diff_now){
                rlt::copy(device, device_init, actor_training_buffers, actor_training_buffers_init2);
                rlt::copy(device, device_init, actor_critic, actor_critic_init2);
                T diff_output = rlt::abs_diff(device_init, actor_critic_init.actor.output_layer.output, actor_critic_init2.actor.output_layer.output);
                std::cout << "step: " << step_i << " " << "diff_output: " << diff_output << std::endl;
                ASSERT_LT(diff_output, epsilon*epsilon_decay);
                T diff_actions = rlt::abs_diff(device_init, actor_training_buffers_init.actions, actor_training_buffers_init2.actions);
                std::cout << "step: " << step_i << " " << "diff_actions: " << diff_actions << std::endl;
                ASSERT_LT(diff_actions, epsilon*epsilon_decay);
                T diff_after = rlt::abs_diff(device_init, actor_critic_init.actor, actor_critic_init2.actor);
                std::cout << "step: " << step_i << " " << "actor diff after: " << diff_after << std::endl;
                ASSERT_LT(diff_after, 1e-10*epsilon_decay);
            }
//            cudaDeviceSynchronize();
//            auto end = std::chrono::high_resolution_clock::now();
//            auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//                    std::cout << "actor: " << duration_microseconds << "us" << std::endl;
            if(epsilon_decay == 1){
                epsilon_decay *=  10000000;
            }
            else{
                epsilon_decay *= epsilon_decay_rate;
            }
        }

        if(step_i > rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_CRITIC && step_i % rlp::ACTOR_CRITIC_PARAMETERS::CRITIC_TARGET_UPDATE_INTERVAL == 0) {
            {
//                cudaDeviceSynchronize();
//                auto start = std::chrono::high_resolution_clock::now();
                rlt::update_critic_targets(device, actor_critic);
                rlt::update_critic_targets(device_init, actor_critic_init);
//                cudaDeviceSynchronize();
//                auto end = std::chrono::high_resolution_clock::now();
//                auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//                    std::cout << "update: " << duration_microseconds << "us" << std::endl;
            }
        }
    }
    ASSERT_GT(returns_acc/returns_acc_count, -200);
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
        // 90s, 15x of CPU BLAS => todo: investigate individual kernel timings
        // on device rollout: 24s, 6x of CPU BLAS => todo: investigate individual kernel timings
        // no device sync: 14s, 2.5x of CPU BLAS => todo: investigate individual kernel timings

    }
    rlt::free(device, critic_batch);
    rlt::free(device, critic_training_buffers);
    rlt::free(device, actor_batch);
    rlt::free(device, actor_training_buffers);
    rlt::free(device, off_policy_runner);
    rlt::free(device, actor_critic);
}
