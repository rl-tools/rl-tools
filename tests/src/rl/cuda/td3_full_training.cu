// ------------ Groups 1 ------------
#include <backprop_tools/operations/cuda/group_1.h>
#include <backprop_tools/operations/cpu_mkl/group_1.h>
#include <backprop_tools/operations/cpu_tensorboard/group_1.h>
// ------------ Groups 2 ------------
#include <backprop_tools/operations/cuda/group_2.h>
#include <backprop_tools/operations/cpu_mkl/group_2.h>
#include <backprop_tools/operations/cpu_tensorboard/group_2.h>
// ------------ Groups 3 ------------
#include <backprop_tools/operations/cuda/group_3.h>
#include <backprop_tools/operations/cpu_mkl/group_3.h>
#include <backprop_tools/operations/cpu_tensorboard/group_3.h>

namespace bpt = backprop_tools;

#include <backprop_tools/nn/operations_cuda.h>
#include <backprop_tools/nn/operations_cpu_mkl.h>
using DEV_SPEC_INIT = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, bpt::devices::logging::CPU_TENSORBOARD>;
using DEVICE_INIT = bpt::devices::CPU<DEV_SPEC_INIT>;
//using DEVICE = bpt::devices::CPU_MKL<DEV_SPEC_INIT>;
using DEVICE = bpt::devices::DefaultCUDA;
using DEV_SPEC = DEVICE::SPEC;

#include "td3_full_training_parameters_pendulum.h"

#include <backprop_tools/nn_models/operations_generic.h>
#include <backprop_tools/rl/components/off_policy_runner/operations_cuda.h>
#include <backprop_tools/rl/algorithms/td3/operations_cuda.h>
#include <backprop_tools/rl/algorithms/td3/operations_generic.h>

#include <backprop_tools/rl/utils/evaluation.h>


#include <gtest/gtest.h>
#include <filesystem>

using DTYPE = float;


using p = parameters_pendulum_0<DEVICE, DTYPE>;
using rlp = p::rl<p::env::ENVIRONMENT>;

static_assert(rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

TEST(BACKPROP_TOOLS_RL_CUDA_TD3, TEST_FULL_TRAINING) {
    DEVICE_INIT::SPEC::LOGGING logger;
    DEVICE device;
    DEVICE_INIT device_init;
    rlp::OPTIMIZER optimizer;

    rlp::ACTOR_CRITIC_TYPE actor_critic_init;
    rlp::ACTOR_CRITIC_TYPE actor_critic;
    rlp::OFF_POLICY_RUNNER_TYPE off_policy_runner_init, off_policy_runner;
    rlp::OFF_POLICY_RUNNER_TYPE* off_policy_runner_pointer;

    rlp::CRITIC_BATCH_TYPE critic_batch;
    rlp::CRITIC_BATCH_TYPE* critic_batch_pointer;
    rlp::CRITIC_TRAINING_BUFFERS_TYPE critic_training_buffers;
    rlp::CRITIC_NETWORK_TYPE::BuffersForwardBackward<rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];

    rlp::ACTOR_BATCH_TYPE actor_batch;
    rlp::ACTOR_BATCH_TYPE* actor_batch_pointer;
    rlp::ACTOR_TRAINING_BUFFERS_TYPE actor_training_buffers;
    rlp::ACTOR_NETWORK_TYPE::Buffers<rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
    rlp::ACTOR_NETWORK_TYPE::Buffers<rlp::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    rlp::ACTOR_NETWORK_TYPE::Buffers<rlp::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval_init;

    bpt::init(device);
    device_init.logger = &logger;
    bpt::construct(device_init, device_init.logger);
    auto rng_init = bpt::random::default_engine(DEVICE_INIT::SPEC::RANDOM());
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    p::env::ENVIRONMENT envs[decltype(off_policy_runner_init)::N_ENVIRONMENTS];
    bool ui = false;
    
    
    bpt::malloc(device_init, actor_critic_init);
    bpt::malloc(device, actor_critic);
    bpt::malloc(device_init, off_policy_runner_init);
    bpt::malloc(device, off_policy_runner);
    cudaMalloc(&off_policy_runner_pointer, sizeof(rlp::OFF_POLICY_RUNNER_TYPE));
    bpt::check_status(device);

    bpt::malloc(device, critic_batch);
    cudaMalloc(&critic_batch_pointer, sizeof(rlp::CRITIC_BATCH_TYPE));
    bpt::check_status(device);
    bpt::malloc(device, critic_training_buffers);
    bpt::malloc(device, critic_buffers[0]);
    bpt::malloc(device, critic_buffers[1]);

    bpt::malloc(device, actor_batch);
    cudaMalloc(&actor_batch_pointer, sizeof(rlp::ACTOR_BATCH_TYPE));
    bpt::check_status(device);
    bpt::malloc(device, actor_training_buffers);
    bpt::malloc(device, actor_buffers_eval);
    bpt::malloc(device_init, actor_buffers_eval_init);
    bpt::malloc(device, actor_buffers[0]);
    bpt::malloc(device, actor_buffers[1]);

    bpt::init(device_init, actor_critic_init, rng_init);
    bpt::copy(device, device_init, actor_critic, actor_critic_init);
//    for(int i = 0; i < decltype(off_policy_runner_init)::N_ENVIRONMENTS; i += 1){
//        auto parameters = p::env::parameters;
//        envs[i].parameters = parameters;
//    }
    bpt::init(device_init, off_policy_runner_init, envs);
    bpt::copy(device, device_init, off_policy_runner, off_policy_runner_init);
    cudaMemcpy(off_policy_runner_pointer, &off_policy_runner, sizeof(rlp::OFF_POLICY_RUNNER_TYPE), cudaMemcpyHostToDevice);
    bpt::check_status(device);
    cudaMemcpy(actor_batch_pointer, &actor_batch, sizeof(rlp::ACTOR_BATCH_TYPE), cudaMemcpyHostToDevice);
    bpt::check_status(device);
    cudaMemcpy(critic_batch_pointer, &critic_batch, sizeof(rlp::CRITIC_BATCH_TYPE), cudaMemcpyHostToDevice);
    bpt::check_status(device);

    auto start_time = std::chrono::high_resolution_clock::now();

    constexpr DEVICE::index_t step_limit = 500000;
    for(int step_i = 0; step_i < step_limit; step_i += 1){
        rng = bpt::random::next(DEVICE::SPEC::RANDOM(), rng);
        bpt::rl::components::off_policy_runner::prologue(device, off_policy_runner_pointer, rng);
        rng = bpt::random::next(DEVICE::SPEC::RANDOM(), rng);
        bpt::rl::components::off_policy_runner::interlude(device, off_policy_runner, actor_critic.actor, actor_buffers_eval);
        rng = bpt::random::next(DEVICE::SPEC::RANDOM(), rng);
        bpt::rl::components::off_policy_runner::epilogue(device, off_policy_runner_pointer, rng);

        if(step_i % 1000 == 0){
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = current_time - start_time;
            std::cout << "step_i: " << step_i << " " << elapsed_seconds.count() << "s" << std::endl;
        }

        if(step_i > rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_CRITIC && step_i % rlp::ACTOR_CRITIC_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0) {
            for (int critic_i = 0; critic_i < 2; critic_i++) {
                rng = bpt::random::next(DEVICE::SPEC::RANDOM(), rng);
//                cudaDeviceSynchronize();
//                auto start = std::chrono::high_resolution_clock::now();
                bpt::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                rng = bpt::random::next(DEVICE::SPEC::RANDOM(), rng);
                bpt::gather_batch(device, off_policy_runner_pointer, critic_batch_pointer, rng);
                bpt::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, optimizer, actor_buffers[critic_i], critic_buffers[critic_i], critic_training_buffers);
//                cudaDeviceSynchronize();
//                auto end = std::chrono::high_resolution_clock::now();
//                auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//                std::cout << "critic_i: " << critic_i << " " << duration_microseconds << "us" << std::endl;
            }
        }

        if(step_i > rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_ACTOR && step_i % rlp::ACTOR_CRITIC_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0) {
            cudaDeviceSynchronize();
//            auto start = std::chrono::high_resolution_clock::now();
            rng = bpt::random::next(DEVICE::SPEC::RANDOM(), rng);
            bpt::gather_batch(device, off_policy_runner_pointer, actor_batch_pointer, rng);
            bpt::train_actor(device, actor_critic, actor_batch, optimizer, actor_buffers[0], critic_buffers[0], actor_training_buffers);
//            cudaDeviceSynchronize();
//            auto end = std::chrono::high_resolution_clock::now();
//            auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//                    std::cout << "actor: " << duration_microseconds << "us" << std::endl;
        }

        if(step_i > rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_CRITIC && step_i % rlp::ACTOR_CRITIC_PARAMETERS::CRITIC_TARGET_UPDATE_INTERVAL == 0) {
            {
//                cudaDeviceSynchronize();
//                auto start = std::chrono::high_resolution_clock::now();
                bpt::update_critic_targets(device, actor_critic);
//                cudaDeviceSynchronize();
//                auto end = std::chrono::high_resolution_clock::now();
//                auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//                    std::cout << "update: " << duration_microseconds << "us" << std::endl;
            }
        }
        if(step_i > rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_ACTOR && step_i % rlp::ACTOR_CRITIC_PARAMETERS::ACTOR_TARGET_UPDATE_INTERVAL == 0) {
            {
//                cudaDeviceSynchronize();
//                auto start = std::chrono::high_resolution_clock::now();
                bpt::update_actor_target(device, actor_critic);
//                cudaDeviceSynchronize();
//                auto end = std::chrono::high_resolution_clock::now();
//                auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//                    std::cout << "update: " << duration_microseconds << "us" << std::endl;
            }
        }
        if(step_i % 20000 == 0){
            bpt::copy(device_init, device, actor_critic_init, actor_critic);
            auto results = bpt::evaluate(device_init, envs[0], ui, actor_critic_init.actor, bpt::rl::utils::evaluation::Specification<1, rlp::ENVIRONMENT_STEP_LIMIT>(), rng_init, true);
            std::cout << "Mean return: " << results.mean << std::endl;
        }
    }
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
        // 90s, 15x of CPU BLAS => todo: investigate individual kernel timings
        // on device rollout: 24s, 6x of CPU BLAS => todo: investigate individual kernel timings
        // no device sync: 14s, 2.5x of CPU BLAS => todo: investigate individual kernel timings

    }
    bpt::free(device, critic_batch);
    bpt::free(device, critic_training_buffers);
    bpt::free(device, actor_batch);
    bpt::free(device, actor_training_buffers);
    bpt::free(device, off_policy_runner);
    bpt::free(device, actor_critic);
}
