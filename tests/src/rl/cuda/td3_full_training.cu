// ------------ Groups 1 ------------
#include <layer_in_c/operations/cuda/group_1.h>
#include <layer_in_c/operations/cpu_mkl/group_1.h>
#include <layer_in_c/operations/cpu_tensorboard/group_1.h>
// ------------ Groups 2 ------------
#include <layer_in_c/operations/cuda/group_2.h>
#include <layer_in_c/operations/cpu_mkl/group_2.h>
#include <layer_in_c/operations/cpu_tensorboard/group_2.h>
// ------------ Groups 3 ------------
#include <layer_in_c/operations/cuda/group_3.h>
#include <layer_in_c/operations/cpu_mkl/group_3.h>
#include <layer_in_c/operations/cpu_tensorboard/group_3.h>

namespace lic = layer_in_c;

#include <layer_in_c/nn/operations_cuda.h>
#include <layer_in_c/nn/operations_cpu_mkl.h>
using DEV_SPEC_INIT = lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, lic::devices::logging::CPU_TENSORBOARD>;
using DEVICE_INIT = lic::devices::CPU<DEV_SPEC_INIT>;
//using DEVICE = lic::devices::CPU_MKL<DEV_SPEC_INIT>;
using DEVICE = lic::devices::DefaultCUDA;
using DEV_SPEC = DEVICE::SPEC;

#include "td3_full_training_parameters_pendulum.h"
#include "td3_full_training_parameters_multirotor.h"

#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/rl/components/off_policy_runner/operations_cuda.h>
#include <layer_in_c/rl/algorithms/td3/operations_cuda.h>
#include <layer_in_c/rl/algorithms/td3/operations_generic.h>

#include <layer_in_c/rl/utils/evaluation.h>


#include <gtest/gtest.h>
#include <filesystem>

using DTYPE = float;


using p = parameters_multirotor_0<DEVICE, DTYPE>;
//using p = parameters_pendulum_0<DEVICE, DTYPE>;
using rlp = p::rl<p::env::ENVIRONMENT>;

static_assert(rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == rlp::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

TEST(LAYER_IN_C_RL_CUDA_TD3, TEST_FULL_TRAINING) {
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

    lic::init(device);
    device_init.logger = &logger;
    lic::construct(device_init, device_init.logger);
    auto rng_init = lic::random::default_engine(DEVICE_INIT::SPEC::RANDOM());
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    p::env::ENVIRONMENT envs[decltype(off_policy_runner_init)::N_ENVIRONMENTS];
    bool ui = false;
    
    
    lic::malloc(device_init, actor_critic_init);
    lic::malloc(device, actor_critic);
    lic::malloc(device_init, off_policy_runner_init);
    lic::malloc(device, off_policy_runner);
    cudaMalloc(&off_policy_runner_pointer, sizeof(rlp::OFF_POLICY_RUNNER_TYPE));
    lic::check_status(device);

    lic::malloc(device, critic_batch);
    cudaMalloc(&critic_batch_pointer, sizeof(rlp::CRITIC_BATCH_TYPE));
    lic::check_status(device);
    lic::malloc(device, critic_training_buffers);
    lic::malloc(device, critic_buffers[0]);
    lic::malloc(device, critic_buffers[1]);

    lic::malloc(device, actor_batch);
    cudaMalloc(&actor_batch_pointer, sizeof(rlp::ACTOR_BATCH_TYPE));
    lic::check_status(device);
    lic::malloc(device, actor_training_buffers);
    lic::malloc(device, actor_buffers_eval);
    lic::malloc(device_init, actor_buffers_eval_init);
    lic::malloc(device, actor_buffers[0]);
    lic::malloc(device, actor_buffers[1]);

    lic::init(device_init, actor_critic_init, optimizer, rng_init);
    lic::copy(device, device_init, actor_critic, actor_critic_init);
    for(int i = 0; i < decltype(off_policy_runner_init)::N_ENVIRONMENTS; i += 1){
        auto parameters = p::env::parameters;
        envs[i].parameters = parameters;
    }
    lic::init(device_init, off_policy_runner_init, envs);
    lic::copy(device, device_init, off_policy_runner, off_policy_runner_init);
    cudaMemcpy(off_policy_runner_pointer, &off_policy_runner, sizeof(rlp::OFF_POLICY_RUNNER_TYPE), cudaMemcpyHostToDevice);
    lic::check_status(device);
    cudaMemcpy(actor_batch_pointer, &actor_batch, sizeof(rlp::ACTOR_BATCH_TYPE), cudaMemcpyHostToDevice);
    lic::check_status(device);
    cudaMemcpy(critic_batch_pointer, &critic_batch, sizeof(rlp::CRITIC_BATCH_TYPE), cudaMemcpyHostToDevice);
    lic::check_status(device);

    auto start_time = std::chrono::high_resolution_clock::now();

    constexpr DEVICE::index_t step_limit = 500000;
    for(int step_i = 0; step_i < step_limit; step_i += 1){
        rng = lic::random::next(DEVICE::SPEC::RANDOM(), rng);
        lic::rl::components::off_policy_runner::prologue(device, off_policy_runner_pointer, rng);
        rng = lic::random::next(DEVICE::SPEC::RANDOM(), rng);
        lic::rl::components::off_policy_runner::interlude(device, off_policy_runner, actor_critic.actor, actor_buffers_eval);
        rng = lic::random::next(DEVICE::SPEC::RANDOM(), rng);
        lic::rl::components::off_policy_runner::epilogue(device, off_policy_runner_pointer, rng);

        if(step_i % 1000 == 0){
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = current_time - start_time;
            std::cout << "step_i: " << step_i << " " << elapsed_seconds.count() << "s" << std::endl;
        }

        if(step_i > rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_CRITIC && step_i % rlp::ACTOR_CRITIC_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0) {
            for (int critic_i = 0; critic_i < 2; critic_i++) {
                rng = lic::random::next(DEVICE::SPEC::RANDOM(), rng);
//                cudaDeviceSynchronize();
//                auto start = std::chrono::high_resolution_clock::now();
                lic::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                rng = lic::random::next(DEVICE::SPEC::RANDOM(), rng);
                lic::gather_batch(device, off_policy_runner_pointer, critic_batch_pointer, rng);
                lic::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, optimizer, actor_buffers[critic_i], critic_buffers[critic_i], critic_training_buffers);
//                cudaDeviceSynchronize();
//                auto end = std::chrono::high_resolution_clock::now();
//                auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//                std::cout << "critic_i: " << critic_i << " " << duration_microseconds << "us" << std::endl;
            }
        }

        if(step_i > rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_ACTOR && step_i % rlp::ACTOR_CRITIC_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0) {
            cudaDeviceSynchronize();
//            auto start = std::chrono::high_resolution_clock::now();
            rng = lic::random::next(DEVICE::SPEC::RANDOM(), rng);
            lic::gather_batch(device, off_policy_runner_pointer, actor_batch_pointer, rng);
            lic::train_actor(device, actor_critic, actor_batch, optimizer, actor_buffers[0], critic_buffers[0], actor_training_buffers);
//            cudaDeviceSynchronize();
//            auto end = std::chrono::high_resolution_clock::now();
//            auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//                    std::cout << "actor: " << duration_microseconds << "us" << std::endl;
        }

        if(step_i > rlp::ACTOR_CRITIC_PARAMETERS::N_WARMUP_STEPS_CRITIC && step_i % rlp::ACTOR_CRITIC_PARAMETERS::CRITIC_TARGET_UPDATE_INTERVAL == 0) {
            {
//                cudaDeviceSynchronize();
//                auto start = std::chrono::high_resolution_clock::now();
                lic::update_critic_targets(device, actor_critic);
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
                lic::update_actor_target(device, actor_critic);
//                cudaDeviceSynchronize();
//                auto end = std::chrono::high_resolution_clock::now();
//                auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
//                    std::cout << "update: " << duration_microseconds << "us" << std::endl;
            }
        }
        if(step_i % 20000 == 0){
            lic::copy(device_init, device, actor_critic_init, actor_critic);
            auto results = lic::evaluate(device_init, envs[0], ui, actor_critic_init.actor, lic::rl::utils::evaluation::Specification<1, rlp::ENVIRONMENT_STEP_LIMIT>(), rng_init, true);
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
        // multirotor no device sync: 500k 140s, 2x of CPU BLAS => todo: investigate individual kernel timings

    }
    lic::free(device, critic_batch);
    lic::free(device, critic_training_buffers);
    lic::free(device, actor_batch);
    lic::free(device, actor_training_buffers);
    lic::free(device, off_policy_runner);
    lic::free(device, actor_critic);
}
