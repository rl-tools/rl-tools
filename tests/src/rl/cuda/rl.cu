// Group 1
#include <layer_in_c/operations/cpu/group_1.h>
#include <layer_in_c/operations/cuda/group_1.h>

// Group 2
#include <layer_in_c/operations/cpu/group_2.h>
#include <layer_in_c/operations/cuda/group_2.h>

// Group 3
#include <layer_in_c/operations/cpu/group_3.h>
#include <layer_in_c/operations/cuda/group_3.h>

#include <layer_in_c/nn/operations_cuda.h>
#include <layer_in_c/nn/loss_functions/mse/operations_cuda.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/nn_models/operations_cpu.h>

#include <layer_in_c/rl/components/replay_buffer/operations_cpu.h>
#include <layer_in_c/rl/components/replay_buffer/persist.h>
#include <layer_in_c/rl/components/off_policy_runner/operations_cpu.h>

#include <layer_in_c/rl/environments/pendulum/operations_cpu.h>

#include <layer_in_c/rl/components/off_policy_runner/operations_cuda.h>

#include "../components/replay_buffer.h"


#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

namespace lic = layer_in_c;


TEST(LAYER_IN_C_RL_CUDA, GATHER_BATCH) {
    using DEVICE_CPU = lic::devices::DefaultCPU;
    using DEVICE_GPU = lic::devices::DefaultCUDA;
    using DTYPE = float;
    constexpr DEVICE_CPU::index_t OBSERVATION_DIM = 5;
    constexpr DEVICE_CPU::index_t ACTION_DIM = 2;
    constexpr DEVICE_CPU::index_t CAPACITY = 2000;
//    using REPLAY_BUFFER_SPEC = lic::rl::components::replay_buffer::Specification<DTYPE, DEVICE_CPU::index_t, OBSERVATION_DIM, ACTION_DIM, CAPACITY>;
//    using REPLAY_BUFFER = lic::rl::components::ReplayBuffer<REPLAY_BUFFER_SPEC>;
    using PENDULUM_SPEC = lic::rl::environments::pendulum::Specification<DTYPE, DEVICE_CPU::index_t, lic::rl::environments::pendulum::DefaultParameters<DTYPE>>;
    using ENVIRONMENT = lic::rl::environments::Pendulum<PENDULUM_SPEC>;
    using OFF_POLICY_RUNNER_SPEC = lic::rl::components::off_policy_runner::Specification<DTYPE, DEVICE_CPU::index_t, ENVIRONMENT, 1, CAPACITY, 100, lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>>;
    using OFF_POLICY_RUNNER_TYPE = lic::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    auto rng = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    OFF_POLICY_RUNNER_TYPE off_policy_runner_cpu;
    OFF_POLICY_RUNNER_TYPE off_policy_runner_cpu_2;
    OFF_POLICY_RUNNER_TYPE off_policy_runner_gpu_cpu;
    OFF_POLICY_RUNNER_TYPE* off_policy_runner_gpu_struct;
    lic::malloc(device_cpu, off_policy_runner_cpu);
    lic::malloc(device_cpu, off_policy_runner_cpu_2);
    lic::malloc(device_gpu, off_policy_runner_gpu_cpu);
    for(DEVICE_CPU::index_t rb_i = 0; rb_i < OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; rb_i++) {
        lic::test::rl::components::replay_buffer::sample(device_cpu, off_policy_runner_cpu.replay_buffers[rb_i], rng);
        lic::copy(device_gpu, device_cpu, off_policy_runner_gpu_cpu.replay_buffers[rb_i], off_policy_runner_cpu.replay_buffers[rb_i]);
    }

    cudaMalloc(&off_policy_runner_gpu_struct, sizeof(OFF_POLICY_RUNNER_TYPE));
    cudaMemcpy(off_policy_runner_gpu_struct, &off_policy_runner_gpu_cpu, sizeof(OFF_POLICY_RUNNER_TYPE), cudaMemcpyHostToDevice);

    for(DEVICE_CPU::index_t rb_i = 0; rb_i < OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; rb_i++) {
        lic::copy(device_cpu, device_gpu, off_policy_runner_cpu_2.replay_buffers[rb_i], off_policy_runner_gpu_cpu.replay_buffers[rb_i]);
        auto abs_diff = lic::abs_diff(device_cpu, off_policy_runner_cpu.replay_buffers[rb_i], off_policy_runner_cpu_2.replay_buffers[rb_i]);
        ASSERT_FLOAT_EQ(abs_diff, 0);
    }

    lic::free(device_cpu, off_policy_runner_cpu);
    lic::free(device_gpu, off_policy_runner_gpu_cpu);
    cudaFree(off_policy_runner_gpu_struct);
}
