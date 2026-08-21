#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environments/operations_generic_batch.h>
#include <rl_tools/rl/environments/operations_cuda_batch.h>
#include <rl_tools/rl/environments/multi_environment/operations_cuda.h>

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;

constexpr TI NUMBER_OF_ENVIRONMENTS = 4;
constexpr TI INSTANCES = 64;
using MULTI_ENVIRONMENT = rlt::rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>;

template <typename T_DEVICE, bool DYNAMIC = true>
struct Tensors {
    rlt::Tensor<rlt::tensor::Specification<typename ENVIRONMENT::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>, DYNAMIC>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename ENVIRONMENT::State, TI, rlt::tensor::Shape<TI, INSTANCES>, DYNAMIC>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>, DYNAMIC>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, ENVIRONMENT::ACTION_DIM>, DYNAMIC>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, ENVIRONMENT::Observation::DIM>, DYNAMIC>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES>, DYNAMIC>> rewards;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>, DYNAMIC>> terminated_flags;

    void allocate(T_DEVICE& device){
        rlt::malloc(device, parameters);
        rlt::malloc(device, states);
        rlt::malloc(device, next_states);
        rlt::malloc(device, reset_mask);
        rlt::malloc(device, actions);
        rlt::malloc(device, observations);
        rlt::malloc(device, rewards);
        rlt::malloc(device, terminated_flags);
    }
    void deallocate(T_DEVICE& device){
        rlt::free(device, parameters);
        rlt::free(device, states);
        rlt::free(device, next_states);
        rlt::free(device, reset_mask);
        rlt::free(device, actions);
        rlt::free(device, observations);
        rlt::free(device, rewards);
        rlt::free(device, terminated_flags);
    }
};

static bool cuda_available(){
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

template <typename ENV_LIKE>
void cuda_rollout(DEVICE& device, DEVICE_GPU& device_gpu, ENV_LIKE& env, TI seed, std::vector<typename ENVIRONMENT::State>& final_states, std::vector<T>& final_rewards){
    RNG_GPU rng;
    rlt::malloc(device_gpu, rng);
    rlt::init(device_gpu, rng, seed);
    Tensors<DEVICE_GPU> tensors;
    tensors.allocate(device_gpu);
    Tensors<DEVICE> host;
    host.allocate(device);

    rlt::set_all(device_gpu, tensors.reset_mask, true);
    rlt::set_all(device_gpu, tensors.actions, (T)0.5);
    rlt::sample_initial_parameters(device_gpu, env, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device_gpu, env, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    for(TI step_i = 0; step_i < 20; step_i++){
        rlt::observe(device_gpu, env, tensors.parameters, tensors.states, typename ENVIRONMENT::Observation{}, tensors.observations, rng);
        rlt::step(device_gpu, env, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
        rlt::reward(device_gpu, env, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, tensors.rewards, rng);
        rlt::terminated(device_gpu, env, tensors.parameters, tensors.states, tensors.terminated_flags, rng);
        rlt::copy(device_gpu, device_gpu, tensors.next_states, tensors.states);
    }
    rlt::check_status(device_gpu);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, tensors.states, host.states);
    rlt::copy(device_gpu, device, tensors.rewards, host.rewards);
    final_states.resize(INSTANCES);
    final_rewards.resize(INSTANCES);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        final_states[instance_i] = rlt::get(device, host.states, instance_i);
        final_rewards[instance_i] = rlt::get(device, host.rewards, instance_i);
    }
    tensors.deallocate(device_gpu);
    host.deallocate(device);
    rlt::free(device_gpu, rng);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT_CUDA, MAPPED_DEFAULT_DETERMINISM){
    if(!cuda_available()){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device_gpu);
    ENVIRONMENT env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    std::vector<typename ENVIRONMENT::State> states_a, states_b;
    std::vector<T> rewards_a, rewards_b;
    cuda_rollout(device, device_gpu, env, 1337, states_a, rewards_a);
    cuda_rollout(device, device_gpu, env, 1337, states_b, rewards_b);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        ASSERT_EQ(std::memcmp(&states_a[instance_i], &states_b[instance_i], sizeof(typename ENVIRONMENT::State)), 0);
        ASSERT_EQ(rewards_a[instance_i], rewards_b[instance_i]);
    }
    bool distinct = false;
    for(TI instance_i = 1; instance_i < INSTANCES; instance_i++){
        distinct = distinct || std::memcmp(&states_a[0], &states_a[instance_i], sizeof(typename ENVIRONMENT::State)) != 0;
    }
    EXPECT_TRUE(distinct) << "per-instance RNG states should decorrelate the instances";
    rlt::free(device, env);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT_CUDA, COMPOSITE_DETERMINISM){
    if(!cuda_available()){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device_gpu);
    MULTI_ENVIRONMENT multi_env;
    rlt::malloc(device, multi_env);
    rlt::init(device, multi_env);
    std::vector<typename ENVIRONMENT::State> states_a, states_b;
    std::vector<T> rewards_a, rewards_b;
    cuda_rollout(device, device_gpu, multi_env, 1337, states_a, rewards_a);
    cuda_rollout(device, device_gpu, multi_env, 1337, states_b, rewards_b);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        ASSERT_EQ(std::memcmp(&states_a[instance_i], &states_b[instance_i], sizeof(typename ENVIRONMENT::State)), 0);
        ASSERT_EQ(rewards_a[instance_i], rewards_b[instance_i]);
    }
    rlt::free(device, multi_env);
}

// same fixed initial state, zero actions: the CUDA mapped kernels compute the same dynamics as
// the CPU loop within float tolerance
TEST(RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT_CUDA, CPU_CUDA_CONSISTENCY){
    if(!cuda_available()){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device_gpu);
    ENVIRONMENT env;
    rlt::malloc(device, env);
    rlt::init(device, env);

    Tensors<DEVICE> host;
    host.allocate(device);
    typename DEVICE::SPEC::RANDOM::ENGINE<> rng_cpu;
    rlt::malloc(device, rng_cpu);
    rlt::init(device, rng_cpu, 0);
    rlt::set_all(device, host.reset_mask, true);
    rlt::set_all(device, host.actions, (T)0);
    rlt::sample_initial_parameters(device, env, host.parameters, host.reset_mask, rng_cpu);
    rlt::sample_initial_state(device, env, host.parameters, host.states, host.reset_mask, rng_cpu);

    Tensors<DEVICE_GPU> gpu;
    gpu.allocate(device_gpu);
    rlt::copy(device, device_gpu, host.parameters, gpu.parameters);
    rlt::copy(device, device_gpu, host.states, gpu.states);
    rlt::copy(device, device_gpu, host.actions, gpu.actions);
    RNG_GPU rng_gpu;
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_gpu, rng_gpu, 0);

    for(TI step_i = 0; step_i < 10; step_i++){
        rlt::step(device, env, host.parameters, host.states, host.actions, host.next_states, rng_cpu);
        rlt::copy(device, device, host.next_states, host.states);
        rlt::step(device_gpu, env, gpu.parameters, gpu.states, gpu.actions, gpu.next_states, rng_gpu);
        rlt::copy(device_gpu, device_gpu, gpu.next_states, gpu.states);
    }
    cudaDeviceSynchronize();
    Tensors<DEVICE> downloaded;
    downloaded.allocate(device);
    rlt::copy(device_gpu, device, gpu.states, downloaded.states);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename ENVIRONMENT::State cpu_state = rlt::get(device, host.states, instance_i);
        typename ENVIRONMENT::State gpu_state = rlt::get(device, downloaded.states, instance_i);
        EXPECT_NEAR(cpu_state.theta, gpu_state.theta, 1e-4);
        EXPECT_NEAR(cpu_state.theta_dot, gpu_state.theta_dot, 1e-4);
    }
    host.deallocate(device);
    gpu.deallocate(device_gpu);
    downloaded.deallocate(device);
    rlt::free(device, env);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
