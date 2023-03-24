#define LAYER_IN_C_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <layer_in_c/operations/cpu_mux.h>
#include <layer_in_c/nn/operations_cpu_mux.h>
#include <layer_in_c/nn_models/operations_cpu.h>
#include <layer_in_c/nn_models/persist.h>
namespace lic = layer_in_c;
#include "../parameters_ppo.h"
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu_mkl.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
#include <layer_in_c/rl/algorithms/ppo/operations_generic.h>
#include <layer_in_c/rl/utils/evaluation.h>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>


namespace parameters = parameters_0;

using LOGGER = lic::devices::logging::CPU_TENSORBOARD;

using DEV_SPEC_SUPER = lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, LOGGER>;
using TI = typename DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints{
    struct HINTS: lic::rl::components::on_policy_runner::ExecutionHints<TI, 16>{};
}
struct DEV_SPEC: DEV_SPEC_SUPER{
    using EXECUTION_HINTS = execution_hints::HINTS;
};
using DEVICE = DEVICE_FACTORY<DEV_SPEC>;


using DEVICE = DEVICE_FACTORY<DEV_SPEC>;
using DEVICE_CUDA = DEVICE_FACTORY_GPU<lic::devices::DefaultCUDASpecification>;
using T = float;
using TI = typename DEVICE::index_t;
using envp = parameters::environment<double, TI>;
using rlp = parameters::rl<T, TI, envp::ENVIRONMENT>;
using STATE = envp::ENVIRONMENT::State;

TEST(LAYER_IN_C_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_MULTI_CORE_SPAWNING_CUDA){
    constexpr TI NUM_ROLLOUT_STEPS = 760;
    constexpr TI NUM_STEPS_PER_ENVIRONMENT = 64;
    constexpr TI NUM_ENVIRONMENTS = 64;
    constexpr TI NUM_THREADS = 16;
    using ACTOR_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, envp::ENVIRONMENT::OBSERVATION_DIM, envp::ENVIRONMENT::ACTION_DIM, 3, 256, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::IDENTITY>;
    using ACTOR_SPEC = lic::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = lic::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;

    DEVICE device;
    DEVICE_CUDA device_cuda;
    lic::init(device_cuda);
    STATE states[NUM_ENVIRONMENTS], next_states[NUM_ENVIRONMENTS];
    envp::ENVIRONMENT envs[NUM_ENVIRONMENTS];
    ACTOR_TYPE actor_cpu, actor_gpu;
    ACTOR_TYPE::Buffers<NUM_ENVIRONMENTS> actor_buffers;
    lic::Matrix<lic::matrix::Specification<T, TI, NUM_ENVIRONMENTS, envp::ENVIRONMENT::ACTION_DIM>> actions, actions_gpu;
    lic::Matrix<lic::matrix::Specification<T, TI, NUM_ENVIRONMENTS, envp::ENVIRONMENT::OBSERVATION_DIM>> observations, observations_gpu;
    auto proto_rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    decltype(proto_rng) rngs[NUM_THREADS];

    lic::malloc(device, actions);
    lic::malloc(device, observations);
    lic::malloc(device, actor_cpu);
    lic::malloc(device_cuda, actions_gpu);
    lic::malloc(device_cuda, observations_gpu);
    lic::malloc(device_cuda, actor_gpu);
    lic::malloc(device_cuda, actor_buffers);
    for(TI env_i = 0; env_i < NUM_ENVIRONMENTS; env_i++){
        lic::malloc(device, envs[env_i]);
    }

    lic::randn(device, actions, proto_rng);
    lic::init_weights(device, actor_cpu, proto_rng);
    lic::copy(device_cuda, device, actor_gpu, actor_cpu);

    for(TI env_i = 0; env_i < NUM_ENVIRONMENTS; env_i++){
        lic::sample_initial_state(device, envs[env_i], states[env_i], proto_rng);
        auto observation = lic::view(device, observations, lic::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
        lic::observe(device,envs[env_i], states[env_i], observation);
    }


    auto start = std::chrono::high_resolution_clock::now();
    for(TI rollout_step_i = 0; rollout_step_i < NUM_ROLLOUT_STEPS; rollout_step_i++){
        std::cout << "Rollout step " << rollout_step_i << std::endl;
        for(TI step_i = 0; step_i < NUM_STEPS_PER_ENVIRONMENT; step_i++) {
            std::thread threads[NUM_THREADS];
            for(TI thread_i = 0; thread_i < NUM_THREADS; thread_i++){
                threads[thread_i] = std::thread([&device, &rngs, &actions, &observations, &envs, &states, &next_states, thread_i, step_i](){
                    for(TI env_i = thread_i; env_i < NUM_ENVIRONMENTS; env_i += NUM_THREADS){
                        auto rng = rngs[thread_i];
                        auto& env = envs[env_i];
                        auto& state = states[env_i];
                        auto& next_state = next_states[env_i];
                        auto action = lic::view(device, actions, lic::matrix::ViewSpec<1, envp::ENVIRONMENT::ACTION_DIM>(), env_i, 0);
                        auto observation = lic::view(device, observations, lic::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
                        lic::step(device, env, state, action, next_state);
                        if(step_i % 1000 == 0 || lic::terminated(device, env, next_state, rng)) {
                            lic::sample_initial_state(device, env, state, rng);
                        }
                        else{
                            next_state = state;
                        }
                        lic::observe(device, env, next_state, observation);
                    }
                });
            }
            for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
                threads[env_i].join();
            }
            lic::copy(device_cuda, device, observations_gpu, observations);
            lic::evaluate(device_cuda, actor_gpu, observations_gpu, actions_gpu, actor_buffers);
            lic::copy(device, device_cuda, actions, actions_gpu);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto steps_per_second = NUM_STEPS_PER_ENVIRONMENT * NUM_ENVIRONMENTS * NUM_ROLLOUT_STEPS * 1000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    std::cout << "Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps)" << std::endl;
}
