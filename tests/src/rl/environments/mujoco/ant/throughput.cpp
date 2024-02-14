#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/operations_cpu.h>
#include <rl_tools/nn_models/persist.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
#include "../../../../../src/rl/environments/mujoco/ant/ppo/parameters.h"
#ifdef RL_TOOLS_BACKEND_ENABLE_MKL
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mkl.h>
#else
#ifdef RL_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
#include <rl_tools/rl/algorithms/ppo/operations_generic.h>
#include <rl_tools/rl/utils/evaluation.h>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>


namespace parameters = parameters_0;

using LOGGER = rlt::devices::logging::CPU_TENSORBOARD<>;

using DEV_SPEC_SUPER = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
using TI = typename rlt::devices::DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints{
    struct HINTS: rlt::rl::components::on_policy_runner::ExecutionHints<TI, 16>{};
}
struct DEV_SPEC: DEV_SPEC_SUPER{
    using EXECUTION_HINTS = execution_hints::HINTS;
};
using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;


using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
using T = float;
using TI = typename DEVICE::index_t;
using envp = parameters::environment<double, TI>;
using rlp = parameters::rl<T, TI, envp::ENVIRONMENT>;
using STATE = envp::ENVIRONMENT::State;

TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_SINGLE_CORE){
    constexpr TI NUM_STEPS = 10000;

    DEVICE device;
    envp::ENVIRONMENT env;
    STATE state, next_state;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, envp::ENVIRONMENT::ACTION_DIM>> action;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    rlt::malloc(device, env);
    rlt::malloc(device, action);

    rlt::sample_initial_state(device, env, state, rng);
    auto start = std::chrono::high_resolution_clock::now();
    for(TI step_i = 0; step_i < NUM_STEPS; step_i++){
        rlt::step(device, env, state, action, next_state, rng);
        if(step_i % 1000 == 0 || rlt::terminated(device, env, next_state, rng)) {
            rlt::sample_initial_state(device, env, state, rng);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto steps_per_second = NUM_STEPS * 1000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    std::cout << "Single Core Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps)" << std::endl;
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_MULTI_CORE_INDEPENDENT){
    constexpr TI NUM_STEPS_PER_THREAD = 1000;
    constexpr TI NUM_THREADS = 16;

    DEVICE device;
    envp::ENVIRONMENT envs[NUM_THREADS];
    std::thread threads[NUM_THREADS];
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, NUM_THREADS, envp::ENVIRONMENT::ACTION_DIM>> actions;
    auto proto_rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    decltype(proto_rng) rngs[NUM_THREADS];

    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        rlt::malloc(device, envs[env_i]);
    }
    rlt::malloc(device, actions);


    auto start = std::chrono::high_resolution_clock::now();
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i] = std::thread([&device, &rngs, &actions, &envs, env_i, NUM_STEPS_PER_THREAD](){
            STATE state, next_state;
            auto rng = rngs[env_i];
            auto& env = envs[env_i];
            auto action = rlt::view(device, actions, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::ACTION_DIM>(), env_i, 0);
            rlt::randn(device, action, rng);
            rlt::sample_initial_state(device, env, state, rng);
            for(TI step_i = 0; step_i < NUM_STEPS_PER_THREAD; step_i++){
                rlt::step(device, env, state, action, next_state, rng);
                if(step_i % 1000 == 0 || rlt::terminated(device, env, next_state, rng)) {
                    rlt::sample_initial_state(device, env, state, rng);
                }
            }
        });
    }
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto steps_per_second = NUM_STEPS_PER_THREAD * NUM_THREADS * 1000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    std::cout << "Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps)" << std::endl;
}


template <TI NUM_THREADS>
class TwoWayBarrier {
public:
    TwoWayBarrier() : count(0), waiting(0), waiting2(0) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        ++count;
        ++waiting;
        if (count < NUM_THREADS) {
            cond.wait(lock, [this] {
                return this->count == NUM_THREADS;
            });
        } else {
            cond.notify_all();
            waiting2 = 0;
        }
        --waiting;
        ++waiting2;
        if (waiting > 0) {
            cond.wait(lock, [this] {
                return this->waiting == 0;
            });
        } else {
            cond.notify_all();
            count = 0;
        }
        --waiting2;
        if (waiting2 > 0) {
            cond.wait(lock, [this] {
                return this->waiting2 == 0;
            });
        } else {
            cond.notify_all();
        }
    }

private:
    int count;
    int waiting;
    int waiting2;
    std::condition_variable cond;
    std::mutex mutex;
};
//
TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_MULTI_CORE_LOCKSTEP){
    constexpr TI NUM_STEPS_PER_THREAD = 1000;
    constexpr TI NUM_THREADS = 16;

    DEVICE device;
    envp::ENVIRONMENT envs[NUM_THREADS];
    std::thread threads[NUM_THREADS];
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, NUM_THREADS, envp::ENVIRONMENT::ACTION_DIM>> actions;
    auto proto_rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    decltype(proto_rng) rngs[NUM_THREADS];

    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        rlt::malloc(device, envs[env_i]);
        rngs[env_i] = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), env_i);
    }
    rlt::malloc(device, actions);

    TwoWayBarrier<NUM_THREADS> barrier;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> order;
    std::mutex order_mutex;
    T step_time[NUM_THREADS] = {0};
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i] = std::thread([&device, &rngs, &actions, &envs, &barrier, &order, &order_mutex, &step_time, env_i, NUM_STEPS_PER_THREAD](){
            STATE state, next_state;
            auto rng = rngs[env_i];
            auto& env = envs[env_i];
            auto action = rlt::view(device, actions, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::ACTION_DIM>(), env_i, 0);
            rlt::randn(device, action, rng);
            rlt::sample_initial_state(device, env, state, rng);
            for(TI step_i = 0; step_i < NUM_STEPS_PER_THREAD; step_i++){
                rlt::randn(device, action, rng);
                rlt::step(device, env, state, action, next_state, rng);
                if(step_i % 1000 == 0 || rlt::terminated(device, env, next_state, rng)) {
                    rlt::sample_initial_state(device, env, state, rng);
                }
                {
                    std::lock_guard<std::mutex> lock(order_mutex);
                    order.push_back(env_i);
                }
                barrier.wait();
            }
        });
    }
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto steps_per_second = NUM_STEPS_PER_THREAD * NUM_THREADS * 1000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    std::cout << "Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps)" << std::endl;


    for(TI i = 0; i < order.size()/NUM_THREADS; i++){
        bool found[NUM_THREADS] = {false};
        for(TI j = 0; j < NUM_THREADS; j++){
            found[order[i*NUM_THREADS + j]] = true;
        }
        for(TI j = 0; j < NUM_THREADS; j++){
            ASSERT_TRUE(found[j]);
        }
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_MULTI_CORE_SPAWNING){
    constexpr TI NUM_ROLLOUT_STEPS = 100;
    constexpr TI NUM_STEPS_PER_ENVIRONMENT = 64;
    constexpr TI NUM_ENVIRONMENTS = 64;
    constexpr TI NUM_THREADS = 16;
    using ACTOR_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<T, TI, envp::ENVIRONMENT::OBSERVATION_DIM, envp::ENVIRONMENT::ACTION_DIM, 3, 256, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR_SPEC = rlt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;

    DEVICE device;
    STATE states[NUM_ENVIRONMENTS], next_states[NUM_ENVIRONMENTS];
    envp::ENVIRONMENT envs[NUM_ENVIRONMENTS];
    ACTOR_TYPE actor;
    ACTOR_TYPE::Buffer<NUM_ENVIRONMENTS> actor_buffers;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, NUM_ENVIRONMENTS, envp::ENVIRONMENT::ACTION_DIM>> actions;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, NUM_ENVIRONMENTS, envp::ENVIRONMENT::OBSERVATION_DIM>> observations;
    auto proto_rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    decltype(proto_rng) rngs[NUM_THREADS];

    rlt::malloc(device, actions);
    rlt::malloc(device, observations);
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffers);
    for(TI env_i = 0; env_i < NUM_ENVIRONMENTS; env_i++){
        rlt::malloc(device, envs[env_i]);
    }

    rlt::randn(device, actions, proto_rng);
    rlt::init_weights(device, actor, proto_rng);
    for(TI env_i = 0; env_i < NUM_ENVIRONMENTS; env_i++){
        rlt::sample_initial_state(device, envs[env_i], states[env_i], proto_rng);
        auto observation = rlt::view(device, observations, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
        rlt::observe(device,envs[env_i], states[env_i], observation, rngs[0]);
    }


    auto start = std::chrono::high_resolution_clock::now();
    for(TI rollout_step_i = 0; rollout_step_i < NUM_ROLLOUT_STEPS; rollout_step_i++){
        std::cout << "Rollout step " << rollout_step_i << std::endl;
        for(TI step_i = 0; step_i < NUM_STEPS_PER_ENVIRONMENT; step_i++) {
            std::thread threads[NUM_THREADS];
            for(TI thread_i = 0; thread_i < NUM_THREADS; thread_i++){
                threads[thread_i] = std::thread([&device, &rngs, &actions, &observations, &envs, &states, &next_states, thread_i, step_i, NUM_ENVIRONMENTS, NUM_THREADS](){
                    for(TI env_i = thread_i; env_i < NUM_ENVIRONMENTS; env_i += NUM_THREADS){
                        auto rng = rngs[thread_i];
                        auto& env = envs[env_i];
                        auto& state = states[env_i];
                        auto& next_state = next_states[env_i];
                        auto action = rlt::view(device, actions, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::ACTION_DIM>(), env_i, 0);
                        auto observation = rlt::view(device, observations, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
                        rlt::step(device, env, state, action, next_state, rng);
                        if(step_i % 1000 == 0 || rlt::terminated(device, env, next_state, rng)) {
                            rlt::sample_initial_state(device, env, state, rng);
                        }
                        else{
                            next_state = state;
                        }
                        rlt::observe(device, env, next_state, observation, rng);
                    }
                });
            }
            for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
                threads[env_i].join();
            }
            rlt::evaluate(device, actor, observations, actions, actor_buffers);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto steps_per_second = NUM_STEPS_PER_ENVIRONMENT * NUM_ENVIRONMENTS * NUM_ROLLOUT_STEPS * 1000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    std::cout << "Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps)" << std::endl;
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_MULTI_CORE_INDEPENDENT_FORWARD_PASS){
    constexpr TI NUM_STEPS_PER_THREAD = 1000;
    constexpr TI NUM_THREADS = 16;
    using ACTOR_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<T, TI, envp::ENVIRONMENT::OBSERVATION_DIM, envp::ENVIRONMENT::ACTION_DIM, 3, 256, rlt::nn::activation_functions::ActivationFunction::TANH, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR_SPEC = rlt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;


    DEVICE device;
    envp::ENVIRONMENT envs[NUM_THREADS];
    std::thread threads[NUM_THREADS];
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, NUM_THREADS, envp::ENVIRONMENT::ACTION_DIM>> actions;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, NUM_THREADS, envp::ENVIRONMENT::OBSERVATION_DIM>> observations;
    ACTOR_TYPE actors[NUM_THREADS];
    ACTOR_TYPE::Buffer<1> actor_buffers[NUM_THREADS];
    auto proto_rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    decltype(proto_rng) rngs[NUM_THREADS];

    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        rlt::malloc(device, envs[env_i]);
        rlt::malloc(device, actors[env_i]);
        rlt::malloc(device, actor_buffers[env_i]);
        rlt::init_weights(device, actors[env_i], proto_rng);
    }
    rlt::malloc(device, actions);
    rlt::malloc(device, observations);



    auto start = std::chrono::high_resolution_clock::now();
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i] = std::thread([&device, &actors, &actor_buffers, &rngs, &observations, &actions, &envs, env_i, NUM_STEPS_PER_THREAD](){
            STATE state, next_state;
            auto rng = rngs[env_i];
            auto& env = envs[env_i];
            auto& actor = actors[env_i];
            auto& actor_buffer = actor_buffers[env_i];
            auto action = rlt::view(device, actions, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::ACTION_DIM>(), env_i, 0);
            auto observation = rlt::view(device, observations, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
            rlt::sample_initial_state(device, env, state, rng);
            for(TI step_i = 0; step_i < NUM_STEPS_PER_THREAD; step_i++){
                rlt::observe(device, env, state, observation, rng);
                rlt::evaluate(device, actor, observation, action, actor_buffer);
                rlt::step(device, env, state, action, next_state, rng);
                if(step_i % 1000 == 0 || rlt::terminated(device, env, next_state, rng)) {
                    rlt::sample_initial_state(device, env, state, rng);
                }
            }
        });
    }
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto steps_per_second = NUM_STEPS_PER_THREAD * NUM_THREADS * 1000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    std::cout << "Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps)" << std::endl;
}
//
TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_MULTI_CORE_COLLECTIVE_FORWARD_PASS_TAKE_AWAY){
    constexpr TI NUM_STEPS_PER_ENVIRONMENT = 10000;
    constexpr TI NUM_THREADS = 16;
    constexpr TI NUM_ENVIRONMENTS = 64;
    using ACTOR_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<T, TI, envp::ENVIRONMENT::OBSERVATION_DIM, envp::ENVIRONMENT::ACTION_DIM, 3, 256, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR_SPEC = rlt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;


    DEVICE device;
    envp::ENVIRONMENT envs[NUM_ENVIRONMENTS];
    STATE states[NUM_ENVIRONMENTS], next_states[NUM_ENVIRONMENTS];
    std::thread threads[NUM_THREADS];
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, NUM_ENVIRONMENTS, envp::ENVIRONMENT::ACTION_DIM>> actions;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, NUM_ENVIRONMENTS, envp::ENVIRONMENT::OBSERVATION_DIM>> observations;
    ACTOR_TYPE actor;
    ACTOR_TYPE::Buffer<NUM_ENVIRONMENTS> actor_buffers;
    auto proto_rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    decltype(proto_rng) rngs[NUM_THREADS];
    TwoWayBarrier<NUM_THREADS> barrier_1, barrier_2;

    for(TI env_i = 0; env_i < NUM_ENVIRONMENTS; env_i++){
        rlt::malloc(device, envs[env_i]);
    }
    rlt::malloc(device, actions);
    rlt::malloc(device, observations);
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffers);

    rlt::init_weights(device, actor, proto_rng);

    std::atomic<unsigned long> barrier_1_wait_time = 0, barrier_2_wait_time = 0, evaluation_time = 0;
    TI next_env = 0;
    std::mutex next_env_lock;

    auto start = std::chrono::high_resolution_clock::now();
    for (TI env_i = 0; env_i < NUM_ENVIRONMENTS; env_i++) {
        rlt::sample_initial_state(device, envs[env_i], states[env_i], proto_rng);
        auto observation = rlt::view(device, observations, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
        rlt::observe(device, envs[env_i], states[env_i], observation, rngs[0]);
    }
    for(TI thread_i = 0; thread_i < NUM_THREADS; thread_i++){
        threads[thread_i] = std::thread([&device, &states, &next_states, &actor_buffers, &next_env_lock, &next_env, &barrier_1, &evaluation_time, &barrier_1_wait_time, &barrier_2_wait_time, &barrier_2, &actor, &rngs, &observations, &actions, &envs, thread_i, NUM_STEPS_PER_ENVIRONMENT, NUM_ENVIRONMENTS](){
            auto rng = rngs[thread_i];
            for(TI step_i = 0; step_i < NUM_STEPS_PER_ENVIRONMENT; step_i++){
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    barrier_1.wait();
                    auto end = std::chrono::high_resolution_clock::now();
                    barrier_1_wait_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                }
                if(thread_i == 0){
                    next_env = 0;
                    {
                        auto start = std::chrono::high_resolution_clock::now();
                        rlt::evaluate(device, actor, observations, actions, actor_buffers);
                        auto end = std::chrono::high_resolution_clock::now();
                        evaluation_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                    }
                }
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    barrier_2.wait();
                    auto end = std::chrono::high_resolution_clock::now();
                    barrier_2_wait_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                }
                TI current_env = 0;
                while(current_env < NUM_ENVIRONMENTS){
                    {
                        std::lock_guard<std::mutex> lock(next_env_lock);
                        current_env = next_env;
                        if(next_env < NUM_ENVIRONMENTS){
//                            std::cout << "Thread " << thread_i << " is working on environment " << next_env << std::endl;
                            next_env++;
                        }
                        else{
                            break;
                        }
                    }
                    auto action = rlt::view(device, actions, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::ACTION_DIM>(), current_env, 0);
                    rlt::step(device, envs[current_env], states[current_env], action, next_states[current_env], rng);
                    if(step_i % 1000 == 0 || rlt::terminated(device, envs[current_env], next_states[current_env], rng)) {
                        rlt::sample_initial_state(device, envs[current_env], states[current_env], rng);
                    }
                    auto observation = rlt::view(device, observations, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), current_env, 0);
                    rlt::observe(device, envs[current_env], states[current_env], observation, rng);
                }
            }
        });
    }
    for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
        threads[env_i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto steps_per_second = NUM_STEPS_PER_ENVIRONMENT * NUM_ENVIRONMENTS * 1000000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    auto barrier_1_wait_time_fraction = (double)barrier_1_wait_time / duration.count() / NUM_THREADS;
    auto barrier_2_wait_time_fraction = (double)barrier_2_wait_time / duration.count() / NUM_THREADS;
    auto evaluation_time_per_eval = (double)evaluation_time / (double)NUM_STEPS_PER_ENVIRONMENT;
    auto evaluation_time_fraction = (double)evaluation_time / duration.count();
    std::cout << "Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps) barrier_wait_time_fractions: " << barrier_1_wait_time_fraction << " : " << barrier_2_wait_time_fraction << " evaluation time per step: " << evaluation_time_per_eval << " fraction: " << evaluation_time_fraction << std::endl;
}
//
TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, THROUGHPUT_MULTI_CORE_COLLECTIVE_FORWARD_PASS_FIXED_ASSIGNMENT){
    constexpr TI NUM_FULL_STEPS = 100;
    constexpr TI NUM_STEPS_PER_ENVIRONMENT = 64;
    constexpr TI NUM_THREADS = 16;
    constexpr TI NUM_ENVIRONMENTS = 64;
    using ACTOR_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<T, TI, envp::ENVIRONMENT::OBSERVATION_DIM, envp::ENVIRONMENT::ACTION_DIM, 3, 256, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR_SPEC = rlt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;


    DEVICE device;
    envp::ENVIRONMENT envs[NUM_ENVIRONMENTS];
    STATE states[NUM_ENVIRONMENTS], next_states[NUM_ENVIRONMENTS];
    std::thread threads[NUM_THREADS];
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, NUM_ENVIRONMENTS, envp::ENVIRONMENT::ACTION_DIM>> actions;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, NUM_ENVIRONMENTS, envp::ENVIRONMENT::OBSERVATION_DIM>> observations;
    ACTOR_TYPE actor;
    ACTOR_TYPE::Buffer<NUM_ENVIRONMENTS> actor_buffers;
    auto proto_rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    decltype(proto_rng) rngs[NUM_THREADS];
    TwoWayBarrier<NUM_THREADS> barrier_1, barrier_2;

    for(TI env_i = 0; env_i < NUM_ENVIRONMENTS; env_i++){
        rlt::malloc(device, envs[env_i]);
    }
    rlt::malloc(device, actions);
    rlt::malloc(device, observations);
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffers);

    rlt::init_weights(device, actor, proto_rng);

    std::atomic<unsigned long> barrier_1_wait_time = 0, barrier_2_wait_time = 0, evaluation_time = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for(TI full_step_i = 0; full_step_i < NUM_FULL_STEPS; full_step_i++){
        for (TI env_i = 0; env_i < NUM_ENVIRONMENTS; env_i++) {
            rlt::sample_initial_state(device, envs[env_i], states[env_i], proto_rng);
            auto observation = rlt::view(device, observations, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
            rlt::observe(device, envs[env_i], states[env_i], observation, proto_rng);
        }
        for(TI thread_i = 0; thread_i < NUM_THREADS; thread_i++){
            threads[thread_i] = std::thread([&device, &states, &next_states, &actor_buffers, &barrier_1, &evaluation_time, &barrier_1_wait_time, &barrier_2_wait_time, &barrier_2, &actor, &rngs, &observations, &actions, &envs, thread_i, NUM_STEPS_PER_ENVIRONMENT, NUM_ENVIRONMENTS, NUM_THREADS](){
                auto rng = rngs[thread_i];
                for(TI step_i = 0; step_i < NUM_STEPS_PER_ENVIRONMENT; step_i++){
                    {
                        auto start = std::chrono::high_resolution_clock::now();
                        barrier_1.wait();
                        auto end = std::chrono::high_resolution_clock::now();
                        barrier_1_wait_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                    }
                    if(thread_i == 0){
                        {
                            auto start = std::chrono::high_resolution_clock::now();
                            rlt::evaluate(device, actor, observations, actions, actor_buffers);
                            auto end = std::chrono::high_resolution_clock::now();
                            evaluation_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                        }
                    }
                    {
                        auto start = std::chrono::high_resolution_clock::now();
                        barrier_2.wait();
                        auto end = std::chrono::high_resolution_clock::now();
                        barrier_2_wait_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                    }
                    for(TI env_i = thread_i; env_i < NUM_ENVIRONMENTS; env_i += NUM_THREADS){
                        auto action = rlt::view(device, actions, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::ACTION_DIM>(), env_i, 0);
                        auto observation = rlt::view(device, observations, rlt::matrix::ViewSpec<1, envp::ENVIRONMENT::OBSERVATION_DIM>(), env_i, 0);
                        rlt::step(device, envs[env_i], states[env_i], action, next_states[env_i], rng);
                        if(step_i % 1000 == 0 || rlt::terminated(device, envs[env_i], next_states[env_i], rng)) {
                            rlt::sample_initial_state(device, envs[env_i], states[env_i], rng);
                        }
                        rlt::observe(device, envs[env_i], states[env_i], observation, rng);
                    }
                }
            });
        }
        for(TI env_i = 0; env_i < NUM_THREADS; env_i++){
            threads[env_i].join();
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto steps_per_second = NUM_STEPS_PER_ENVIRONMENT * NUM_ENVIRONMENTS * NUM_FULL_STEPS * 1000000.0 / duration.count();
    auto frames_per_second = steps_per_second * envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP;
    auto barrier_1_wait_time_fraction = (double)barrier_1_wait_time / duration.count() / NUM_THREADS;
    auto barrier_2_wait_time_fraction = (double)barrier_2_wait_time / duration.count() / NUM_THREADS;
    auto evaluation_time_per_eval = (double)evaluation_time / (double)NUM_STEPS_PER_ENVIRONMENT;
    auto evaluation_time_fraction = (double)evaluation_time / duration.count();
    std::cout << "Throughput: " << steps_per_second << " steps/s (frameskip: " << envp::ENVIRONMENT::SPEC::PARAMETERS::FRAME_SKIP << " -> " << frames_per_second << " fps) barrier_wait_time_fractions: " << barrier_1_wait_time_fraction << " : " << barrier_2_wait_time_fraction << " evaluation time per step: " << evaluation_time_per_eval << " fraction: " << evaluation_time_fraction << std::endl;
}
