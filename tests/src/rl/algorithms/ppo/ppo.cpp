// ------------ Groups 1 ------------
#include <layer_in_c/operations/cpu_tensorboard/group_1.h>
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/operations/cpu_mkl/group_1.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/operations/cpu_accelerate/group_1.h>
#else
#include <layer_in_c/operations/cpu/group_1.h>
#endif
#endif
// ------------ Groups 2 ------------
#include <layer_in_c/operations/cpu_tensorboard/group_2.h>
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/operations/cpu_mkl/group_2.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/operations/cpu_accelerate/group_2.h>
#else
#include <layer_in_c/operations/cpu/group_2.h>
#endif
#endif
// ------------ Groups 3 ------------
#include <layer_in_c/operations/cpu_tensorboard/group_3.h>
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/operations/cpu_mkl/group_3.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/operations/cpu_accelerate/group_3.h>
#else
#include <layer_in_c/operations/cpu/group_3.h>
#endif
#endif

#include <layer_in_c/rl/environments/pendulum/operations_cpu.h>


#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/nn/operations_cpu_mkl.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/nn/operations_cpu_accelerate.h>
#endif
#endif
#include <layer_in_c/nn_models/operations_cpu.h>


#include <layer_in_c/rl/components/on_policy_runner/operations_generic.h>
#include <layer_in_c/rl/algorithms/ppo/operations_generic.h>


namespace lic = layer_in_c;


#include <gtest/gtest.h>



namespace TEST_DEFINITIONS{
    using LOGGER = lic::devices::logging::CPU_TENSORBOARD;
//    using LOGGER = lic::devices::logging::CPU;
    using DEVSPEC = lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, LOGGER>;
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
    using DEVICE = lic::devices::CPU_MKL<DEVSPEC>;
#else
    #ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
    using DEVICE = lic::devices::CPU_ACCELERATE<DEVSPEC>;
#else
    using DEVICE = lic::devices::CPU<DEVSPEC>;
#endif
#endif
    using T = float;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = lic::rl::environments::pendulum::Specification<T, TI>;
    using ENVIRONMENT = lic::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
    constexpr TI BATCH_SIZE = 64;
    using ACTOR_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::ActivationFunction::TANH, lic::nn::activation_functions::IDENTITY, BATCH_SIZE>;

    struct OPTIMIZER_PARAMETERS: lic::nn::optimizers::adam::DefaultParametersTorch<T>{
        static constexpr T ALPHA = 0.0001;
    };
    using OPTIMIZER = lic::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    using ACTOR_SPEC = lic::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = lic::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;
    using CRITIC_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, 1, 3, 64, lic::nn::activation_functions::ActivationFunction::TANH, lic::nn::activation_functions::IDENTITY, BATCH_SIZE>;
    using CRITIC_SPEC = lic::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
    using CRITIC_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<CRITIC_SPEC>;

    struct PPO_PARAMETERS: lic::rl::algorithms::ppo::DefaultParameters<T, TI>{
        static constexpr TI N_EPOCHS = 1;
    };
    using PPO_SPEC = lic::rl::algorithms::ppo::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, CRITIC_TYPE, PPO_PARAMETERS>;
    using PPO_TYPE = lic::rl::algorithms::PPO<PPO_SPEC>;
    using PPO_BUFFERS_TYPE = lic::rl::algorithms::ppo::Buffers<PPO_SPEC>;

    constexpr TI ON_POLICY_RUNNER_STEP_LIMIT = 200;
    constexpr TI N_ENVIRONMENTS = 10;
    using ON_POLICY_RUNNER_SPEC = lic::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ON_POLICY_RUNNER_STEP_LIMIT>;
    using ON_POLICY_RUNNER_TYPE = lic::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
    constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 200;
    using ON_POLICY_RUNNER_BUFFER_SPEC = lic::rl::components::on_policy_runner::BufferSpecification<ON_POLICY_RUNNER_SPEC, ON_POLICY_RUNNER_STEPS_PER_ENV>;
    using ON_POLICY_RUNNER_BUFFER_TYPE = lic::rl::components::on_policy_runner::Buffer<ON_POLICY_RUNNER_BUFFER_SPEC>;

    using ACTOR_EVAL_BUFFERS = typename ACTOR_TYPE::template Buffers<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;
    using ACTOR_BUFFERS = typename ACTOR_TYPE::template BuffersForwardBackward<BATCH_SIZE>;
    using CRITIC_BUFFERS = typename CRITIC_TYPE::template BuffersForwardBackward<BATCH_SIZE>;
}

TEST(LAYER_IN_C_RL_ALGORITHMS_PPO, TEST){
    using namespace TEST_DEFINITIONS;

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    OPTIMIZER optimizer;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    PPO_TYPE ppo;
    PPO_BUFFERS_TYPE ppo_buffers;
    ON_POLICY_RUNNER_TYPE on_policy_runner;
    ON_POLICY_RUNNER_BUFFER_TYPE on_policy_runner_buffer;
    ACTOR_EVAL_BUFFERS actor_eval_buffers;
    ACTOR_BUFFERS actor_buffers;
    CRITIC_BUFFERS critic_buffers;

    lic::malloc(device, ppo);
    lic::malloc(device, ppo_buffers);
    lic::malloc(device, on_policy_runner_buffer);
    lic::malloc(device, on_policy_runner);
    lic::malloc(device, actor_eval_buffers);
    lic::malloc(device, actor_buffers);
    lic::malloc(device, critic_buffers);

    ENVIRONMENT envs[N_ENVIRONMENTS];
    lic::init(device, on_policy_runner, envs, rng);
    lic::init(device, ppo, optimizer, rng);
    device.logger = &logger;
    lic::construct(device, device.logger);
    auto training_start = std::chrono::high_resolution_clock::now();
    for(TI ppo_step_i = 0; ppo_step_i < 100000; ppo_step_i++) {
        device.logger->step = on_policy_runner.step;

        if(ppo_step_i % 10 == 0){
            std::chrono::duration<T> training_elapsed = std::chrono::high_resolution_clock::now() - training_start;
            std::cout << "PPO step: " << ppo_step_i << " elapsed: " << training_elapsed.count() << "s" << std::endl;
        }
        for (TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++) {
            T action_log_std = lic::get(ppo.actor.action_log_std.parameters, 0, action_i);
            std::stringstream topic;
            topic << "actor/action_std/" << action_i;
            lic::add_scalar(device, device.logger, topic.str(), lic::math::exp(DEVICE::SPEC::MATH(), action_log_std));
        }
        auto start = std::chrono::high_resolution_clock::now();
        {
            auto start = std::chrono::high_resolution_clock::now();
            lic::collect(device, on_policy_runner_buffer, on_policy_runner, ppo.actor, actor_eval_buffers, rng);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
//            std::cout << "Rollout: " << elapsed.count() << " s" << std::endl;
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            lic::estimate_generalized_advantages(device, ppo, on_policy_runner_buffer);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
//            std::cout << "GAE: " << elapsed.count() << " s" << std::endl;
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            lic::train(device, ppo, on_policy_runner_buffer, optimizer, ppo_buffers, actor_buffers, critic_buffers, rng);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
//            std::cout << "Train: " << elapsed.count() << " s" << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> elapsed = end - start;
//        std::cout << "Total: " << elapsed.count() << " s" << std::endl;
    }


    lic::free(device, ppo);
    lic::free(device, ppo_buffers);
    lic::free(device, on_policy_runner_buffer);
    lic::free(device, on_policy_runner);
    lic::free(device, actor_eval_buffers);
    lic::free(device, actor_buffers);
    lic::free(device, critic_buffers);

}