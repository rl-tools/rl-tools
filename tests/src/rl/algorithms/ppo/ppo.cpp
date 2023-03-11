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


template <typename T>
struct AdamParameters: lic::nn::optimizers::adam::DefaultParametersTorch<T>{
    static constexpr T ALPHA = 0.0001;
};



TEST(LAYER_IN_C_RL_ALGORITHMS_PPO, TEST){
    using DEVSPEC = lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, lic::devices::logging::CPU_TENSORBOARD>;
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
    using OPTIMIZER_PARAMETERS = AdamParameters<T>;
    using OPTIMIZER = lic::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    using ACTOR_SPEC = lic::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = lic::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;
    using CRITIC_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, 1, 3, 64, lic::nn::activation_functions::ActivationFunction::TANH, lic::nn::activation_functions::IDENTITY, BATCH_SIZE>;
    using CRITIC_SPEC = lic::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
    using CRITIC_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<CRITIC_SPEC>;

    using PPO_SPEC = lic::rl::algorithms::ppo::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, CRITIC_TYPE, lic::rl::algorithms::ppo::DefaultParameters<T, TI>>;
    using PPO_TYPE = lic::rl::algorithms::PPO<PPO_SPEC>;

    constexpr TI ON_POLICY_RUNNER_STEP_LIMIT = 200;
    constexpr TI N_ENVIRONMENTS = 1;
    using ON_POLICY_RUNNER_SPEC = lic::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ON_POLICY_RUNNER_STEP_LIMIT>;
    using ON_POLICY_RUNNER_TYPE = lic::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
    constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 10000;
    using ON_POLICY_RUNNER_BUFFER_SPEC = lic::rl::components::on_policy_runner::BufferSpecification<ON_POLICY_RUNNER_SPEC, ON_POLICY_RUNNER_STEPS_PER_ENV>;
    using ON_POLICY_RUNNER_BUFFER_TYPE = lic::rl::components::on_policy_runner::Buffer<ON_POLICY_RUNNER_BUFFER_SPEC>;

    using ACTOR_EVAL_BUFFERS = typename ACTOR_TYPE::template Buffers<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;
    using ACTOR_BUFFERS = typename ACTOR_TYPE::template BuffersForwardBackward<BATCH_SIZE>;
    using CRITIC_BUFFERS = typename CRITIC_TYPE::template BuffersForwardBackward<BATCH_SIZE>;

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    OPTIMIZER optimizer;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    PPO_TYPE ppo;
    ON_POLICY_RUNNER_TYPE on_policy_runner;
    ON_POLICY_RUNNER_BUFFER_TYPE on_policy_runner_buffer;
    ACTOR_EVAL_BUFFERS actor_eval_buffers;
    ACTOR_BUFFERS actor_buffers;
    CRITIC_BUFFERS critic_buffers;

    lic::malloc(device, ppo);
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

    for(TI ppo_step_i = 0; ppo_step_i < 1000; ppo_step_i++){
        device.logger->step = on_policy_runner.step;
        std::cout << "PPO step: " << ppo_step_i << std::endl;
        for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
            T action_log_std = lic::get(ppo.actor.action_log_std.parameters, 0, action_i);
            std::stringstream topic;
            topic << "actor/action_std/" << action_i;
            lic::add_scalar(device, device.logger, topic.str(), lic::math::exp(DEVICE::SPEC::MATH(), action_log_std));
        }
        lic::collect(device, on_policy_runner_buffer, on_policy_runner, ppo.actor, actor_eval_buffers, rng);
//        lic::print(device, on_policy_runner_buffer.data);
        lic::estimate_generalized_advantages(device, ppo, on_policy_runner_buffer);
//        lic::print(device, on_policy_runner_buffer.data);
        lic::train(device, ppo, on_policy_runner_buffer, optimizer, actor_buffers, critic_buffers, rng);
//        lic::print(device, on_policy_runner_buffer.data);
    }



}