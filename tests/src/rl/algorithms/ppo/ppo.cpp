#include <layer_in_c/operations/cpu_mkl.h>

#include <layer_in_c/rl/environments/pendulum/operations_cpu.h>


#include <layer_in_c/nn/operations_cpu_mkl.h>
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
    using DEVICE = lic::devices::DefaultCPU_MKL;
    using T = float;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = lic::rl::environments::pendulum::Specification<T, TI>;
    using ENVIRONMENT = lic::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
    constexpr TI BATCH_SIZE = 64;
    using ACTOR_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::ActivationFunction::TANH, lic::nn::activation_functions::TANH, BATCH_SIZE>;
    using ACTOR_SPEC = lic::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC, AdamParameters<T>>;
    using ACTOR_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_SPEC>;
    using CRITIC_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, 1, 3, 64, lic::nn::activation_functions::ActivationFunction::TANH, lic::nn::activation_functions::IDENTITY, BATCH_SIZE>;
    using CRITIC_SPEC = lic::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC, AdamParameters<T>>;
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

    DEVICE device;
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
    lic::init(device, on_policy_runner, envs);
    lic::init(device, ppo, rng);

    for(TI ppo_step_i = 0; ppo_step_i < 1000; ppo_step_i++){
        std::cout << "PPO step: " << ppo_step_i << std::endl;
        lic::collect(device, on_policy_runner_buffer, on_policy_runner, ppo, actor_eval_buffers, rng);
//        lic::print(device, on_policy_runner_buffer.data);
        lic::estimate_generalized_advantages(device, ppo, on_policy_runner_buffer);
//        lic::print(device, on_policy_runner_buffer.data);
        lic::train(device, ppo, on_policy_runner_buffer, actor_buffers, critic_buffers, rng);
//        lic::print(device, on_policy_runner_buffer.data);
    }



}