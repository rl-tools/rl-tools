#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
namespace bpt = backprop_tools;




#include <backprop_tools/rl/environments/operations_generic.h>
#include <backprop_tools/nn_models/operations_generic.h>
#include <backprop_tools/rl/operations_generic.h>


#include <backprop_tools/rl/utils/evaluation.h>

#include <filesystem>


#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
#include <backprop_tools/rl/environments/pendulum/ui.h>
#include <backprop_tools/rl/utils/evaluation_visual.h>
#endif


#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
#include "plot_policy_and_value_function.h"
#endif

#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_DISABLE_TENSORBOARD)
    using LOGGER = bpt::devices::logging::CPU_TENSORBOARD;
#else
    using LOGGER = bpt::devices::logging::CPU;
#endif

#if defined(BACKPROP_TOOLS_BACKEND_ENABLE_MKL) || defined(BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE) || defined(BACKPROP_TOOLS_BACKEND_ENABLE_OPENBLAS) && !defined(BACKPROP_TOOLS_BACKEND_DISABLE_BLAS)
using DEV_SPEC = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, LOGGER>;
using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
#else
using DEVICE = bpt::devices::DefaultCPU;
#endif


using T = float;
using TI = typename DEVICE::index_t;

typedef bpt::rl::environments::pendulum::Specification<T, TI, bpt::rl::environments::pendulum::DefaultParameters<T>> PENDULUM_SPEC;
typedef bpt::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
typedef bpt::rl::environments::pendulum::UI<T> UI;
#endif


struct TD3_PENDULUM_PARAMETERS: bpt::rl::algorithms::td3::DefaultParameters<T, TI>{
    constexpr static TI CRITIC_BATCH_SIZE = 100;
    constexpr static TI ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3_PENDULUM_PARAMETERS;

using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
using CRITIC_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;


using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<T>;
using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
using ACTOR_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
using ACTOR_NETWORK_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = bpt::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
using ACTOR_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
using CRITIC_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = backprop_tools::nn_models::mlp::InferenceSpecification<CRITIC_STRUCTURE_SPEC>;
using CRITIC_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = bpt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
using ACTOR_CRITIC_TYPE = bpt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;

#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_DEBUG
constexpr TI STEP_LIMIT = 1000;
#else
constexpr TI STEP_LIMIT = 10000;
#endif

constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
constexpr TI EPISODE_STEP_LIMIT = 200;
using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<
        T,
        TI,
        ENVIRONMENT,
        1,
        REPLAY_BUFFER_CAP,
        EPISODE_STEP_LIMIT,
        bpt::rl::components::off_policy_runner::DefaultParameters<T>
>;
using OFF_POLICY_RUNNER_TYPE = bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
OFF_POLICY_RUNNER_TYPE off_policy_runner;
ACTOR_CRITIC_TYPE actor_critic;
const T STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

void run(){
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
    UI ui;
#endif
    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;

    OPTIMIZER optimizer;

    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 4);
    bpt::malloc(device, actor_critic);
    bpt::init(device, actor_critic, optimizer, rng);

    bool ui = false;

    bpt::construct(device, device.logger);

    bpt::malloc(device, off_policy_runner);
    ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
    bpt::init(device, off_policy_runner, envs);

//    bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
    OFF_POLICY_RUNNER_TYPE::Batch<TD3_PARAMETERS::CRITIC_BATCH_SIZE> critic_batch;
    bpt::rl::algorithms::td3::CriticTrainingBuffers<ACTOR_CRITIC_TYPE::SPEC> critic_training_buffers;
    CRITIC_NETWORK_TYPE::BuffersForwardBackward<ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];
    bpt::malloc(device, critic_batch);
    bpt::malloc(device, critic_training_buffers);
    bpt::malloc(device, critic_buffers[0]);
    bpt::malloc(device, critic_buffers[1]);

    OFF_POLICY_RUNNER_TYPE::Batch<TD3_PARAMETERS::ACTOR_BATCH_SIZE> actor_batch;
    bpt::rl::algorithms::td3::ActorTrainingBuffers<ACTOR_CRITIC_TYPE::SPEC> actor_training_buffers;
    ACTOR_NETWORK_TYPE::Buffers<ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
    ACTOR_NETWORK_TYPE::Buffers<OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    bpt::malloc(device, actor_batch);
    bpt::malloc(device, actor_training_buffers);
    bpt::malloc(device, actor_buffers_eval);
    bpt::malloc(device, actor_buffers[0]);
    bpt::malloc(device, actor_buffers[1]);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_mean;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_std;
    bpt::malloc(device, observations_mean);
    bpt::malloc(device, observations_std);
    bpt::set_all(device, observations_mean, 0);
    bpt::set_all(device, observations_std, 1);


    auto start_time = std::chrono::high_resolution_clock::now();

    for(int step_i = 0; step_i < STEP_LIMIT; step_i+=OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS){
        bpt::set_step(device, device.logger, step_i);
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
        if(step_i % 20 == 0){
            plot_policy_and_value_function<T, ENVIRONMENT, decltype(actor_critic.actor), decltype(actor_critic.critic_1)>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
        }
#endif
        bpt::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
        bpt::set_state(ui, off_policy_runner.state);
#endif

        if(step_i > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = current_time - start_time;
                std::cout << "step_i: " << step_i << " " << elapsed_seconds.count() << "s" << std::endl;
            }

            for(int critic_i = 0; critic_i < 2; critic_i++){
                bpt::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                bpt::gather_batch(device, off_policy_runner, critic_batch, rng);
                bpt::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, optimizer, actor_buffers[critic_i], critic_buffers[critic_i], critic_training_buffers);
            }

//            T critic_1_loss = bpt::train_critic(device, actor_critic, actor_critic.critic_1, off_policy_runner.replay_buffer, rng);
//            bpt::train_critic(device, actor_critic, actor_critic.critic_2, off_policy_runner.replay_buffer, rng);
//            std::cout << "Critic 1 loss: " << critic_1_loss << std::endl;
            if(step_i % 2 == 0){
                {
                    bpt::gather_batch(device, off_policy_runner, actor_batch, rng);
                    bpt::train_actor(device, actor_critic, actor_batch, optimizer, actor_buffers[0], critic_buffers[0], actor_training_buffers);
                }

                bpt::update_critic_targets(device, actor_critic);
                bpt::update_actor_target(device, actor_critic);
            }
        }
#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_PENDULUM_DISABLE_EVALUATION
        if(step_i % 1000 == 0){
//            auto result = bpt::evaluate(device, envs[0], ui, actor_critic.actor, bpt::rl::utils::evaluation::Specification<1, EPISODE_STEP_LIMIT>(), rng, true);
            auto result = bpt::evaluate(device, envs[0], ui, actor_critic.actor, bpt::rl::utils::evaluation::Specification<10, EPISODE_STEP_LIMIT>(), observations_mean, observations_std, rng);
            std::cout << "Mean return: " << result.mean << std::endl;
            bpt::add_scalar(device, device.logger, "mean_return", result.mean);
//            if(step_i >= 6000){
//                ASSERT_GT(mean_return, -1000);
//            }
//            if(step_i >= 14000){
//                ASSERT_GT(mean_return, -400);
//            }

//#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
//            plot_policy_and_value_function<T, ENVIRONMENT, ACTOR_CRITIC_TYPE::ACTOR_NETWORK_TYPE, ACTOR_CRITIC_TYPE::CRITIC_NETWORK_TYPE>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
//#endif
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
            //            for(int evaluation_i = 0; evaluation_i < 10; evaluation_i++){
//                ENVIRONMENT::State initial_state;
//                bpt::sample_initial_state(env, initial_state, rng);
//                bpt::evaluate_visual<ENVIRONMENT, UI, ACTOR_CRITIC_TYPE::ACTOR_NETWORK_TYPE, EPISODE_STEP_LIMIT, 5>(env, ui, actor_critic.actor, initial_state);
//            }
#endif
        }
#endif
    }
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
    }
    bpt::free(device, critic_batch);
    bpt::free(device, critic_training_buffers);
    bpt::free(device, actor_batch);
    bpt::free(device, actor_training_buffers);
    bpt::free(device, off_policy_runner);
    bpt::free(device, actor_critic);
    bpt::free(device, observations_mean);
    bpt::free(device, observations_std);

    bpt::destruct(device, device.logger);
}

