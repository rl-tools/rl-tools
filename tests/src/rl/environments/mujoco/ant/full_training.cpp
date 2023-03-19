// ------------ Groups 1 ------------
#include <layer_in_c/operations/cpu_tensorboard/group_1.h>
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/operations/cpu_mkl/group_1.h>
template <typename DEV_SPEC>
using DEVICE_FACTORY = layer_in_c::devices::CPU_MKL<DEV_SPEC>;
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/operations/cpu_accelerate/group_1.h>
template <typename DEV_SPEC>
using DEVICE_FACTORY = layer_in_c::devices::CPU_ACCELERATE<DEV_SPEC>;
#else
#include <layer_in_c/operations/cpu/group_1.h>
template <typename DEV_SPEC>
using DEVICE_FACTORY = layer_in_c::devices::CPU<DEV_SPEC>;
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

#include <layer_in_c/rl/components/off_policy_runner/off_policy_runner.h>
namespace lic = layer_in_c;
using DEV_SPEC_SUPER = lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, lic::devices::logging::CPU_TENSORBOARD>;
using TI = typename DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints{
    struct HINTS: lic::rl::components::off_policy_runner::ExecutionHints<TI, 1>{};
}
struct DEV_SPEC: DEV_SPEC_SUPER{
    using EXECUTION_HINTS = execution_hints::HINTS;
};
using DEVICE = DEVICE_FACTORY<DEV_SPEC>;

#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/nn/operations_cpu_mkl.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/nn/operations_cpu_accelerate.h>
#else
#include <layer_in_c/nn/operations_generic.h>
#endif
#endif

// generic nn_model operations use the specialized layer operations depending on the backend device
#include <layer_in_c/nn_models/operations_generic.h>
// simulation is run on the cpu and the environments functions are required in the off_policy_runner operations included afterwards
#include <layer_in_c/rl/environments/mujoco/ant/operations_cpu.h>

#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/rl/algorithms/td3/operations_cpu_mkl.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/rl/algorithms/td3/operations_cpu_accelerate.h>
#else
#include <layer_in_c/rl/algorithms/td3/operations_cpu.h>
#endif
#endif

// additional includes for the ui and persisting
#include <layer_in_c/nn_models/persist.h>
#include <layer_in_c/rl/components/replay_buffer/persist.h>

#include <layer_in_c/rl/utils/evaluation.h>

#include "parameters.h"

#include <gtest/gtest.h>
#include <iostream>
#include <highfive/H5File.hpp>
#include <filesystem>
#include <thread>
#include <future>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

using DTYPE = float;

namespace parameter_set = parameters_0;

using parameters_environment = parameter_set::environment<DEVICE, double>;
using ENVIRONMENT = typename parameters_environment::ENVIRONMENT;

using parameters_rl = parameter_set::rl<DEVICE, DTYPE, ENVIRONMENT>;
static_assert(parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

constexpr DEVICE::index_t performance_logging_interval = 100;
constexpr DEVICE::index_t ACTOR_CRITIC_EVALUATION_INTERVAL = 100;
constexpr DEVICE::index_t ACTOR_CHECKPOINT_INTERVAL = 5000;
const std::string ACTOR_CHECKPOINT_DIRECTORY = "actor_checkpoints";


#ifdef LAYER_IN_C_TEST_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_DEBUG
TEST(LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR, TEST_FULL_TRAINING_DEBUG) {
#else
TEST(LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR, TEST_FULL_TRAINING) {
#endif
    std::string DATA_FILE_PATH = "learning_curves.h5";
    std::vector<std::vector<DTYPE>> episode_step;
    std::vector<std::vector<DTYPE>> episode_returns;
    std::vector<std::vector<DTYPE>> episode_steps;

    std::vector<std::vector<DTYPE>> eval_step;
    std::vector<std::vector<DTYPE>> eval_return;

    for(typename DEVICE::index_t run_i = 0; run_i < 1; run_i++){
        std::string run_name;
        {
            auto now = std::chrono::system_clock::now();
            auto local_time = std::chrono::system_clock::to_time_t(now);
            std::tm* tm = std::localtime(&local_time);

            std::ostringstream oss;
            oss << std::put_time(tm, "%FT%T%z");
            run_name = oss.str();
        }

        episode_step.push_back({});
        episode_returns.push_back({});
        episode_steps.push_back({});

        eval_step.push_back({});
        eval_return.push_back({});

        auto& run_episode_step = episode_step.back();
        auto& run_episode_returns = episode_returns.back();
        auto& run_episode_steps = episode_steps.back();

        auto& run_eval_step = eval_step.back();
        auto& run_eval_return = eval_return.back();

        auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), run_i);

        // device
        typename DEVICE::SPEC::LOGGING logger;
        DEVICE device;
        device.logger = &logger;
        lic::construct(device, device.logger);

        // optimizer
        parameters_rl::OPTIMIZER optimizer[2];

        // environment
        DTYPE ui_speed_factor = 1;
//        auto parameters = parameters_environment::parameters;
        bool ui = false;

        // rl
        parameters_rl::ActorCriticType actor_critic;
        lic::malloc(device, actor_critic);
        lic::init(device, actor_critic, optimizer, rng);

        lic::rl::components::OffPolicyRunner<parameters_rl::OFF_POLICY_RUNNER_SPEC> off_policy_runner;
        lic::malloc(device, off_policy_runner);

        ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
        for (auto& env : envs) {
            lic::malloc(device, env);
        }

        lic::init(device, off_policy_runner, envs);

        using CRITIC_BATCH_SPEC = lic::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>;
        lic::rl::components::off_policy_runner::Batch<CRITIC_BATCH_SPEC> critic_batches[2];
        lic::rl::algorithms::td3::CriticTrainingBuffers<parameters_rl::ActorCriticType::SPEC> critic_training_buffers[2];
        parameters_rl::CRITIC_NETWORK_TYPE::BuffersForwardBackward<> critic_buffers[2];
        lic::malloc(device, critic_batches[0]);
        lic::malloc(device, critic_batches[1]);
        lic::malloc(device, critic_training_buffers[0]);
        lic::malloc(device, critic_training_buffers[1]);
        lic::malloc(device, critic_buffers[0]);
        lic::malloc(device, critic_buffers[1]);

        using ACTOR_BATCH_SPEC = lic::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>;
        lic::rl::components::off_policy_runner::Batch<ACTOR_BATCH_SPEC> actor_batch;
        lic::rl::algorithms::td3::ActorTrainingBuffers<parameters_rl::ActorCriticType::SPEC> actor_training_buffers;
        parameters_rl::ACTOR_NETWORK_TYPE::Buffers<> actor_buffers[2];
        parameters_rl::ACTOR_NETWORK_TYPE::Buffers<decltype(off_policy_runner)::N_ENVIRONMENTS> actor_buffers_eval;
        lic::malloc(device, actor_batch);
        lic::malloc(device, actor_training_buffers);
        lic::malloc(device, actor_buffers[0]);
        lic::malloc(device, actor_buffers[1]);
        lic::malloc(device, actor_buffers_eval);


        // training
#ifdef LAYER_IN_C_TEST_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_DEBUG
        constexpr DEVICE::index_t step_limit = parameters_rl::N_WARMUP_STEPS_ACTOR + 5000;
#else
        constexpr DEVICE::index_t step_limit = parameters_rl::REPLAY_BUFFER_CAP;
#endif
        for(int step_i = 0; step_i < step_limit; step_i++){
            auto step_start = std::chrono::high_resolution_clock::now();
            device.logger->step = step_i;
            lic::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);
            if(step_i % 1000 == 0){
                std::cout << "run_i: " << run_i << " step_i: " << step_i << std::endl;
            }
            if(step_i > std::max(parameters_rl::ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE, parameters_rl::ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE)){
                if(step_i >= parameters_rl::N_WARMUP_STEPS_CRITIC){
                    if(step_i % parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_TRAINING_INTERVAL == 0) {
                        auto train_critic = [&device, &actor_critic, &off_policy_runner](parameters_rl::CRITIC_NETWORK_TYPE& critic, decltype(critic_batches[0])& critic_batch, decltype(optimizer[0])& optimizer, decltype(actor_buffers[0])& actor_buffers, decltype(critic_buffers[0])& critic_buffers, decltype(critic_training_buffers[0])& critic_training_buffers, decltype(rng)& rng){
                            auto gather_batch_start = std::chrono::high_resolution_clock::now();
                            lic::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                            lic::gather_batch(device, off_policy_runner, critic_batch, rng);
                            auto gather_batch_end = std::chrono::high_resolution_clock::now();
                            lic::add_scalar(device, device.logger, "performance/gather_batch_duration", std::chrono::duration_cast<std::chrono::microseconds>(gather_batch_end - gather_batch_start).count(), performance_logging_interval);
                            auto critic_training_start = std::chrono::high_resolution_clock::now();
                            lic::train_critic(device, actor_critic, critic, critic_batch, optimizer, actor_buffers, critic_buffers, critic_training_buffers);
                            auto critic_training_end = std::chrono::high_resolution_clock::now();
                            lic::add_scalar(device, device.logger, "performance/critic_training_duration", std::chrono::duration_cast<std::chrono::microseconds>(critic_training_end - critic_training_start).count(), performance_logging_interval);
                        };
                        auto rng1 = lic::random::default_engine(DEVICE::SPEC::RANDOM(), std::uniform_int_distribution<DEVICE::index_t>()(rng));
                        auto rng2 = lic::random::default_engine(DEVICE::SPEC::RANDOM(), std::uniform_int_distribution<DEVICE::index_t>()(rng));

                        if(std::getenv("LAYER_IN_C_TEST_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_CONCURRENT") != nullptr){
                            auto critic_1_training = std::async([&](){return train_critic(actor_critic.critic_1, critic_batches[0], optimizer[0], actor_buffers[0], critic_buffers[0], critic_training_buffers[0], rng1);});
                            auto critic_2_training = std::async([&](){return train_critic(actor_critic.critic_2, critic_batches[1], optimizer[1], actor_buffers[1], critic_buffers[1], critic_training_buffers[1], rng2);});
                            critic_1_training.wait();
                            critic_2_training.wait();
                        }
                        else{
                            train_critic(actor_critic.critic_1, critic_batches[0], optimizer[0], actor_buffers[0], critic_buffers[0], critic_training_buffers[0], rng1);
                            train_critic(actor_critic.critic_2, critic_batches[1], optimizer[1], actor_buffers[1], critic_buffers[1], critic_training_buffers[1], rng2);
                        }
                    }
                    if(step_i % parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_TARGET_UPDATE_INTERVAL == 0) {
                        auto update_critic_targets_start = std::chrono::high_resolution_clock::now();
                        lic::update_critic_targets(device, actor_critic);
                        auto update_critic_targets_end = std::chrono::high_resolution_clock::now();
                        lic::add_scalar(device, device.logger, "performance/update_critic_targets_duration", std::chrono::duration_cast<std::chrono::microseconds>(update_critic_targets_end - update_critic_targets_start).count(), performance_logging_interval);
                    }
                }
                if(step_i >= parameters_rl::N_WARMUP_STEPS_ACTOR){
                    if(step_i % parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_TRAINING_INTERVAL == 0){
                        lic::gather_batch(device, off_policy_runner, actor_batch, rng);
                        auto actor_training_start = std::chrono::high_resolution_clock::now();
                        lic::train_actor(device, actor_critic, actor_batch, optimizer[0], actor_buffers[0], critic_buffers[0], actor_training_buffers);
                        auto actor_training_end = std::chrono::high_resolution_clock::now();
                        lic::add_scalar(device, device.logger, "performance/actor_training_duration", std::chrono::duration_cast<std::chrono::microseconds>(actor_training_end - actor_training_start).count(), performance_logging_interval);
                    }
                    if(step_i % parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_TARGET_UPDATE_INTERVAL == 0) {
                        lic::update_actor_target(device, actor_critic);
                    }
                }
                if(step_i % ACTOR_CRITIC_EVALUATION_INTERVAL == 0){
                    lic::gather_batch(device, off_policy_runner, critic_batches[0], rng);
                    DTYPE critic_1_loss = lic::critic_loss(device, actor_critic, actor_critic.critic_1, critic_batches[0], actor_buffers[0], critic_buffers[0], critic_training_buffers[0]);
                    lic::add_scalar(device, device.logger, "critic_1_loss", critic_1_loss, 100);

                    lic::gather_batch(device, off_policy_runner, actor_batch, rng);
                    DTYPE actor_value = lic::mean(device, actor_training_buffers.state_action_value);
                    lic::add_scalar(device, device.logger, "actor_value", actor_value, 100);

                    {
                        typename DEVICE::index_t num_episodes = 0;
                        DTYPE mean_return = 0;
                        DTYPE mean_steps = 0;

                        for(typename DEVICE::index_t env_i = 0; env_i < parameters_rl::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; env_i++){
                            auto& episode_stats = off_policy_runner.episode_stats[env_i];
                            if(episode_stats.next_episode_i > 0){
                                for(typename DEVICE::index_t episode_i = 0; episode_i < episode_stats.next_episode_i - 1; episode_i++){
                                    mean_return += get(episode_stats.returns, episode_i, 0);
                                    mean_steps  += get(episode_stats.steps  , episode_i, 0);
                                    num_episodes++;
                                }
                                episode_stats.next_episode_i = 1;
                            }
                        }
                        if(num_episodes > 0){
                            mean_return /= num_episodes;
                            mean_steps /= num_episodes;

                            lic::add_scalar(device, device.logger, "episode/return", mean_return);
                            lic::add_scalar(device, device.logger, "episode/steps", mean_steps);
                            run_episode_step.push_back(step_i);
                            run_episode_returns.push_back(mean_return);
                            run_episode_steps.push_back(mean_steps);
                        }
                    }
                }
            }
            auto step_end = std::chrono::high_resolution_clock::now();
            lic::add_scalar(device, device.logger, "performance/step_duration", std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count(), performance_logging_interval);
            if(step_i % 1000 == 0){
//                DTYPE mean_return = lic::evaluate<DEVICE, ENVIRONMENT, decltype(ui), decltype(actor_critic.actor), decltype(rng), parameters_rl::ENVIRONMENT_STEP_LIMIT, true>(device, envs[0], ui, actor_critic.actor, 1, rng);
//                std::cout << "Mean return: " << mean_return << std::endl;
//                run_eval_step.push_back(step_i);
//                run_eval_return.push_back(mean_return);

//            if(step_i > 250000){
//                ASSERT_GT(mean_return, 1000);
//            }
            }
            if(step_i % ACTOR_CHECKPOINT_INTERVAL == 0){
                std::filesystem::path actor_output_dir = std::filesystem::path(ACTOR_CHECKPOINT_DIRECTORY) / run_name;
                try {
                    std::filesystem::create_directories(actor_output_dir);
                }
                catch (std::exception& e) {
                }
                std::stringstream checkpoint_name;
                checkpoint_name << "actor_" << std::setw(15) << std::setfill('0') << step_i << ".h5";
                std::filesystem::path actor_output_path = actor_output_dir / checkpoint_name.str();
                try{
                    auto actor_file = HighFive::File(actor_output_path, HighFive::File::Overwrite);
                    lic::save(device, actor_critic.actor, actor_file.createGroup("actor"));
                }
                catch(HighFive::Exception& e){
                    std::cout << "Error while saving actor: " << e.what() << std::endl;
                }
            }
        }
        {
            std::string rb_output_path = "replay_buffer.h5";
            try{
                auto actor_file = HighFive::File(rb_output_path, HighFive::File::Overwrite);
                auto replay_buffer_group = actor_file.createGroup("replay_buffer");
                for(typename DEVICE::index_t env_i = 0; env_i < decltype(off_policy_runner)::N_ENVIRONMENTS; env_i++){
                    lic::save(device, off_policy_runner.replay_buffers[env_i], replay_buffer_group.createGroup(std::to_string(env_i)));
                }
            }
            catch(HighFive::Exception& e){
                std::cout << "Error while saving actor: " << e.what() << std::endl;
            }
        }
        lic::destruct(device, device.logger);

        lic::free(device, actor_critic);
        lic::free(device, off_policy_runner);

        lic::free(device, critic_batches[0]);
        lic::free(device, critic_batches[1]);
        lic::free(device, critic_training_buffers[0]);
        lic::free(device, critic_training_buffers[1]);
        lic::free(device, critic_buffers[0]);
        lic::free(device, critic_buffers[1]);

        lic::free(device, actor_batch);
        lic::free(device, actor_training_buffers);
        lic::free(device, actor_buffers[0]);
        lic::free(device, actor_buffers[1]);
        lic::free(device, actor_buffers_eval);
    }


    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::Overwrite);
    for(typename DEVICE::index_t run_i = 0; run_i < episode_step.size(); run_i++){
        auto group = data_file.createGroup(std::to_string(run_i));
        group.createDataSet("episode_step", episode_step[run_i]);
        group.createDataSet("episode_returns", episode_returns[run_i]);
        group.createDataSet("episode_steps", episode_steps[run_i]);
        group.createDataSet("eval_step", eval_step[run_i]);
        group.createDataSet("eval_return", eval_return[run_i]);
    }
}
