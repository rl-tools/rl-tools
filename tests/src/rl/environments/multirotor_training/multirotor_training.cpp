// ------------ Groups 1 ------------
#include <backprop_tools/operations/cpu_tensorboard/group_1.h>
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_1.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/operations/cpu_accelerate/group_1.h>
#else
#include <backprop_tools/operations/cpu/group_1.h>
#endif
#endif
// ------------ Groups 2 ------------
#include <backprop_tools/operations/cpu_tensorboard/group_2.h>
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_2.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/operations/cpu_accelerate/group_2.h>
#else
#include <backprop_tools/operations/cpu/group_2.h>
#endif
#endif
// ------------ Groups 3 ------------
#include <backprop_tools/operations/cpu_tensorboard/group_3.h>
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_3.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/operations/cpu_accelerate/group_3.h>
#else
#include <backprop_tools/operations/cpu/group_3.h>
#endif
#endif

namespace bpt = backprop_tools;
using DEV_SPEC = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, bpt::devices::logging::CPU_TENSORBOARD>;

#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/nn/operations_cpu_mkl.h>
using DEVICE = bpt::devices::CPU_MKL<DEV_SPEC>;
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/nn/operations_cpu_accelerate.h>
using DEVICE = bpt::devices::CPU_ACCELERATE<DEV_SPEC>;
#else
#include <backprop_tools/nn/operations_generic.h>
using DEVICE = bpt::devices::CPU<DEV_SPEC>;
#endif
#endif

// generic nn_model operations use the specialized layer operations depending on the backend device
#include <backprop_tools/nn_models/operations_generic.h>
// simulation is run on the cpu and the environments functions are required in the off_policy_runner operations included afterwards
#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>
#include <backprop_tools/rl/algorithms/td3/operations_cpu.h>

// additional includes for the ui and persisting
#include <backprop_tools/rl/environments/multirotor/ui.h>
#include <backprop_tools/nn_models/persist.h>
#include <backprop_tools/rl/components/replay_buffer/persist.h>

#include <backprop_tools/rl/utils/evaluation.h>

#include "parameters.h"

#include <gtest/gtest.h>
#include <iostream>
#include <highfive/H5File.hpp>
#include <thread>
#include <future>
#include <filesystem>

using DTYPE = float;


namespace parameter_set = parameters_0;

using parameters_environment = parameter_set::environment<DEVICE, DTYPE>;
using ENVIRONMENT = typename parameters_environment::ENVIRONMENT;

using parameters_rl = parameter_set::rl<DEVICE, DTYPE, ENVIRONMENT>;
static_assert(parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

constexpr DEVICE::index_t performance_logging_interval = 100;
constexpr DEVICE::index_t ACTOR_CRITIC_EVALUATION_INTERVAL = 100;

#ifdef BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_DEBUG
TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR, TEST_FULL_TRAINING_DEBUG) {
#else
TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR, TEST_FULL_TRAINING) {
#endif
    std::string DATA_FILE_PATH = "learning_curves.h5";
    std::vector<std::vector<DTYPE>> episode_step;
    std::vector<std::vector<DTYPE>> episode_returns;
    std::vector<std::vector<DTYPE>> episode_steps;

    std::vector<std::vector<DTYPE>> eval_step;
    std::vector<std::vector<DTYPE>> eval_return;

    for(typename DEVICE::index_t run_i = 0; run_i < 1; run_i++){
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

        std::mt19937 rng(run_i);

        // device
        typename DEVICE::SPEC::LOGGING logger;
        DEVICE device;
        device.logger = &logger;
        bpt::construct(device, device.logger);

        // optimizer
        parameters_rl::OPTIMIZER optimizer[2];

        // environment
        DTYPE ui_speed_factor = 1;
        auto parameters = parameters_environment::parameters;
#if BACKPROP_TOOLS_ENABLE_MULTIROTOR_UI
        bpt::rl::environments::multirotor::UI<ENVIRONMENT> ui;
    ui.host = "localhost";
    ui.port = "8080";
    bpt::init(device, env, ui);
#else
        bool ui = false;
#endif

        // rl
        parameters_rl::ActorCriticType actor_critic;
        bpt::malloc(device, actor_critic);
        bpt::init(device, actor_critic, optimizer, rng);

        bpt::rl::components::OffPolicyRunner<parameters_rl::OFF_POLICY_RUNNER_SPEC> off_policy_runner;
        bpt::malloc(device, off_policy_runner);

        ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
        for (auto& env : envs) {
            env.parameters = parameters;
        }

        bpt::init(device, off_policy_runner, envs);

        using CRITIC_BATCH_SPEC = bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>;
        bpt::rl::components::off_policy_runner::Batch<CRITIC_BATCH_SPEC> critic_batches[2];
        bpt::rl::algorithms::td3::CriticTrainingBuffers<parameters_rl::ActorCriticType::SPEC> critic_training_buffers[2];
        parameters_rl::CRITIC_NETWORK_TYPE::BuffersForwardBackward<> critic_buffers[2];
        bpt::malloc(device, critic_batches[0]);
        bpt::malloc(device, critic_batches[1]);
        bpt::malloc(device, critic_training_buffers[0]);
        bpt::malloc(device, critic_training_buffers[1]);
        bpt::malloc(device, critic_buffers[0]);
        bpt::malloc(device, critic_buffers[1]);

        using ACTOR_BATCH_SPEC = bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>;
        bpt::rl::components::off_policy_runner::Batch<ACTOR_BATCH_SPEC> actor_batch;
        bpt::rl::algorithms::td3::ActorTrainingBuffers<parameters_rl::ActorCriticType::SPEC> actor_training_buffers;
        parameters_rl::ACTOR_NETWORK_TYPE::Buffers<> actor_buffers[2];
        parameters_rl::ACTOR_NETWORK_TYPE::Buffers<decltype(off_policy_runner)::N_ENVIRONMENTS> actor_buffers_eval;
        bpt::malloc(device, actor_batch);
        bpt::malloc(device, actor_training_buffers);
        bpt::malloc(device, actor_buffers[0]);
        bpt::malloc(device, actor_buffers[1]);
        bpt::malloc(device, actor_buffers_eval);


        // training
#ifdef BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_DEBUG
        constexpr DEVICE::index_t step_limit = parameters_rl::N_WARMUP_STEPS_ACTOR + 5000;
#else
        constexpr DEVICE::index_t step_limit = parameters_rl::REPLAY_BUFFER_CAP;
#endif
        for(int step_i = 0; step_i < step_limit; step_i++){
            auto step_start = std::chrono::high_resolution_clock::now();
            device.logger->step = step_i;
            bpt::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);
            if(step_i % 1000 == 0){
                std::cout << "run_i: " << run_i << " step_i: " << step_i << std::endl;
            }
            if(step_i > std::max(parameters_rl::ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE, parameters_rl::ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE)){
                if(step_i >= parameters_rl::N_WARMUP_STEPS_CRITIC){
                    if(step_i % parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_TRAINING_INTERVAL == 0) {
                        auto train_critic = [&device, &actor_critic, &off_policy_runner](parameters_rl::CRITIC_NETWORK_TYPE& critic, decltype(critic_batches[0])& critic_batch, decltype(optimizer[0])& optimizer, decltype(actor_buffers[0])& actor_buffers, decltype(critic_buffers[0])& critic_buffers, decltype(critic_training_buffers[0])& critic_training_buffers, decltype(rng)& rng){
                            auto gather_batch_start = std::chrono::high_resolution_clock::now();
                            bpt::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                            bpt::gather_batch(device, off_policy_runner, critic_batch, rng);
                            auto gather_batch_end = std::chrono::high_resolution_clock::now();
                            bpt::add_scalar(device, device.logger, "performance/gather_batch_duration", std::chrono::duration_cast<std::chrono::microseconds>(gather_batch_end - gather_batch_start).count(), performance_logging_interval);
                            auto critic_training_start = std::chrono::high_resolution_clock::now();
                            bpt::train_critic(device, actor_critic, critic, critic_batch, optimizer, actor_buffers, critic_buffers, critic_training_buffers);
                            auto critic_training_end = std::chrono::high_resolution_clock::now();
                            bpt::add_scalar(device, device.logger, "performance/critic_training_duration", std::chrono::duration_cast<std::chrono::microseconds>(critic_training_end - critic_training_start).count(), performance_logging_interval);
                        };
                        std::mt19937 rng1(std::uniform_int_distribution<DEVICE::index_t>()(rng));
                        std::mt19937 rng2(std::uniform_int_distribution<DEVICE::index_t>()(rng));

                        if(std::getenv("BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_CONCURRENT") != nullptr){
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
                        bpt::update_critic_targets(device, actor_critic);
                        auto update_critic_targets_end = std::chrono::high_resolution_clock::now();
                        bpt::add_scalar(device, device.logger, "performance/update_critic_targets_duration", std::chrono::duration_cast<std::chrono::microseconds>(update_critic_targets_end - update_critic_targets_start).count(), performance_logging_interval);
                    }
                }
                if(step_i >= parameters_rl::N_WARMUP_STEPS_ACTOR){
                    if(step_i % parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_TRAINING_INTERVAL == 0){
                        bpt::gather_batch(device, off_policy_runner, actor_batch, rng);
                        auto actor_training_start = std::chrono::high_resolution_clock::now();
                        bpt::train_actor(device, actor_critic, actor_batch, optimizer[0], actor_buffers[0], critic_buffers[0], actor_training_buffers);
                        auto actor_training_end = std::chrono::high_resolution_clock::now();
                        bpt::add_scalar(device, device.logger, "performance/actor_training_duration", std::chrono::duration_cast<std::chrono::microseconds>(actor_training_end - actor_training_start).count(), performance_logging_interval);
                    }
                    if(step_i % parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_TARGET_UPDATE_INTERVAL == 0) {
                        bpt::update_actor_target(device, actor_critic);
                    }
                }
                if(step_i % ACTOR_CRITIC_EVALUATION_INTERVAL == 0){
                    bpt::gather_batch(device, off_policy_runner, critic_batches[0], rng);
                    DTYPE critic_1_loss = bpt::critic_loss(device, actor_critic, actor_critic.critic_1, critic_batches[0], actor_buffers[0], critic_buffers[0], critic_training_buffers[0]);
                    bpt::add_scalar(device, device.logger, "critic_1_loss", critic_1_loss, 100);

                    bpt::gather_batch(device, off_policy_runner, actor_batch, rng);
                    DTYPE actor_value = bpt::mean(device, actor_training_buffers.state_action_value);
                    bpt::add_scalar(device, device.logger, "actor_value", actor_value, 100);

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

                            bpt::add_scalar(device, device.logger, "episode/return", mean_return);
                            bpt::add_scalar(device, device.logger, "episode/length", mean_steps);
                            run_episode_step.push_back(step_i);
                            run_episode_returns.push_back(mean_return);
                            run_episode_steps.push_back(mean_steps);
                        }
                    }
                }
            }

            auto step_end = std::chrono::high_resolution_clock::now();
            bpt::add_scalar(device, device.logger, "performance/step_duration", std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count(), performance_logging_interval);
            if(step_i % 1000 == 0){
                auto results = bpt::evaluate(device, envs[0], ui, actor_critic.actor, bpt::rl::utils::evaluation::Specification<1, parameters_rl::ENVIRONMENT_STEP_LIMIT>(), rng, true);
                std::cout << "Mean return: " << results.mean << std::endl;
                run_eval_step.push_back(step_i);
                run_eval_return.push_back(results.mean);

//            if(step_i > 250000){
//                ASSERT_GT(mean_return, 1000);
//            }
            }
        }
        // 300000 steps: 28s on M1
        std::filesystem::path data_output_dir = "data_test";
        {
            try {
                if (std::filesystem::create_directories(data_output_dir)) {
                    std::cout << "Directories created successfully: " << data_output_dir << std::endl;
                } else {
                    std::cout << "Directories already exist or failed to create: " << data_output_dir << std::endl;
                }
            } catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }
        {
            try{
                auto actor_file = HighFive::File(data_output_dir / "actor.h5", HighFive::File::Overwrite);
                bpt::save(device, actor_critic.actor, actor_file.createGroup("actor"));
            }
            catch(HighFive::Exception& e){
                std::cout << "Error while saving actor: " << e.what() << std::endl;
            }
        }
        {
            std::filesystem::path rb_output_path = data_output_dir / "replay_buffer.h5";
            try{
                auto actor_file = HighFive::File(rb_output_path, HighFive::File::Overwrite);
                auto replay_buffer_group = actor_file.createGroup("replay_buffer");
                for(typename DEVICE::index_t env_i = 0; env_i < decltype(off_policy_runner)::N_ENVIRONMENTS; env_i++){
                    bpt::save(device, off_policy_runner.replay_buffers[env_i], replay_buffer_group.createGroup(std::to_string(env_i)));
                }
            }
            catch(HighFive::Exception& e){
                std::cout << "Error while saving actor: " << e.what() << std::endl;
            }
        }
        bpt::destruct(device, device.logger);

        bpt::free(device, actor_critic);
        bpt::free(device, off_policy_runner);

        bpt::free(device, critic_batches[0]);
        bpt::free(device, critic_batches[1]);
        bpt::free(device, critic_training_buffers[0]);
        bpt::free(device, critic_training_buffers[1]);
        bpt::free(device, critic_buffers[0]);
        bpt::free(device, critic_buffers[1]);

        bpt::free(device, actor_batch);
        bpt::free(device, actor_training_buffers);
        bpt::free(device, actor_buffers[0]);
        bpt::free(device, actor_buffers[1]);
        bpt::free(device, actor_buffers_eval);
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
