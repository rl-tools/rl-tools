#include <backprop_tools/operations/cpu_mux.h>

namespace bpt = backprop_tools;
using DEV_SPEC = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, bpt::devices::logging::CPU_TENSORBOARD>;

#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/nn/operations_cpu_mkl.h>
using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
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
//#include <backprop_tools/rl/environments/multirotor/ui.h>
#include <backprop_tools/nn_models/persist.h>
#include <backprop_tools/nn_models/persist_code.h>
#include <backprop_tools/rl/components/replay_buffer/persist.h>

#include <backprop_tools/rl/utils/evaluation.h>

#include "parameters.h"

#include "../assessment/full_assessment.h"

#include <iostream>
#include <highfive/H5File.hpp>
#include <thread>
#include <future>
#include <filesystem>

using DTYPE = float;


namespace parameter_set = parameters_0;

using TI = typename DEVICE::index_t;
using parameters_environment = parameter_set::environment<DTYPE, TI>;
using ENVIRONMENT = typename parameters_environment::ENVIRONMENT;

using parameters_rl = parameter_set::rl<DTYPE, TI, ENVIRONMENT>;
static_assert(parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

constexpr TI performance_logging_interval = 100;
constexpr TI ACTOR_CRITIC_EVALUATION_INTERVAL = 100;
#if defined(ENABLE_MULTI_CONFIG)
constexpr TI NUM_RUNS = 1;
constexpr TI BASE_SEED = 100 + ( JOB_ID );
#else
constexpr TI NUM_RUNS = 100;
constexpr TI BASE_SEED = 403;
#endif
#ifdef BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_DEBUG
constexpr DEVICE::index_t step_limit = parameters_rl::N_WARMUP_STEPS_ACTOR + 5000;
#else
constexpr DEVICE::index_t step_limit = parameters_rl::REPLAY_BUFFER_CAP;
#endif
constexpr bool ACTOR_ENABLE_CHECKPOINTS = true;
constexpr TI ACTOR_CHECKPOINT_INTERVAL = 50000;
constexpr TI ASSESSMENT_INTERVAL = 100000;
constexpr bool ACTOR_OVERWRITE_CHECKPOINTS = false;
const std::string ACTOR_CHECKPOINT_DIRECTORY = "checkpoints/multirotor_td3";


using ACTOR_CHECKPOINT_TYPE = bpt::nn_models::mlp::NeuralNetwork<bpt::nn_models::mlp::InferenceSpecification<parameters_rl::ACTOR_STRUCTURE_SPEC>>;


std::string sanitize_file_name(const std::string &input){
    std::string output = input;

    const std::string invalid_chars = R"(<>:\"/\|?*)";

    std::replace_if(output.begin(), output.end(), [&invalid_chars](const char &c) {
        return invalid_chars.find(c) != std::string::npos;
    }, '_');

    return output;
}

int main(){
    std::string DATA_FILE_PATH = "learning_curves.h5";
    std::vector<std::vector<DTYPE>> episode_step;
    std::vector<std::vector<DTYPE>> episode_returns;
    std::vector<std::vector<DTYPE>> episode_steps;

    std::vector<std::vector<DTYPE>> eval_step;
    std::vector<std::vector<DTYPE>> eval_return;

    for(typename DEVICE::index_t run_i = 0; run_i < NUM_RUNS; run_i++){
        TI seed = BASE_SEED + run_i;
        std::stringstream run_name_ss;
        run_name_ss << "multirotor_td3_" + std::to_string(seed);
#if defined(ENABLE_MULTI_CONFIG)
        run_name_ss << "[" << JOB_ID << "]";
#endif
        std::string run_name = run_name_ss.str();
        {
            auto now = std::chrono::system_clock::now();
            auto local_time = std::chrono::system_clock::to_time_t(now);
            std::tm* tm = std::localtime(&local_time);

            std::ostringstream oss;
            oss << std::put_time(tm, "%FT%T%z");
            run_name = sanitize_file_name(oss.str()) + "_" + run_name;
        }
        std::cout << "Run " << run_i << " of " << NUM_RUNS << " with seed " << seed << " and name " << run_name << std::endl;
        std::cout << "Checkpoints: " << (ACTOR_ENABLE_CHECKPOINTS ? "enabled" : "disabled") << std::endl;
        std::cout << "Observation dim: " << parameters_environment::ENVIRONMENT::OBSERVATION_DIM << " action dim: " << parameters_environment::ENVIRONMENT::ACTION_DIM << std::endl;

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

        auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), seed);

        // device
        typename DEVICE::SPEC::LOGGING logger;
        DEVICE device;
        device.logger = &logger;
        bpt::construct(device, device.logger, std::string("logs"), run_name);

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

        parameters_rl::OFF_POLICY_RUNNER_TYPE off_policy_runner;
        off_policy_runner.parameters = parameters_rl::off_policy_runner_parameters;
        bpt::malloc(device, off_policy_runner);

        ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
        for (auto& env : envs) {
            env.parameters = parameters;
        }

        bpt::init(device, off_policy_runner, envs);

        using CRITIC_BATCH_SPEC = bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>;
        bpt::rl::components::off_policy_runner::Batch<CRITIC_BATCH_SPEC> critic_batches[2];
        bpt::rl::algorithms::td3::CriticTrainingBuffers<parameters_rl::ActorCriticType::SPEC> critic_training_buffers[2];
        parameters_rl::CRITIC_TYPE::BuffersForwardBackward<> critic_buffers[2];
        bpt::malloc(device, critic_batches[0]);
        bpt::malloc(device, critic_batches[1]);
        bpt::malloc(device, critic_training_buffers[0]);
        bpt::malloc(device, critic_training_buffers[1]);
        bpt::malloc(device, critic_buffers[0]);
        bpt::malloc(device, critic_buffers[1]);

        using ACTOR_BATCH_SPEC = bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>;
        bpt::rl::components::off_policy_runner::Batch<ACTOR_BATCH_SPEC> actor_batch;
        bpt::rl::algorithms::td3::ActorTrainingBuffers<parameters_rl::ActorCriticType::SPEC> actor_training_buffers;
        parameters_rl::ACTOR_TYPE::Buffers<> actor_buffers[2];
        parameters_rl::ACTOR_TYPE::Buffers<decltype(off_policy_runner)::N_ENVIRONMENTS> actor_buffers_eval;
        bpt::malloc(device, actor_batch);
        bpt::malloc(device, actor_training_buffers);
        bpt::malloc(device, actor_buffers[0]);
        bpt::malloc(device, actor_buffers[1]);
        bpt::malloc(device, actor_buffers_eval);


        // training
        for(int step_i = 0; step_i < step_limit; step_i++){
            if(ACTOR_ENABLE_CHECKPOINTS && (step_i % ACTOR_CHECKPOINT_INTERVAL == 0)){
                std::filesystem::path actor_output_dir = std::filesystem::path(ACTOR_CHECKPOINT_DIRECTORY) / run_name;
                try {
                    std::filesystem::create_directories(actor_output_dir);
                }
                catch (std::exception& e) {
                }
                std::string checkpoint_name = "latest.h5";
                if(!ACTOR_OVERWRITE_CHECKPOINTS){
                    std::stringstream checkpoint_name_ss;
                    checkpoint_name_ss << "actor_" << std::setw(15) << std::setfill('0') << step_i;
                    checkpoint_name = checkpoint_name_ss.str();
                }
#if defined(BACKPROP_TOOLS_ENABLE_HDF5) && !defined(BACKPROP_TOOLS_DISABLE_HDF5)
                std::filesystem::path actor_output_path_hdf5 = actor_output_dir / (checkpoint_name + ".h5");
                std::cout << "Saving actor checkpoint " << actor_output_path_hdf5 << std::endl;
                try{
                    auto actor_file = HighFive::File(actor_output_path_hdf5.string(), HighFive::File::Overwrite);
                    bpt::save(device, actor_critic.actor, actor_file.createGroup("actor"));
                }
                catch(HighFive::Exception& e){
                    std::cout << "Error while saving actor: " << e.what() << std::endl;
                }
#endif
#if !defined(ENABLE_MULTI_CONFIG)
                {
                    // Since checkpointing a full Adam model to code (including gradients and moments of the weights and biases currently does not work)
                    ACTOR_CHECKPOINT_TYPE actor_checkpoint;
                    bpt::malloc(device, actor_checkpoint);
                    bpt::copy(device, device, actor_checkpoint, actor_critic.actor);
                    std::filesystem::path actor_output_path_code = actor_output_dir / (checkpoint_name + ".h");
                    auto actor_weights = bpt::save(device, actor_checkpoint, std::string("backprop_tools::checkpoint::actor"), true);
                    std::ofstream actor_output_file(actor_output_path_code);
                    actor_output_file << actor_weights;
                    {
                        parameters_environment::ENVIRONMENT::State state;
//                        while true
//                            q = rand(UnitQuaternion); angle = acos((q * [0, 0, 1])[3])
//                            if angle / pi * 180 < 20
//                                break
//                            end
//                        end
//                        state.position[0] = 0.1;
//                        state.position[1] = 0.1;
//                        state.position[2] = 0.1;
//                        state.orientation[0] = 0.924297;
//                        state.orientation[1] = 0.0688265;
//                        state.orientation[2] = 0.0552117;
//                        state.orientation[3] = -0.371335;
//                        state.linear_velocity[0] = 1;
//                        state.linear_velocity[1] = 2;
//                        state.linear_velocity[2] = 3;
//                        state.angular_velocity[0] = 1;
//                        state.angular_velocity[1] = 2;
//                        state.angular_velocity[2] = 3;
//                        if constexpr(parameters_environment::ENVIRONMENT::STATE_TYPE == bpt::rl::environments::multirotor::StateType::BaseRotorsHistory){
//                            for(TI step_i = 0; step_i < parameters_environment::ENVIRONMENT::ACTION_HISTORY_LENGTH; step_i++){
//                                for(TI action_i = 0; action_i < parameters_environment::ENVIRONMENT::ACTION_DIM; action_i++){
//                                    state.action_history[step_i][action_i] = ((DTYPE)(step_i * parameters_environment::ENVIRONMENT::ACTION_DIM + action_i))/(parameters_environment::ENVIRONMENT::ACTION_HISTORY_LENGTH * parameters_environment::ENVIRONMENT::ACTION_DIM) * 2 - 1;
//                                }
//                            }
//                        }
//                        bpt::sample_initial_state(device, envs[0], state, rng);
//                        bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, decltype(state)::DIM>> state_flat;
//                        bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, parameters_environment::ENVIRONMENT::OBSERVATION_DIM>> observation;
//                        bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, parameters_environment::ENVIRONMENT::ACTION_DIM>> action;
//                        bpt::malloc(device, state_flat);
//                        bpt::malloc(device, observation);
//                        bpt::malloc(device, action);
//                        bpt::serialize(device, state, state_flat);
//                        bpt::set_all(device, state_flat, 0);
//                        auto rng_copy = rng;
//                        bpt::observe(device, envs[0], state, observation, rng_copy);
//                        bpt::evaluate(device, actor_critic.actor, observation, action);
//                        bpt::evaluate(device, actor_checkpoint, observation, action);
//                        actor_output_file << "\n" << bpt::save(device, state_flat, std::string("backprop_tools::checkpoint::state"), true);
//                        actor_output_file << "\n" << bpt::save(device, observation, std::string("backprop_tools::checkpoint::observation"), true);
//                        actor_output_file << "\n" << bpt::save(device, action, std::string("backprop_tools::checkpoint::action"), true);
//                        bpt::free(device, state_flat);
//                        bpt::free(device, observation);
//                        bpt::free(device, action);
                    }
                    {
                        actor_output_file << "#include <backprop_tools/rl/environments/multirotor/multirotor.h>\n";
                        actor_output_file << "namespace backprop_tools::checkpoint::environment{\n";
                        static_assert(parameters_environment::ENVIRONMENT::OBSERVATION_TYPE != bpt::rl::environments::multirotor::ObservationType::DoubleQuaternion);
                        if constexpr(parameters_environment::ENVIRONMENT::OBSERVATION_TYPE == bpt::rl::environments::multirotor::ObservationType::Normal){
                            actor_output_file << "    " << "constexpr backprop_tools::rl::environments::multirotor::ObservationType OBSERVATION_TYPE = backprop_tools::rl::environments::multirotor::ObservationType::Normal;\n";
                        }
                        if constexpr(parameters_environment::ENVIRONMENT::OBSERVATION_TYPE == bpt::rl::environments::multirotor::ObservationType::RotationMatrix){
                            actor_output_file << "    " << "constexpr backprop_tools::rl::environments::multirotor::ObservationType OBSERVATION_TYPE = backprop_tools::rl::environments::multirotor::ObservationType::RotationMatrix;\n";
                        }

                        if constexpr(parameters_environment::ENVIRONMENT::STATE_TYPE == bpt::rl::environments::multirotor::StateType::Base){
                            actor_output_file << "    " << "constexpr backprop_tools::rl::environments::multirotor::ObservationType STATE_TYPE = backprop_tools::rl::environments::multirotor::StateType::Base;\n";
                        }
                        if constexpr(parameters_environment::ENVIRONMENT::STATE_TYPE == bpt::rl::environments::multirotor::StateType::BaseRotors){
                            actor_output_file << "    " << "constexpr backprop_tools::rl::environments::multirotor::ObservationType STATE_TYPE = backprop_tools::rl::environments::multirotor::StateType::BaseRotors;\n";
                        }
                        if constexpr(parameters_environment::ENVIRONMENT::STATE_TYPE == bpt::rl::environments::multirotor::StateType::BaseRotorsHistory){
                            actor_output_file << "    " << "constexpr backprop_tools::rl::environments::multirotor::StateType STATE_TYPE = backprop_tools::rl::environments::multirotor::StateType::BaseRotorsHistory;\n";
                            actor_output_file << "    " << "constexpr int ACTION_HISTORY_LENGTH = " << parameters_environment::ENVIRONMENT::ACTION_HISTORY_LENGTH << ";\n";
                        }
                        else{
                            actor_output_file << "    " << "constexpr int ACTION_HISTORY_LENGTH = " << 0 << ";\n";
                        }
                        actor_output_file << "}\n";
                    }
                    bpt::free(device, actor_checkpoint);
                }
#endif
            }
            if(step_i != 0 && step_i % 100000 == 0){
//                constexpr DTYPE decay = 0.96;
                constexpr DTYPE decay = 0.5;
                off_policy_runner.parameters.exploration_noise *= decay;
//                actor_critic.target_next_action_noise_std *= decay;
//                actor_critic.target_next_action_noise_clip *= decay;
                off_policy_runner.parameters.exploration_noise = off_policy_runner.parameters.exploration_noise < 0.05 ? 0.05 : off_policy_runner.parameters.exploration_noise;
//                actor_critic.target_next_action_noise_std = actor_critic.target_next_action_noise_std < 0.05 ? 0.05 : actor_critic.target_next_action_noise_std;
//                actor_critic.target_next_action_noise_clip = actor_critic.target_next_action_noise_clip < 0.15 ? 0.15 : actor_critic.target_next_action_noise_clip;
                bpt::add_scalar(device, device.logger, "td3/target_next_action_noise_std", actor_critic.target_next_action_noise_std);
                bpt::add_scalar(device, device.logger, "td3/target_next_action_noise_clip", actor_critic.target_next_action_noise_clip);
                bpt::add_scalar(device, device.logger, "off_policy_runner/exploration_noise", off_policy_runner.parameters.exploration_noise);


//                if(step_i > 1000000){
//                    for (auto& env : off_policy_runner.envs) {
//                        env.parameters.mdp.reward.angular_acceleration = 0.01;
//                    }
//                }
//                if(step_i <= 2000000){
//                    for (auto& env : off_policy_runner.envs) {
//                        env.parameters.mdp.reward.scale *= 2;
//                    }
//                }
            }
            auto step_start = std::chrono::high_resolution_clock::now();
            device.logger->step = step_i;
            if(step_i % ASSESSMENT_INTERVAL == 0){
                full_assessment<DEVICE, ENVIRONMENT, parameters_rl::ACTOR_TYPE>(device, actor_critic.actor, parameters_environment::parameters, true);
            }
            bpt::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);
            if(step_i % 1000 == 0){
                std::cout << "run_i: " << run_i << " step_i: " << step_i << std::endl;
            }
            if(step_i > std::max(parameters_rl::ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE, parameters_rl::ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE)){
                if(step_i >= parameters_rl::N_WARMUP_STEPS_CRITIC){
                    if(step_i % parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_TRAINING_INTERVAL == 0) {
                        auto train_critic = [&device, &actor_critic, &off_policy_runner](parameters_rl::CRITIC_TYPE& critic, decltype(critic_batches[0])& critic_batch, decltype(optimizer[0])& optimizer, decltype(actor_buffers[0])& actor_buffers, decltype(critic_buffers[0])& critic_buffers, decltype(critic_training_buffers[0])& critic_training_buffers, decltype(rng)& rng){
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
                        decltype(rng) rng1(std::uniform_int_distribution<DEVICE::index_t>()(rng));
                        decltype(rng) rng2(std::uniform_int_distribution<DEVICE::index_t>()(rng));

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
                    // this is undefined (takes the state action value of the previous step (there should be some evaluate() on the collected batch
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
                            bpt::add_scalar(device, device.logger, "episode/return_per_step", mean_return/mean_steps);
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
            if(step_i % 10000 == 0){
                auto results = bpt::evaluate(device, envs[0], ui, actor_critic.actor, bpt::rl::utils::evaluation::Specification<1, parameters_rl::ENVIRONMENT_STEP_LIMIT>(), rng, false);
                std::cout << "Mean return: " << results.mean << std::endl;
                run_eval_step.push_back(step_i);
                run_eval_return.push_back(results.mean);
                bpt::add_scalar(device, device.logger, "evaluation/return_mean", results.mean);
                bpt::add_scalar(device, device.logger, "evaluation/return_std", results.std);

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
    return 0;
}
