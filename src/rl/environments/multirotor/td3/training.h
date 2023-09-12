#include <backprop_tools/operations/cpu_mux.h>

namespace bpt = backprop_tools;
using LOGGING_DEVICE = bpt::devices::logging::CPU_TENSORBOARD;
//using LOGGING_DEVICE = bpt::devices::logging::CPU;
using DEV_SPEC = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, LOGGING_DEVICE>;

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

using T = float;
using TI = typename DEVICE::index_t;


std::string sanitize_file_name(const std::string &input){
    std::string output = input;

    const std::string invalid_chars = R"(<>:\"/\|?*)";

    std::replace_if(output.begin(), output.end(), [&invalid_chars](const char &c) {
        return invalid_chars.find(c) != std::string::npos;
    }, '_');

    return output;
}

template <typename ABLATION_SPEC>
std::string name(){
    std::string n = "";
    n += std::string("d") + (ABLATION_SPEC::DISTURBANCE ? "+"  : "-");
    n += std::string("o") + (ABLATION_SPEC::OBSERVATION_NOISE ? "+"  : "-");
    n += std::string("a") + (ABLATION_SPEC::ASYMMETRIC_ACTOR_CRITIC ? "+"  : "-");
    n += std::string("r") + (ABLATION_SPEC::ROTOR_DELAY ? "+"  : "-");
    n += std::string("h") + (ABLATION_SPEC::ACTION_HISTORY ? "+"  : "-");
    n += std::string("c") + (ABLATION_SPEC::ENABLE_CURRICULUM ? "+"  : "-");
    n += std::string("f") + (ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION ? "+"  : "-");
    n += std::string("w") + (ABLATION_SPEC::RECALCULATE_REWARDS ? "+"  : "-");
    return n;
}

template <typename BASE_SPEC>
struct SpecEval: BASE_SPEC{
    static constexpr bool DISTURBANCE = true;
    static constexpr bool OBSERVATION_NOISE = true;
    static constexpr bool ROTOR_DELAY = true;
    static constexpr bool ACTION_HISTORY = BASE_SPEC::ROTOR_DELAY && BASE_SPEC::ACTION_HISTORY;
    static constexpr bool USE_INITIAL_REWARD_FUNCTION = false;
    static constexpr bool INIT_NORMAL = false;
};

template <typename ABLATION_SPEC>
void train(TI run_id){
    static_assert(!ABLATION_SPEC::ENABLE_CURRICULUM || ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION);

    namespace parameter_set = parameters_0;


    using parameters_environment = parameter_set::environment<T, TI, ABLATION_SPEC>;
    using ENVIRONMENT = typename parameters_environment::ENVIRONMENT;
    using ABLATION_SPEC_EVAL = SpecEval<ABLATION_SPEC>;
    using parameters_environment_eval = parameter_set::environment<T, TI, ABLATION_SPEC_EVAL>;
    using ENVIRONMENT_EVAL = typename parameters_environment_eval::ENVIRONMENT;

    using parameters_rl = parameter_set::rl<T, TI, ENVIRONMENT>;
    static_assert(parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

    constexpr TI NUM_RUNS = 1;
    constexpr TI BASE_SEED = 0;
#ifdef BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_TRAINING_DEBUG
    constexpr DEVICE::index_t step_limit = parameters_rl::N_WARMUP_STEPS_ACTOR + 5000;
#else
    constexpr DEVICE::index_t step_limit = parameters_rl::REPLAY_BUFFER_CAP;
#endif
    constexpr bool ACTOR_ENABLE_CHECKPOINTS = true;
    constexpr TI ACTOR_CHECKPOINT_INTERVAL = 50000;
    constexpr TI ASSESSMENT_INTERVAL = 100000;
    constexpr bool ENABLE_ASSESSMENT = false;
    constexpr bool ACTOR_OVERWRITE_CHECKPOINTS = false;
    const std::string ACTOR_CHECKPOINT_DIRECTORY = "checkpoints/multirotor_td3";
    constexpr bool SAVE_REPLAY_BUFFER = false;
    constexpr TI performance_logging_interval = 10000;
    constexpr bool ENABLE_ACTOR_CRITIC_EVALUATION = true;
    constexpr TI ACTOR_CRITIC_EVALUATION_INTERVAL = 1000;
    constexpr bool ENABLE_EVALUATION = true;
    constexpr TI EVALUATION_INTERVAL = 10000;

    using ACTOR_CHECKPOINT_TYPE = bpt::nn_models::mlp::NeuralNetwork<bpt::nn_models::mlp::InferenceSpecification<typename parameters_rl::ACTOR_STRUCTURE_SPEC>>;
    std::string ablation_name = name<ABLATION_SPEC>();
    std::string DATA_FILE_PATH = std::string("learning_curves_") + ablation_name + std::to_string(run_id) + ".h5";
    std::cout << "Saving stats to " << DATA_FILE_PATH << std::endl;
    std::vector<std::vector<T>> training_stats_step;
    std::vector<std::vector<T>> training_stats_returns;
    std::vector<std::vector<T>> training_stats_episode_lengths;

    std::vector<std::vector<T>> eval_stats_step;
    std::vector<std::vector<T>> eval_stats_returns;
    std::vector<std::vector<T>> eval_stats_episode_lengths;

    for(typename DEVICE::index_t run_i = 0; run_i < NUM_RUNS; run_i++){
        auto run_start_time = std::chrono::high_resolution_clock::now();
        TI seed = BASE_SEED + run_id * NUM_RUNS + run_i;
        std::stringstream run_name_ss;
        run_name_ss << "multirotor_td3_" << ablation_name << "_" << std::to_string(seed);
        std::string run_name = run_name_ss.str();
        {
            auto now = std::chrono::system_clock::now();
            auto local_time = std::chrono::system_clock::to_time_t(now);
            std::tm* tm = std::localtime(&local_time);

            std::ostringstream oss;
            oss << std::put_time(tm, "%FT%T%z");
            run_name = sanitize_file_name(oss.str()) + "_" + run_name;
        }
//        run_name = "latest";
        std::cout << "Run " << run_i << " of " << NUM_RUNS << " with seed " << seed << " and name " << run_name << std::endl;
        std::cout << "Checkpoints: " << (ACTOR_ENABLE_CHECKPOINTS ? "enabled" : "disabled") << std::endl;
        std::cout << "Observation dim: " << parameters_environment::ENVIRONMENT::OBSERVATION_DIM << " privileged: " << parameters_environment::ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED << " action dim: " << parameters_environment::ENVIRONMENT::ACTION_DIM << std::endl;

        training_stats_step.push_back({});
        training_stats_returns.push_back({});
        training_stats_episode_lengths.push_back({});

        eval_stats_step.push_back({});
        eval_stats_returns.push_back({});
        eval_stats_episode_lengths.push_back({});

        auto& run_training_stats_step            = training_stats_step.back();
        auto& run_training_stats_returns         = training_stats_returns.back();
        auto& run_training_stats_episode_lengths = training_stats_episode_lengths.back();

        auto& run_eval_stats_step            = eval_stats_step.back();
        auto& run_eval_stats_returns         = eval_stats_returns.back();
        auto& run_eval_stats_episode_lengths = eval_stats_episode_lengths.back();

        auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), seed);
        auto rng_eval = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), seed);

        // device
        typename DEVICE::SPEC::LOGGING logger;
        DEVICE device;
        device.logger = &logger;
        bpt::construct(device, device.logger, std::string("logs"), run_name);

//        // optimizer
//        parameters_rl::OPTIMIZER optimizer[2];

        // environment
        T ui_speed_factor = 1;
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
        typename parameters_rl::ActorCriticType actor_critic;
        bpt::malloc(device, actor_critic);
        bpt::init(device, actor_critic, rng);

        typename parameters_rl::OFF_POLICY_RUNNER_TYPE off_policy_runner;
        off_policy_runner.parameters = parameters_rl::off_policy_runner_parameters;
        bpt::malloc(device, off_policy_runner);

        ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
        for (auto& env : envs) {
            env.parameters = parameters;
        }
        ENVIRONMENT_EVAL env_eval;
        env_eval.parameters = parameters_environment_eval::parameters;

        bpt::init(device, off_policy_runner, envs);

        using CRITIC_BATCH_SPEC = bpt::rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>;
        bpt::rl::components::off_policy_runner::Batch<CRITIC_BATCH_SPEC> critic_batches[2];
        bpt::rl::algorithms::td3::CriticTrainingBuffers<typename parameters_rl::ActorCriticType::SPEC> critic_training_buffers[2];
        typename parameters_rl::CRITIC_TYPE::template Buffers<> critic_buffers[2];
        bpt::malloc(device, critic_batches[0]);
        bpt::malloc(device, critic_batches[1]);
        bpt::malloc(device, critic_training_buffers[0]);
        bpt::malloc(device, critic_training_buffers[1]);
        bpt::malloc(device, critic_buffers[0]);
        bpt::malloc(device, critic_buffers[1]);

        using ACTOR_BATCH_SPEC = bpt::rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>;
        bpt::rl::components::off_policy_runner::Batch<ACTOR_BATCH_SPEC> actor_batch;
        bpt::rl::algorithms::td3::ActorTrainingBuffers<typename parameters_rl::ActorCriticType::SPEC> actor_training_buffers;
        typename parameters_rl::ACTOR_TYPE::template Buffers<> actor_buffers[2];
        typename parameters_rl::ACTOR_TYPE::template Buffers<decltype(off_policy_runner)::N_ENVIRONMENTS> actor_buffers_eval;
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
                        typename parameters_environment::ENVIRONMENT::State state;
                        bpt::sample_initial_state(device, envs[0], state, rng_eval);
                        bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, parameters_environment::ENVIRONMENT::OBSERVATION_DIM>> observation;
                        bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, parameters_environment::ENVIRONMENT::ACTION_DIM>> action;
                        bpt::malloc(device, observation);
                        bpt::malloc(device, action);
                        auto rng_copy = rng_eval;
                        bpt::observe(device, envs[0], state, observation, rng_copy);
                        bpt::evaluate(device, actor_critic.actor, observation, action);
                        bpt::evaluate(device, actor_checkpoint, observation, action);
                        actor_output_file << "\n" << bpt::save(device, observation, std::string("backprop_tools::checkpoint::observation"), true);
                        actor_output_file << "\n" << bpt::save(device, action, std::string("backprop_tools::checkpoint::action"), true);
                        bpt::free(device, observation);
                        bpt::free(device, action);
                    }
                    bpt::free(device, actor_checkpoint);
                }
#endif
            }
            if(ENABLE_EVALUATION && step_i % EVALUATION_INTERVAL == 0){
                auto results = bpt::evaluate(device, env_eval, ui, actor_critic.actor, bpt::rl::utils::evaluation::Specification<10, parameters_rl::ENVIRONMENT_STEP_LIMIT>(), actor_buffers[0], rng_eval, false);
                std::cout << "Mean return: " << results.returns_mean << std::endl;
                run_eval_stats_step.push_back(step_i);
                run_eval_stats_returns.push_back(results.returns_mean);
                run_eval_stats_episode_lengths.push_back(results.episode_length_mean);
                bpt::add_scalar(device, device.logger, "evaluation/return_mean", results.returns_mean);
                bpt::add_scalar(device, device.logger, "evaluation/return_std", results.returns_std);
                bpt::add_scalar(device, device.logger, "evaluation/episode_length_mean", results.episode_length_mean);
                bpt::add_scalar(device, device.logger, "evaluation/episode_length_std", results.episode_length_std);

//            if(step_i > 250000){
//                ASSERT_GT(mean_return, 1000);
//            }
            }
            if(ENABLE_ACTOR_CRITIC_EVALUATION && step_i % ACTOR_CRITIC_EVALUATION_INTERVAL == 0){
                if(step_i > std::max(parameters_rl::ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE, parameters_rl::ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE)){
                    bpt::gather_batch(device, off_policy_runner, critic_batches[0], rng_eval);
                    T critic_1_loss = bpt::critic_loss(device, actor_critic, actor_critic.critic_1, critic_batches[0], actor_buffers[0], critic_buffers[0], critic_training_buffers[0]);
                    bpt::add_scalar(device, device.logger, "critic_1_loss", critic_1_loss);

                    bpt::gather_batch(device, off_policy_runner, actor_batch, rng_eval);
                    // this is undefined (takes the state action value of the previous step (there should be some evaluate() on the collected batch
                    T actor_value = bpt::mean(device, actor_training_buffers.state_action_value);
                    bpt::add_scalar(device, device.logger, "actor_value", actor_value);
                }

                {
                    typename DEVICE::index_t num_episodes = 0;
                    T mean_return = 0;
                    T mean_steps = 0;

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
                        run_training_stats_step.push_back(step_i);
                        run_training_stats_returns.push_back(mean_return);
                        run_training_stats_episode_lengths.push_back(mean_steps);
                    }
                }
            }
            if(step_i != 0 && step_i % 100000 == 0){
//                constexpr T decay = 0.96;
                constexpr T decay = 0.75;
//                off_policy_runner.parameters.exploration_noise *= decay;
//                actor_critic.target_next_action_noise_std *= decay;
//                actor_critic.target_next_action_noise_clip *= decay;
//                off_policy_runner.parameters.exploration_noise = off_policy_runner.parameters.exploration_noise < 0.2 ? 0.2 : off_policy_runner.parameters.exploration_noise;
//                actor_critic.target_next_action_noise_std = actor_critic.target_next_action_noise_std < 0.05 ? 0.05 : actor_critic.target_next_action_noise_std;
//                actor_critic.target_next_action_noise_clip = actor_critic.target_next_action_noise_clip < 0.15 ? 0.15 : actor_critic.target_next_action_noise_clip;
                bpt::add_scalar(device, device.logger, "td3/target_next_action_noise_std", actor_critic.target_next_action_noise_std);
                bpt::add_scalar(device, device.logger, "td3/target_next_action_noise_clip", actor_critic.target_next_action_noise_clip);
                bpt::add_scalar(device, device.logger, "off_policy_runner/exploration_noise", off_policy_runner.parameters.exploration_noise);


                // sq exp
//                {
//                    for (auto& env : off_policy_runner.envs) {
//                        T action_weight = env.parameters.mdp.reward.angular_acceleration;
//                        action_weight *= 1.2;
//                        T action_weight_limit = 0.1 / 250.0 * 2;
//                        action_weight = action_weight > action_weight_limit ? action_weight_limit : action_weight;
//                        env.parameters.mdp.reward.angular_acceleration = action_weight;
//                    }
//                    bpt::add_scalar(device, device.logger, "reward_function/action_weight", off_policy_runner.envs[0].parameters.mdp.reward.action);
//                    bpt::add_scalar(device, device.logger, "reward_function/angular_acceleration_weight", off_policy_runner.envs[0].parameters.mdp.reward.angular_acceleration);
//                }
//                sq
                if constexpr(ABLATION_SPEC::ENABLE_CURRICULUM == true){
                    for (auto& env : off_policy_runner.envs) {
                        {
                            T action_weight = env.parameters.mdp.reward.action;
                            action_weight *= 1.4;
                            T action_weight_limit = 0.5;
                            action_weight = action_weight > action_weight_limit ? action_weight_limit : action_weight;
                            env.parameters.mdp.reward.action = action_weight;
                        }
                        {
                            T position_weight = env.parameters.mdp.reward.position;
                            position_weight *= 1.2;
                            T position_weight_limit = 20;
                            position_weight = position_weight > position_weight_limit ? position_weight_limit : position_weight;
                            env.parameters.mdp.reward.position = position_weight;
                        }
                        {
                            T linear_velocity_weight = env.parameters.mdp.reward.linear_velocity;
                            linear_velocity_weight *= 1.4;
                            T linear_velocity_weight_limit = 1;
                            linear_velocity_weight = linear_velocity_weight > linear_velocity_weight_limit ? linear_velocity_weight_limit : linear_velocity_weight;
                            env.parameters.mdp.reward.linear_velocity = linear_velocity_weight;
                        }
                    }
                    bpt::add_scalar(device, device.logger, "reward_function/position_weight", off_policy_runner.envs[0].parameters.mdp.reward.position);
                    bpt::add_scalar(device, device.logger, "reward_function/linear_velocity_weight", off_policy_runner.envs[0].parameters.mdp.reward.linear_velocity);
                    bpt::add_scalar(device, device.logger, "reward_function/action_weight", off_policy_runner.envs[0].parameters.mdp.reward.action);
                    bpt::add_scalar(device, device.logger, "reward_function/angular_acceleration_weight", off_policy_runner.envs[0].parameters.mdp.reward.angular_acceleration);
                    if constexpr(ABLATION_SPEC::RECALCULATE_REWARDS == true){
                        auto start = std::chrono::high_resolution_clock::now();
                        bpt::recalculate_rewards(device, off_policy_runner.replay_buffers[0], off_policy_runner.envs[0], rng);
                        auto end = std::chrono::high_resolution_clock::now();
                        std::cout << "recalculate_rewards: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
                    }
                }



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
            bpt::set_step(device, device.logger, step_i);
            if (ENABLE_ASSESSMENT && step_i % ASSESSMENT_INTERVAL == 0){
                full_assessment<DEVICE, ENVIRONMENT, typename parameters_rl::ACTOR_TYPE>(device, actor_critic.actor, parameters_environment::parameters, true);
            }
            bpt::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);
            if(step_i % 1000 == 0){
                auto now = std::chrono::high_resolution_clock::now();
                T seconds_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - run_start_time).count() / 1000.0;
                std::cout << "run: " << run_i << " step: " << step_i << " (" << step_i / seconds_elapsed << " steps/s)" << std::endl;
            }
            if(step_i > std::max(parameters_rl::ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE, parameters_rl::ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE)){
                if(step_i >= parameters_rl::N_WARMUP_STEPS_CRITIC){
                    if(step_i % parameters_rl::ActorCriticType::SPEC::PARAMETERS::CRITIC_TRAINING_INTERVAL == 0) {
                        auto train_critic = [&device, &actor_critic, &off_policy_runner](typename parameters_rl::CRITIC_TYPE& critic, decltype(critic_batches[0])& critic_batch, decltype(actor_critic.critic_optimizers[0])& optimizer, decltype(actor_buffers[0])& actor_buffers, decltype(critic_buffers[0])& critic_buffers, decltype(critic_training_buffers[0])& critic_training_buffers, decltype(rng)& rng){
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
                            auto critic_1_training = std::async([&](){return train_critic(actor_critic.critic_1, critic_batches[0], actor_critic.critic_optimizers[0], actor_buffers[0], critic_buffers[0], critic_training_buffers[0], rng1);});
                            auto critic_2_training = std::async([&](){return train_critic(actor_critic.critic_2, critic_batches[1], actor_critic.critic_optimizers[1], actor_buffers[1], critic_buffers[1], critic_training_buffers[1], rng2);});
                            critic_1_training.wait();
                            critic_2_training.wait();
                        }
                        else{
                            train_critic(actor_critic.critic_1, critic_batches[0], actor_critic.critic_optimizers[0], actor_buffers[0], critic_buffers[0], critic_training_buffers[0], rng1);
                            train_critic(actor_critic.critic_2, critic_batches[1], actor_critic.critic_optimizers[1], actor_buffers[1], critic_buffers[1], critic_training_buffers[1], rng2);
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
                        bpt::train_actor(device, actor_critic, actor_batch, actor_critic.critic_optimizers[0], actor_buffers[0], critic_buffers[0], actor_training_buffers);
                        auto actor_training_end = std::chrono::high_resolution_clock::now();
                        bpt::add_scalar(device, device.logger, "performance/actor_training_duration", std::chrono::duration_cast<std::chrono::microseconds>(actor_training_end - actor_training_start).count(), performance_logging_interval);
                    }
                    if(step_i % parameters_rl::ActorCriticType::SPEC::PARAMETERS::ACTOR_TARGET_UPDATE_INTERVAL == 0) {
                        bpt::update_actor_target(device, actor_critic);
                    }
                }
            }

            auto step_end = std::chrono::high_resolution_clock::now();
            bpt::add_scalar(device, device.logger, "performance/step_duration", std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count(), performance_logging_interval);
        }
        // 300000 steps: 28s on M1
        std::filesystem::path data_output_dir = "data_test";
        if(false){
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
        if constexpr(SAVE_REPLAY_BUFFER){
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

        auto run_end_time = std::chrono::high_resolution_clock::now();
        std::cout << "FINISHED in " << std::chrono::duration_cast<std::chrono::seconds>(run_end_time - run_start_time).count() << "s" << std::endl;
    }


    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::Overwrite);
    for(typename DEVICE::index_t run_i = 0; run_i < training_stats_step.size(); run_i++){
        auto group = data_file.createGroup(std::to_string(run_i));
        group.createDataSet("episode_step", training_stats_step[run_i]);
        group.createDataSet("episode_returns", training_stats_returns[run_i]);
        group.createDataSet("episode_lengths", training_stats_episode_lengths[run_i]);
        group.createDataSet("eval_step", eval_stats_step[run_i]);
        group.createDataSet("eval_returns", eval_stats_returns[run_i]);
        group.createDataSet("eval_episode_lengths", eval_stats_episode_lengths[run_i]);
    }
}
