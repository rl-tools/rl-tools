#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>

#include <backprop_tools/nn_models/sequential/operations_generic.h>

#include <backprop_tools/rl/algorithms/td3/loop.h>

#include <backprop_tools/containers/persist.h>
#include <backprop_tools/nn/parameters/persist.h>
#include <backprop_tools/nn/layers/dense/persist.h>
#include <backprop_tools/nn_models/sequential/persist.h>

#include <backprop_tools/containers/persist_code.h>
#include <backprop_tools/nn/parameters/persist_code.h>
#include <backprop_tools/nn/layers/dense/persist_code.h>
#include <backprop_tools/nn_models/sequential/persist_code.h>
namespace bpt = BACKPROP_TOOLS_NAMESPACE_WRAPPER ::backprop_tools;

#include "../td3/parameters.h"
namespace multirotor_training{
    namespace config {
        using namespace bpt::nn_models::sequential::interface; // to simplify the model definition we import the sequential interface but we don't want to pollute the global namespace hence we do it in a model definition namespace
        struct CoreConfig{
            using DEV_SPEC = bpt::devices::DefaultCPUSpecification;
//    using DEVICE = bpt::devices::CPU<DEV_SPEC>;
            using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
            using T = float;
            using TI = typename DEVICE::index_t;

            using ENVIRONMENT = parameters_0::environment<T, TI>::ENVIRONMENT;
            using UI = bool;

            struct DEVICE_SPEC: bpt::devices::DefaultCPUSpecification {
                using LOGGING = bpt::devices::logging::CPU;
            };
            struct TD3PendulumParameters: bpt::rl::algorithms::td3::DefaultParameters<T, TI>{
//                constexpr static typename TI CRITIC_BATCH_SIZE = 100;
//                constexpr static typename TI ACTOR_BATCH_SIZE = 100;
//                constexpr static T GAMMA = 0.997;
                static constexpr TI ACTOR_BATCH_SIZE = 256;
                static constexpr TI CRITIC_BATCH_SIZE = 256;
                static constexpr TI CRITIC_TRAINING_INTERVAL = 10;
                static constexpr TI ACTOR_TRAINING_INTERVAL = 20;
                static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 10;
                static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = 20;
//            static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 1.0;
//            static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.5;
                static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
                static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
                static constexpr T GAMMA = 0.99;
                static constexpr bool IGNORE_TERMINATION = false;
            };

            using TD3_PARAMETERS = TD3PendulumParameters;

            static constexpr bool ASYMMETRIC_OBSERVATIONS = ENVIRONMENT::PRIVILEGED_OBSERVATION_AVAILABLE;
            static constexpr TI CRITIC_OBSERVATION_DIM = ASYMMETRIC_OBSERVATIONS ? ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED : ENVIRONMENT::OBSERVATION_DIM;

            template <typename PARAMETER_TYPE>
            struct ACTOR{
                static constexpr TI HIDDEN_DIM = 64;
                static constexpr TI BATCH_SIZE = TD3_PARAMETERS::ACTOR_BATCH_SIZE;
                static constexpr auto ACTIVATION_FUNCTION = bpt::nn::activation_functions::FAST_TANH;
                using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
                using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
                using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ENVIRONMENT::ACTION_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            template <typename ACTOR>
            struct ACTOR_CHECKPOINT{
                using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ACTOR::HIDDEN_DIM, ACTOR::ACTIVATION_FUNCTION, bpt::nn::parameters::Plain>;
                using LAYER_1 = bpt::nn::layers::dense::Layer<LAYER_1_SPEC>;
                using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, ACTOR::HIDDEN_DIM, ACTOR::HIDDEN_DIM, ACTOR::ACTIVATION_FUNCTION, bpt::nn::parameters::Plain>;
                using LAYER_2 = bpt::nn::layers::dense::Layer<LAYER_2_SPEC>;
                using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, ACTOR::HIDDEN_DIM, ENVIRONMENT::ACTION_DIM, ACTOR::ACTIVATION_FUNCTION, bpt::nn::parameters::Plain>;
                using LAYER_3 = bpt::nn::layers::dense::Layer<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            template <typename PARAMETER_TYPE>
            struct CRITIC{
                static constexpr TI HIDDEN_DIM = 64;
                static constexpr TI BATCH_SIZE = TD3_PARAMETERS::CRITIC_BATCH_SIZE;

                static constexpr auto ACTIVATION_FUNCTION = bpt::nn::activation_functions::FAST_TANH;
                using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, CRITIC_OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
                using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
                using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, 1, bpt::nn::activation_functions::ActivationFunction::IDENTITY, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<T, TI>;
            using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
            using ACTOR_TYPE = typename ACTOR<bpt::nn::parameters::Adam>::MODEL;
            using ACTOR_CHECKPOINT_TYPE = typename ACTOR_CHECKPOINT<ACTOR<bpt::nn::parameters::Plain>>::MODEL;
            using ACTOR_TARGET_TYPE = typename ACTOR<bpt::nn::parameters::Plain>::MODEL;
            using CRITIC_TYPE = typename CRITIC<bpt::nn::parameters::Adam>::MODEL;
            using CRITIC_TARGET_TYPE = typename CRITIC<bpt::nn::parameters::Plain>::MODEL;

            using ACTOR_CRITIC_SPEC = bpt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_TYPE, CRITIC_TYPE, CRITIC_TARGET_TYPE, OPTIMIZER, TD3_PARAMETERS>;
            using ACTOR_CRITIC_TYPE = bpt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;


            static constexpr bool ACTOR_ENABLE_CHECKPOINTS = true;
            static constexpr TI ACTOR_CHECKPOINT_INTERVAL = 100000;
            static constexpr TI N_WARMUP_STEPS = ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
            static constexpr bool DETERMINISTIC_EVALUATION = true;
            static constexpr TI EVALUATION_INTERVAL = 50000;
            static constexpr TI NUM_EVALUATION_EPISODES = 10;
            static constexpr bool COLLECT_EPISODE_STATS = false;
            static constexpr TI EPISODE_STATS_BUFFER_SIZE = 1000;
            static constexpr TI N_ENVIRONMENTS = 1;
            static constexpr TI STEP_LIMIT = 1500001;
            static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
            static constexpr TI ENVIRONMENT_STEP_LIMIT = 500;
            static constexpr TI SEED = 3;
            using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ASYMMETRIC_OBSERVATIONS, REPLAY_BUFFER_CAP, ENVIRONMENT_STEP_LIMIT, bpt::rl::components::off_policy_runner::DefaultParameters<T>, false, true, 1000>;
            using OFF_POLICY_RUNNER_TYPE = bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
            static constexpr bpt::rl::components::off_policy_runner::DefaultParameters<T> off_policy_runner_parameters = {
                    0.5
            };

            static constexpr TI N_WARMUP_STEPS_CRITIC = 15000;
            static constexpr TI N_WARMUP_STEPS_ACTOR = 30000;
            static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        };
        struct Config: CoreConfig{
            static constexpr bool ENABLE_CURRICULUM = true;
            static constexpr bool RECALCULATE_REWARDS = true;
        };
    }

    namespace operations{
        struct CustomTrainingState: bpt::rl::algorithms::td3::loop::TrainingState<config::Config>{
            std::string run_name;
        };
        using TrainingState = CustomTrainingState;
        void init(TrainingState& ts){
            using CONFIG = config::Config;
            using TI = typename CONFIG::TI;
            for (auto& env : ts.envs) {
                env.parameters = parameters_0::environment<config::Config::T, config::Config::TI>::parameters;
            }
            ts.env_eval.parameters = ts.envs[0].parameters;
            bpt::rl::algorithms::td3::loop::init(ts, CONFIG::SEED);
            ts.off_policy_runner.parameters = parameters_0::rl<config::Config::T, config::Config::TI, config::Config::ENVIRONMENT>::off_policy_runner_parameters;
            {
                std::stringstream run_name_ss;
                run_name_ss << "multirotor_td3_";
                std::string run_name = run_name_ss.str();
                auto now = std::chrono::system_clock::now();
                auto local_time = std::chrono::system_clock::to_time_t(now);
                std::tm* tm = std::localtime(&local_time);
                run_name_ss << std::put_time(tm, "%Y%m%d%H%M%S");
                ts.run_name = run_name_ss.str();
            }
        }

        void step_checkpoint(TrainingState& ts){
            using CONFIG = config::Config;
            using T = CONFIG::T;
            using TI = CONFIG::TI;
            if(CONFIG::ACTOR_ENABLE_CHECKPOINTS && (ts.step % CONFIG::ACTOR_CHECKPOINT_INTERVAL == 0)){
                const std::string ACTOR_CHECKPOINT_DIRECTORY = "checkpoints/multirotor_td3";
                std::filesystem::path actor_output_dir = std::filesystem::path(ACTOR_CHECKPOINT_DIRECTORY) / ts.run_name;
                try {
                    std::filesystem::create_directories(actor_output_dir);
                }
                catch (std::exception& e) {
                }
                std::stringstream checkpoint_name_ss;
                checkpoint_name_ss << "actor_" << std::setw(15) << std::setfill('0') << ts.step;
                std::string checkpoint_name = checkpoint_name_ss.str();

#if defined(BACKPROP_TOOLS_ENABLE_HDF5) && !defined(BACKPROP_TOOLS_DISABLE_HDF5)
                std::filesystem::path actor_output_path_hdf5 = actor_output_dir / (checkpoint_name + ".h5");
                std::cout << "Saving actor checkpoint " << actor_output_path_hdf5 << std::endl;
                try{
                    auto actor_file = HighFive::File(actor_output_path_hdf5.string(), HighFive::File::Overwrite);
                    bpt::save(ts.device, ts.actor_critic.actor, actor_file.createGroup("actor"));
                }
                catch(HighFive::Exception& e){
                    std::cout << "Error while saving actor: " << e.what() << std::endl;
                }
#endif
                {
                    // Since checkpointing a full Adam model to code (including gradients and moments of the weights and biases currently does not work)
                    CONFIG::ACTOR_CHECKPOINT_TYPE actor_checkpoint;
                    decltype(ts.actor_critic.actor)::DoubleBuffer<1> actor_buffer;
                    decltype(actor_checkpoint)::DoubleBuffer<1> actor_checkpoint_buffer;
                    bpt::malloc(ts.device, actor_checkpoint);
                    bpt::malloc(ts.device, actor_buffer);
                    bpt::malloc(ts.device, actor_checkpoint_buffer);
                    bpt::copy(ts.device, ts.device, actor_checkpoint, ts.actor_critic.actor);
                    std::filesystem::path actor_output_path_code = actor_output_dir / (checkpoint_name + ".h");
                    auto actor_weights = bpt::save_code(ts.device, actor_checkpoint, std::string("backprop_tools::checkpoint::actor"), true);
                    std::ofstream actor_output_file(actor_output_path_code);
                    actor_output_file << actor_weights;
                    {
                        typename CONFIG::ENVIRONMENT::State state;
                        bpt::sample_initial_state(ts.device, ts.envs[0], state, ts.rng_eval);
                        bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT::OBSERVATION_DIM>> observation;
                        bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT::ACTION_DIM>> action;
                        bpt::malloc(ts.device, observation);
                        bpt::malloc(ts.device, action);
                        auto rng_copy = ts.rng_eval;
                        bpt::observe(ts.device, ts.env_eval, state, observation, rng_copy);
                        bpt::evaluate(ts.device, ts.actor_critic.actor, observation, action, actor_buffer);
                        bpt::evaluate(ts.device, actor_checkpoint, observation, action, actor_checkpoint_buffer);
                        actor_output_file << "\n" << bpt::save_code(ts.device, observation, std::string("backprop_tools::checkpoint::observation"), true);
                        actor_output_file << "\n" << bpt::save_code(ts.device, action, std::string("backprop_tools::checkpoint::action"), true);
                        actor_output_file << "\n" << "namespace backprop_tools::checkpoint::meta{";
                        actor_output_file << "\n" << "   " << "char name[] = \"" << ts.run_name << "_" << checkpoint_name << "\";";
                        actor_output_file << "\n" << "}";
                        bpt::free(ts.device, observation);
                        bpt::free(ts.device, action);
                    }
                    bpt::free(ts.device, actor_checkpoint);
                    bpt::free(ts.device, actor_buffer);
                    bpt::free(ts.device, actor_checkpoint_buffer);
                }
            }
        }

        void step_curriculum(TrainingState& ts){
            using CONFIG = config::Config;
            using T = CONFIG::T;
            using TI = CONFIG::TI;
            if(ts.step != 0 && ts.step % 100000 == 0){
//                constexpr T decay = 0.96;
                constexpr T decay = 0.75;
//                off_policy_runner.parameters.exploration_noise *= decay;
//                actor_critic.target_next_action_noise_std *= decay;
//                actor_critic.target_next_action_noise_clip *= decay;
//                off_policy_runner.parameters.exploration_noise = off_policy_runner.parameters.exploration_noise < 0.2 ? 0.2 : off_policy_runner.parameters.exploration_noise;
//                actor_critic.target_next_action_noise_std = actor_critic.target_next_action_noise_std < 0.05 ? 0.05 : actor_critic.target_next_action_noise_std;
//                actor_critic.target_next_action_noise_clip = actor_critic.target_next_action_noise_clip < 0.15 ? 0.15 : actor_critic.target_next_action_noise_clip;
                bpt::add_scalar(ts.device, ts.device.logger, "td3/target_next_action_noise_std", ts.actor_critic.target_next_action_noise_std);
                bpt::add_scalar(ts.device, ts.device.logger, "td3/target_next_action_noise_clip", ts.actor_critic.target_next_action_noise_clip);
                bpt::add_scalar(ts.device, ts.device.logger, "off_policy_runner/exploration_noise", ts.off_policy_runner.parameters.exploration_noise);


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
                if constexpr(CONFIG::ENABLE_CURRICULUM == true){
                    for (auto& env : ts.off_policy_runner.envs) {
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
                    bpt::add_scalar(ts.device, ts.device.logger, "reward_function/position_weight", ts.off_policy_runner.envs[0].parameters.mdp.reward.position);
                    bpt::add_scalar(ts.device, ts.device.logger, "reward_function/linear_velocity_weight",ts. off_policy_runner.envs[0].parameters.mdp.reward.linear_velocity);
                    bpt::add_scalar(ts.device, ts.device.logger, "reward_function/action_weight", ts.off_policy_runner.envs[0].parameters.mdp.reward.action);
                    bpt::add_scalar(ts.device, ts.device.logger, "reward_function/angular_acceleration_weight", ts.off_policy_runner.envs[0].parameters.mdp.reward.angular_acceleration);
                    if constexpr(CONFIG::RECALCULATE_REWARDS == true){
                        auto start = std::chrono::high_resolution_clock::now();
                        bpt::recalculate_rewards(ts.device, ts.off_policy_runner.replay_buffers[0], ts.off_policy_runner.envs[0], ts.rng_eval);
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
        }
        void step(TrainingState& ts){
            step_checkpoint(ts);
            step_curriculum(ts);
            bpt::rl::algorithms::td3::loop::step(ts);
        }
    }
}
