#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>
#include <backprop_tools/rl/environments/multirotor/metrics.h>

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

#include <backprop_tools/rl/utils/validation_analysis.h>


namespace bpt = BACKPROP_TOOLS_NAMESPACE_WRAPPER ::backprop_tools;

#include "../td3/parameters.h"

#include <vector>
#include <queue>
#include <mutex>


namespace multirotor_training{
    namespace config {
        using namespace bpt::nn_models::sequential::interface; // to simplify the model definition we import the sequential interface but we don't want to pollute the global namespace hence we do it in a model definition namespace
        struct DEFAULT_ABLATION_SPEC{
            static constexpr bool DISTURBANCE = true;
            static constexpr bool OBSERVATION_NOISE = true;
            static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
            static constexpr bool ROTOR_DELAY = true;
            static constexpr bool ACTION_HISTORY = true;
            static constexpr bool ENABLE_CURRICULUM = true;
            static constexpr bool RECALCULATE_REWARDS = true;
            static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
            static constexpr bool INIT_NORMAL = true;
            static constexpr bool EXPLORATION_NOISE_DECAY = true;
        };
        template <typename T_ABLATION_SPEC>
        struct ABLATION_SPEC_EVAL: T_ABLATION_SPEC{
            // override everything but ACTION_HISTORY because that changes the observation space
            static constexpr bool DISTURBANCE = true;
            static constexpr bool OBSERVATION_NOISE = true;
            static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
            static constexpr bool ROTOR_DELAY = true;
            static constexpr bool ENABLE_CURRICULUM = true;
            static constexpr bool RECALCULATE_REWARDS = true;
            static constexpr bool USE_INITIAL_REWARD_FUNCTION = false; // Use target reward function as metric
            static constexpr bool INIT_NORMAL = true;
            static constexpr bool EXPLORATION_NOISE_DECAY = true;
        };
        template <typename T_ABLATION_SPEC>
        struct CoreConfig{
            static constexpr bool BENCHMARK = false;
            using ABLATION_SPEC = T_ABLATION_SPEC;
            using DEV_SPEC = bpt::utils::typing::conditional_t<BENCHMARK, bpt::devices::DefaultCPUSpecification, bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, bpt::devices::logging::CPU_TENSORBOARD>>;
//    using DEVICE = bpt::devices::CPU<DEV_SPEC>;
            using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
            using T = float;
            using TI = typename DEVICE::index_t;


            using ENVIRONMENT = typename parameters_0::environment<T, TI, ABLATION_SPEC>::ENVIRONMENT;
            using ABLATION_SPEC_EVAL_INSTANCE = ABLATION_SPEC_EVAL<ABLATION_SPEC>;
            static_assert(ABLATION_SPEC_EVAL_INSTANCE::ROTOR_DELAY == true);
            using ENVIRONMENT_EVALUATION = typename parameters_0::environment<T, TI, ABLATION_SPEC_EVAL_INSTANCE>::ENVIRONMENT;
            static_assert(ENVIRONMENT::OBSERVATION_DIM == ENVIRONMENT_EVALUATION::OBSERVATION_DIM);
            static_assert(ENVIRONMENT::ACTION_DIM == ENVIRONMENT_EVALUATION::ACTION_DIM);
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
                static constexpr TI TRAINING_INTERVAL = 10;
                static constexpr TI CRITIC_TRAINING_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr TI ACTOR_TRAINING_INTERVAL = 2 * TRAINING_INTERVAL;
                static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 1 * TRAINING_INTERVAL;
                static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = 2 * TRAINING_INTERVAL;
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
                using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, bpt::nn::parameters::groups::Input>;
                using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
                using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, bpt::nn::parameters::groups::Normal>;
                using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
                using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ENVIRONMENT::ACTION_DIM, bpt::nn::activation_functions::FAST_TANH, PARAMETER_TYPE, BATCH_SIZE, bpt::nn::parameters::groups::Output>;
                using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            template <typename ACTOR>
            struct ACTOR_CHECKPOINT{
                using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ACTOR::HIDDEN_DIM, ACTOR::ACTIVATION_FUNCTION, bpt::nn::parameters::Plain, 1, bpt::nn::parameters::groups::Input>;
                using LAYER_1 = bpt::nn::layers::dense::Layer<LAYER_1_SPEC>;
                using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, ACTOR::HIDDEN_DIM, ACTOR::HIDDEN_DIM, ACTOR::ACTIVATION_FUNCTION, bpt::nn::parameters::Plain, 1, bpt::nn::parameters::groups::Normal>;
                using LAYER_2 = bpt::nn::layers::dense::Layer<LAYER_2_SPEC>;
                using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, ACTOR::HIDDEN_DIM, ENVIRONMENT::ACTION_DIM, bpt::nn::activation_functions::FAST_TANH, bpt::nn::parameters::Plain, 1, bpt::nn::parameters::groups::Output>;
                using LAYER_3 = bpt::nn::layers::dense::Layer<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            template <typename PARAMETER_TYPE>
            struct CRITIC{
                static constexpr TI HIDDEN_DIM = 64;
                static constexpr TI BATCH_SIZE = TD3_PARAMETERS::CRITIC_BATCH_SIZE;

                static constexpr auto ACTIVATION_FUNCTION = bpt::nn::activation_functions::FAST_TANH;
                using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, CRITIC_OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, bpt::nn::parameters::groups::Input>;
                using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
                using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE, bpt::nn::parameters::groups::Normal>;
                using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
                using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, 1, bpt::nn::activation_functions::ActivationFunction::IDENTITY, PARAMETER_TYPE, BATCH_SIZE, bpt::nn::parameters::groups::Output>;
                using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            struct OPTIMIZER_PARAMETERS: bpt::nn::optimizers::adam::DefaultParameters<T, TI>{
                static constexpr T WEIGHT_DECAY = 0.0001;
                static constexpr T WEIGHT_DECAY_INPUT = 0.0001;
                static constexpr T WEIGHT_DECAY_OUTPUT = 0.0001;
                static constexpr T BIAS_LR_FACTOR = 1;
            };
            using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
            using ACTOR_TYPE = typename ACTOR<bpt::nn::parameters::Adam>::MODEL;
            using ACTOR_CHECKPOINT_TYPE = typename ACTOR_CHECKPOINT<ACTOR<bpt::nn::parameters::Plain>>::MODEL;
            using ACTOR_TARGET_TYPE = typename ACTOR<bpt::nn::parameters::Plain>::MODEL;
            using CRITIC_TYPE = typename CRITIC<bpt::nn::parameters::Adam>::MODEL;
            using CRITIC_TARGET_TYPE = typename CRITIC<bpt::nn::parameters::Plain>::MODEL;

            using ACTOR_CRITIC_SPEC = bpt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_TYPE, CRITIC_TYPE, CRITIC_TARGET_TYPE, OPTIMIZER, TD3_PARAMETERS>;
            using ACTOR_CRITIC_TYPE = bpt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;


            static constexpr bool ACTOR_ENABLE_CHECKPOINTS = true; //!BENCHMARK;
            static constexpr TI ACTOR_CHECKPOINT_INTERVAL = 100000;
            static constexpr bool DETERMINISTIC_EVALUATION = !BENCHMARK;
            static constexpr TI EVALUATION_INTERVAL = 10000;
            static constexpr TI NUM_EVALUATION_EPISODES = 1000;
            static constexpr bool COLLECT_EPISODE_STATS = false;
            static constexpr TI EPISODE_STATS_BUFFER_SIZE = 1000;
            static constexpr TI N_ENVIRONMENTS = 1;
            static constexpr TI STEP_LIMIT = 300001;
//            static constexpr TI REPLAY_BUFFER_LIMIT = 3000000;
            static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
            static constexpr TI ENVIRONMENT_STEP_LIMIT = 500;
            static constexpr TI BASE_SEED = 0;
            static constexpr bool CONSTRUCT_LOGGER = true;
            using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ASYMMETRIC_OBSERVATIONS, REPLAY_BUFFER_CAP, ENVIRONMENT_STEP_LIMIT, bpt::rl::components::off_policy_runner::DefaultParameters<T>, false, true, 1000>;
            using OFF_POLICY_RUNNER_TYPE = bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
            static constexpr bpt::rl::components::off_policy_runner::DefaultParameters<T> off_policy_runner_parameters = {
                    0.1
            };
            // good policy stats:
            // angular_acc error < 60
            // action cost < 0.7
            // angular_velocity < 4
            // linear acceleration < 4
            // orientation < 0.08
            // position < 0.25
            // linear_velocity < 0.75

            static constexpr TI N_WARMUP_STEPS_CRITIC = 15000;
            static constexpr TI N_WARMUP_STEPS_ACTOR = 30000;
            static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        };
        template <typename T_ABLATION_SPEC>
        struct Config: CoreConfig<T_ABLATION_SPEC>{
            using SUPER = CoreConfig<T_ABLATION_SPEC>;
            using T = typename SUPER::T;
            using TI = typename SUPER::TI;
            using ENVIRONMENT = typename SUPER::ENVIRONMENT;
            using VALIDATION_SPEC = bpt::rl::utils::validation::Specification<T, TI, ENVIRONMENT>;
            static constexpr TI VALIDATION_N_EPISODES = 10;
            static constexpr TI VALIDATION_MAX_EPISODE_LENGTH = SUPER::ENVIRONMENT_STEP_LIMIT;
            using TASK_SPEC = bpt::rl::utils::validation::TaskSpecification<VALIDATION_SPEC, VALIDATION_N_EPISODES, VALIDATION_MAX_EPISODE_LENGTH>;
            using ADDITIONAL_METRICS = bpt::rl::utils::validation::set::Component<
                    bpt::rl::utils::validation::metrics::SettlingFractionPosition<TI, 200>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorMean<bpt::rl::utils::validation::metrics::multirotor::POSITION, TI, 100>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorStd <bpt::rl::utils::validation::metrics::multirotor::POSITION, TI, 100>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorMean<bpt::rl::utils::validation::metrics::multirotor::POSITION, TI, 200>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorStd <bpt::rl::utils::validation::metrics::multirotor::POSITION, TI, 200>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorMean<bpt::rl::utils::validation::metrics::multirotor::ANGLE, TI, 100>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorStd <bpt::rl::utils::validation::metrics::multirotor::ANGLE, TI, 200>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorMean<bpt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 100>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorStd <bpt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 100>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorMean<bpt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 200>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorStd <bpt::rl::utils::validation::metrics::multirotor::LINEAR_VELOCITY, TI, 200>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorMean<bpt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 100>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorStd <bpt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 100>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorMean<bpt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 200>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorStd <bpt::rl::utils::validation::metrics::multirotor::ANGULAR_VELOCITY, TI, 200>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorMean<bpt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 100>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorStd <bpt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 100>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorMean<bpt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 200>,
                    bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::MaxErrorStd <bpt::rl::utils::validation::metrics::multirotor::ANGULAR_ACCELERATION, TI, 200>,
                    bpt::rl::utils::validation::set::FinalComponent>>>>>>>>>>>>>>>>>>>;
            using METRICS = bpt::rl::utils::validation::DefaultMetrics<ADDITIONAL_METRICS>;
        };
    }

    namespace operations{
        template <typename T_CONFIG>
        struct CustomTrainingState: bpt::rl::algorithms::td3::loop::TrainingState<T_CONFIG>{
            using CONFIG = T_CONFIG;
            std::string run_name;
            std::queue<std::vector<typename CONFIG::ENVIRONMENT::State>> trajectories;
            std::mutex trajectories_mutex;
            std::vector<typename CONFIG::ENVIRONMENT::State> episode;
            // validation
            bpt::rl::utils::validation::Task<typename CONFIG::TASK_SPEC> task;
            typename CONFIG::ENVIRONMENT validation_envs[CONFIG::VALIDATION_N_EPISODES];
            typename CONFIG::ACTOR_TYPE::template DoubleBuffer<CONFIG::VALIDATION_N_EPISODES> validation_actor_buffers;
        };

        template <typename ABLATION_SPEC>
        std::string ablation_name(){
            std::string n = "";
            n += std::string("d") + (ABLATION_SPEC::DISTURBANCE ? "+"  : "-");
            n += std::string("o") + (ABLATION_SPEC::OBSERVATION_NOISE ? "+"  : "-");
            n += std::string("a") + (ABLATION_SPEC::ASYMMETRIC_ACTOR_CRITIC ? "+"  : "-");
            n += std::string("r") + (ABLATION_SPEC::ROTOR_DELAY ? "+"  : "-");
            n += std::string("h") + (ABLATION_SPEC::ACTION_HISTORY ? "+"  : "-");
            n += std::string("c") + (ABLATION_SPEC::ENABLE_CURRICULUM ? "+"  : "-");
            n += std::string("f") + (ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION ? "+"  : "-");
            n += std::string("w") + (ABLATION_SPEC::RECALCULATE_REWARDS ? "+"  : "-");
            n += std::string("e") + (ABLATION_SPEC::EXPLORATION_NOISE_DECAY ? "+"  : "-");
            return n;
        }

        template <typename T_CONFIG>
        using TrainingState = CustomTrainingState<T_CONFIG>;
        template <typename T_CONFIG>
        void init(TrainingState<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
            using CONFIG = T_CONFIG;
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            using ABLATION_SPEC = typename CONFIG::ABLATION_SPEC;
            auto env_parameters = parameters_0::environment<T, TI, ABLATION_SPEC>::parameters;
            auto env_parameters_eval = parameters_0::environment<T, TI, config::ABLATION_SPEC_EVAL<ABLATION_SPEC>>::parameters;
            for (auto& env : ts.envs) {
                env.parameters = env_parameters;
            }
            ts.env_eval.parameters = env_parameters_eval;
            TI effective_seed = CONFIG::BASE_SEED + seed;
            {
                std::stringstream run_name_ss;
                run_name_ss << "";
                auto now = std::chrono::system_clock::now();
                auto local_time = std::chrono::system_clock::to_time_t(now);
                std::tm* tm = std::localtime(&local_time);
                run_name_ss << "" << std::put_time(tm, "%Y_%m_%d_%H_%M_%S");
                if constexpr(CONFIG::BENCHMARK){
                    run_name_ss << "_BENCHMARK";
                }
                run_name_ss << "_" << ablation_name<ABLATION_SPEC>();
                run_name_ss << "_" << std::setw(3) << std::setfill('0') << effective_seed;
                ts.run_name = run_name_ss.str();
            }
            {
                bpt::construct(ts.device, ts.device.logger, std::string("logs"), ts.run_name);
            }
            bpt::set_step(ts.device, ts.device.logger, 0);
            bpt::add_scalar(ts.device, ts.device.logger, "loop/seed", effective_seed);
            bpt::rl::algorithms::td3::loop::init(ts, effective_seed);
            ts.off_policy_runner.parameters = CONFIG::off_policy_runner_parameters;

            for(typename CONFIG::ENVIRONMENT& env: ts.validation_envs){
                env.parameters = parameters_0::environment<typename CONFIG::T, TI, ABLATION_SPEC>::parameters;
            }
            bpt::malloc(ts.device, ts.validation_actor_buffers);
            bpt::init(ts.device, ts.task, ts.validation_envs, ts.rng_eval);

            // info

            std::cout << "Environment Info: \n";
            std::cout << "\t" << "Observation dim: " << CONFIG::ENVIRONMENT::OBSERVATION_DIM << std::endl;
            std::cout << "\t" << "Observation dim privileged: " << CONFIG::ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED << std::endl;
            std::cout << "\t" << "Action dim: " << CONFIG::ENVIRONMENT::ACTION_DIM << std::endl;
        }

        template <typename T_CONFIG>
        void step_logger(TrainingState<T_CONFIG>& ts){
            bpt::set_step(ts.device, ts.device.logger, ts.step);
        }

        template <typename T_CONFIG>
        void step_checkpoint(TrainingState<T_CONFIG>& ts){
            using CONFIG = T_CONFIG;
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
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
                    typename CONFIG::ACTOR_CHECKPOINT_TYPE actor_checkpoint;
                    typename decltype(ts.actor_critic.actor)::template DoubleBuffer<1> actor_buffer;
                    typename decltype(actor_checkpoint)::template DoubleBuffer<1> actor_checkpoint_buffer;
                    bpt::malloc(ts.device, actor_checkpoint);
                    bpt::malloc(ts.device, actor_buffer);
                    bpt::malloc(ts.device, actor_checkpoint_buffer);
                    bpt::copy(ts.device, ts.device, ts.actor_critic.actor, actor_checkpoint);
                    std::filesystem::path actor_output_path_code = actor_output_dir / (checkpoint_name + ".h");
                    auto actor_weights = bpt::save_code(ts.device, actor_checkpoint, std::string("backprop_tools::checkpoint::actor"), true);
                    std::cout << "Saving checkpoint at: " << actor_output_path_code << std::endl;
                    std::ofstream actor_output_file(actor_output_path_code);
                    actor_output_file << actor_weights;
                    {
                        typename CONFIG::ENVIRONMENT_EVALUATION::State state;
                        bpt::sample_initial_state(ts.device, ts.envs[0], state, ts.rng_eval);
                        bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT_EVALUATION::OBSERVATION_DIM>> observation;
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
                        actor_output_file << "\n" << "   " << "char commit_hash[] = \"" << BACKPROP_TOOLS_STRINGIFY(BACKPROP_TOOLS_COMMIT_HASH) << "\";";
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

        template <typename CONFIG>
        void step_curriculum(TrainingState<CONFIG>& ts){
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            if constexpr(CONFIG::ABLATION_SPEC::ENABLE_CURRICULUM == true) {
                if(ts.step != 0 && ts.step % 100000 == 0 && ts.step != (CONFIG::STEP_LIMIT - 1)){
//                constexpr T decay = 0.96;
//                constexpr T decay = 0.75;
//                off_policy_runner.parameters.exploration_noise *= decay;
//                actor_critic.target_next_action_noise_std *= decay;
//                actor_critic.target_next_action_noise_clip *= decay;
//                off_policy_runner.parameters.exploration_noise = off_policy_runner.parameters.exploration_noise < 0.2 ? 0.2 : off_policy_runner.parameters.exploration_noise;
//                actor_critic.target_next_action_noise_std = actor_critic.target_next_action_noise_std < 0.05 ? 0.05 : actor_critic.target_next_action_noise_std;
//                actor_critic.target_next_action_noise_clip = actor_critic.target_next_action_noise_clip < 0.15 ? 0.15 : actor_critic.target_next_action_noise_clip;
//                constexpr T noise_decay_base = 0.95;
//                ts.off_policy_runner.parameters.exploration_noise *= noise_decay_base;
//                ts.actor_critic.target_next_action_noise_std *= noise_decay_base;
//                ts.actor_critic.target_next_action_noise_clip *= noise_decay_base;
//                if constexpr(CONFIG::ABLATION_SPEC::ENABLE_CURRICULUM == true){
//                    T gamma = ts.actor_critic.gamma;
//                    gamma += 0.001;
//                    T gamma_limit = 0.997;
//                    gamma = gamma > gamma_limit ? gamma_limit : gamma;
//                    ts.actor_critic.gamma = gamma;
//                }
//                if (CONFIG::ABLATION_SPEC::ENABLE_CURRICULUM == true && ts.step == 300000){
//                    for (auto& env : ts.off_policy_runner.envs) {
//                        env.parameters.mdp.init = backprop_tools::rl::environments::multirotor::parameters::init::all_around_2<T, TI, 4, CONFIG::ENVIRONMENT::PARAMETERS::MDP::REWARD_FUNCTION>;
//                    }
//                }
                    bpt::add_scalar(ts.device, ts.device.logger, "td3/gamma", ts.actor_critic.gamma);
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
                    for(auto& env : ts.off_policy_runner.envs){
                        {
                            T action_weight = env.parameters.mdp.reward.action;
                            action_weight *= 1.4;
                            T action_weight_limit = 1.0;
                            action_weight = action_weight > action_weight_limit ? action_weight_limit : action_weight;
                            env.parameters.mdp.reward.action = action_weight;
                            bpt::add_scalar(ts.device, ts.device.logger, "reward_function/action_weight", action_weight);
                        }
                        {
                            T position_weight = env.parameters.mdp.reward.position;
                            position_weight *= 1.2;
                            T position_weight_limit = 40;
                            position_weight = position_weight > position_weight_limit ? position_weight_limit : position_weight;
                            env.parameters.mdp.reward.position = position_weight;
                            bpt::add_scalar(ts.device, ts.device.logger, "reward_function/position_weight", position_weight);
                        }
                        {
                            T linear_velocity_weight = env.parameters.mdp.reward.linear_velocity;
                            linear_velocity_weight *= 1.4;
                            T linear_velocity_weight_limit = 1;
                            linear_velocity_weight = linear_velocity_weight > linear_velocity_weight_limit ? linear_velocity_weight_limit : linear_velocity_weight;
                            env.parameters.mdp.reward.linear_velocity = linear_velocity_weight;
                            bpt::add_scalar(ts.device, ts.device.logger, "reward_function/linear_velocity_weight", linear_velocity_weight);
                        }
                    }
                    if constexpr(CONFIG::ABLATION_SPEC::RECALCULATE_REWARDS == true){
                        auto start = std::chrono::high_resolution_clock::now();
                        bpt::recalculate_rewards(ts.device, ts.off_policy_runner.replay_buffers[0], ts.off_policy_runner.envs[0], ts.rng_eval);
                        auto end = std::chrono::high_resolution_clock::now();
                        std::cout << "recalculate_rewards: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms\n";
                    }
                }
            }
            if(CONFIG::ABLATION_SPEC::EXPLORATION_NOISE_DECAY == true){
                if(ts.step % 100000 == 0 && ts.step >= 500000){
                    constexpr T noise_decay_base = 0.90;
                    ts.off_policy_runner.parameters.exploration_noise *= noise_decay_base;
                    ts.actor_critic.target_next_action_noise_std *= noise_decay_base;
                    ts.actor_critic.target_next_action_noise_clip *= noise_decay_base;
                }
            }
        }
        template <typename CONFIG>
        void step_critic_reset(TrainingState<CONFIG>& ts){
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            if(ts.step == 500000) {
                std::cout << "Resetting critic" << std::endl;
                bpt::init_weights(ts.device, ts.actor_critic.critic_1, ts.rng);
                bpt::init_weights(ts.device, ts.actor_critic.critic_2, ts.rng);
                bpt::reset_optimizer_state(ts.device, ts.actor_critic.critic_optimizers[0], ts.actor_critic.critic_1);
                bpt::reset_optimizer_state(ts.device, ts.actor_critic.critic_optimizers[1], ts.actor_critic.critic_2);
            }
            if(ts.step == 600000){
                std::cout << "Resetting actor" << std::endl;
                bpt::init_weights(ts.device, ts.actor_critic.actor, ts.rng);
                bpt::reset_optimizer_state(ts.device, ts.actor_critic.actor_optimizer, ts.actor_critic.actor);
            }
        }

        template <typename CONFIG>
        void step_trajectory_collection(TrainingState<CONFIG>& ts){
            using TI = typename CONFIG::TI;
            auto& rb = ts.off_policy_runner.replay_buffers[0];
            TI current_pos = rb.position == 0 ? CONFIG::REPLAY_BUFFER_CAP - 1 : rb.position - 1;
            typename CONFIG::ENVIRONMENT::State s = get(rb.states, current_pos, 0);
            ts.episode.push_back(s);
            if(bpt::get(rb.terminated, current_pos, 0) == 1.0){
                {
                    std::lock_guard<std::mutex> lock(ts.trajectories_mutex);
                    ts.trajectories.push(ts.episode);
                }
                ts.episode.clear();
            }
        }
        template <typename CONFIG>
        void step_network_stats(TrainingState<CONFIG>& ts){
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            if(ts.step % 100000 == 0){
                {
                    T rmse = bpt::math::sqrt(ts.device.math, bpt::mean_of_squares(ts.device, bpt::get_layer(ts.device, ts.actor_critic.actor, bpt::Constant<0>{}).weights.parameters));
                    bpt::add_scalar(ts.device, ts.device.logger, std::string("td3/actor/layer_0/weights_rmse"), rmse);
                }
                {
                    T rmse = bpt::math::sqrt(ts.device.math, bpt::mean_of_squares(ts.device, bpt::get_layer(ts.device, ts.actor_critic.actor, bpt::Constant<1>{}).weights.parameters));
                    bpt::add_scalar(ts.device, ts.device.logger, std::string("td3/actor/layer_1/weights_rmse"), rmse);
                }
                {
                    T rmse = bpt::math::sqrt(ts.device.math, bpt::mean_of_squares(ts.device, bpt::get_layer(ts.device, ts.actor_critic.actor, bpt::Constant<2>{}).weights.parameters));
                    bpt::add_scalar(ts.device, ts.device.logger, std::string("td3/actor/layer_2/weights_rmse"), rmse);
                }
            }
        }
        template <typename CONFIG>
        void step_network_analysis(TrainingState<CONFIG>& ts){
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            if(ts.step % 100000 == 0){
                {
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 100, 100>> image;
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED + CONFIG::ENVIRONMENT::ACTION_DIM>> critic_input;
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 1>> critic_output;
                    typename CONFIG::CRITIC_TYPE::template DoubleBuffer<1> critic_buffer;
                    bpt::malloc(ts.device, image);
                    bpt::malloc(ts.device, critic_input);
                    bpt::malloc(ts.device, critic_output);
                    bpt::malloc(ts.device, critic_buffer);
                    auto observation = bpt::view(ts.device, critic_input, bpt::matrix::ViewSpec<1, CONFIG::ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED>{});
                    auto action = bpt::view(ts.device, critic_input, bpt::matrix::ViewSpec<1, CONFIG::ENVIRONMENT::ACTION_DIM>{}, 0, CONFIG::ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED);
                    bpt::set_all(ts.device, action, 0);
                    typename CONFIG::ENVIRONMENT::State state;
                    bpt::observe_privileged(ts.device, ts.env_eval, state, observation, ts.rng_eval);
                    bpt::initial_state(ts.device, ts.envs[0], state);
                    for(TI action_0 = 0; action_0 < decltype(image)::COLS; action_0++){
                        for(TI action_1 = 0; action_1 < decltype(image)::ROWS; action_1++){
                            T action_0_value = -1.0 + 2.0 / (decltype(image)::COLS - 1) * action_0;
                            T action_1_value = -1.0 + 2.0 / (decltype(image)::ROWS - 1) * action_1;
                            bpt::set(action, 0, 0, action_0_value);
                            bpt::set(action, 0, 1, action_1_value);
                            bpt::evaluate(ts.device, ts.actor_critic.critic_1, critic_input, critic_output, critic_buffer);
                            bpt::set(image, action_0, action_1, bpt::get(critic_output, 0, 0));
                        }
                    }
                    bpt::normalize(ts.device, image);
                    bpt::multiply_all(ts.device, image, 1/3.0);
                    bpt::add_image(ts.device, ts.device.logger, std::string("td3/action_value_0_1"), image);
                    bpt::free(ts.device, image);
                    bpt::free(ts.device, critic_input);
                    bpt::free(ts.device, critic_output);
                    bpt::free(ts.device, critic_buffer);
                }
                {
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 100, 100>> image;
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED + CONFIG::ENVIRONMENT::ACTION_DIM>> critic_input;
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 1>> critic_output;
                    typename CONFIG::CRITIC_TYPE::template DoubleBuffer<1> critic_buffer;
                    bpt::malloc(ts.device, image);
                    bpt::malloc(ts.device, critic_input);
                    bpt::malloc(ts.device, critic_output);
                    bpt::malloc(ts.device, critic_buffer);
                    auto observation = bpt::view(ts.device, critic_input, bpt::matrix::ViewSpec<1, CONFIG::ENVIRONMENT_EVALUATION::OBSERVATION_DIM_PRIVILEGED>{});
                    auto action = bpt::view(ts.device, critic_input, bpt::matrix::ViewSpec<1, CONFIG::ENVIRONMENT_EVALUATION::ACTION_DIM>{}, 0, CONFIG::ENVIRONMENT_EVALUATION::OBSERVATION_DIM_PRIVILEGED);
                    bpt::set_all(ts.device, action, 0);
                    typename CONFIG::ENVIRONMENT_EVALUATION::State state;
                    bpt::initial_state(ts.device, ts.envs[0], state);
                    for(TI x_i = 0; x_i < decltype(image)::COLS; x_i++){
                        for(TI y_i = 0; y_i < decltype(image)::ROWS; y_i++){
                            T x = -1.0 + 2.0 / (decltype(image)::COLS - 1) * x_i;
                            T y = -1.0 + 2.0 / (decltype(image)::ROWS - 1) * y_i;
                            state.position[0] = x;
                            state.position[1] = y;
                            bpt::observe_privileged(ts.device, ts.env_eval, state, observation, ts.rng_eval);
                            bpt::evaluate(ts.device, ts.actor_critic.critic_1, critic_input, critic_output, critic_buffer);
                            bpt::set(image, x_i, y_i, bpt::get(critic_output, 0, 0));
                        }
                    }
                    bpt::normalize(ts.device, image);
                    bpt::multiply_all(ts.device, image, 1/3.0);
                    bpt::add_image(ts.device, ts.device.logger, std::string("td3/position_value"), image);
                    bpt::free(ts.device, image);
                    bpt::free(ts.device, critic_input);
                    bpt::free(ts.device, critic_output);
                    bpt::free(ts.device, critic_buffer);
                }
                {
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 100, 100>> image;
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT_EVALUATION::OBSERVATION_DIM_PRIVILEGED + CONFIG::ENVIRONMENT_EVALUATION::ACTION_DIM>> critic_input;
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 1>> critic_output;
                    typename CONFIG::CRITIC_TYPE::template DoubleBuffer<1> critic_buffer;
                    bpt::malloc(ts.device, image);
                    bpt::malloc(ts.device, critic_input);
                    bpt::malloc(ts.device, critic_output);
                    bpt::malloc(ts.device, critic_buffer);
                    auto observation = bpt::view(ts.device, critic_input, bpt::matrix::ViewSpec<1, CONFIG::ENVIRONMENT_EVALUATION::OBSERVATION_DIM_PRIVILEGED>{});
                    auto action = bpt::view(ts.device, critic_input, bpt::matrix::ViewSpec<1, CONFIG::ENVIRONMENT_EVALUATION::ACTION_DIM>{}, 0, CONFIG::ENVIRONMENT_EVALUATION::OBSERVATION_DIM_PRIVILEGED);
                    bpt::set_all(ts.device, action, 0);
                    typename CONFIG::ENVIRONMENT_EVALUATION::State state;
                    bpt::initial_state(ts.device, ts.envs[0], state);
                    for(TI x_i = 0; x_i < decltype(image)::COLS; x_i++){
                        for(TI y_i = 0; y_i < decltype(image)::ROWS; y_i++){
                            T x = -1.0 + 2.0 / (decltype(image)::COLS - 1) * x_i;
                            T y = -1.0 + 2.0 / (decltype(image)::ROWS - 1) * y_i;
                            state.linear_velocity[0] = x;
                            state.linear_velocity[1] = y;
                            bpt::observe_privileged(ts.device, ts.env_eval, state, observation, ts.rng_eval);
                            bpt::evaluate(ts.device, ts.actor_critic.critic_1, critic_input, critic_output, critic_buffer);
                            bpt::set(image, x_i, y_i, bpt::get(critic_output, 0, 0));
                        }
                    }
                    bpt::normalize(ts.device, image);
                    bpt::multiply_all(ts.device, image, 1/3.0);
                    bpt::add_image(ts.device, ts.device.logger, std::string("td3/velocity_value"), image);
                    bpt::free(ts.device, image);
                    bpt::free(ts.device, critic_input);
                    bpt::free(ts.device, critic_output);
                    bpt::free(ts.device, critic_buffer);
                }
                {
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 100, 100>> image_0, image_1, image_2, image_3;
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT_EVALUATION::OBSERVATION_DIM>> actor_input;
                    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT_EVALUATION::ACTION_DIM>> action;
                    typename CONFIG::ACTOR_TYPE::template DoubleBuffer<1> actor_buffer;
                    bpt::malloc(ts.device, image_0);
                    bpt::malloc(ts.device, image_1);
                    bpt::malloc(ts.device, image_2);
                    bpt::malloc(ts.device, image_3);
                    bpt::malloc(ts.device, actor_input);
                    bpt::malloc(ts.device, action);
                    bpt::malloc(ts.device, actor_buffer);
                    bpt::set_all(ts.device, action, 0);
                    typename CONFIG::ENVIRONMENT_EVALUATION::State state;
                    bpt::initial_state(ts.device, ts.envs[0], state);
                    for(TI x_i = 0; x_i < decltype(image_0)::COLS; x_i++){
                        for(TI y_i = 0; y_i < decltype(image_0)::ROWS; y_i++){
                            T x = -1.0 + 2.0 / (decltype(image_0)::COLS - 1) * x_i;
                            T y = -1.0 + 2.0 / (decltype(image_0)::ROWS - 1) * y_i;
                            state.position[0] = x;
                            state.position[1] = y;
                            bpt::observe(ts.device, ts.env_eval, state, actor_input, ts.rng_eval);
                            bpt::evaluate(ts.device, ts.actor_critic.actor, actor_input, action, actor_buffer);
                            bpt::set(image_0, x_i, y_i, bpt::get(action, 0, 0));
                            bpt::set(image_1, x_i, y_i, bpt::get(action, 0, 1));
                            bpt::set(image_2, x_i, y_i, bpt::get(action, 0, 2));
                            bpt::set(image_3, x_i, y_i, bpt::get(action, 0, 3));
                        }
                    }
                    {
                        bpt::normalize(ts.device, image_0);
                        bpt::multiply_all(ts.device, image_0, 1/3.0);
                        bpt::add_image(ts.device, ts.device.logger, std::string("td3/position_action_0"), image_0);
                    }
                    {
                        bpt::normalize(ts.device, image_1);
                        bpt::multiply_all(ts.device, image_1, 1/3.0);
                        bpt::add_image(ts.device, ts.device.logger, std::string("td3/position_action_1"), image_1);
                    }
                    {
                        bpt::normalize(ts.device, image_2);
                        bpt::multiply_all(ts.device, image_2, 1/3.0);
                        bpt::add_image(ts.device, ts.device.logger, std::string("td3/position_action_2"), image_2);
                    }
                    {
                        bpt::normalize(ts.device, image_3);
                        bpt::multiply_all(ts.device, image_3, 1/3.0);
                        bpt::add_image(ts.device, ts.device.logger, std::string("td3/position_action_3"), image_3);
                    }
                    bpt::free(ts.device, image_0);
                    bpt::free(ts.device, image_1);
                    bpt::free(ts.device, image_2);
                    bpt::free(ts.device, image_3);
                    bpt::free(ts.device, actor_input);
                    bpt::free(ts.device, action);
                    bpt::free(ts.device, actor_buffer);
                }
            }
        }
        template <typename CONFIG>
        void step_validation(TrainingState<CONFIG>& ts){
            if(ts.step % 50000 == 0){
                bpt::reset(ts.device, ts.task, ts.rng_eval);
                bool completed = false;
                while(!completed){
                    completed = bpt::step(ts.device, ts.task, ts.actor_critic.actor, ts.validation_actor_buffers, ts.rng_eval);
                }
                bpt::analyse_log(ts.device, ts.task, typename TrainingState<CONFIG>::SPEC::METRICS{});
            }
        }
        template <typename CONFIG>
        void step(TrainingState<CONFIG>& ts){
            step_logger(ts);
            step_checkpoint(ts);
            step_validation(ts);
            step_curriculum(ts);
            bpt::rl::algorithms::td3::loop::step(ts);
            step_trajectory_collection(ts);
//            step_critic_reset(ts);
            step_network_stats(ts);
//            step_network_analysis(ts);
        }
        template <typename CONFIG>
        void destroy(TrainingState<CONFIG>& ts){
            bpt::rl::algorithms::td3::loop::destroy(ts);
            bpt::destroy(ts.device, ts.task);
            bpt::free(ts.device, ts.validation_actor_buffers);
        }
    }
}
