#define BACKPROP_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <backprop_tools/operations/cpu_mux.h>
// -------------- added for cuda training ----------------
#include <backprop_tools/nn/optimizers/adam/operations_cuda.h>
// -------------------------------------------------------
#include <backprop_tools/nn/operations_cpu_mux.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/nn_models/persist.h>
namespace bpt = backprop_tools;
// --------------- changed for cuda training -----------------
#include "../parameters_ppo.h"
// -------------------------------------------------------
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu_mkl.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
// -------------- added for cuda training ----------------
#include <backprop_tools/rl/components/on_policy_runner/operations_generic_extensions.h>
// -------------------------------------------------------
#include <backprop_tools/rl/algorithms/ppo/operations_generic.h>
// -------------- added for cuda training ----------------
#include <backprop_tools/rl/algorithms/ppo/operations_generic_extensions.h>
// -------------------------------------------------------
#include <backprop_tools/rl/components/running_normalizer/operations_generic.h>
#include <backprop_tools/rl/components/running_normalizer/persist.h>
#include <backprop_tools/rl/utils/evaluation.h>

#include <highfive/H5File.hpp>
#include <CLI/CLI.hpp>


namespace parameters = parameters_0;

#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_TESTS_RL_ENVIRONMENTS_MUJOCO_ANT_BENCHMARK)
using LOGGER = bpt::devices::logging::CPU_TENSORBOARD;
#else
using LOGGER = bpt::devices::logging::CPU;
#endif

using DEV_SPEC_SUPER = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, LOGGER>;
using TI = typename bpt::DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints{
    struct HINTS: bpt::rl::components::on_policy_runner::ExecutionHints<TI, 16>{};
}
struct DEV_SPEC: DEV_SPEC_SUPER{
    using EXECUTION_HINTS = execution_hints::HINTS;
};

using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
// -------------- added for cuda training ----------------
using DEVICE_GPU = bpt::DEVICE_FACTORY_GPU<bpt::devices::DefaultCUDASpecification>;
// -------------------------------------------------------
using T = float;
using TI = typename DEVICE::index_t;


constexpr TI BASE_SEED = 600;
constexpr TI ACTOR_CHECKPOINT_INTERVAL = 100000;
#if !defined(BACKPROP_TOOLS_TESTS_RL_ENVIRONMENTS_MUJOCO_ANT_BENCHMARK)
constexpr bool ENABLE_EVALUATION = true;
constexpr bool ACTOR_ENABLE_CHECKPOINTS = true;
#else
constexpr bool ENABLE_EVALUATION = false;
constexpr bool ACTOR_ENABLE_CHECKPOINTS = false;
#endif
constexpr TI NUM_EVALUATION_EPISODES = 10;
constexpr TI EVALUATION_INTERVAL = 100000;
constexpr bool ACTOR_OVERWRITE_CHECKPOINTS = false;

// --------------- changed for cuda training -----------------
int main(int argc, char** argv){
    std::string actor_checkpoints_dir_stub = "checkpoints";
    std::string logs_dir = "logs";
    TI job_seed = 0;
    TI num_runs = 1;
    {
        CLI::App app;
        app.add_option("--checkpoints", actor_checkpoints_dir_stub, "path to the checkpoint directory");
        app.add_option("--logs", logs_dir, "path to the logs directory");
        app.add_option("--seed", job_seed, "seed for this job");
        app.add_option("--runs", num_runs, "number of runs with different seeds");
        CLI11_PARSE(app, argc, argv);
    }
    std::string actor_checkpoints_dir = actor_checkpoints_dir_stub + "/ppo_ant";
    if (ACTOR_ENABLE_CHECKPOINTS){
        std::cout << "Saving actor checkpoints to: " << actor_checkpoints_dir << std::endl;
    }
// -------------------------------------------------------
    for(TI run_i = 0; run_i < num_runs; ++run_i){
        using penv = parameters::environment<double, TI>;
        using prl = parameters::rl<T, TI, penv::ENVIRONMENT>;
        // -------------- added for cuda training ----------------
        using ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE = bpt::rl::components::on_policy_runner::CollectionEvaluationBuffer<prl::ON_POLICY_RUNNER_SPEC>;
        using PPO_TRAINING_HYBRID_BUFFER_TYPE = bpt::rl::algorithms::ppo::TrainingBuffersHybrid<prl::PPO_SPEC>;
        // -------------------------------------------------------
        TI seed = BASE_SEED + job_seed * num_runs + run_i;
        std::stringstream run_name_ss;
        run_name_ss << "ppo_ant_" + std::to_string(seed);
        if(prl::PPO_SPEC::PARAMETERS::LEARN_ACTION_STD){
            run_name_ss << "_learn_astd";
        }
        if(prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS){
            run_name_ss << "_normobs";
        }
        if(prl::PPO_SPEC::PARAMETERS::ADAPTIVE_LEARNING_RATE){
            run_name_ss << "_adapt_lr";
        }
        if(prl::PPO_SPEC::PARAMETERS::NORMALIZE_ADVANTAGE){
            run_name_ss << "_norm_adv";
        }
        std::string run_name = run_name_ss.str();
        {
            auto now = std::chrono::system_clock::now();
            auto local_time = std::chrono::system_clock::to_time_t(now);
            std::tm* tm = std::localtime(&local_time);

            std::ostringstream oss;
            oss << std::put_time(tm, "%FT%T%z");
            run_name = oss.str() + "_" + run_name;
        }

        DEVICE::SPEC::LOGGING logger;
        DEVICE device;
        // -------------- added for cuda training ----------------
        DEVICE_GPU device_gpu;
        // -------------------------------------------------------
        prl::ACTOR_OPTIMIZER actor_optimizer;
        prl::CRITIC_OPTIMIZER critic_optimizer;
        auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), seed);
        auto evaluation_rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 12);
        prl::PPO_TYPE ppo;
        // -------------- added for cuda training ----------------
        prl::PPO_TYPE ppo_gpu;
        // -------------------------------------------------------
        prl::PPO_BUFFERS_TYPE ppo_buffers;
        prl::ON_POLICY_RUNNER_TYPE on_policy_runner;
        prl::ON_POLICY_RUNNER_DATASET_TYPE on_policy_runner_dataset;
        // -------------- added for cuda training ----------------
        ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE on_policy_runner_collection_eval_buffer_gpu, on_policy_runner_collection_eval_buffer_cpu;
        PPO_TRAINING_HYBRID_BUFFER_TYPE ppo_training_hybrid_buffer_cpu, ppo_training_hybrid_buffer_gpu;
        bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, decltype(on_policy_runner_dataset.data)::ROWS, prl::PPO_SPEC::ENVIRONMENT::OBSERVATION_DIM>> gae_all_observations;
        bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, decltype(on_policy_runner_dataset.data)::ROWS, 1>> gae_all_values;
        // -------------------------------------------------------
        // -------------- replaced for cuda training ----------------
        prl::ACTOR_EVAL_BUFFERS actor_eval_buffers, actor_eval_buffers_gpu;
        // ----------------------------------------------------------
        prl::ACTOR_BUFFERS actor_buffers;
        prl::CRITIC_BUFFERS critic_buffers;
        prl::CRITIC_BUFFERS_GAE critic_buffers_gae;
        bpt::rl::components::RunningNormalizer<bpt::rl::components::running_normalizer::Specification<T, TI, penv::ENVIRONMENT::OBSERVATION_DIM>> observation_normalizer;
        penv::ENVIRONMENT envs[prl::N_ENVIRONMENTS];
        penv::ENVIRONMENT evaluation_env;
        bool ui = false;
        TI next_checkpoint_id = 0;
        TI next_evaluation_id = 0;

        // -------------- added for cuda training ----------------
        bpt::init(device_gpu);
        // -------------------------------------------------------
        bpt::malloc(device, ppo);
        bpt::malloc(device, ppo_buffers);
        bpt::malloc(device, on_policy_runner_dataset);
        // -------------- added for cuda training ----------------
        bpt::malloc(device, on_policy_runner_collection_eval_buffer_cpu);
        bpt::malloc(device, ppo_training_hybrid_buffer_cpu);
        // -------------------------------------------------------
        bpt::malloc(device, on_policy_runner);
        bpt::malloc(device, actor_eval_buffers);
        // ------------- removed for cuda training ---------------
//        bpt::malloc(device, actor_buffers);
//        bpt::malloc(device, critic_buffers);
//        bpt::malloc(device, critic_buffers_gae);
        // -------------------------------------------------------
        bpt::malloc(device, observation_normalizer);
        for(auto& env : envs){
            bpt::malloc(device, env);
        }
        bpt::malloc(device, evaluation_env);
        // -------------- added for cuda training ----------------
        bpt::malloc(device_gpu, actor_buffers);
        bpt::malloc(device_gpu, critic_buffers);
        bpt::malloc(device_gpu, critic_buffers_gae);
        bpt::malloc(device_gpu, ppo_gpu);
        bpt::malloc(device_gpu, on_policy_runner_collection_eval_buffer_gpu);
        bpt::malloc(device_gpu, ppo_training_hybrid_buffer_gpu);
        bpt::malloc(device_gpu, actor_eval_buffers_gpu);
        bpt::malloc(device_gpu, gae_all_observations);
        bpt::malloc(device_gpu, gae_all_values);
        // -------------------------------------------------------

        auto on_policy_runner_dataset_all_observations = prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS ? on_policy_runner_dataset.all_observations_normalized : on_policy_runner_dataset.all_observations;
        auto on_policy_runner_dataset_observations = prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS ? on_policy_runner_dataset.observations_normalized : on_policy_runner_dataset.observations;

        bpt::init(device, on_policy_runner, envs, rng);
        bpt::init(device, observation_normalizer);
        bpt::init(device, ppo, actor_optimizer, critic_optimizer, rng);
        // -------------- added for cuda training ----------------
        bpt::copy(device_gpu, device, ppo_gpu, ppo);
        // -------------------------------------------------------
        device.logger = &logger;
        bpt::construct(device, device.logger, logs_dir, run_name);
        auto training_start = std::chrono::high_resolution_clock::now();
        if(prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS){
            for(TI observation_normalization_warmup_step_i = 0; observation_normalization_warmup_step_i < prl::OBSERVATION_NORMALIZATION_WARMUP_STEPS; observation_normalization_warmup_step_i++) {
                bpt::collect(device, on_policy_runner_dataset, on_policy_runner, ppo.actor, actor_eval_buffers, observation_normalizer.mean, observation_normalizer.std, rng);
                bpt::update(device, observation_normalizer, on_policy_runner_dataset.observations);
            }
            std::cout << "Observation means: " << std::endl;
            bpt::print(device, observation_normalizer.mean);
            std::cout << "Observation std: " << std::endl;
            bpt::print(device, observation_normalizer.std);
            bpt::init(device, on_policy_runner, envs, rng); // reinitializing the on_policy_runner to reset the episode counters
        }
        for(TI ppo_step_i = 0; ppo_step_i < 2500; ppo_step_i++) {
            // -------------- added for cuda training ----------------
            bpt::copy(device, device_gpu, ppo, ppo_gpu);
            // -------------------------------------------------------
            if(ACTOR_ENABLE_CHECKPOINTS && (on_policy_runner.step / ACTOR_CHECKPOINT_INTERVAL == next_checkpoint_id)){
                std::filesystem::path actor_output_dir = std::filesystem::path(actor_checkpoints_dir) / run_name;
                try {
                    std::filesystem::create_directories(actor_output_dir);
                }
                catch (std::exception& e) {
                }
                std::string checkpoint_name = "latest.h5";
                if(!ACTOR_OVERWRITE_CHECKPOINTS){
                    std::stringstream checkpoint_name_ss;
                    checkpoint_name_ss << "actor_" << std::setw(15) << std::setfill('0') << next_checkpoint_id << "_" << std::setw(15) << std::setfill('0') << on_policy_runner.step << ".h5";
                    checkpoint_name = checkpoint_name_ss.str();
                }
                std::filesystem::path actor_output_path = actor_output_dir / checkpoint_name;
                try{
                    auto actor_file = HighFive::File(actor_output_path, HighFive::File::Overwrite);
                    bpt::save(device, ppo.actor, actor_file.createGroup("actor"));
                    bpt::save(device, observation_normalizer, actor_file.createGroup("observation_normalizer"));
                }
                catch(HighFive::Exception& e){
                    std::cout << "Error while saving actor: " << e.what() << std::endl;
                }
                next_checkpoint_id++;
            }
            if(ENABLE_EVALUATION && (on_policy_runner.step / EVALUATION_INTERVAL == next_evaluation_id)){
                auto result = bpt::evaluate(device, evaluation_env, ui, ppo.actor, bpt::rl::utils::evaluation::Specification<NUM_EVALUATION_EPISODES, prl::ON_POLICY_RUNNER_STEP_LIMIT>(), observation_normalizer.mean, observation_normalizer.std, evaluation_rng);
//                bpt::add_scalar(device, device.logger, "evaluation/return/mean", result.mean);
//                bpt::add_scalar(device, device.logger, "evaluation/return/std", result.std);
                bpt::add_histogram(device, device.logger, "evaluation/return", result.returns, decltype(result)::N_EPISODES);
                std::cout << "Evaluation return mean: " << result.mean << " (std: " << result.std << ")" << std::endl;
                next_evaluation_id++;
            }
            bpt::set_step(device, device.logger, on_policy_runner.step);

//            for (TI action_i = 0; action_i < penv::ENVIRONMENT::ACTION_DIM; action_i++) {
//                T action_log_std = bpt::get(ppo.actor.log_std.parameters, 0, action_i);
//                std::stringstream topic;
//                topic << "actor/action_std/" << action_i;
//                bpt::add_scalar(device, device.logger, topic.str(), bpt::math::exp(DEVICE::SPEC::MATH(), action_log_std));
//            }
//            auto start = std::chrono::high_resolution_clock::now();
            auto training_step_start = std::chrono::high_resolution_clock::now();
            {
//                auto start = std::chrono::high_resolution_clock::now();
                // -------------- replaced for cuda training ----------------
                bpt::collect_hybrid(device, device_gpu, on_policy_runner_dataset, on_policy_runner, ppo.actor, ppo_gpu.actor, actor_eval_buffers_gpu, on_policy_runner_collection_eval_buffer_cpu, on_policy_runner_collection_eval_buffer_gpu, observation_normalizer.mean, observation_normalizer.std, rng);
                // ----------------------------------------------------------
                if(prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS){
                    bpt::update(device, observation_normalizer, on_policy_runner_dataset.observations);
                    for(TI state_i = 0; state_i < penv::ENVIRONMENT::OBSERVATION_DIM; state_i++){
//                        bpt::add_scalar(device, device.logger, std::string("observation_normalizer/mean_") + std::to_string(state_i), get(observation_normalizer.mean, 0, state_i));
//                        bpt::add_scalar(device, device.logger, std::string("observation_normalizer/std") + std::to_string(state_i), get(observation_normalizer.std, 0, state_i));
                    }
                }
//                bpt::add_scalar(device, device.logger, "opr/observation/mean", bpt::mean(device, on_policy_runner_dataset.observations));
//                bpt::add_scalar(device, device.logger, "opr/observation/std", bpt::std(device, on_policy_runner_dataset.observations));
//                bpt::add_scalar(device, device.logger, "opr/action/mean", bpt::mean(device, on_policy_runner_dataset.actions));
//                bpt::add_scalar(device, device.logger, "opr/action/std", bpt::std(device, on_policy_runner_dataset.actions));
//                bpt::add_scalar(device, device.logger, "opr/rewards/mean", bpt::mean(device, on_policy_runner_dataset.rewards));
//                bpt::add_scalar(device, device.logger, "opr/rewards/std", bpt::std(device, on_policy_runner_dataset.rewards));
//                auto end = std::chrono::high_resolution_clock::now();
//                std::chrono::duration<T> elapsed = end - start;
//                std::cout << "Rollout: " << elapsed.count() << " s" << std::endl;
            }
            {
//                auto start = std::chrono::high_resolution_clock::now();
                // -------------- replaced for cuda training ----------------
                copy(device_gpu, device, gae_all_observations, on_policy_runner_dataset_all_observations);
                evaluate(device_gpu, ppo_gpu.critic, gae_all_observations, gae_all_values, critic_buffers_gae);
                copy(device, device_gpu, on_policy_runner_dataset.all_values, gae_all_values);
                // ----------------------------------------------------------
                bpt::estimate_generalized_advantages(device, on_policy_runner_dataset, prl::PPO_TYPE::SPEC::PARAMETERS{});
//                auto end = std::chrono::high_resolution_clock::now();
//                std::chrono::duration<T> elapsed = end - start;
//                std::cout << "GAE: " << elapsed.count() << " s" << std::endl;
            }
            {
//                auto start = std::chrono::high_resolution_clock::now();
                // -------------- replaced for cuda training ----------------
                bpt::train_hybrid(device, device_gpu, ppo, ppo_gpu, on_policy_runner_dataset, actor_optimizer, critic_optimizer, ppo_buffers, ppo_training_hybrid_buffer_gpu, actor_buffers, critic_buffers, rng);
                // ----------------------------------------------------------
//                auto end = std::chrono::high_resolution_clock::now();
//                std::chrono::duration<T> elapsed = end - start;
//                std::cout << "Train: " << elapsed.count() << " s" << std::endl;
            }
//            auto end = std::chrono::high_resolution_clock::now();
//            std::chrono::duration<T> elapsed = end - start;
//            std::cout << "Total: " << elapsed.count() << " s" << std::endl;
            if(ppo_step_i % 1 == 0){
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<T> training_elapsed = now - training_start;
                std::chrono::duration<T> step_elapsed = now - training_step_start;
                T steps_per_second_lifetime = on_policy_runner.step / training_elapsed.count();
                T steps_per_second_current = prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS * prl::ON_POLICY_RUNNER_STEPS_PER_ENV / step_elapsed.count();
                std::cout << "PPO step: " << std::setw(10) << ppo_step_i << " elapsed: " << std::setw(10) << std::setprecision(2) << training_elapsed.count() << "s (lifetime: " << std::setw(10) << std::setprecision(2) << steps_per_second_lifetime << " steps/s, current: " << std::setw(10) << std::setprecision(2) << steps_per_second_current << " steps/s)" << std::endl;
//                bpt::add_scalar(device, device.logger, "ppo/step", ppo_step_i);
//                bpt::add_scalar(device, device.logger, "ppo/actor_learning_rate", actor_optimizer.alpha);
//                bpt::add_scalar(device, device.logger, "ppo/critic_learning_rate", critic_optimizer.alpha);
            }
        }

        bpt::free(device, ppo);
        bpt::free(device, ppo_buffers);
        bpt::free(device, on_policy_runner_dataset);
        // -------------- added for cuda training ----------------
        bpt::free(device, on_policy_runner_collection_eval_buffer_cpu);
        bpt::free(device, ppo_training_hybrid_buffer_cpu);
        // -------------------------------------------------------
        bpt::free(device, on_policy_runner);
        bpt::free(device, actor_eval_buffers);
        // ------------- removed for cuda training ---------------
//        bpt::free(device, actor_buffers);
//        bpt::free(device, critic_buffers);
//        bpt::free(device, critic_buffers_gae);
        // -------------------------------------------------------
        bpt::free(device, observation_normalizer);
        for(auto& env : envs){
            bpt::free(device, env);
        }
        bpt::free(device, evaluation_env);
        // -------------- added for cuda training ----------------
        bpt::free(device_gpu, actor_buffers);
        bpt::free(device_gpu, critic_buffers);
        bpt::free(device_gpu, critic_buffers_gae);
        bpt::free(device_gpu, ppo_gpu);
        bpt::free(device_gpu, on_policy_runner_collection_eval_buffer_gpu);
        bpt::free(device_gpu, ppo_training_hybrid_buffer_gpu);
        bpt::free(device_gpu, actor_eval_buffers_gpu);
        bpt::free(device_gpu, gae_all_observations);
        bpt::free(device_gpu, gae_all_values);
        // -------------------------------------------------------
    }

    return 0;
}
