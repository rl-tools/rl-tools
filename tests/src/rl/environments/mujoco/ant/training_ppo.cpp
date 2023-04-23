#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/nn_models/persist.h>
namespace bpt = backprop_tools;
#include "parameters_ppo.h"
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu_mkl.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
#include <backprop_tools/rl/algorithms/ppo/operations_generic.h>
#include <backprop_tools/rl/components/running_normalizer/operations_generic.h>
#include <backprop_tools/rl/components/running_normalizer/persist.h>
#include <backprop_tools/rl/utils/evaluation.h>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>


namespace parameters = parameters_0;

using LOGGER = bpt::devices::logging::CPU;
//using LOGGER = bpt::devices::logging::CPU_TENSORBOARD;

using DEV_SPEC_SUPER = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, LOGGER>;
using TI = typename bpt::DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints{
    struct HINTS: bpt::rl::components::on_policy_runner::ExecutionHints<TI, 16>{};
}
struct DEV_SPEC: DEV_SPEC_SUPER{
    using EXECUTION_HINTS = execution_hints::HINTS;
};

using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
using T = float;
using TI = typename DEVICE::index_t;


constexpr TI BASE_SEED = 600;
constexpr TI NUM_RUNS = 100;
constexpr TI ACTOR_CHECKPOINT_INTERVAL = 100000;
constexpr bool ENABLE_EVALUATION = false;
constexpr TI NUM_EVALUATION_EPISODES = 10;
constexpr TI EVALUATION_INTERVAL = 100000;
constexpr bool ACTOR_ENABLE_CHECKPOINTS = false;
constexpr bool ACTOR_OVERWRITE_CHECKPOINTS = false;
const std::string ACTOR_CHECKPOINT_DIRECTORY = "checkpoints/ppo_ant";

TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, TRAINING_PPO){
    for(TI run_i = 0; run_i < NUM_RUNS; ++run_i){
        using penv = parameters::environment<double, TI>;
        using prl = parameters::rl<T, TI, penv::ENVIRONMENT>;
        TI seed = BASE_SEED + run_i;
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
        prl::ACTOR_OPTIMIZER actor_optimizer;
        prl::CRITIC_OPTIMIZER critic_optimizer;
        auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), seed);
        auto evaluation_rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 12);
        prl::PPO_TYPE ppo;
        prl::PPO_BUFFERS_TYPE ppo_buffers;
        prl::ON_POLICY_RUNNER_TYPE on_policy_runner;
        prl::ON_POLICY_RUNNER_DATASET_TYPE on_policy_runner_dataset;
        prl::ACTOR_EVAL_BUFFERS actor_eval_buffers;
        prl::ACTOR_BUFFERS actor_buffers;
        prl::CRITIC_BUFFERS critic_buffers;
        prl::CRITIC_BUFFERS_GAE critic_buffers_gae;
        bpt::rl::components::RunningNormalizer<bpt::rl::components::running_normalizer::Specification<T, TI, penv::ENVIRONMENT::OBSERVATION_DIM>> observation_normalizer;
        penv::ENVIRONMENT envs[prl::N_ENVIRONMENTS];
        penv::ENVIRONMENT evaluation_env;
        bool ui = false;
        TI next_checkpoint_id = 0;
        TI next_evaluation_id = 0;

        bpt::malloc(device, ppo);
        bpt::malloc(device, ppo_buffers);
        bpt::malloc(device, on_policy_runner_dataset);
        bpt::malloc(device, on_policy_runner);
        bpt::malloc(device, actor_eval_buffers);
        bpt::malloc(device, actor_buffers);
        bpt::malloc(device, critic_buffers);
        bpt::malloc(device, critic_buffers_gae);
        bpt::malloc(device, observation_normalizer);
        for(auto& env : envs){
            bpt::malloc(device, env);
        }
        bpt::malloc(device, evaluation_env);

        auto on_policy_runner_dataset_all_observations = prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS ? on_policy_runner_dataset.all_observations_normalized : on_policy_runner_dataset.all_observations;
        auto on_policy_runner_dataset_observations = prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS ? on_policy_runner_dataset.observations_normalized : on_policy_runner_dataset.observations;

        bpt::init(device, on_policy_runner, envs, rng);
        bpt::init(device, observation_normalizer);
        bpt::init(device, ppo, actor_optimizer, critic_optimizer, rng);
        device.logger = &logger;
        bpt::construct(device, device.logger, std::string("logs"), run_name);
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
            if(ACTOR_ENABLE_CHECKPOINTS && (on_policy_runner.step / ACTOR_CHECKPOINT_INTERVAL == next_checkpoint_id)){
                std::filesystem::path actor_output_dir = std::filesystem::path(ACTOR_CHECKPOINT_DIRECTORY) / run_name;
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
                bpt::add_scalar(device, device.logger, "evaluation/return/mean", result.mean);
                bpt::add_scalar(device, device.logger, "evaluation/return/std", result.std);
                bpt::add_histogram(device, device.logger, "evaluation/return", result.returns, decltype(result)::N_EPISODES);
                std::cout << "Evaluation return mean: " << result.mean << " (std: " << result.std << ")" << std::endl;
                next_evaluation_id++;
            }
            bpt::set_step(device, device.logger, on_policy_runner.step);

            if(ppo_step_i % 1 == 0){
                std::chrono::duration<T> training_elapsed = std::chrono::high_resolution_clock::now() - training_start;
                std::cout << "PPO step: " << ppo_step_i << " elapsed: " << training_elapsed.count() << "s" << std::endl;
                bpt::add_scalar(device, device.logger, "ppo/step", ppo_step_i);
                bpt::add_scalar(device, device.logger, "ppo/actor_learning_rate", actor_optimizer.alpha);
                bpt::add_scalar(device, device.logger, "ppo/critic_learning_rate", critic_optimizer.alpha);
            }
            for (TI action_i = 0; action_i < penv::ENVIRONMENT::ACTION_DIM; action_i++) {
                T action_log_std = bpt::get(ppo.actor.log_std.parameters, 0, action_i);
                std::stringstream topic;
                topic << "actor/action_std/" << action_i;
                bpt::add_scalar(device, device.logger, topic.str(), bpt::math::exp(DEVICE::SPEC::MATH(), action_log_std));
            }
            auto start = std::chrono::high_resolution_clock::now();
            bpt::collect(device, on_policy_runner_dataset, on_policy_runner, ppo.actor, actor_eval_buffers, observation_normalizer.mean, observation_normalizer.std, rng);
            if(prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS){
                bpt::update(device, observation_normalizer, on_policy_runner_dataset.observations);
                for(TI state_i = 0; state_i < penv::ENVIRONMENT::OBSERVATION_DIM; state_i++){
                    bpt::add_scalar(device, device.logger, std::string("observation_normalizer/mean_") + std::to_string(state_i), get(observation_normalizer.mean, 0, state_i));
                    bpt::add_scalar(device, device.logger, std::string("observation_normalizer/std") + std::to_string(state_i), get(observation_normalizer.std, 0, state_i));
                }
            }
            bpt::add_scalar(device, device.logger, "opr/observation/mean", bpt::mean(device, on_policy_runner_dataset.observations));
            bpt::add_scalar(device, device.logger, "opr/observation/std", bpt::std(device, on_policy_runner_dataset.observations));
            bpt::add_scalar(device, device.logger, "opr/action/mean", bpt::mean(device, on_policy_runner_dataset.actions));
            bpt::add_scalar(device, device.logger, "opr/action/std", bpt::std(device, on_policy_runner_dataset.actions));
            bpt::add_scalar(device, device.logger, "opr/rewards/mean", bpt::mean(device, on_policy_runner_dataset.rewards));
            bpt::add_scalar(device, device.logger, "opr/rewards/std", bpt::std(device, on_policy_runner_dataset.rewards));
            evaluate(device, ppo.critic, on_policy_runner_dataset_all_observations, on_policy_runner_dataset.all_values, critic_buffers_gae);
            bpt::estimate_generalized_advantages(device, on_policy_runner_dataset, prl::PPO_TYPE::SPEC::PARAMETERS{});
            bpt::train(device, ppo, on_policy_runner_dataset, actor_optimizer, critic_optimizer, ppo_buffers, actor_buffers, critic_buffers, rng);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
            std::cout << "Total: " << elapsed.count() << " s" << std::endl;
        }

        bpt::free(device, ppo);
        bpt::free(device, ppo_buffers);
        bpt::free(device, on_policy_runner_dataset);
        bpt::free(device, on_policy_runner);
        bpt::free(device, actor_eval_buffers);
        bpt::free(device, actor_buffers);
        bpt::free(device, critic_buffers);
        bpt::free(device, critic_buffers_gae);
        bpt::free(device, observation_normalizer);
        for(auto& env : envs){
            bpt::free(device, env);
        }
        bpt::free(device, evaluation_env);
    }

}
