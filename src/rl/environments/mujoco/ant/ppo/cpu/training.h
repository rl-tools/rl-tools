#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/operations_cpu.h>
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
#include <rl_tools/nn_models/persist.h>
#endif
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
#include "../parameters.h"
#if defined(RL_TOOLS_BACKEND_ENABLE_MKL) && !defined(RL_TOOLS_BACKEND_DISABLE_BLAS)
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mkl.h>
#else
#if defined(RL_TOOLS_BACKEND_ENABLE_ACCELERATE) && !defined(RL_TOOLS_BACKEND_DISABLE_BLAS)
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
#include <rl_tools/rl/algorithms/ppo/operations_generic.h>
#include <rl_tools/rl/components/running_normalizer/operations_generic.h>
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
#include <rl_tools/rl/components/running_normalizer/persist.h>
#endif
#include <rl_tools/rl/utils/evaluation.h>

#include <filesystem>
#include <sstream>
#include <string>
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
#include <highfive/H5File.hpp>
#endif


#ifdef RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_TRAINING_TEST
#include <gtest/gtest.h>
#endif


namespace parameters = parameters_0;

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
using LOGGER = rlt::devices::logging::CPU_TENSORBOARD<>;
#else
using LOGGER = rlt::devices::logging::CPU;
#endif


using DEV_SPEC_SUPER = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
using TI = typename rlt::devices::DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints{
    struct HINTS: rlt::rl::components::on_policy_runner::ExecutionHints<TI, 16>{};
}
struct DEV_SPEC: DEV_SPEC_SUPER{
    using EXECUTION_HINTS = execution_hints::HINTS;
};

using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
using T = float;
using TI = typename DEVICE::index_t;


constexpr TI BASE_SEED = 600;
constexpr DEVICE::index_t NUM_RUNS = 1;
#if !defined(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_TRAINING_TEST)
constexpr DEVICE::index_t NUM_STEPS = 2500;
#else
constexpr DEVICE::index_t NUM_STEPS = 400;
#endif
constexpr TI ACTOR_CHECKPOINT_INTERVAL = 100000;
#if defined(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_DISABLE_EVALUATION)
constexpr bool ENABLE_EVALUATION = false;
#else
constexpr bool ENABLE_EVALUATION = true;
#endif
constexpr TI NUM_EVALUATION_EPISODES = 10;
constexpr TI EVALUATION_INTERVAL = 100000;
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
constexpr bool ACTOR_ENABLE_CHECKPOINTS = true;
#else
constexpr bool ACTOR_ENABLE_CHECKPOINTS = false;
#endif
constexpr bool ACTOR_OVERWRITE_CHECKPOINTS = false;
const std::string ACTOR_CHECKPOINT_DIRECTORY = "checkpoints/ppo_ant";

std::string sanitize_file_name(const std::string &input) {
    std::string output = input;

    const std::string invalid_chars = R"(<>:\"/\|?*)";

    std::replace_if(output.begin(), output.end(), [&invalid_chars](const char &c) {
        return invalid_chars.find(c) != std::string::npos;
    }, '_');

    return output;
}

void run(){
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
            run_name = sanitize_file_name(oss.str()) + "_" + run_name;
        }
        std::cout << "Run " << run_i << " of " << NUM_RUNS << " with seed " << seed << " and name " << run_name << std::endl;
        std::cout << "Checkpoints: " << (ACTOR_ENABLE_CHECKPOINTS ? "enabled" : "disabled") << std::endl;

        DEVICE::SPEC::LOGGING logger;
        DEVICE device;
        prl::ACTOR_OPTIMIZER actor_optimizer;
        prl::CRITIC_OPTIMIZER critic_optimizer;
        actor_optimizer.parameters.alpha = 3e-4;
        critic_optimizer.parameters.alpha = 3e-4 * 2;
        auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), seed);
        auto evaluation_rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 12);
        prl::PPO_TYPE ppo;
        prl::PPO_BUFFERS_TYPE ppo_buffers;
        prl::ON_POLICY_RUNNER_TYPE on_policy_runner;
        prl::ON_POLICY_RUNNER_DATASET_TYPE on_policy_runner_dataset;
        prl::ACTOR_EVAL_BUFFERS actor_eval_buffers;
        prl::PPO_TYPE::SPEC::ACTOR_TYPE::Buffer<1> actor_deterministic_eval_buffers;
        prl::ACTOR_BUFFERS actor_buffers;
        prl::CRITIC_BUFFERS critic_buffers;
        prl::CRITIC_BUFFERS_GAE critic_buffers_gae;
        rlt::rl::components::RunningNormalizer<rlt::rl::components::running_normalizer::Specification<T, TI, penv::ENVIRONMENT::OBSERVATION_DIM>> observation_normalizer;
        penv::ENVIRONMENT envs[prl::N_ENVIRONMENTS];
        penv::ENVIRONMENT evaluation_env;
        rlt::rl::environments::DummyUI ui;
        TI next_checkpoint_id = 0;
        TI next_evaluation_id = 0;

        rlt::malloc(device, ppo);
        rlt::malloc(device, ppo_buffers);
        rlt::malloc(device, on_policy_runner_dataset);
        rlt::malloc(device, on_policy_runner);
        rlt::malloc(device, actor_eval_buffers);
        rlt::malloc(device, actor_deterministic_eval_buffers);
        rlt::malloc(device, actor_buffers);
        rlt::malloc(device, critic_buffers);
        rlt::malloc(device, critic_buffers_gae);
        rlt::malloc(device, observation_normalizer);
        for(auto& env : envs){
            rlt::malloc(device, env);
        }
        rlt::malloc(device, evaluation_env);

        auto on_policy_runner_dataset_all_observations = prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS ? on_policy_runner_dataset.all_observations_normalized : on_policy_runner_dataset.all_observations;
        auto on_policy_runner_dataset_observations = prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS ? on_policy_runner_dataset.observations_normalized : on_policy_runner_dataset.observations;

        rlt::init(device);
        rlt::init(device, on_policy_runner, envs, rng);
        rlt::init(device, observation_normalizer);
        rlt::init(device, ppo, actor_optimizer, critic_optimizer, rng);
        rlt::init(device, device.logger);
        auto training_start = std::chrono::high_resolution_clock::now();
        if(prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS){
            for(TI observation_normalization_warmup_step_i = 0; observation_normalization_warmup_step_i < prl::OBSERVATION_NORMALIZATION_WARMUP_STEPS; observation_normalization_warmup_step_i++) {
                rlt::collect(device, on_policy_runner_dataset, on_policy_runner, ppo.actor, actor_eval_buffers, observation_normalizer.mean, observation_normalizer.std, rng);
                rlt::update(device, observation_normalizer, on_policy_runner_dataset.observations);
            }
            std::cout << "Observation means: " << std::endl;
            rlt::print(device, observation_normalizer.mean);
            std::cout << "Observation std: " << std::endl;
            rlt::print(device, observation_normalizer.std);
            rlt::init(device, on_policy_runner, envs, rng); // reinitializing the on_policy_runner to reset the episode counters
        }
        for(TI ppo_step_i = 0; ppo_step_i < NUM_STEPS; ppo_step_i++) {
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
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
                try{
                    auto actor_file = HighFive::File(actor_output_path.string(), HighFive::File::Overwrite);
                    rlt::save(device, ppo.actor, actor_file.createGroup("actor"));
                    rlt::save(device, observation_normalizer, actor_file.createGroup("observation_normalizer"));
                }
                catch(HighFive::Exception& e){
                    std::cout << "Error while saving actor: " << e.what() << std::endl;
                }
#endif
                next_checkpoint_id++;
            }
            if(ENABLE_EVALUATION && (on_policy_runner.step / EVALUATION_INTERVAL == next_evaluation_id)){
                auto result = rlt::evaluate(device, evaluation_env, ui, ppo.actor, rlt::rl::utils::evaluation::Specification<NUM_EVALUATION_EPISODES, prl::ON_POLICY_RUNNER_STEP_LIMIT>(), observation_normalizer.mean, observation_normalizer.std, actor_deterministic_eval_buffers, evaluation_rng);
                rlt::add_scalar(device, device.logger, "evaluation/return/mean", result.returns_mean);
                rlt::add_scalar(device, device.logger, "evaluation/return/std", result.returns_std);
                rlt::add_histogram(device, device.logger, "evaluation/return", result.returns, decltype(result)::N_EPISODES);
                std::cout << "Evaluation return mean: " << result.returns_mean << " (std: " << result.returns_std << ")" << std::endl;
#ifdef RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT_TRAINING_TEST
                if(on_policy_runner.step > 1000000){
                    ASSERT_GT(result.returns_mean + result.returns_std, 2000);
                }
#endif

                next_evaluation_id++;
            }
            rlt::set_step(device, device.logger, on_policy_runner.step);

            if(ppo_step_i % 1 == 0){
                std::chrono::duration<T> training_elapsed = std::chrono::high_resolution_clock::now() - training_start;
                std::cout << "PPO step: " << ppo_step_i << " environment step: " << on_policy_runner.step << " elapsed: " << training_elapsed.count() << "s" << std::endl;
                rlt::add_scalar(device, device.logger, "ppo/step", ppo_step_i);
                rlt::add_scalar(device, device.logger, "ppo/actor_learning_rate", actor_optimizer.parameters.alpha);
                rlt::add_scalar(device, device.logger, "ppo/critic_learning_rate", critic_optimizer.parameters.alpha);
            }
            for (TI action_i = 0; action_i < penv::ENVIRONMENT::ACTION_DIM; action_i++) {
                T action_log_std = rlt::get(ppo.actor.log_std.parameters, 0, action_i);
                std::stringstream topic;
                topic << "actor/action_std/" << action_i;
                rlt::add_scalar(device, device.logger, topic.str(), rlt::math::exp(DEVICE::SPEC::MATH(), action_log_std));
            }
            auto start = std::chrono::high_resolution_clock::now();
            rlt::collect(device, on_policy_runner_dataset, on_policy_runner, ppo.actor, actor_eval_buffers, observation_normalizer.mean, observation_normalizer.std, rng);
            if(prl::PPO_SPEC::PARAMETERS::NORMALIZE_OBSERVATIONS){
                rlt::update(device, observation_normalizer, on_policy_runner_dataset.observations);
                for(TI state_i = 0; state_i < penv::ENVIRONMENT::OBSERVATION_DIM; state_i++){
                    rlt::add_scalar(device, device.logger, std::string("observation_normalizer/mean_") + std::to_string(state_i), get(observation_normalizer.mean, 0, state_i));
                    rlt::add_scalar(device, device.logger, std::string("observation_normalizer/std") + std::to_string(state_i), get(observation_normalizer.std, 0, state_i));
                }
            }
            rlt::add_scalar(device, device.logger, "opr/observation/mean", rlt::mean(device, on_policy_runner_dataset.observations));
            rlt::add_scalar(device, device.logger, "opr/observation/std", rlt::std(device, on_policy_runner_dataset.observations));
            rlt::add_scalar(device, device.logger, "opr/action/mean", rlt::mean(device, on_policy_runner_dataset.actions));
            rlt::add_scalar(device, device.logger, "opr/action/std", rlt::std(device, on_policy_runner_dataset.actions));
            rlt::add_scalar(device, device.logger, "opr/rewards/mean", rlt::mean(device, on_policy_runner_dataset.rewards));
            rlt::add_scalar(device, device.logger, "opr/rewards/std", rlt::std(device, on_policy_runner_dataset.rewards));
            evaluate(device, ppo.critic, on_policy_runner_dataset_all_observations, on_policy_runner_dataset.all_values, critic_buffers_gae);
            rlt::estimate_generalized_advantages(device, on_policy_runner_dataset, prl::PPO_TYPE::SPEC::PARAMETERS{});
            rlt::train(device, ppo, on_policy_runner_dataset, actor_optimizer, critic_optimizer, ppo_buffers, actor_buffers, critic_buffers, rng);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
            std::cout << "Total: " << elapsed.count() << " s" << std::endl;
        }

        rlt::free(device, ppo);
        rlt::free(device, ppo_buffers);
        rlt::free(device, on_policy_runner_dataset);
        rlt::free(device, on_policy_runner);
        rlt::free(device, actor_eval_buffers);
        rlt::free(device, actor_buffers);
        rlt::free(device, critic_buffers);
        rlt::free(device, critic_buffers_gae);
        rlt::free(device, observation_normalizer);
        for(auto& env : envs){
            rlt::free(device, env);
        }
        rlt::free(device, evaluation_env);
        rlt::free(device, device.logger);
    }

}
