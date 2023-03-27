#define LAYER_IN_C_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <layer_in_c/operations/cpu_mux.h>
#include <layer_in_c/nn/optimizers/adam/operations_cuda.h>
#include <layer_in_c/nn/operations_cpu_mux.h>
#include <layer_in_c/nn_models/operations_cpu.h>
#include <layer_in_c/nn_models/persist.h>
namespace lic = layer_in_c;
#include "../parameters_ppo.h"
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu_mkl.h>
#include <layer_in_c/rl/components/on_policy_runner/operations_generic_extensions.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
#include <layer_in_c/rl/algorithms/ppo/operations_generic.h>
#include <layer_in_c/rl/algorithms/ppo/operations_generic_extensions.h>
#include <layer_in_c/rl/utils/evaluation.h>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>


namespace parameters = parameters_0;

using LOGGER = lic::devices::logging::CPU_TENSORBOARD;

using DEV_SPEC_SUPER = lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, LOGGER>;
using TI = typename lic::DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints{
    struct HINTS: lic::rl::components::on_policy_runner::ExecutionHints<TI, 16>{};
}
struct DEV_SPEC: DEV_SPEC_SUPER{
    using EXECUTION_HINTS = execution_hints::HINTS;
};
using DEVICE = lic::DEVICE_FACTORY<DEV_SPEC>;
using DEVICE_GPU = lic::DEVICE_FACTORY_GPU<lic::devices::DefaultCUDASpecification>;


using DEVICE = lic::DEVICE_FACTORY<DEV_SPEC>;
using T = float;
using TI = typename DEVICE::index_t;


constexpr TI ACTOR_CHECKPOINT_INTERVAL = 150;
constexpr bool ENABLE_EVALUATION = false;
constexpr TI EVALUATION_INTERVAL = 150;
constexpr bool ACTOR_ENABLE_CHECKPOINTS = false;
constexpr bool ACTOR_OVERWRITE_CHECKPOINTS = false;
const std::string ACTOR_CHECKPOINT_DIRECTORY = "checkpoints/ppo_ant";

TEST(LAYER_IN_C_RL_ENVIRONMENTS_MUJOCO_ANT, TRAINING_PPO_CUDA){
    std::string run_name;
    {
        auto now = std::chrono::system_clock::now();
        auto local_time = std::chrono::system_clock::to_time_t(now);
        std::tm* tm = std::localtime(&local_time);

        std::ostringstream oss;
        oss << std::put_time(tm, "%FT%T%z");
        run_name = oss.str();
    }
    using penv = parameters::environment<double, TI>;
    using prl = parameters::rl<T, TI, penv::ENVIRONMENT>;
    using ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE = lic::rl::components::on_policy_runner::CollectionEvaluationBuffer<prl::ON_POLICY_RUNNER_SPEC>;
    using PPO_TRAINING_HYBRID_BUFFER_TYPE = lic::rl::algorithms::ppo::TrainingBuffersHybrid<prl::PPO_SPEC>;

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    DEVICE_GPU device_gpu;
    lic::init(device_gpu);
    prl::OPTIMIZER optimizer;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 13);
    auto evaluation_rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 12);
    prl::PPO_TYPE ppo, ppo_cpu;
    prl::ACTOR_TYPE_INFERENCE actor_gpu;
    prl::PPO_BUFFERS_TYPE ppo_buffers;
    prl::ON_POLICY_RUNNER_TYPE on_policy_runner;
    prl::ON_POLICY_RUNNER_DATASET_TYPE dataset;
    ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE on_policy_runner_collection_eval_buffer_gpu, on_policy_runner_collection_eval_buffer_cpu;
    PPO_TRAINING_HYBRID_BUFFER_TYPE ppo_training_hybrid_buffer_cpu, ppo_training_hybrid_buffer_gpu;
    prl::ACTOR_EVAL_BUFFERS actor_eval_buffers, actor_eval_buffers_gpu;
    prl::ACTOR_BUFFERS actor_buffers;
    prl::CRITIC_BUFFERS critic_buffers;
    prl::CRITIC_BUFFERS_GAE critic_buffers_gae;
    lic::Matrix<lic::matrix::Specification<T, TI, decltype(dataset.data)::ROWS, prl::PPO_SPEC::ENVIRONMENT::OBSERVATION_DIM>> gae_all_observations;
    lic::Matrix<lic::matrix::Specification<T, TI, decltype(dataset.data)::ROWS, 1>> gae_all_values;
    penv::ENVIRONMENT envs[prl::N_ENVIRONMENTS];
    penv::ENVIRONMENT evaluation_env;
    bool ui = false;

    lic::malloc(device, ppo_cpu);
    lic::malloc(device, ppo_buffers);
    lic::malloc(device, dataset);
    lic::malloc(device, on_policy_runner_collection_eval_buffer_cpu);
    lic::malloc(device, ppo_training_hybrid_buffer_cpu);
    lic::malloc(device, on_policy_runner);
    lic::malloc(device, actor_eval_buffers);
    lic::malloc(device_gpu, actor_buffers);
    lic::malloc(device_gpu, critic_buffers);
    lic::malloc(device_gpu, critic_buffers_gae);
    lic::malloc(device_gpu, ppo);
    lic::malloc(device_gpu, actor_gpu);
    lic::malloc(device_gpu, on_policy_runner_collection_eval_buffer_gpu);
    lic::malloc(device_gpu, ppo_training_hybrid_buffer_gpu);
    lic::malloc(device_gpu, actor_eval_buffers_gpu);
    lic::malloc(device_gpu, gae_all_observations);
    lic::malloc(device_gpu, gae_all_values);
    for(auto& env : envs){
        lic::malloc(device, env);
    }
    lic::malloc(device, evaluation_env);

    lic::init(device, on_policy_runner, envs, rng);
    lic::init(device, ppo_cpu, optimizer, rng);
    lic::copy(device_gpu, device, ppo, ppo_cpu);
    device.logger = &logger;
    lic::construct(device, device.logger);
    auto training_start = std::chrono::high_resolution_clock::now();

    for(TI ppo_step_i = 0; ppo_step_i < 10000; ppo_step_i++) {
        device.logger->step = on_policy_runner.step;

        if(ppo_step_i % 1 == 0){
            std::chrono::duration<T> training_elapsed = std::chrono::high_resolution_clock::now() - training_start;
            T steps_per_second = on_policy_runner.step / training_elapsed.count();
            std::cout << "PPO step: " << ppo_step_i << " elapsed: " << training_elapsed.count() << "s (" << steps_per_second << " steps/s)" << std::endl;
            lic::add_scalar(device, device.logger, "ppo/step", ppo_step_i);
        }
        lic::copy(device, device_gpu, ppo_cpu, ppo);
        for (TI action_i = 0; action_i < penv::ENVIRONMENT::ACTION_DIM; action_i++) {
            T action_log_std = lic::get(ppo_cpu.actor.log_std.parameters, 0, action_i);
            std::stringstream topic;
            topic << "actor/action_std/" << action_i;
            lic::add_scalar(device, device.logger, topic.str(), lic::math::exp(DEVICE::SPEC::MATH(), action_log_std));
        }
        auto start = std::chrono::high_resolution_clock::now();
        {
            auto start = std::chrono::high_resolution_clock::now();
//            lic::collect(device, dataset, on_policy_runner, ppo.actor, actor_eval_buffers, rng);
//            lic::copy(device_gpu, device, actor_gpu, ppo.actor);
            lic::collect_hybrid(device, device_gpu, dataset, on_policy_runner, ppo_cpu.actor, ppo.actor, actor_eval_buffers_gpu, on_policy_runner_collection_eval_buffer_cpu, on_policy_runner_collection_eval_buffer_gpu, rng);

            lic::add_scalar(device, device.logger, "opr/observation/mean", lic::mean(device, dataset.observations));
            lic::add_scalar(device, device.logger, "opr/observation/std", lic::std(device, dataset.observations));
            lic::add_scalar(device, device.logger, "opr/action/mean", lic::mean(device, dataset.actions));
            lic::add_scalar(device, device.logger, "opr/action/std", lic::std(device, dataset.actions));
            lic::add_scalar(device, device.logger, "opr/rewards/mean", lic::mean(device, dataset.rewards));
            lic::add_scalar(device, device.logger, "opr/rewards/std", lic::std(device, dataset.rewards));
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
            std::cout << "Rollout: " << elapsed.count() << " s" << std::endl;
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            copy(device_gpu, device, gae_all_observations, dataset.all_observations);
            evaluate(device_gpu, ppo.critic, gae_all_observations, gae_all_values, critic_buffers_gae);
            lic::check_status(device_gpu);
            copy(device, device_gpu, dataset.all_values, gae_all_values);
            lic::estimate_generalized_advantages(device, dataset, prl::PPO_TYPE::SPEC::PARAMETERS{});
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
            std::cout << "GAE: " << elapsed.count() << " s" << std::endl;
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            lic::train_hybrid(device, device_gpu, ppo_cpu, ppo, dataset, optimizer, ppo_buffers, ppo_training_hybrid_buffer_gpu, actor_buffers, critic_buffers, rng);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
            std::cout << "Train: " << elapsed.count() << " s" << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> elapsed = end - start;
        std::cout << "Total: " << elapsed.count() << " s" << std::endl;
        if(ENABLE_EVALUATION && (ppo_step_i % EVALUATION_INTERVAL == 0)){
            auto result = lic::evaluate(device, evaluation_env, ui, ppo.actor, lic::rl::utils::evaluation::Specification<10, prl::ON_POLICY_RUNNER_STEP_LIMIT>(), evaluation_rng);
            lic::add_scalar(device, device.logger, "evaluation/return/mean", result.mean);
            lic::add_scalar(device, device.logger, "evaluation/return/std", result.std);
            lic::add_histogram(device, device.logger, "evaluation/return", result.returns, decltype(result)::N_EPISODES);
            std::cout << "Evaluation return mean: " << result.mean << " (std: " << result.std << ")" << std::endl;

//            if(step_i > 250000){
//                ASSERT_GT(mean_return, 1000);
//            }
        }
        if(ACTOR_ENABLE_CHECKPOINTS && ppo_step_i % ACTOR_CHECKPOINT_INTERVAL == 0){
            std::filesystem::path actor_output_dir = std::filesystem::path(ACTOR_CHECKPOINT_DIRECTORY) / run_name;
            try {
                std::filesystem::create_directories(actor_output_dir);
            }
            catch (std::exception& e) {
            }
            std::string checkpoint_name = "latest.h5";
            if(!ACTOR_OVERWRITE_CHECKPOINTS){
                std::stringstream checkpoint_name_ss;
                checkpoint_name_ss << "actor_" << std::setw(15) << std::setfill('0') << ppo_step_i << ".h5";
                checkpoint_name = checkpoint_name_ss.str();
            }
            std::filesystem::path actor_output_path = actor_output_dir / checkpoint_name;
            try{
                auto actor_file = HighFive::File(actor_output_path, HighFive::File::Overwrite);
                lic::save(device, ppo.actor, actor_file.createGroup("actor"));
            }
            catch(HighFive::Exception& e){
                std::cout << "Error while saving actor: " << e.what() << std::endl;
            }
        }
    }

    for(auto& env : envs){
        lic::malloc(device, env);
    }
    lic::free(device, ppo);
    lic::free(device, ppo_buffers);
    lic::free(device, dataset);
    lic::free(device, on_policy_runner);
    lic::free(device, actor_eval_buffers);
    lic::free(device, actor_buffers);
    lic::free(device, critic_buffers);

}
