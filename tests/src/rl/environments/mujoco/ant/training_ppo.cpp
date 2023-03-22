#include <layer_in_c/operations/cpu_mux.h>
#include <layer_in_c/nn/operations_cpu_mux.h>
#include <layer_in_c/nn_models/operations_cpu.h>
#include <layer_in_c/nn_models/persist.h>
namespace lic = layer_in_c;
#include "parameters_ppo.h"
#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu_mkl.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
#include <layer_in_c/rl/algorithms/ppo/operations_generic.h>
#include <layer_in_c/rl/utils/evaluation.h>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>


namespace parameters = parameters_0;

using LOGGER = lic::devices::logging::CPU_TENSORBOARD;

using DEV_SPEC_SUPER = lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, LOGGER>;
using TI = typename DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints{
    struct HINTS: lic::rl::components::on_policy_runner::ExecutionHints<TI, 8>{};
}
struct DEV_SPEC: DEV_SPEC_SUPER{
    using EXECUTION_HINTS = execution_hints::HINTS;
};
using DEVICE = DEVICE_FACTORY<DEV_SPEC>;


using DEVICE = DEVICE_FACTORY<DEV_SPEC>;
using T = float;
using TI = typename DEVICE::index_t;


constexpr TI ACTOR_CHECKPOINT_INTERVAL = 150;
constexpr TI EVALUATION_INTERVAL = 150;
constexpr bool ACTOR_ENABLE_CHECKPOINTS = true;
constexpr bool ACTOR_OVERWRITE_CHECKPOINTS = false;
const std::string ACTOR_CHECKPOINT_DIRECTORY = "checkpoints/ppo_ant";

TEST(LAYER_IN_C_RL_ALGORITHMS_PPO, TEST){
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

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    prl::OPTIMIZER optimizer;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 12);
    auto evaluation_rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 12);
    prl::PPO_TYPE ppo;
    prl::PPO_BUFFERS_TYPE ppo_buffers;
    prl::ON_POLICY_RUNNER_TYPE on_policy_runner;
    prl::ON_POLICY_RUNNER_BUFFER_TYPE on_policy_runner_buffer;
    prl::ACTOR_EVAL_BUFFERS actor_eval_buffers;
    prl::ACTOR_BUFFERS actor_buffers;
    prl::CRITIC_BUFFERS critic_buffers;
    prl::CRITIC_BUFFERS_GAE critic_buffers_gae;
    penv::ENVIRONMENT envs[prl::N_ENVIRONMENTS];
    penv::ENVIRONMENT evaluation_env;
    bool ui = false;

    lic::malloc(device, ppo);
    lic::malloc(device, ppo_buffers);
    lic::malloc(device, on_policy_runner_buffer);
    lic::malloc(device, on_policy_runner);
    lic::malloc(device, actor_eval_buffers);
    lic::malloc(device, actor_buffers);
    lic::malloc(device, critic_buffers);
    lic::malloc(device, critic_buffers_gae);
    for(auto& env : envs){
        lic::malloc(device, env);
    }
    lic::malloc(device, evaluation_env);

    lic::init(device, on_policy_runner, envs, rng);
    lic::init(device, ppo, optimizer, rng);
    device.logger = &logger;
    lic::construct(device, device.logger);
    auto training_start = std::chrono::high_resolution_clock::now();
    for(TI ppo_step_i = 0; ppo_step_i < 100000; ppo_step_i++) {
        device.logger->step = on_policy_runner.step;

        if(ppo_step_i % 1 == 0){
            std::chrono::duration<T> training_elapsed = std::chrono::high_resolution_clock::now() - training_start;
            std::cout << "PPO step: " << ppo_step_i << " elapsed: " << training_elapsed.count() << "s" << std::endl;
            lic::add_scalar(device, device.logger, "ppo/step", ppo_step_i);
        }
        for (TI action_i = 0; action_i < penv::ENVIRONMENT::ACTION_DIM; action_i++) {
            T action_log_std = lic::get(ppo.actor.action_log_std.parameters, 0, action_i);
            std::stringstream topic;
            topic << "actor/action_std/" << action_i;
            lic::add_scalar(device, device.logger, topic.str(), lic::math::exp(DEVICE::SPEC::MATH(), action_log_std));
        }
        auto start = std::chrono::high_resolution_clock::now();
        {
            auto start = std::chrono::high_resolution_clock::now();
            lic::collect(device, on_policy_runner_buffer, on_policy_runner, ppo.actor, actor_eval_buffers, rng);
            lic::add_scalar(device, device.logger, "opr/observation/mean", lic::mean(device, on_policy_runner_buffer.observations));
            lic::add_scalar(device, device.logger, "opr/observation/std", lic::std(device, on_policy_runner_buffer.observations));
            lic::add_scalar(device, device.logger, "opr/action/mean", lic::mean(device, on_policy_runner_buffer.actions));
            lic::add_scalar(device, device.logger, "opr/action/std", lic::std(device, on_policy_runner_buffer.actions));
            lic::add_scalar(device, device.logger, "opr/rewards/mean", lic::mean(device, on_policy_runner_buffer.rewards));
            lic::add_scalar(device, device.logger, "opr/rewards/std", lic::std(device, on_policy_runner_buffer.rewards));
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
            std::cout << "Rollout: " << elapsed.count() << " s" << std::endl;
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            lic::estimate_generalized_advantages(device, ppo, on_policy_runner_buffer, critic_buffers_gae);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
            std::cout << "GAE: " << elapsed.count() << " s" << std::endl;
        }
        {
            auto start = std::chrono::high_resolution_clock::now();
            lic::train(device, ppo, on_policy_runner_buffer, optimizer, ppo_buffers, actor_buffers, critic_buffers, rng);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = end - start;
            std::cout << "Train: " << elapsed.count() << " s" << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> elapsed = end - start;
        std::cout << "Total: " << elapsed.count() << " s" << std::endl;
        if(ppo_step_i % EVALUATION_INTERVAL == 0){
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
    lic::free(device, on_policy_runner_buffer);
    lic::free(device, on_policy_runner);
    lic::free(device, actor_eval_buffers);
    lic::free(device, actor_buffers);
    lic::free(device, critic_buffers);

}
