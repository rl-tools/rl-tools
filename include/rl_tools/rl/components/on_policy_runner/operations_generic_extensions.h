#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_EXTENSIONS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_EXTENSIONS_H

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::components::on_policy_runner{
        template <typename T_SPEC>
        struct CollectionEvaluationBuffer{
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            Matrix<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::OBSERVATION::DIM, SPEC::DYNAMIC_ALLOCATION>> observations;
            Matrix<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::BATCH_ENVIRONMENT::ACTION_DIM, SPEC::DYNAMIC_ALLOCATION>> actions;
        };
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::on_policy_runner::CollectionEvaluationBuffer<SPEC>& buffer){
        malloc(device, buffer.observations);
        malloc(device, buffer.actions);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::on_policy_runner::CollectionEvaluationBuffer<SPEC>& buffer){
        free(device, buffer.observations);
        free(device, buffer.actions);
    }
    template <typename DEVICE, typename DEVICE_EVALUATION, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename ACTOR, typename ACTOR_EVALUATION, typename POLICY_EVAL_BUFFERS, typename RNG, typename RNG_EVALUATION>
    void collect_hybrid(DEVICE& device, DEVICE_EVALUATION& device_evaluation, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& runner_buffer, ENVIRONMENT& environment, ACTOR& actor, ACTOR_EVALUATION& actor_evaluation, POLICY_EVAL_BUFFERS& policy_eval_buffers, rl::components::on_policy_runner::CollectionEvaluationBuffer<SPEC>& evaluation_buffer, rl::components::on_policy_runner::CollectionEvaluationBuffer<SPEC>& evaluation_buffer_evaluation, RNG& rng, RNG_EVALUATION& rng_evaluation){
        static_assert(utils::typing::is_same_v<typename DATASET_SPEC::SPEC, SPEC>, "the dataset must be specified over the runner's specification");
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect_hybrid: runner not initialized");
#endif
        using TI = typename SPEC::TI;
        if constexpr(SPEC::TRUNCATE_ON_EACH_ITERATION){
            reset(device, runner, environment, rng);
        }
        prologue(device, dataset, runner, environment, rng);
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            auto observations = view_range(device, dataset.all_observations, step_i * SPEC::N_ENVIRONMENTS, tensor::ViewSpec<0, SPEC::N_ENVIRONMENTS>{});
            auto observations_matrix = matrix_view(device, observations);
            copy(device, device_evaluation, observations_matrix, evaluation_buffer_evaluation.observations);
            evaluate(device_evaluation, actor_evaluation, evaluation_buffer_evaluation.observations, evaluation_buffer_evaluation.actions, policy_eval_buffers, rng_evaluation);
            copy(device_evaluation, device, evaluation_buffer_evaluation.actions, evaluation_buffer.actions);

            auto actions_mean = view(device, dataset.actions_mean, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::BATCH_ENVIRONMENT::ACTION_DIM>(), step_i * SPEC::N_ENVIRONMENTS, 0);
            copy(device, device, evaluation_buffer.actions, actions_mean);
            auto& last_layer = get_last_layer(actor);
            auto log_std = matrix_view(device, last_layer.log_std.parameters);
            sample_actions(device, dataset, log_std, runner_buffer.actions, step_i, rng);
            epilogue(device, dataset, runner, runner_buffer, environment, rng, step_i);
        }
        runner.step += SPEC::N_ENVIRONMENTS * DATASET_SPEC::STEPS_PER_ENV;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
