namespace layer_in_c{
    namespace rl::components::on_policy_runner{
        template <typename T_SPEC>
        struct CollectionEvaluationBuffer{
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            Matrix<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>> observations;
            Matrix<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>> actions;
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
    template <typename DEVICE, typename DEVICE_EVALUATION, typename BUFFER_SPEC, typename ACTOR, typename RNG> // todo: make this not PPO but general policy with output distribution
    void collect_hybrid(DEVICE& device, DEVICE_EVALUATION& device_evaluation, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, ACTOR& actor, ACTOR& actor_evaluation, typename ACTOR::template Buffers<BUFFER_SPEC::SPEC::N_ENVIRONMENTS>& policy_eval_buffers, rl::components::on_policy_runner::CollectionEvaluationBuffer<typename BUFFER_SPEC::SPEC> evaluation_buffer, rl::components::on_policy_runner::CollectionEvaluationBuffer<typename BUFFER_SPEC::SPEC>& evaluation_buffer_evaluation, RNG& rng){
#ifdef LAYER_IN_C_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        using SPEC = typename BUFFER_SPEC::SPEC;
        using BUFFER = rl::components::on_policy_runner::Buffer<SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI step_i = 0; step_i < BUFFER_SPEC::STEPS_PER_ENV; step_i++){
            auto actions = view(device, buffer.actions, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations = view(device, buffer.observations, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);

            rl::components::on_policy_runner::prologue(device, evaluation_buffer.observations, runner, rng, step_i);
            copy(device_evaluation, device, evaluation_buffer_evaluation.observations, evaluation_buffer.observations);
            evaluate(device_evaluation, actor_evaluation, evaluation_buffer_evaluation.observations, evaluation_buffer_evaluation.actions, policy_eval_buffers);
            copy(device, device, observations, evaluation_buffer.observations);
            copy(device, device_evaluation, evaluation_buffer.actions, evaluation_buffer_evaluation.actions);
            copy(device, device, actions, evaluation_buffer.actions);
            rl::components::on_policy_runner::epilogue(device, buffer, runner, actions, actor.action_log_std.parameters, rng, step_i);
        }
        // final observation
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            auto& env = get(runner.environments, 0, env_i);
            auto& state = get(runner.states, 0, env_i);
            TI pos = BUFFER_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i;
            auto observation = view(device, buffer.all_observations, matrix::ViewSpec<1, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), pos, 0);
            observe(device, env, state, observation);
        }
        runner.step += SPEC::N_ENVIRONMENTS * BUFFER_SPEC::STEPS_PER_ENV;
    }

}