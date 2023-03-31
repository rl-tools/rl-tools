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
    template <typename DEVICE, typename DEVICE_EVALUATION, typename DATASET_SPEC, typename ACTOR, typename ACTOR_EVALUATION, typename RNG> // todo: make this not PPO but general policy with output distribution
    void collect_hybrid(DEVICE& device, DEVICE_EVALUATION& device_evaluation, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, ACTOR& actor, ACTOR_EVALUATION& actor_evaluation, typename ACTOR_EVALUATION::template Buffers<DATASET_SPEC::SPEC::N_ENVIRONMENTS>& policy_eval_buffers, rl::components::on_policy_runner::CollectionEvaluationBuffer<typename DATASET_SPEC::SPEC> evaluation_buffer, rl::components::on_policy_runner::CollectionEvaluationBuffer<typename DATASET_SPEC::SPEC>& evaluation_buffer_evaluation, RNG& rng){
#ifdef LAYER_IN_C_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        using SPEC = typename DATASET_SPEC::SPEC;
        using BUFFER = rl::components::on_policy_runner::Dataset<SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        TI prologue_time = 0;
        TI copy_observations_time = 0;
        TI evaluate_time = 0;
        TI copy_back_time = 0;
        TI epilogue_time = 0;
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            auto actions_mean = view(device, dataset.actions_mean, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>()     , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto actions      = view(device, dataset.actions     , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>()     , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations = view(device, dataset.observations, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), step_i*SPEC::N_ENVIRONMENTS, 0);

            {
                auto start = std::chrono::high_resolution_clock::now();
                rl::components::on_policy_runner::prologue(device, evaluation_buffer.observations, runner, rng, step_i);
                auto end = std::chrono::high_resolution_clock::now();
                prologue_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            }
            {
                auto start = std::chrono::high_resolution_clock::now();
                copy(device_evaluation, device, evaluation_buffer_evaluation.observations, evaluation_buffer.observations);
                auto end = std::chrono::high_resolution_clock::now();
                copy_observations_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            }
            {
                auto start = std::chrono::high_resolution_clock::now();
                evaluate(device_evaluation, actor_evaluation, evaluation_buffer_evaluation.observations, evaluation_buffer_evaluation.actions, policy_eval_buffers);
                cudaDeviceSynchronize();
                auto end = std::chrono::high_resolution_clock::now();
                evaluate_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            }
            {

                auto start = std::chrono::high_resolution_clock::now();
                copy(device, device, observations, evaluation_buffer.observations);
                copy(device, device_evaluation, evaluation_buffer.actions, evaluation_buffer_evaluation.actions);
                copy(device, device, actions_mean, evaluation_buffer.actions);
                auto end = std::chrono::high_resolution_clock::now();
                copy_back_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            }
            {
                auto start = std::chrono::high_resolution_clock::now();
                rl::components::on_policy_runner::epilogue(device, dataset, runner, actions_mean, actions, actor.log_std.parameters, rng, step_i);
                auto end = std::chrono::high_resolution_clock::now();
                epilogue_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            }
        }
        std::cout << "prologue_time: " << prologue_time << std::endl;
        std::cout << "copy_observations_time: " << copy_observations_time << std::endl;
        std::cout << "evaluate_time: " << evaluate_time << std::endl;
        std::cout << "copy_back_time: " << copy_back_time << std::endl;
        std::cout << "epilogue_time: " << epilogue_time << std::endl;

        // final observation
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            auto& env = get(runner.environments, 0, env_i);
            auto& state = get(runner.states, 0, env_i);
            TI pos = DATASET_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i;
            auto observation = view(device, dataset.all_observations, matrix::ViewSpec<1, SPEC::ENVIRONMENT::OBSERVATION_DIM>(), pos, 0);
            observe(device, env, state, observation);
        }
        runner.step += SPEC::N_ENVIRONMENTS * DATASET_SPEC::STEPS_PER_ENV;
    }

}