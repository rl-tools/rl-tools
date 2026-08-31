template <typename T>
struct TeacherMeta{
    T steady_state_position_offset[3];
};

template <typename ENVIRONMENT, typename DEVICE, typename POLICY, typename PARAMETERS, typename RESULT, typename DATA, typename RNG>
void sample_trajectories(DEVICE& device, POLICY& policy, const PARAMETERS& parameters, RESULT& result, DATA& data, RNG& rng){
    using TI = typename DEVICE::index_t;
    ENVIRONMENT base_env;
    rlt::init(device, base_env);
    auto init_old = base_env.parameters.mdp.init; // legacy for v3, remove for next run
    base_env.parameters = parameters;
    base_env.parameters.mdp.init = init_old;
    rlt::sample_initial_parameters(device, base_env, base_env.parameters, rng);
    using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename POLICY::template CHANGE_BATCH_SIZE<TI, RESULT::SPEC::N_EPISODES>;
    using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
    rlt::rl::environments::DummyUI ui;
    EVALUATION_ACTOR_TYPE evaluation_actor;
    typename EVALUATION_ACTOR_TYPE::template Buffer<true> eval_buffer;
    rlt::malloc(device, evaluation_actor);
    rlt::malloc(device, eval_buffer);
    rlt::copy(device, device, policy, evaluation_actor);

    ENVIRONMENT env_eval;
    rlt::init(device, env_eval);
    env_eval.parameters = base_env.parameters;

    rlt::Mode<rlt::mode::Evaluation<rlt::nn::layers::sample_and_squash::mode::DisableEntropy<rlt::mode::Final>>> evaluation_mode;
    rlt::evaluate(device, env_eval, ui, evaluation_actor, result, data, rng, evaluation_mode);
    // rlt::add_scalar(device, device.logger, "evaluation/return/mean", result.returns_mean);
    // rlt::add_scalar(device, device.logger, "evaluation/return/std", result.returns_std);
    // rlt::add_scalar(device, device.logger, "evaluation/episode_length/mean", result.episode_length_mean);
    // rlt::add_scalar(device, device.logger, "evaluation/episode_length/std", result.episode_length_std);
    // rlt::add_scalar(device, device.logger, "evaluation/share_terminated", result.share_terminated);
    // rlt::log(device, device.logger, "Mean return: ", result.returns_mean, " Mean episode length: ", result.episode_length_mean, " Share terminated: ", result.share_terminated * 100, "%");

    rlt::free(device, evaluation_actor);
    rlt::free(device, eval_buffer);
    rlt::free(device, rng);
}

template <typename ENVIRONMENT, typename TEACHER_OBSERVATION, typename STUDENT_OBSERVATION, bool TEACHER_DETERMINISTIC, typename DEVICE, typename TEACHER_ORIG, typename TEACHER_META_SPEC, typename DATA, typename DS_EPISODE_START_INDICES, typename INPUT_SPEC, typename OUTPUT_SPEC, typename TRUNCATED_SPEC, typename RESET_SPEC, typename RNG, typename TI=typename DEVICE::index_t>
TI add_to_dataset(DEVICE& device, DATA& data, TEACHER_ORIG& teacher, TeacherMeta<TEACHER_META_SPEC>& teacher_meta, rlt::Tensor<DS_EPISODE_START_INDICES>& dataset_episode_start_indices, rlt::Tensor<INPUT_SPEC>& dataset_input_student, rlt::Tensor<OUTPUT_SPEC>& dataset_output, rlt::Tensor<TRUNCATED_SPEC>& truncated, rlt::Tensor<RESET_SPEC>& reset, TI& current_episode, TI& current_index, RNG& rng){
    using T = typename INPUT_SPEC::T;
    TI initial_index = current_index;
    ENVIRONMENT env_eval;
    rlt::init(device, env_eval);
    static constexpr TI DATASET_SIZE = INPUT_SPEC::SHAPE::FIRST;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DATASET_SIZE, TEACHER_OBSERVATION::DIM>>> input_teacher;
    rlt::malloc(device, input_teacher);
    bool reset_flag = true;
    for (TI episode_i = 0; episode_i < DATA::SPEC::N_EPISODES; episode_i++){
        rlt::set(device, dataset_episode_start_indices, current_index, current_episode);
        current_episode++;
        typename ENVIRONMENT::Parameters env_eval_parameters = get(device, data.parameters, episode_i);
        TI current_step_i;
        for (current_step_i = 0; current_step_i < ENVIRONMENT::EPISODE_STEP_LIMIT; current_step_i++){
            auto observation_student_tensor = rlt::view(device, dataset_input_student, current_index + current_step_i);
            auto observation_teacher_tensor = rlt::view(device, input_teacher, current_index + current_step_i);
            auto observation_student = rlt::matrix_view(device, observation_student_tensor);
            auto observation_teacher = rlt::matrix_view(device, observation_teacher_tensor);
            auto state = get(device, data.states, episode_i, current_step_i);
            rlt::observe(device, env_eval, env_eval_parameters, state, TEACHER_OBSERVATION{}, observation_teacher, rng);
            rlt::observe(device, env_eval, env_eval_parameters, state, STUDENT_OBSERVATION{}, observation_student, rng);
            for (TI dim_i=0; dim_i < 3; dim_i++){
                T position = rlt::get(observation_student, 0, dim_i);
                rlt::set(observation_student, 0, dim_i, position - teacher_meta.steady_state_position_offset[dim_i]);
            }
            bool truncated_flag = get(device, data.terminated, episode_i, current_step_i) || current_step_i == (ENVIRONMENT::EPISODE_STEP_LIMIT - 1);
            rlt::set(device, truncated, truncated_flag, current_index + current_step_i);
            rlt::set(device, reset, reset_flag, current_index + current_step_i);
            if (get(device, data.terminated, episode_i, current_step_i)){
                reset_flag = true;
                current_step_i++;
                break;
            }
        }
        current_index += current_step_i;
        static_assert(DATA::SPEC::N_EPISODES > 0);
        if (current_index >= INPUT_SPEC::SHAPE::FIRST && episode_i < (DATA::SPEC::N_EPISODES - 1)){
            std::cerr << "Dataset size exceeded" << std::endl;
            break;
        }
    }

    static constexpr TI BATCH_SIZE = 1;
    using TEACHER = typename TEACHER_ORIG::template CHANGE_BATCH_SIZE<TI, BATCH_SIZE>::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
    typename TEACHER::template Buffer<true> policy_teacher_buffer;
    rlt::malloc(device, policy_teacher_buffer);
    typename TEACHER::template State<true> teacher_state;
    rlt::malloc(device, teacher_state);
    rlt::reset(device, teacher, teacher_state, rng);
    for(TI step_i=initial_index; step_i < current_index; ++step_i){
        static_assert(BATCH_SIZE == 1, "Batch size needs to be one for sequential state tracking (reset / evaluate_step)");
        auto input_chunk = rlt::view_range(device, input_teacher, step_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
        auto output_chunk = rlt::view_range(device, dataset_output, step_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
        auto reset_chunk = rlt::view_range(device, reset, step_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
        if (rlt::get(device, reset_chunk, 0)){
            rlt::reset(device, teacher, teacher_state, rng);
        }
        rlt::utils::typing::conditional_t<TEACHER_DETERMINISTIC, rlt::Mode<rlt::mode::Evaluation<>>, rlt::Mode<rlt::mode::Default<>>> mode;
        rlt::evaluate_step(device, teacher, input_chunk, teacher_state, output_chunk, policy_teacher_buffer, rng, mode);
    }
    rlt::free(device, input_teacher);
    rlt::free(device, teacher_state);
    rlt::free(device, policy_teacher_buffer);
    rlt::utils::assert_exit(device, current_index >= initial_index, "Current index out of range");
    return current_index - initial_index;
}


template <typename ENVIRONMENT, typename TEACHER_OBSERVATION, typename STUDENT_OBSERVATION, auto NUM_EPISODES, bool TEACHER_DETERMINISTIC, typename DEVICE, typename STUDENT, typename TEACHER, typename TEACHER_META_SPEC, typename PARAMETERS, typename DS_EPISODE_START_INDICES_SPEC, typename DS_INPUT_SPEC, typename DS_OUTPUT_SPEC, typename DS_TRUNCATED_SPEC, typename DS_RESET_SPEC, typename RNG, typename TI=typename DEVICE::index_t>
auto gather_epoch(DEVICE& device, TEACHER& teacher, TeacherMeta<TEACHER_META_SPEC>& teacher_meta, PARAMETERS& parameters, STUDENT& student, rlt::Tensor<DS_EPISODE_START_INDICES_SPEC>& dataset_episode_start_indices, rlt::Tensor<DS_INPUT_SPEC>& dataset_input, rlt::Tensor<DS_OUTPUT_SPEC>& dataset_output_target, rlt::Tensor<DS_TRUNCATED_SPEC>& dataset_truncated, rlt::Tensor<DS_RESET_SPEC>& dataset_reset, TI& current_episode, TI& current_index, RNG& rng){
    using T = typename DS_INPUT_SPEC::T;
    using RESULT = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES, ENVIRONMENT::EPISODE_STEP_LIMIT>>;
    RESULT result;
    rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<typename RESULT::SPEC>> data;
    rlt::malloc(device, data);
    sample_trajectories<ENVIRONMENT>(device, student, parameters, result, data, rng);
    add_to_dataset<ENVIRONMENT, TEACHER_OBSERVATION, STUDENT_OBSERVATION, TEACHER_DETERMINISTIC>(device, data, teacher, teacher_meta, dataset_episode_start_indices, dataset_input, dataset_output_target, dataset_truncated, dataset_reset, current_episode, current_index, rng);
    rlt::free(device, data);
    return result;
}

auto split_by_comma(const std::string& s) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) result.push_back(item);
    return result;
};

