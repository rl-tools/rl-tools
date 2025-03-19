template <typename ENVIRONMENT, typename DEVICE, typename POLICY, typename RESULT, typename DATA, typename RNG>
void sample_trajectories(DEVICE& device, POLICY& policy, RESULT& result, DATA& data, RNG& rng){
    using TI = typename DEVICE::index_t;
    ENVIRONMENT base_env;
    rlt::sample_initial_parameters<DEVICE, typename ENVIRONMENT::SPEC, RNG, true>(device, base_env, base_env.parameters, rng);
    using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename POLICY::template CHANGE_BATCH_SIZE<TI, RESULT::SPEC::N_EPISODES>;
    using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
    rlt::rl::environments::DummyUI ui;
    EVALUATION_ACTOR_TYPE evaluation_actor;
    typename EVALUATION_ACTOR_TYPE::template Buffer<true> eval_buffer;
    rlt::malloc(device, evaluation_actor);
    rlt::malloc(device, eval_buffer);
    rlt::copy(device, device, policy, evaluation_actor);

    ENVIRONMENT env_eval;
    typename ENVIRONMENT::Parameters env_eval_parameters;
    rlt::init(device, env_eval);
    env_eval.parameters = base_env.parameters;
    rlt::initial_parameters(device, env_eval, env_eval_parameters);

    rlt::Mode<rlt::mode::Evaluation<rlt::nn::layers::sample_and_squash::mode::DisableEntropy<rlt::mode::Final>>> evaluation_mode;
    rlt::evaluate(device, env_eval, env_eval_parameters, ui, evaluation_actor, result, data, eval_buffer, rng, evaluation_mode, false, true);

    rlt::free(device, evaluation_actor);
    rlt::free(device, eval_buffer);
    rlt::free(device, rng);
}


