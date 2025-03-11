template <typename DEVICE, typename RNG_PARAMS, typename RNG, typename RNG_PARAMS_DEVICE, typename ENVIRONMENT_PT, typename T, typename TI, TI NUM_EPISODES, typename RESULT, typename DATA>
auto generate_data(DEVICE& device, rlt::utils::extrack::Path checkpoint_path, typename DEVICE::index_t seed, RESULT& result, DATA& data){

    RNG_PARAMS rng_params;
    {
        rlt::malloc(device, rng_params);
        rlt::init(device, rng_params, seed);
        // warmup
        for(TI i=0; i < (TI)std::stoi(checkpoint_path.attributes["rng-warmup"]); i++){
            rlt::random::uniform_int_distribution(RNG_PARAMS_DEVICE{}, 0, 1, rng_params);
        }
        rlt::free(device, rng_params);
    }
    auto actor_file = HighFive::File(checkpoint_path.checkpoint_path.string(), HighFive::File::ReadOnly);
    ENVIRONMENT_PT base_env;
    rlt::sample_initial_parameters<DEVICE, typename ENVIRONMENT_PT::SPEC, RNG_PARAMS, true>(device, base_env, base_env.parameters, rng_params);
    using LOOP_CORE_CONFIG_PRE_TRAINING = typename builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING>::LOOP_CORE_CONFIG;
    using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename LOOP_CORE_CONFIG_PRE_TRAINING::NN::ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, NUM_EPISODES>;
    using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<LOOP_CORE_CONFIG_PRE_TRAINING::DYNAMIC_ALLOCATION>>;
    rlt::rl::environments::DummyUI ui;
    EVALUATION_ACTOR_TYPE evaluation_actor;
    typename EVALUATION_ACTOR_TYPE::template Buffer<LOOP_CORE_CONFIG_PRE_TRAINING::DYNAMIC_ALLOCATION> eval_buffer;
    rlt::malloc(device, evaluation_actor);
    rlt::malloc(device, eval_buffer);
    rlt::load(device, evaluation_actor, actor_file.getGroup("actor"));

    ENVIRONMENT_PT env_eval;
    typename ENVIRONMENT_PT::Parameters env_eval_parameters;
    rlt::init(device, env_eval);
    env_eval.parameters = base_env.parameters;
    rlt::initial_parameters(device, env_eval, env_eval_parameters);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    rlt::Mode<rlt::mode::Evaluation<rlt::nn::layers::sample_and_squash::mode::DisableEntropy<rlt::mode::Final>>> evaluation_mode;
    rlt::evaluate(device, env_eval, env_eval_parameters, ui, evaluation_actor, result, data, eval_buffer, rng, evaluation_mode, false, true);

    rlt::free(device, evaluation_actor);
    rlt::free(device, eval_buffer);
    rlt::free(device, rng);
    return base_env.parameters;
}


