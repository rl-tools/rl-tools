template <typename DEVICE, typename SPEC, typename ACTOR_TYPE, auto VARIATION, typename RNG>
std::tuple<typename SPEC::T, typename SPEC::T> assess(DEVICE& device, ACTOR_TYPE& actor, typename SPEC::ENVIRONMENT::PARAMETERS nominal_parameters, RNG& rng){
    using T = typename SPEC::T;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT = typename SPEC::ENVIRONMENT;

    ENVIRONMENT env;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    typename ENVIRONMENT::State state, next_state;

    bpt::malloc(device, env);
    bpt::malloc(device, action);
    bpt::malloc(device, observation);

    T total_rewards = 0;
    T total_steps = 0;
    constexpr TI NUM_EPISODES = 100;
    for(TI episode_i = 0; episode_i < NUM_EPISODES; episode_i++){
        env.parameters = nominal_parameters;
        VARIATION(env, rng);
        T reward_acc = 0;
        bpt::sample_initial_state(device, env, state, rng);
        for(int step_i = 0; step_i < SPEC::MAX_EPISODE_LENGTH; step_i++){
            auto start = std::chrono::high_resolution_clock::now();
            bpt::observe(device, env, state, observation, rng);
            bpt::evaluate(device, actor, observation, action);
//            for(TI action_i = 0; action_i < penv::ENVIRONMENT::ACTION_DIM; action_i++){
//                increment(action, 0, action_i, bpt::random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, (T)(T)prl::OFF_POLICY_RUNNER_PARAMETERS::EXPLORATION_NOISE, rng));
//            }
            bpt::clamp(device, action, (T)-1, (T)1);
            T dt = bpt::step(device, env, state, action, next_state, rng);
            bool terminated_flag = bpt::terminated(device, env, next_state, rng);
            T reward = bpt::reward(device, env, state, action, next_state, rng);
            if(std::isnan(reward)){
                std::cout << "NAN reward" << std::endl;
            }
            reward_acc += reward;
            state = next_state;
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end-start;
            if(terminated_flag || step_i == (SPEC::MAX_EPISODE_LENGTH - 1)){
//                std::cout << "Episode terminated after " << step_i << " steps with reward " << reward_acc << std::endl;
                total_rewards += reward_acc;
                total_steps += step_i + 1;
                break;
            }
        }
    }
    bpt::free(device, action);
    bpt::free(device, observation);
    return {total_rewards / NUM_EPISODES, total_steps / NUM_EPISODES / SPEC::MAX_EPISODE_LENGTH};
}

