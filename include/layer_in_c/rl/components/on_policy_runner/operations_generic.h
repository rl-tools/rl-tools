namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer){
        malloc(device, buffer.data);
        using BUFFER = rl::components::on_policy_runner::Buffer<SPEC>;
        using DATA_SPEC = typename decltype(buffer.data)::SPEC;
        using TI = typename SPEC::SPEC::TI;
        TI pos = 0;
        buffer.observations     = view<DEVICE, DATA_SPEC, BUFFER::STEPS_TOTAL, decltype(buffer.observations    )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.observations    )::COLS;
        buffer.actions          = view<DEVICE, DATA_SPEC, BUFFER::STEPS_TOTAL, decltype(buffer.actions         )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.actions         )::COLS;
        buffer.action_log_probs = view<DEVICE, DATA_SPEC, BUFFER::STEPS_TOTAL, decltype(buffer.action_log_probs)::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.action_log_probs)::COLS;
        buffer.rewards          = view<DEVICE, DATA_SPEC, BUFFER::STEPS_TOTAL, decltype(buffer.rewards         )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.rewards         )::COLS;
        buffer.terminated       = view<DEVICE, DATA_SPEC, BUFFER::STEPS_TOTAL, decltype(buffer.terminated      )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.terminated      )::COLS;
        buffer.truncated        = view<DEVICE, DATA_SPEC, BUFFER::STEPS_TOTAL, decltype(buffer.truncated       )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.truncated       )::COLS;
        buffer.value            = view<DEVICE, DATA_SPEC, BUFFER::STEPS_TOTAL, decltype(buffer.value           )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.value           )::COLS;
        buffer.advantage        = view<DEVICE, DATA_SPEC, BUFFER::STEPS_TOTAL, decltype(buffer.advantage       )::COLS>(device, buffer.data, 0, pos); pos += decltype(buffer.advantage       )::COLS;
        buffer.target_value     = view<DEVICE, DATA_SPEC, BUFFER::STEPS_TOTAL, decltype(buffer.target_value    )::COLS>(device, buffer.data, 0, pos);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::on_policy_runner::Buffer<SPEC>& buffer){
        free(device, buffer.data);
        buffer.observations    ._data = nullptr;
        buffer.actions         ._data = nullptr;
        buffer.action_log_probs._data = nullptr;
        buffer.rewards         ._data = nullptr;
        buffer.terminated      ._data = nullptr;
        buffer.truncated       ._data = nullptr;
        buffer.value           ._data = nullptr;
        buffer.advantage       ._data = nullptr;
        buffer.target_value    ._data = nullptr;
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        malloc(device, runner.environments);
        malloc(device, runner.states);
        malloc(device, runner.episode_step);
        malloc(device, runner.truncated);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        free(device, runner.environments);
        free(device, runner.states);
        free(device, runner.episode_step);
        free(device, runner.truncated);
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void init(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, typename SPEC::ENVIRONMENT environments[SPEC::N_ENVIRONMENTS], RNG& rng){
        using TI = typename SPEC::TI;
        set_all(device, runner.episode_step, 0);
        set_all(device, runner.truncated, true);
    }
    template <typename DEVICE, typename BUFFER_SPEC, typename RNG>
    void collect(DEVICE& device, rl::components::on_policy_runner::Buffer<BUFFER_SPEC>& buffer, rl::components::OnPolicyRunner<typename BUFFER_SPEC::SPEC>& runner, RNG& rng){
        using SPEC = typename BUFFER_SPEC::SPEC;
        using BUFFER = rl::components::on_policy_runner::Buffer<SPEC>;
        using TI = typename SPEC::TI;
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            auto& env = get(runner.environments, 0, env_i);
            auto& state = get(runner.states, 0, env_i);
            for(TI step_i = 0; step_i < BUFFER_SPEC::STEPS_PER_ENV; step_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                if(get(runner.truncated, 0, env_i)){
                    set(runner.truncated, 0, env_i, false);
                    set(runner.episode_step, 0, env_i, 0);
                    sample_initial_state(device, env, state, rng);
                }
                auto observation = view<DEVICE, typename decltype(buffer.observations)::SPEC, 1, SPEC::ENVIRONMENT::OBSERVATION_DIM>(device, buffer.observations, pos, 0);
                observe(device, env, state, observation);

                auto action = view<DEVICE, typename decltype(buffer.actions)::SPEC, 1, SPEC::ENVIRONMENT::ACTION_DIM>(device, buffer.actions, pos, 0);
                for(TI action_i = 0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++){
                    set(action, 0, action_i, random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -1, 1, rng));
                }
                set(buffer.action_log_probs, pos, 0, random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -1, 1, rng));
                typename SPEC::ENVIRONMENT::State next_state;
                step(device, env, state, action, next_state);
                bool terminated_flag = terminated(device, env, next_state);
                set(buffer.terminated, pos, 0, terminated_flag);
                increment(runner.episode_step, 0, env_i, 1);
                bool truncated = terminated_flag || (SPEC::STEP_LIMIT > 0 && get(runner.episode_step, 0, env_i) >= SPEC::STEP_LIMIT);
                set(buffer.truncated, pos, 0, truncated);
                set(runner.truncated, 0, env_i, truncated);
                state = next_state;
            }
        }
    }
}