#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include "on_policy_runner.h"
#include "operations_generic_per_env.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        malloc(device, dataset.all_observations);
        malloc(device, dataset.all_observations_privileged);
        malloc(device, dataset.scalar_data);
        using SCALAR_DATA_SPEC = typename decltype(dataset.scalar_data)::SPEC;
        using TI = typename SPEC::SPEC::TI;
        TI pos = 0;
        dataset.actions_mean     = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.actions_mean    )::ROWS, decltype(dataset.actions_mean    )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.actions_mean    )::COLS;
        dataset.actions          = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.actions         )::ROWS, decltype(dataset.actions         )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.actions         )::COLS;
        dataset.action_log_probs = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.action_log_probs)::ROWS, decltype(dataset.action_log_probs)::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.action_log_probs)::COLS;
        dataset.rewards          = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.rewards         )::ROWS, decltype(dataset.rewards         )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.rewards         )::COLS;
        dataset.terminated       = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.terminated      )::ROWS, decltype(dataset.terminated      )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.terminated      )::COLS;
        dataset.truncated        = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.truncated       )::ROWS, decltype(dataset.truncated       )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.truncated       )::COLS;
        dataset.all_reset        = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.all_reset       )::ROWS, decltype(dataset.all_reset       )::COLS>(device, dataset.scalar_data, 0, pos);
        dataset.reset            = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.reset           )::ROWS, decltype(dataset.reset           )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.reset           )::COLS;
        dataset.all_values       = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.all_values      )::ROWS, decltype(dataset.all_values      )::COLS>(device, dataset.scalar_data, 0, pos);
        dataset.values           = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.values          )::ROWS, decltype(dataset.values          )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.values          )::COLS;
        dataset.advantages       = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.advantages      )::ROWS, decltype(dataset.advantages      )::COLS>(device, dataset.scalar_data, 0, pos); pos += decltype(dataset.advantages      )::COLS;
        dataset.target_values    = view<DEVICE, SCALAR_DATA_SPEC, decltype(dataset.target_values   )::ROWS, decltype(dataset.target_values   )::COLS>(device, dataset.scalar_data, 0, pos);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset){
        free(device, dataset.all_observations);
        free(device, dataset.all_observations_privileged);
        free(device, dataset.scalar_data);
        dataset.actions_mean               ._data = nullptr;
        dataset.actions                    ._data = nullptr;
        dataset.action_log_probs           ._data = nullptr;
        dataset.rewards                    ._data = nullptr;
        dataset.terminated                 ._data = nullptr;
        dataset.truncated                  ._data = nullptr;
        dataset.all_values                 ._data = nullptr;
        dataset.values                     ._data = nullptr;
        dataset.advantages                 ._data = nullptr;
        dataset.target_values              ._data = nullptr;
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        malloc(device, runner.environments);
        malloc(device, runner.env_parameters);
        malloc(device, runner.states);
        malloc(device, runner.policy_state);
        malloc(device, runner.episode_step);
        malloc(device, runner.episode_return);
        malloc(device, runner.truncated);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner){
        free(device, runner.environments);
        free(device, runner.env_parameters);
        free(device, runner.states);
        free(device, runner.policy_state);
        free(device, runner.episode_step);
        free(device, runner.episode_return);
        free(device, runner.truncated);
    }
    template <typename DEVICE, typename SPEC, typename ENV_SPEC, typename PARAM_SPEC, typename ACTOR, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, Tensor<ENV_SPEC> environments, Tensor<PARAM_SPEC> parameters, ACTOR& actor, RNG& rng){
        using TI = typename SPEC::TI;
        set_all(device, runner.episode_step, 0);
        set_all(device, runner.episode_return, 0);
        set_all(device, runner.truncated, true);
        for(TI env_i=0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            set(runner.environments, 0, env_i, get_ref(device, environments, env_i));
            set(runner.env_parameters, 0, env_i, get_ref(device, parameters, env_i));
        }
        reset(device, actor, runner.policy_state, rng);
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
    }
    namespace rl::components::on_policy_runner{
        template <typename DEVICE, typename OBS_PRIV_SPEC, typename OBS_SPEC, typename SPEC, typename ARRAY_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void prologue(DEVICE& device, Tensor<OBS_PRIV_SPEC>& observations_privileged, Tensor<OBS_SPEC>& observations, rl::components::OnPolicyRunner<SPEC>& runner, devices::generic::random::ArrayENGINE<ARRAY_SPEC>& rng, typename DEVICE::index_t step_i){
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                auto& rng_state = get(rng.states, 0, env_i);
                per_env::prologue(device, observations_privileged, observations, runner, rng_state, env_i);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename ACTIONS_MEAN_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename ARRAY_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void epilogue(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, Matrix<ACTIONS_MEAN_SPEC>& actions_mean, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, devices::generic::random::ArrayENGINE<ARRAY_SPEC>& rng, typename DEVICE::index_t step_i){
            using SPEC = typename DATASET_SPEC::SPEC;
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                auto& rng_state = get(rng.states, 0, env_i);
                per_env::epilogue(device, dataset, runner, actions_mean, actions, action_log_std, rng_state, pos, env_i);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename ARRAY_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void final_observations(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, devices::generic::random::ArrayENGINE<ARRAY_SPEC>& rng){
            using SPEC = typename DATASET_SPEC::SPEC;
            using TI = typename SPEC::TI;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                auto& rng_state = get(rng.states, 0, env_i);
                auto& env = get(runner.environments, 0, env_i);
                auto& state = get(runner.states, 0, env_i);
                auto& parameters = get(runner.env_parameters, 0, env_i);
                auto obs_slice = view(device, dataset.all_observations, (TI)(DATASET_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i));
                auto obs_matrix = matrix_view(device, obs_slice);
                observe(device, env, parameters, state, typename SPEC::ENVIRONMENT::Observation{}, obs_matrix, rng_state);
                auto obs_priv_slice = view(device, dataset.all_observations_privileged, (TI)(DATASET_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i));
                auto obs_priv_matrix = matrix_view(device, obs_priv_slice);
                observe(device, env, parameters, state, typename SPEC::ENVIRONMENT::ObservationPrivileged{}, obs_priv_matrix, rng_state);
            }
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename ACTOR, typename ACTOR_BUFFER, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, ACTOR& actor, ACTOR_BUFFER& policy_eval_buffers, RNG& rng){
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        using SPEC = typename DATASET_SPEC::SPEC;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        if constexpr(SPEC::TRUNCATE_ON_EACH_ITERATION){
            set_all(device, runner.truncated, true);
        }
        for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
            set(dataset.reset, env_i, 0, get(runner.truncated, 0, env_i));
        }
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            auto actions_mean            = view(device, dataset.actions_mean               , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>()                , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto actions                 = view(device, dataset.actions                    , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>()                , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations_privileged = view_range(device, dataset.all_observations_privileged, step_i*SPEC::N_ENVIRONMENTS, tensor::ViewSpec<0, SPEC::N_ENVIRONMENTS>{});
            auto observations            = view_range(device, dataset.all_observations          , step_i*SPEC::N_ENVIRONMENTS, tensor::ViewSpec<0, SPEC::N_ENVIRONMENTS>{});
            auto truncated_view = view(device, runner.truncated);
            Mode<mode::sequential::ResetMask<mode::Default<>, mode::sequential::ResetMaskSpecification<decltype(truncated_view)>>> mode_reset_mask;
            mode_reset_mask.mask = truncated_view;
            reset(device, actor, runner.policy_state, rng, mode_reset_mask); // it is important that this happens before prologue because prologue resets the truncated flags on the runner
            rl::components::on_policy_runner::prologue(device, observations_privileged, observations, runner, rng, step_i);
            using OBS_SHAPE = typename SPEC::ENVIRONMENT::Observation::SHAPE;
            using EVAL_INPUT_SHAPE = tensor::Prepend<OBS_SHAPE, SPEC::N_ENVIRONMENTS>;
            auto observations_reshaped = reshape_row_major(device, observations, EVAL_INPUT_SHAPE{});
            auto actions_mean_tensor = to_tensor(device, actions_mean);
            Mode<mode::Rollout<>> mode;
            evaluate_step(device, actor, observations_reshaped, runner.policy_state, actions_mean_tensor, policy_eval_buffers, rng, mode);
            auto& last_layer = get_last_layer(actor);
            auto log_std = matrix_view(device, last_layer.log_std.parameters);
            rl::components::on_policy_runner::epilogue(device, dataset, runner, actions_mean, actions, log_std, rng, step_i);
        }
        rl::components::on_policy_runner::final_observations(device, dataset, runner, rng);
        runner.step += SPEC::N_ENVIRONMENTS * DATASET_SPEC::STEPS_PER_ENV;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::SPEC::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC_1>& d1, rl::components::on_policy_runner::Dataset<SPEC_2>& d2){
        using T = typename SPEC_1::SPEC::TYPE_POLICY::DEFAULT;
        T acc = 0;
        acc += abs_diff(device, d1.all_observations, d2.all_observations);
        acc += abs_diff(device, d1.all_observations_privileged, d2.all_observations_privileged);
        acc += abs_diff(device, d1.actions_mean, d2.actions_mean);
        acc += abs_diff(device, d1.actions, d2.actions);
        acc += abs_diff(device, d1.action_log_probs, d2.action_log_probs);
        acc += abs_diff(device, d1.rewards, d2.rewards);
        acc += abs_diff(device, d1.terminated, d2.terminated);
        acc += abs_diff(device, d1.truncated, d2.truncated);
        acc += abs_diff(device, d1.all_reset, d2.all_reset);
        acc += abs_diff(device, d1.all_values, d2.all_values);
        acc += abs_diff(device, d1.advantages, d2.advantages);
        acc += abs_diff(device, d1.target_values, d2.target_values);
        return acc;
    }
    template <typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, rl::components::OnPolicyRunner<SPEC_1>& r1, rl::components::OnPolicyRunner<SPEC_2>& r2){
        using T = typename SPEC_1::TYPE_POLICY::DEFAULT;
        using TI = typename DEVICE::index_t;
        T acc = 0;
        acc += math::abs(device.math, (T)r1.step - (T)r2.step);
        acc += abs_diff(device, r1.policy_state, r2.policy_state);
        for(TI env_i = 0; env_i < SPEC_1::N_ENVIRONMENTS; env_i++){
            TI episode_step_r1 = get(r1.episode_step, 0, env_i);
            TI episode_step_r2 = get(r2.episode_step, 0, env_i);
            acc += math::abs(device.math, (T)episode_step_r1 - (T)episode_step_r2);
        }
        acc += abs_diff(device, r1.episode_return, r2.episode_return);
        acc += abs_diff(device, r1.truncated, r2.truncated);
        for(TI env_i = 0; env_i < SPEC_1::N_ENVIRONMENTS; env_i++){
            acc += abs_diff(device, get(r1.states, 0, env_i), get(r2.states, 0, env_i));
            acc += abs_diff(device, get(r1.env_parameters, 0, env_i), get(r2.env_parameters, 0, env_i));
        }
        return acc;
    }

    // batched ingest of a step's flags from tensor producers (e.g. hyperdrone::episodes): the row
    // of step_i and the reset column of step_i + 1 (reset = truncation delayed by one step)
    namespace rl::components::on_policy_runner{
        template <typename DEVICE, typename DATASET_SPEC, typename REWARD_SPEC, typename TERMINATED_SPEC, typename TRUNCATED_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void record_step_env(DEVICE& device, Dataset<DATASET_SPEC>& dataset, typename DATASET_SPEC::TI step_i, const Tensor<REWARD_SPEC>& rewards, const Tensor<TERMINATED_SPEC>& terminated, const Tensor<TRUNCATED_SPEC>& truncated, typename DATASET_SPEC::TI env_i){
            using T = typename Dataset<DATASET_SPEC>::T;
            using TI = typename DATASET_SPEC::TI;
            constexpr TI N_ENVIRONMENTS = DATASET_SPEC::SPEC::N_ENVIRONMENTS;
            const TI pos = step_i * N_ENVIRONMENTS + env_i;
            const bool truncated_flag = get(device, truncated, env_i);
            set(dataset.rewards, pos, 0, (T)get(device, rewards, env_i));
            set(dataset.terminated, pos, 0, get(device, terminated, env_i) ? (T)1 : (T)0);
            set(dataset.truncated, pos, 0, truncated_flag ? (T)1 : (T)0);
            set(dataset.all_reset, pos + N_ENVIRONMENTS, 0, truncated_flag ? (T)1 : (T)0);
        }
        template <typename DEVICE, typename DATASET_SPEC, typename RESET_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void record_reset_env(DEVICE& device, Dataset<DATASET_SPEC>& dataset, const Tensor<RESET_SPEC>& reset, typename DATASET_SPEC::TI env_i){
            using T = typename Dataset<DATASET_SPEC>::T;
            set(dataset.reset, env_i, 0, get(device, reset, env_i) ? (T)1 : (T)0);
        }
        template <typename DATASET_SPEC, typename FLAG_SPEC>
        constexpr bool check_batched_flags(){
            static_assert(get<0>(typename FLAG_SPEC::SHAPE{}) == DATASET_SPEC::SPEC::N_ENVIRONMENTS, "the flag tensors must cover all environments");
            return true;
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename REWARD_SPEC, typename TERMINATED_SPEC, typename TRUNCATED_SPEC>
    void record_step(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, typename DATASET_SPEC::TI step_i, const Tensor<REWARD_SPEC>& rewards, const Tensor<TERMINATED_SPEC>& terminated, const Tensor<TRUNCATED_SPEC>& truncated){
        using TI = typename DATASET_SPEC::TI;
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, REWARD_SPEC>());
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, TERMINATED_SPEC>());
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, TRUNCATED_SPEC>());
        utils::assert_exit(device, step_i < DATASET_SPEC::STEPS_PER_ENV, "on_policy_runner::record_step: step index outside the dataset");
        for(TI env_i = 0; env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS; env_i++){
            rl::components::on_policy_runner::record_step_env(device, dataset, step_i, rewards, terminated, truncated, env_i);
        }
    }
    // the reset column of step 0: the mask applied at the start of the rollout
    template <typename DEVICE, typename DATASET_SPEC, typename RESET_SPEC>
    void record_reset(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const Tensor<RESET_SPEC>& reset){
        using TI = typename DATASET_SPEC::TI;
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, RESET_SPEC>());
        for(TI env_i = 0; env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS; env_i++){
            rl::components::on_policy_runner::record_reset_env(device, dataset, reset, env_i);
        }
    }
    // batched environments (OnPolicyRunnerBatched): the environment verbs are tensor-batched, the
    // autoreset happens in the epilogue (the dataset stays linear: the observation after a reset is
    // the first of the new episode) and the phases can be driven individually by custom loops
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::components::OnPolicyRunnerBatched<SPEC>& runner){
        malloc(device, runner.policy_state);
        malloc(device, runner.env_parameters);
        malloc(device, runner.states);
        malloc(device, runner.next_states);
        malloc(device, runner.actions);
        malloc(device, runner.rewards);
        malloc(device, runner.episodes);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::components::OnPolicyRunnerBatched<SPEC>& runner){
        free(device, runner.policy_state);
        free(device, runner.env_parameters);
        free(device, runner.states);
        free(device, runner.next_states);
        free(device, runner.actions);
        free(device, runner.rewards);
        free(device, runner.episodes);
    }
    namespace rl::components::on_policy_runner{
        template <typename DEVICE, typename ENVIRONMENT, typename PARAMETERS, typename STATES, typename MASK, typename = void>
        struct HasRender: rl_tools::utils::typing::false_type{};
        template <typename DEVICE, typename ENVIRONMENT, typename PARAMETERS, typename STATES, typename MASK>
        struct HasRender<DEVICE, ENVIRONMENT, PARAMETERS, STATES, MASK, rl_tools::utils::typing::void_t<decltype(render(rl_tools::utils::typing::declared_lvalue<DEVICE>(), rl_tools::utils::typing::declared_lvalue<ENVIRONMENT>(), rl_tools::utils::typing::declared_lvalue<PARAMETERS>(), rl_tools::utils::typing::declared_lvalue<STATES>(), rl_tools::utils::typing::declared_lvalue<const MASK>()))>>: rl_tools::utils::typing::true_type{};

        // environments with a framebuffer expose render(device, environment, parameters, states, reset_mask)
        template <typename DEVICE, typename SPEC, typename ENVIRONMENT>
        void render_if_available(DEVICE& device, rl::components::OnPolicyRunnerBatched<SPEC>& runner, ENVIRONMENT& environment){
            using PARAMETERS = decltype(runner.env_parameters);
            using STATES = decltype(runner.states);
            using MASK = decltype(runner.episodes.reset);
            if constexpr(HasRender<DEVICE, ENVIRONMENT, PARAMETERS, STATES, MASK>::value){
                render(device, environment, runner.env_parameters, runner.states, runner.episodes.reset);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
        void observe_row(DEVICE& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunnerBatched<SPEC>& runner, ENVIRONMENT& environment, typename SPEC::TI row_i, RNG& rng){
            constexpr typename SPEC::TI N_ENVIRONMENTS = SPEC::N_ENVIRONMENTS;
            auto observations = view_range(device, dataset.all_observations, row_i * N_ENVIRONMENTS, tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            observe(device, environment, runner.env_parameters, runner.states, typename SPEC::OBSERVATION{}, observations, rng);
            if constexpr(SPEC::ASYMMETRIC_OBSERVATIONS){
                auto observations_privileged = view_range(device, dataset.all_observations_privileged, row_i * N_ENVIRONMENTS, tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                observe(device, environment, runner.env_parameters, runner.states, typename SPEC::OBSERVATION_PRIVILEGED{}, observations_privileged, rng);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT void sample_actions_env(DEVICE& device, Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, typename DATASET_SPEC::TI env_i, RNG& rng){
            using SPEC = typename DATASET_SPEC::SPEC;
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            using TI = typename SPEC::TI;
            constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
            const TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
            T action_log_prob = 0;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                const T action_mean = get(dataset.actions_mean, pos, action_i);
                const T action_log_std = get(log_std, 0, action_i);
                const T action_std = math::exp(device.math, action_log_std);
                const T action = random::normal_distribution::sample(device.random, action_mean, action_std, rng);
                action_log_prob += random::normal_distribution::log_prob(device.random, action_mean, action_log_std, action);
                set(dataset.actions, pos, action_i, action);
                set(device, step_actions, action, env_i, action_i);
            }
            set(dataset.action_log_probs, pos, 0, action_log_prob);
        }
        template <typename DEVICE, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
        void sample_actions(DEVICE& device, Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, RNG& rng){
            using TI = typename DATASET_SPEC::TI;
            for(TI env_i = 0; env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS; env_i++){
                sample_actions_env(device, dataset, log_std, step_actions, step_i, env_i, rng);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
        void prologue(DEVICE& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunnerBatched<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
            record_reset(device, dataset, runner.episodes.reset);
            observe_row(device, dataset, runner, environment, 0, rng);
        }
        template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ACTOR, typename ACTOR_BUFFERS, typename RNG>
        void interlude(DEVICE& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunnerBatched<SPEC>& runner, ACTOR& actor, ACTOR_BUFFERS& actor_buffers, RNG& rng, typename SPEC::TI step_i){
            using TI = typename SPEC::TI;
            constexpr TI N_ENVIRONMENTS = SPEC::N_ENVIRONMENTS;
            constexpr TI ACTION_DIM = SPEC::ENVIRONMENT::ACTION_DIM;
            Mode<mode::sequential::ResetMask<mode::Default<>, mode::sequential::ResetMaskSpecification<decltype(runner.episodes.reset)>>> mode_reset_mask;
            mode_reset_mask.mask = runner.episodes.reset;
            reset(device, actor, runner.policy_state, rng, mode_reset_mask);
            auto observations = view_range(device, dataset.all_observations, step_i * N_ENVIRONMENTS, tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            using EVAL_INPUT_SHAPE = tensor::Prepend<typename SPEC::OBSERVATION::SHAPE, N_ENVIRONMENTS>;
            auto observations_reshaped = reshape_row_major(device, observations, EVAL_INPUT_SHAPE{});
            auto actions_mean = view(device, dataset.actions_mean, matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), step_i * N_ENVIRONMENTS, 0);
            auto actions_mean_tensor = to_tensor(device, actions_mean);
            Mode<mode::Rollout<>> mode;
            evaluate_step(device, actor, observations_reshaped, runner.policy_state, actions_mean_tensor, actor_buffers, rng, mode);
            auto& last_layer = get_last_layer(actor);
            auto log_std = matrix_view(device, last_layer.log_std.parameters);
            sample_actions(device, dataset, log_std, runner.actions, step_i, rng);
        }
        template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
        void epilogue(DEVICE& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunnerBatched<SPEC>& runner, ENVIRONMENT& environment, RNG& rng, typename SPEC::TI step_i){
            step(device, environment, runner.env_parameters, runner.states, runner.actions, runner.next_states, rng);
            reward(device, environment, runner.env_parameters, runner.states, runner.actions, runner.next_states, runner.rewards, rng);
            copy(device, device, runner.next_states, runner.states);
            end_step(device, environment, runner.episodes, runner.env_parameters, runner.states, runner.rewards, rng);
            record_step(device, dataset, step_i, runner.rewards, runner.episodes.terminated, runner.episodes.truncated);
            begin_step(device, environment, runner.episodes, runner.env_parameters, runner.states, rng);
            render_if_available(device, runner, environment);
            observe_row(device, dataset, runner, environment, step_i + 1, rng);
        }
    }
    namespace rl::components::on_policy_runner{
        // applies the pending resets outside of a rollout (after force_reset / scene changes): the
        // due instances are re-sampled and the framebuffer state is brought up to date
        template <typename DEVICE, typename SPEC, typename ENVIRONMENT, typename RNG>
        void reset_due(DEVICE& device, rl::components::OnPolicyRunnerBatched<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
            begin_step(device, environment, runner.episodes, runner.env_parameters, runner.states, rng);
            render_if_available(device, runner, environment);
        }
    }
    template <typename DEVICE, typename SPEC, typename ENVIRONMENT, typename RNG>
    void init(DEVICE& device, rl::components::OnPolicyRunnerBatched<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        runner.step = 0;
        set_all(device, runner.actions, (T)0);
        set_all(device, runner.rewards, (T)0);
        init(device, runner.episodes);
        rl::components::on_policy_runner::reset_due(device, runner, environment, rng);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename ACTOR, typename ACTOR_BUFFERS, typename RNG>
    void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunnerBatched<SPEC>& runner, ENVIRONMENT& environment, ACTOR& actor, ACTOR_BUFFERS& actor_buffers, RNG& rng){
        static_assert(rl_tools::utils::typing::is_same_v<typename DATASET_SPEC::SPEC, SPEC>, "the dataset must be specified over the runner's specification");
        using TI = typename SPEC::TI;
        rl::components::on_policy_runner::prologue(device, dataset, runner, environment, rng);
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            rl::components::on_policy_runner::interlude(device, dataset, runner, actor, actor_buffers, rng, step_i);
            rl::components::on_policy_runner::epilogue(device, dataset, runner, environment, rng, step_i);
        }
        runner.step += SPEC::N_ENVIRONMENTS * DATASET_SPEC::STEPS_PER_ENV;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
