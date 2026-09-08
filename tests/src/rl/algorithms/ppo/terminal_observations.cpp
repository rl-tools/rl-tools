#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/batch/environment.h>
#include <rl_tools/rl/components/on_policy_runner/on_policy_runner.h>
#include <gtest/gtest.h>
#include <metra/metra.h>

namespace rl_tools::test_terminal{
    using T = double;
    using TI = devices::DefaultCPU::index_t;
    using TYPE_POLICY = numeric_types::Policy<T>;
    struct Environment{
        using T = test_terminal::T;
        using TI = test_terminal::TI;
        struct Parameters{};
        struct State{ TI age, episode; T observation; };
        struct Observation{
            using SHAPE = tensor::Shape<TI, 1>;
            static constexpr TI DIM = 1;
        };
        using ObservationPrivileged = Observation;
        static constexpr TI ACTION_DIM = 1, N_AGENTS = 1, EPISODE_STEP_LIMIT = 2;
        TI episodes = 0, observations = 0;
        bool terminate = false;
        T reset_observation = 1;
    };
    template <TI N> struct State{ Matrix<matrix::Specification<T, TI, N, 1>> hidden; };
    struct Buffer{};
    template <TI N, bool RECURRENT> struct Critic{
        TI step_calls = 0, bulk_calls = 0;
        template <typename, TI B> using CHANGE_BATCH_SIZE = Critic<B, RECURRENT>;
        template <bool> using State = test_terminal::State<N>;
        template <bool> using Buffer = test_terminal::Buffer;
    };
}
namespace rl_tools{
    template <typename D> void malloc(D&, test_terminal::Environment&){}
    template <typename D> void free(D&, test_terminal::Environment&){}
    template <typename D> void init(D&, test_terminal::Environment&){}
    template <typename D, typename R> void sample_initial_parameters(D&, test_terminal::Environment&, test_terminal::Environment::Parameters&, R&){}
    template <typename D, typename R> void sample_initial_state(D&, test_terminal::Environment& env, test_terminal::Environment::Parameters&, test_terminal::Environment::State& state, R&){
        state = {0, env.episodes++, env.reset_observation};
    }
    template <typename D, typename A, typename R> void step(D&, test_terminal::Environment&, test_terminal::Environment::Parameters&, const test_terminal::Environment::State& state, A&, test_terminal::Environment::State& next, R&){
        next = state;
        next.age++;
        next.observation = 10 * (next.episode + 1) + next.age;
    }
    template <typename D, typename A, typename R> double reward(D&, test_terminal::Environment&, test_terminal::Environment::Parameters&, const test_terminal::Environment::State&, A&, const test_terminal::Environment::State&, R&){ return 1; }
    template <typename D, typename R> bool terminated(D&, test_terminal::Environment& env, test_terminal::Environment::Parameters&, const test_terminal::Environment::State& state, R&){ return env.terminate && state.age == 2; }
    template <typename D, typename O, typename R> void observe(D&, test_terminal::Environment& env, test_terminal::Environment::Parameters&, const test_terminal::Environment::State& state, test_terminal::Environment::Observation, O& observation, R&){ env.observations++; set(observation, 0, 0, state.observation); }
    template <typename D, test_terminal::TI N> void malloc(D& d, test_terminal::State<N>& s){ malloc(d, s.hidden); }
    template <typename D, test_terminal::TI N> void free(D& d, test_terminal::State<N>& s){ free(d, s.hidden); }
    template <typename D, typename E, test_terminal::TI N> void copy(D& d, E& e, test_terminal::State<N>& s, test_terminal::State<N>& t){ copy(d, e, s.hidden, t.hidden); }
    template <typename D> void malloc(D&, test_terminal::Buffer&){}
    template <typename D> void free(D&, test_terminal::Buffer&){}
    template <typename D, test_terminal::TI N, bool RECURRENT, typename R> void reset(D& d, test_terminal::Critic<N, RECURRENT>&, test_terminal::State<N>& s, R&){ set_all(d, s.hidden, 0.0); }
    template <typename D, test_terminal::TI N, bool RECURRENT, typename R, typename M, typename MS> void reset(D&, test_terminal::Critic<N, RECURRENT>&, test_terminal::State<N>& s, R&, const Mode<mode::sequential::ResetMask<M, MS>>& mode){
        for(test_terminal::TI i = 0; i < N; i++) if(get(mode.mask, 0, i)) set(s.hidden, i, 0, 0.0);
    }
    template <typename D, test_terminal::TI N, bool RECURRENT, typename I, typename O, typename R, typename M> void evaluate_step(D& d, test_terminal::Critic<N, RECURRENT>& critic, const I& input, test_terminal::State<N>& s, O& output, test_terminal::Buffer&, R&, const Mode<M>&){
        critic.step_calls++;
        for(test_terminal::TI i = 0; i < N; i++){
            double value = get(d, input, i, 0) + (RECURRENT ? get(s.hidden, i, 0) : 0);
            set(s.hidden, i, 0, value);
            set(d, output, value, i, 0);
        }
    }
    template <typename D, test_terminal::TI N, bool RECURRENT, typename I, typename O, typename R, typename M> void evaluate(D& d, test_terminal::Critic<N, RECURRENT>& critic, const I& input, O& output, test_terminal::Buffer&, R&, const Mode<M>& mode){
        critic.bulk_calls++;
        constexpr auto B = get<1>(typename I::SPEC::SHAPE{});
        double hidden[B]{};
        for(test_terminal::TI t = 0; t < get<0>(typename I::SPEC::SHAPE{}); t++) for(test_terminal::TI i = 0; i < B; i++){
            if(get(d, mode.reset_container, t, i, 0)) hidden[i] = 0;
            hidden[i] = get(d, input, t, i, 0) + (RECURRENT ? hidden[i] : 0);
            set(d, output, hidden[i], t, i, 0);
        }
    }
}
#include <rl_tools/rl/environments/batch/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/operations_generic.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mux.h>

namespace rl_tools::test_terminal{
    template <bool IGNORE, bool BOOTSTRAP = true, bool RECURRENT = false> struct Parameters: rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, 14>{
        static constexpr T GAMMA = 0.9, LAMBDA = 0.8;
        static constexpr bool IGNORE_TERMINATION = IGNORE;
        static constexpr bool BOOTSTRAP_TRUNCATIONS = BOOTSTRAP, STATEFUL_ACTOR_AND_CRITIC = RECURRENT;
    };
    template <bool RECURRENT, bool IGNORE, TI STEPS = 7, bool REAL_GRU = false, bool BOOTSTRAP = true> void check(){
        constexpr TI N = 2;
        constexpr bool CAPTURE = BOOTSTRAP || IGNORE;
        using BATCH = rl::environments::batch::Independent<rl::environments::batch::Specification<Environment, N>>;
        using RS = rl::components::on_policy_runner::Specification<TYPE_POLICY, BATCH, State<N>, Environment::Observation, Environment::ObservationPrivileged, T, T, 2, false, true, CAPTURE>;
        using DS = rl::components::on_policy_runner::DatasetSpecification<RS, STEPS>;
        devices::DefaultCPU device;
        devices::generic::random::ArrayENGINE<devices::generic::random::ArraySpecification<TI, N>> rng;
        malloc(device, rng);
        init(device, rng, 0);
        BATCH environment;
        rl::components::OnPolicyRunner<RS> runner;
        rl::components::on_policy_runner::Buffer<RS> buffer;
        rl::components::on_policy_runner::Dataset<DS> dataset;
        using GRU = nn::layers::gru::BindConfiguration<nn::layers::gru::Configuration<TYPE_POLICY, TI, 1, nn::parameters::groups::Normal, false>>;
        using GRU_CRITIC = nn_models::sequential::Build<nn::capability::Forward<>, nn_models::sequential::Module<GRU>, tensor::Shape<TI, 1, N, 1>>;
        utils::typing::conditional_t<REAL_GRU, GRU_CRITIC, Critic<N, RECURRENT>> critic;
        if constexpr(REAL_GRU){
            malloc(device, critic);
            auto& gru = get_first_layer(critic);
            set_all(device, gru.weights_input.parameters, 0.0);
            set_all(device, gru.weights_hidden.parameters, 0.0);
            set_all(device, gru.biases_input.parameters, 0.0);
            set_all(device, gru.biases_hidden.parameters, 0.0);
            set_all(device, gru.initial_hidden_state.parameters, 0.0);
            set(device, gru.weights_input.parameters, 1.0, 2, 0);
        }
        rl::components::on_policy_runner::ValueState<decltype(critic), DS> critic_state;
        rl::components::on_policy_runner::ValueBuffer<decltype(critic), DS> critic_buffer;
        malloc(device, environment); malloc(device, runner); malloc(device, buffer); malloc(device, dataset); malloc(device, critic_state); malloc(device, critic_buffer);
        get_ref(device, environment.environments, 1).terminate = true;
        init(device, runner, environment, rng);
        for(TI i = 0; i < N; i++) get_ref(device, environment.environments, i).reset_observation = 1e6;
        set_all(device, dataset.scalar_data, 0.0);
        set_all(device, buffer.actions, 0.0);
        prologue(device, dataset, runner, environment, rng);
        T hidden[N]{};
        TI age[N]{}, episode[N]{1, 1};
        for(TI t = 0; t < STEPS; t++){
            evaluate_values(device, dataset, critic, critic_state, critic_buffer, rng, t);
            if constexpr(CAPTURE) for(TI i = 0; i < N; i++){
                TI pos = t * N + i;
                if(get(dataset.reset, pos, 0)) hidden[i] = 0;
                T observation = get(device, dataset.all_observations_privileged, pos, 0);
                hidden[i] = REAL_GRU ? 0.5 * math::tanh(device.math, observation) + 0.5 * hidden[i] : observation + (RECURRENT ? hidden[i] : 0);
                EXPECT_NEAR(get(dataset.values, pos, 0), hidden[i], 1e-12);
            }
            epilogue(device, dataset, runner, buffer, environment, rng, t);
            evaluate_bootstrap_values(device, dataset, buffer.next_observations_privileged, critic, critic_state, critic_buffer, rng, t);
            if constexpr(CAPTURE) for(TI i = 0; i < N; i++){
                if(t > 0 && get(dataset.reset, t * N + i, 0)){ age[i] = 0; episode[i]++; }
                T final_observation = 10 * episode[i] + ++age[i];
                EXPECT_DOUBLE_EQ(get(device, buffer.next_observations_privileged, i, 0), final_observation);
                T expected_bootstrap = REAL_GRU ? 0.5 * math::tanh(device.math, final_observation) + 0.5 * hidden[i] : final_observation + (RECURRENT ? hidden[i] : 0);
                EXPECT_NEAR(get(dataset.bootstrap_values, t * N + i, 0), expected_bootstrap, 1e-12);
                if constexpr(!REAL_GRU){ EXPECT_DOUBLE_EQ(get(critic_state.state.hidden, i, 0), hidden[i]); }
                if(get(dataset.truncated, t * N + i, 0)){ EXPECT_DOUBLE_EQ(get(device, dataset.all_observations_privileged, (t + 1) * N + i, 0), 1e6); }
            }
        }
        evaluate_rollout_values(device, dataset, critic, critic_buffer, rng, Mode<utils::typing::conditional_t<RECURRENT, mode::on_policy_runner::Sequential<mode::on_policy_runner::ActorCritic<>>, mode::on_policy_runner::ActorCritic<>>>{});
        if constexpr(!CAPTURE){
            static_assert(decltype(dataset)::SCALAR_DATA_DIM == 2 * Environment::ACTION_DIM + 8);
            static_assert(utils::typing::is_same_v<decltype(buffer.next_observations_privileged), rl::components::on_policy_runner::NoNextObservations>);
            for(TI t = 0; t <= STEPS; t++) for(TI i = 0; i < N; i++){
                TI pos = t * N + i;
                if(get(dataset.all_reset, pos, 0)) hidden[i] = 0;
                T observation = get(device, dataset.all_observations_privileged, pos, 0);
                hidden[i] = REAL_GRU ? 0.5 * math::tanh(device.math, observation) + 0.5 * hidden[i] : observation + (RECURRENT ? hidden[i] : 0);
                EXPECT_NEAR(get(dataset.all_values, pos, 0), hidden[i], 1e-12);
            }
            EXPECT_EQ(dataset.bootstrap_values._data, dataset.all_values._data + N * decltype(dataset.all_values)::ROW_PITCH);
        }
        if constexpr(!REAL_GRU){
            EXPECT_EQ(critic.step_calls, CAPTURE ? 2 * STEPS : 0);
            EXPECT_EQ(critic.bulk_calls, CAPTURE ? 0 : 1);
        }
        for(TI i = 0; i < N; i++) EXPECT_EQ(get_ref(device, environment.environments, i).observations, (CAPTURE ? 2 : 1) * STEPS + 1);
        estimate_generalized_advantages(device, dataset, dataset.bootstrap_values, Parameters<IGNORE, BOOTSTRAP, RECURRENT>{});
        for(TI t = 0; t < STEPS; t++){
            for(TI i = 0; i < N; i++){
                TI pos = t * N + i;
                if(!get(dataset.truncated, pos, 0)) continue;
                bool stop = get(dataset.terminated, pos, 0) ? !IGNORE : !BOOTSTRAP;
                T bootstrap = stop ? 0 : get(dataset.bootstrap_values, pos, 0);
                T delta = 1 + 0.9 * bootstrap - get(dataset.values, pos, 0);
                EXPECT_NEAR(get(dataset.advantages, pos, 0), delta, 1e-8);
                if(t > 0 && !get(dataset.truncated, pos - N, 0)){
                    T previous_delta = 1 + 0.9 * get(dataset.bootstrap_values, pos - N, 0) - get(dataset.values, pos - N, 0);
                    EXPECT_NEAR(get(dataset.advantages, pos - N, 0), previous_delta + 0.72 * delta, 1e-8);
                }
            }
        }
        if constexpr(STEPS % 2 == 1){
            for(TI i = 0; i < N; i++){
                TI pos = (STEPS - 1) * N + i;
                if(get(dataset.truncated, pos, 0)) continue;
                EXPECT_NEAR(get(dataset.target_values, pos, 0), 1 + 0.9 * get(dataset.bootstrap_values, pos, 0), 1e-8);
            }
        }
        if constexpr(REAL_GRU) free(device, critic);
        metra::log("ppo/terminal_observations/failures", ::testing::Test::HasFailure() ? 1.0 : 0.0);
        free(device, rng); free(device, critic_state); free(device, critic_buffer); free(device, dataset); free(device, buffer); free(device, runner); free(device, environment);
    }
}
TEST(PPO_TERMINAL_OBSERVATIONS, REPEATED_BOUNDARIES){ rl_tools::test_terminal::check<false, false>(); }
TEST(PPO_TERMINAL_OBSERVATIONS, IGNORE_TERMINATION){ rl_tools::test_terminal::check<false, true>(); }
TEST(PPO_TERMINAL_OBSERVATIONS, RECURRENT){ rl_tools::test_terminal::check<true, false>(); }
TEST(PPO_TERMINAL_OBSERVATIONS, RECURRENT_IGNORE_TERMINATION){ rl_tools::test_terminal::check<true, true>(); }
TEST(PPO_TERMINAL_OBSERVATIONS, FINAL_STEP_BOUNDARY){ rl_tools::test_terminal::check<true, false, 6>(); }
TEST(PPO_TERMINAL_OBSERVATIONS, ONE_STEP_ROLLOUT){ rl_tools::test_terminal::check<false, false, 1>(); }

TEST(PPO_TERMINAL_OBSERVATIONS, GRU_CRITIC){ rl_tools::test_terminal::check<true, false, 7, true>(); }

TEST(PPO_TERMINAL_OBSERVATIONS, NO_TRUNCATION_BOOTSTRAP){ rl_tools::test_terminal::check<false, false, 7, false, false>(); }
TEST(PPO_TERMINAL_OBSERVATIONS, NO_BOOTSTRAP_FINAL_BOUNDARY){ rl_tools::test_terminal::check<false, false, 6, false, false>(); }
TEST(PPO_TERMINAL_OBSERVATIONS, NO_BOOTSTRAP_ONE_STEP){ rl_tools::test_terminal::check<false, false, 1, false, false>(); }
TEST(PPO_TERMINAL_OBSERVATIONS, NO_BOOTSTRAP_RECURRENT){ rl_tools::test_terminal::check<true, false, 7, false, false>(); }
TEST(PPO_TERMINAL_OBSERVATIONS, NO_BOOTSTRAP_GRU){ rl_tools::test_terminal::check<true, false, 7, true, false>(); }
TEST(PPO_TERMINAL_OBSERVATIONS, NO_TRUNCATION_BOOTSTRAP_IGNORE_TERMINATION){ rl_tools::test_terminal::check<false, true, 7, false, false>(); }
