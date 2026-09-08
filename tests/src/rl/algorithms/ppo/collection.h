#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/random/operations_generic_array.h>
#include <rl_tools/rl/environments/batch/environment.h>
#include <gtest/gtest.h>
#include <metra/metra.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

namespace rl_tools::test_ppo_collection{
    using T = double;
    using TI = unsigned int;
    using TP = numeric_types::Policy<T>;
    constexpr TI N = 3;
    struct Environment{
        using T = test_ppo_collection::T;
        using TI = test_ppo_collection::TI;
        struct Parameters{ T sample; };
        struct State{ TI age; T value; };
        struct Observation{
            using SHAPE = tensor::Shape<TI, 1>;
            static constexpr TI DIM = 1;
        };
        struct ObservationPrivileged: Observation{};
        static constexpr TI ACTION_DIM = 1, N_AGENTS = 1, EPISODE_STEP_LIMIT = 4;
        TI period = 0, episodes = 0;
    };
}
namespace rl_tools{
    template <typename D> RL_TOOLS_FUNCTION_PLACEMENT void malloc(D&, test_ppo_collection::Environment&){}
    template <typename D> RL_TOOLS_FUNCTION_PLACEMENT void free(D&, test_ppo_collection::Environment&){}
    template <typename D> RL_TOOLS_FUNCTION_PLACEMENT void init(D&, test_ppo_collection::Environment&){}
    template <typename D, typename R> RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_parameters(D& d, test_ppo_collection::Environment&, test_ppo_collection::Environment::Parameters& p, R& rng){
        p.sample = random::uniform_real_distribution(d.random, 0.0, 1.0, rng);
    }
    template <typename D, typename R> RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_state(D&, test_ppo_collection::Environment& env, test_ppo_collection::Environment::Parameters&, test_ppo_collection::Environment::State& state, R&){
        state = {0, -0.4 - 0.01 * env.episodes++};
    }
    template <typename D, typename A, typename R> RL_TOOLS_FUNCTION_PLACEMENT void step(D&, test_ppo_collection::Environment&, test_ppo_collection::Environment::Parameters&, const test_ppo_collection::Environment::State& state, A&, test_ppo_collection::Environment::State& next, R&){
        next = {state.age + 1, state.age == 0 ? 0.1 : state.value + 0.1};
    }
    template <typename D, typename A, typename R> RL_TOOLS_FUNCTION_PLACEMENT double reward(D&, test_ppo_collection::Environment&, test_ppo_collection::Environment::Parameters&, const test_ppo_collection::Environment::State&, A&, const test_ppo_collection::Environment::State&, R&){ return 1; }
    template <typename D, typename R> RL_TOOLS_FUNCTION_PLACEMENT bool terminated(D&, test_ppo_collection::Environment& env, test_ppo_collection::Environment::Parameters&, const test_ppo_collection::Environment::State& state, R&){ return env.period != 0 && state.age == env.period; }
    template <typename D, typename O, typename R> RL_TOOLS_FUNCTION_PLACEMENT void observe(D&, test_ppo_collection::Environment&, test_ppo_collection::Environment::Parameters&, const test_ppo_collection::Environment::State& state, test_ppo_collection::Environment::Observation, O& output, R&){ set(output, 0, 0, state.value); }
    template <typename D, typename O, typename R> RL_TOOLS_FUNCTION_PLACEMENT void observe(D&, test_ppo_collection::Environment&, test_ppo_collection::Environment::Parameters&, const test_ppo_collection::Environment::State& state, test_ppo_collection::Environment::ObservationPrivileged, O& output, R&){ set(output, 0, 0, state.value + 0.05); }
}

#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mux.h>
#include <rl_tools/rl/algorithms/ppo/operations_generic.h>
#ifdef RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/rl/algorithms/ppo/operations_cuda.h>
#endif

namespace rl_tools::test_ppo_collection{
    template <TI STEPS, bool BOOTSTRAP, bool IGNORE> struct Parameters: rl::algorithms::ppo::DefaultParameters<TP, TI, N * STEPS>{
        static constexpr T GAMMA = 0.9, LAMBDA = 0.8;
        static constexpr bool STATEFUL_ACTOR_AND_CRITIC = true, TRUNCATE_ON_EACH_ITERATION = true, SHUFFLE_EPOCH = false;
        static constexpr bool BOOTSTRAP_TRUNCATIONS = BOOTSTRAP, IGNORE_TERMINATION = IGNORE;
        static constexpr TI N_EPOCHS = 1;
    };
    template <typename DEVICE, bool BOOTSTRAP, bool IGNORE, TI STEPS = 5, bool GRU_CRITIC = true>
    void check(DEVICE& device){
        devices::DefaultCPU cpu;
        using P = Parameters<STEPS, BOOTSTRAP, IGNORE>;
        constexpr bool CAPTURE = BOOTSTRAP || IGNORE;
        using BATCH = rl::environments::batch::Independent<rl::environments::batch::Specification<Environment, N>>;
        using GRU = nn::layers::gru::BindConfiguration<nn::layers::gru::Configuration<TP, TI, 1, nn::parameters::groups::Normal, false>>;
        using DENSE = nn::layers::dense::BindConfiguration<nn::layers::dense::Configuration<TP, TI, 1, nn::activation_functions::IDENTITY>>;
        using MLP = nn_models::mlp_unconditional_stddev::BindConfiguration<nn_models::mlp::Configuration<TP, TI, 1, 2, 2, nn::activation_functions::TANH, nn::activation_functions::IDENTITY>>;
        using SHAPE = tensor::Shape<TI, STEPS, N, 1>;
        using ACTOR = nn_models::sequential::Build<nn::capability::Forward<>, nn_models::sequential::Module<GRU, MLP>, SHAPE>;
        using CRITIC = nn_models::sequential::Build<nn::capability::Forward<>, nn_models::sequential::Module<utils::typing::conditional_t<GRU_CRITIC, GRU, DENSE>>, SHAPE>;
        using PS = rl::algorithms::ppo::Specification<TP, TI, Environment, ACTOR, CRITIC, P>;
        using RS = rl::components::on_policy_runner::Specification<TP, BATCH, typename ACTOR::template State<>, Environment::Observation, Environment::ObservationPrivileged, T, T, 4, true, true, CAPTURE>;
        using DS = rl::components::on_policy_runner::DatasetSpecification<RS, STEPS>;
        using RNG = devices::generic::random::ArrayENGINE<devices::generic::random::ArraySpecification<TI, N>>;
        rl::algorithms::PPO<PS> ppo, host_ppo;
        rl::components::OnPolicyRunner<RS> runner;
        rl::components::on_policy_runner::Buffer<RS> buffer;
        rl::components::on_policy_runner::Dataset<DS> dataset, host;
        rl::components::on_policy_runner::ValueState<CRITIC, DS> critic_state;
        rl::components::on_policy_runner::ValueBuffer<CRITIC, DS> critic_buffer;
        typename ACTOR::template Buffer<> actor_buffer, replay_actor_buffer;
        typename CRITIC::template Buffer<> replay_critic_buffer;
        typename CRITIC::template State<> host_critic_state;
        Tensor<tensor::Specification<T, TI, SHAPE>> actor_output, critic_output;
        BATCH environment, host_environment;
        RNG rng, host_rng;
        malloc(cpu, host_rng); init(cpu, host_rng, 19);
        malloc(cpu, host_ppo); init_weights(cpu, host_ppo.actor, host_rng);
        auto& critic_layer = get_first_layer(host_ppo.critic);
        if constexpr(GRU_CRITIC){
            set_all(cpu, critic_layer.weights_input.parameters, 0.0);
            set_all(cpu, critic_layer.weights_hidden.parameters, 0.0);
            set_all(cpu, critic_layer.biases_input.parameters, 0.0);
            set_all(cpu, critic_layer.biases_hidden.parameters, 0.0);
            set_all(cpu, critic_layer.initial_hidden_state.parameters, 0.25);
            set(cpu, critic_layer.weights_input.parameters, 1.0, 2, 0);
        }
        else{
            set_all(cpu, critic_layer.weights.parameters, 0.5);
            set_all(cpu, critic_layer.biases.parameters, 0.25);
        }
        set_all(cpu, get_last_layer(host_ppo.actor).log_std.parameters, -1.0);
        malloc(device, ppo); copy(cpu, device, host_ppo.actor, ppo.actor); copy(cpu, device, host_ppo.critic, ppo.critic);
        malloc(device, rng); copy(cpu, device, host_rng, rng);
        malloc(cpu, host_environment);
        for(TI i = 0; i < N; i++) get_ref(cpu, host_environment.environments, i) = {i == 0 ? 2u : i == 1 ? 0u : 3u, 0};
        malloc(device, environment); copy(cpu, device, host_environment.environments, environment.environments);
        malloc(device, runner); init(device, runner, environment, rng);
        malloc(device, buffer); malloc(device, dataset); malloc(cpu, host); malloc(device, critic_state); malloc(device, critic_buffer);
        malloc(device, actor_buffer); malloc(cpu, replay_actor_buffer); malloc(cpu, replay_critic_buffer);
        malloc(cpu, host_critic_state); malloc(cpu, actor_output); malloc(cpu, critic_output);
        std::uint64_t digest = 14695981039346656037ull;
        auto hash = [&](const auto& value){
            unsigned char bytes[sizeof(value)];
            std::memcpy(bytes, &value, sizeof(value));
            for(unsigned char b: bytes) digest = (digest ^ b) * 1099511628211ull;
        };
        for(TI rollout = 0; rollout < 3; rollout++){
            set_all(device, dataset.scalar_data, 0.0);
            collect(device, dataset, runner, buffer, environment, ppo.actor, actor_buffer, ppo.critic, critic_state, critic_buffer, rng, typename decltype(ppo)::SPEC::COLLECTION_MODE{});
            estimate_generalized_advantages(device, dataset, dataset.bootstrap_values, P{});
            copy(device, cpu, dataset.scalar_data, host.scalar_data);
            copy(device, cpu, dataset.all_observations, host.all_observations);
            copy(device, cpu, dataset.all_observations_privileged, host.all_observations_privileged);
            copy(device, cpu, rng, host_rng);
            T hidden[N]; TI counts[N]{};
            for(TI i = 0; i < N; i++) hidden[i] = 0.25;
            for(TI t = 0; t < STEPS; t++) for(TI i = 0; i < N; i++){
                const TI pos = t * N + i;
                const TI period = i == 0 ? 2 : i == 1 ? 4 : 3;
                EXPECT_EQ(bool(get(host.terminated, pos, 0)), i != 1 && (t + 1) % period == 0);
                EXPECT_EQ(bool(get(host.truncated, pos, 0)), (t + 1) % period == 0);
                EXPECT_EQ(bool(get(host.reset, pos, 0)), t % period == 0);
                if(t == 0 || get(host.reset, pos, 0)){ hidden[i] = 0.25; counts[i] = 0; }
                const T observation = get(cpu, host.all_observations_privileged, pos, 0);
                hidden[i] = GRU_CRITIC ? 0.5 * std::tanh(observation) + 0.5 * hidden[i] : 0.5 * observation + 0.25;
                counts[i]++;
                EXPECT_NEAR(get(host.values, pos, 0), hidden[i], 1e-10);
                const T actor_observation = get(cpu, host.all_observations, pos, 0);
                const T next = actor_observation < 0 ? 0.15 : actor_observation + 0.15;
                if constexpr(CAPTURE){ EXPECT_NEAR(get(host.bootstrap_values, pos, 0), GRU_CRITIC ? 0.5 * std::tanh(next) + 0.5 * hidden[i] : 0.5 * next + 0.25, 1e-10); }
            }
            if constexpr(CAPTURE && GRU_CRITIC){
                copy(device, cpu, critic_state.state, host_critic_state);
                auto& state = nn_models::sequential::content_state<0>(host_critic_state.content_state);
                for(TI i = 0; i < N; i++){
                    EXPECT_NEAR(get(cpu, state.state, i, 0), hidden[i], 1e-10);
                    EXPECT_EQ(get(cpu, state.step, i), counts[i]);
                }
                copy(device, cpu, critic_buffer.bootstrap_state, host_critic_state);
                for(TI i = 0; i < N; i++){
                    EXPECT_NEAR(get(cpu, state.state, i, 0), get(host.bootstrap_values, (STEPS - 1) * N + i, 0), 1e-10);
                    EXPECT_EQ(get(cpu, state.step, i), counts[i] + 1);
                }
            }
            auto input = reshape_row_major(cpu, host.all_observations, tensor::Shape<TI, STEPS + 1, N, 1>{});
            auto privileged = reshape_row_major(cpu, host.all_observations_privileged, tensor::Shape<TI, STEPS + 1, N, 1>{});
            auto actor_input = view_range(cpu, input, 0, tensor::ViewSpec<0, STEPS>{});
            auto critic_input = view_range(cpu, privileged, 0, tensor::ViewSpec<0, STEPS>{});
            auto reset_tensor = to_tensor(cpu, host.reset);
            auto resets = reshape_row_major(cpu, reset_tensor, SHAPE{});
            Mode<mode::sequential::ResetMode<mode::Rollout<>, mode::sequential::ResetModeSpecification<TI, decltype(resets)>>> mode;
            mode.reset_container = resets;
            evaluate(cpu, host_ppo.actor, actor_input, actor_output, replay_actor_buffer, host_rng, mode);
            evaluate(cpu, host_ppo.critic, critic_input, critic_output, replay_critic_buffer, host_rng, mode);
            for(TI t = 0; t < STEPS; t++) for(TI i = 0; i < N; i++){
                EXPECT_NEAR(get(cpu, actor_output, t, i, 0), get(host.actions_mean, t * N + i, 0), 1e-10);
                EXPECT_NEAR(get(cpu, critic_output, t, i, 0), get(host.values, t * N + i, 0), 1e-10);
            }
            for(TI i = 0; i < N; i++){
                T advantage = 0;
                for(TI t = STEPS; t-- > 0;){
                    const TI pos = t * N + i;
                    const bool terminated = get(host.terminated, pos, 0), reset = get(host.truncated, pos, 0);
                    const bool stop = terminated ? !IGNORE : reset && !BOOTSTRAP;
                    const T next = stop ? 0 : get(host.bootstrap_values, pos, 0);
                    advantage = 1 + P::GAMMA * next - get(host.values, pos, 0) + (reset ? 0 : P::GAMMA * P::LAMBDA * advantage);
                    EXPECT_NEAR(get(host.advantages, pos, 0), advantage, 1e-10);
                }
            }
            for(TI row = 0; row < decltype(host.scalar_data)::ROWS; row++) for(TI col = 0; col < decltype(host.scalar_data)::COLS; col++) hash(get(host.scalar_data, row, col));
            for(TI i = 0; i < N; i++) hash(get(host_rng.states, 0, i));
        }
        ::testing::Test::RecordProperty("rollout_rng_digest", std::to_string(digest));
        metra::log("ppo/collection/recurrent_failures", ::testing::Test::HasFailure() ? 1.0 : 0.0);
        free(cpu, critic_output); free(cpu, actor_output); free(cpu, host_critic_state);
        free(cpu, replay_critic_buffer); free(cpu, replay_actor_buffer); free(device, actor_buffer);
        free(device, critic_state); free(device, critic_buffer); free(cpu, host); free(device, dataset); free(device, buffer);
        free(device, runner); free(device, environment); free(cpu, host_environment);
        free(device, rng); free(cpu, host_rng); free(device, ppo); free(cpu, host_ppo);
    }
}
