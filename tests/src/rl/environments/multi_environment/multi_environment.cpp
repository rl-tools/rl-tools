#include <rl_tools/operations/cpu.h>

#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environments/operations_generic_batch.h>
#include <rl_tools/rl/environments/multi_environment/operations_generic.h>

#include <gtest/gtest.h>

#include <cstring>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = double;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;

constexpr TI NUMBER_OF_ENVIRONMENTS = 4;
constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
constexpr TI INSTANCES = NUMBER_OF_ENVIRONMENTS * INSTANCES_PER_ENVIRONMENT;
using MULTI_ENVIRONMENT = rlt::rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>;

using PARAMETERS_TENSOR = rlt::Tensor<rlt::tensor::Specification<typename ENVIRONMENT::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>>;
using STATES_TENSOR = rlt::Tensor<rlt::tensor::Specification<typename ENVIRONMENT::State, TI, rlt::tensor::Shape<TI, INSTANCES>>>;
using RESET_TENSOR = rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>>;
using ACTIONS_TENSOR = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, ENVIRONMENT::ACTION_DIM>>>;
using OBSERVATIONS_TENSOR = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, ENVIRONMENT::Observation::DIM>>>;
using REWARDS_TENSOR = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES>>>;
using TERMINATED_TENSOR = rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>>;

constexpr TI ROLLOUT_STEPS = 50;

struct Rollout {
    typename ENVIRONMENT::State states[ROLLOUT_STEPS][INSTANCES];
    T rewards[ROLLOUT_STEPS][INSTANCES];
    bool terminated_flags[ROLLOUT_STEPS][INSTANCES];
    T observations[ROLLOUT_STEPS][INSTANCES][ENVIRONMENT::Observation::DIM];
};

template <typename ENV_LIKE>
void batch_rollout(DEVICE& device, ENV_LIKE& env, Rollout& rollout, TI seed){
    RNG rng, rng_action;
    rlt::malloc(device, rng);
    rlt::malloc(device, rng_action);
    rlt::init(device, rng, seed);
    rlt::init(device, rng_action, seed + 1);

    PARAMETERS_TENSOR parameters;
    STATES_TENSOR states, next_states;
    RESET_TENSOR reset_mask;
    ACTIONS_TENSOR actions;
    OBSERVATIONS_TENSOR observations;
    REWARDS_TENSOR rewards;
    TERMINATED_TENSOR terminated_flags;
    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, next_states);
    rlt::malloc(device, reset_mask);
    rlt::malloc(device, actions);
    rlt::malloc(device, observations);
    rlt::malloc(device, rewards);
    rlt::malloc(device, terminated_flags);

    rlt::set_all(device, reset_mask, true);
    rlt::sample_initial_parameters(device, env, parameters, reset_mask, rng);
    rlt::sample_initial_state(device, env, parameters, states, reset_mask, rng);

    for(TI step_i = 0; step_i < ROLLOUT_STEPS; step_i++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                rlt::set(device, actions, rlt::random::uniform_real_distribution(device.random, (T)-1, (T)1, rng_action), instance_i, action_i);
            }
        }
        rlt::observe(device, env, parameters, states, typename ENVIRONMENT::Observation{}, observations, rng);
        rlt::step(device, env, parameters, states, actions, next_states, rng);
        rlt::reward(device, env, parameters, states, actions, next_states, rewards, rng);
        rlt::terminated(device, env, parameters, states, terminated_flags, rng);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            rollout.states[step_i][instance_i] = rlt::get(device, next_states, instance_i);
            rollout.rewards[step_i][instance_i] = rlt::get(device, rewards, instance_i);
            rollout.terminated_flags[step_i][instance_i] = rlt::get(device, terminated_flags, instance_i);
            for(TI dim_i = 0; dim_i < ENVIRONMENT::Observation::DIM; dim_i++){
                rollout.observations[step_i][instance_i][dim_i] = rlt::get(device, observations, instance_i, dim_i);
            }
        }
        rlt::copy(device, device, next_states, states);
    }
    rlt::free(device, parameters);
    rlt::free(device, states);
    rlt::free(device, next_states);
    rlt::free(device, reset_mask);
    rlt::free(device, actions);
    rlt::free(device, observations);
    rlt::free(device, rewards);
    rlt::free(device, terminated_flags);
}

void reference_rollout(DEVICE& device, Rollout& rollout, TI seed){
    RNG rng, rng_action;
    rlt::malloc(device, rng);
    rlt::malloc(device, rng_action);
    rlt::init(device, rng, seed);
    rlt::init(device, rng_action, seed + 1);

    ENVIRONMENT env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    typename ENVIRONMENT::Parameters parameters[INSTANCES];
    typename ENVIRONMENT::State states[INSTANCES];
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        rlt::sample_initial_parameters(device, env, parameters[instance_i], rng);
    }
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        rlt::sample_initial_state(device, env, parameters[instance_i], states[instance_i], rng);
    }
    for(TI step_i = 0; step_i < ROLLOUT_STEPS; step_i++){
        T action_values[INSTANCES][ENVIRONMENT::ACTION_DIM];
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                action_values[instance_i][action_i] = rlt::random::uniform_real_distribution(device.random, (T)-1, (T)1, rng_action);
            }
        }
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM, false>> observation;
            rlt::observe(device, env, parameters[instance_i], states[instance_i], typename ENVIRONMENT::Observation{}, observation, rng);
            for(TI dim_i = 0; dim_i < ENVIRONMENT::Observation::DIM; dim_i++){
                rollout.observations[step_i][instance_i][dim_i] = rlt::get(observation, 0, dim_i);
            }
        }
        typename ENVIRONMENT::State next_states[INSTANCES];
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM, false>> action;
            for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                rlt::set(action, 0, action_i, action_values[instance_i][action_i]);
            }
            rlt::step(device, env, parameters[instance_i], states[instance_i], action, next_states[instance_i], rng);
        }
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM, false>> action;
            for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                rlt::set(action, 0, action_i, action_values[instance_i][action_i]);
            }
            rollout.rewards[step_i][instance_i] = rlt::reward(device, env, parameters[instance_i], states[instance_i], action, next_states[instance_i], rng);
        }
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            rollout.terminated_flags[step_i][instance_i] = rlt::terminated(device, env, parameters[instance_i], states[instance_i], rng);
        }
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            rollout.states[step_i][instance_i] = next_states[instance_i];
            states[instance_i] = next_states[instance_i];
        }
    }
    rlt::free(device, env);
}

void expect_rollouts_equal(const Rollout& a, const Rollout& b){
    for(TI step_i = 0; step_i < ROLLOUT_STEPS; step_i++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            ASSERT_EQ(std::memcmp(&a.states[step_i][instance_i], &b.states[step_i][instance_i], sizeof(typename ENVIRONMENT::State)), 0) << "state diverged at step " << step_i << " instance " << instance_i;
            ASSERT_EQ(a.rewards[step_i][instance_i], b.rewards[step_i][instance_i]) << "reward diverged at step " << step_i << " instance " << instance_i;
            ASSERT_EQ(a.terminated_flags[step_i][instance_i], b.terminated_flags[step_i][instance_i]);
            for(TI dim_i = 0; dim_i < ENVIRONMENT::Observation::DIM; dim_i++){
                ASSERT_EQ(a.observations[step_i][instance_i][dim_i], b.observations[step_i][instance_i][dim_i]);
            }
        }
    }
}

// the generic mapped defaults are today's per-instance loops, factored out: bit-exact
TEST(RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT, MAPPED_DEFAULT_BIT_EXACT){
    DEVICE device;
    auto* reference = new Rollout;
    auto* batch = new Rollout;
    reference_rollout(device, *reference, 1337);
    ENVIRONMENT env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    batch_rollout(device, env, *batch, 1337);
    expect_rollouts_equal(*reference, *batch);
    rlt::free(device, env);
    delete reference;
    delete batch;
}

// the composite is itself an environment: fan-out over contiguous member blocks matches the
// flat mapped default bit-exactly
TEST(RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT, COMPOSITE_FAN_OUT_BIT_EXACT){
    DEVICE device;
    auto* reference = new Rollout;
    auto* composite = new Rollout;
    reference_rollout(device, *reference, 1337);
    MULTI_ENVIRONMENT multi_env;
    rlt::malloc(device, multi_env);
    rlt::init(device, multi_env);
    batch_rollout(device, multi_env, *composite, 1337);
    expect_rollouts_equal(*reference, *composite);
    rlt::free(device, multi_env);
    delete reference;
    delete composite;
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT, MASKED_RESET){
    DEVICE device;
    ENVIRONMENT env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    PARAMETERS_TENSOR parameters;
    STATES_TENSOR states;
    RESET_TENSOR reset_mask;
    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, reset_mask);
    rlt::set_all(device, reset_mask, true);
    rlt::sample_initial_parameters(device, env, parameters, reset_mask, rng);
    rlt::sample_initial_state(device, env, parameters, states, reset_mask, rng);
    typename ENVIRONMENT::State before[INSTANCES];
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        before[instance_i] = rlt::get(device, states, instance_i);
    }
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        rlt::set(device, reset_mask, instance_i % 2 == 0, instance_i);
    }
    rlt::sample_initial_state(device, env, parameters, states, reset_mask, rng);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename ENVIRONMENT::State after = rlt::get(device, states, instance_i);
        bool changed = std::memcmp(&before[instance_i], &after, sizeof(typename ENVIRONMENT::State)) != 0;
        if(instance_i % 2 == 0){
            EXPECT_TRUE(changed) << "masked instance " << instance_i << " was not resampled";
        }
        else{
            EXPECT_FALSE(changed) << "unmasked instance " << instance_i << " was resampled";
        }
    }
    rlt::free(device, env);
    rlt::free(device, parameters);
    rlt::free(device, states);
    rlt::free(device, reset_mask);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT, CONTRACT_CONSTANTS){
    static_assert(MULTI_ENVIRONMENT::NUMBER_OF_ENVIRONMENTS == NUMBER_OF_ENVIRONMENTS);
    static_assert(MULTI_ENVIRONMENT::ACTION_DIM == ENVIRONMENT::ACTION_DIM);
    static_assert(MULTI_ENVIRONMENT::N_AGENTS == 1);
    static_assert(rlt::utils::typing::is_same_v<MULTI_ENVIRONMENT::State, ENVIRONMENT::State>);
    // Pendulum does not pin an instance count; the verbs slice by tensor shape
    static_assert(MULTI_ENVIRONMENT::INSTANCES_PER_ENVIRONMENT == 0);
}
