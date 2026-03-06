#include <rl_tools/operations/cpu.h>

#include <rl_tools/rl/environments/reacher/operations_generic.h>
#include <rl_tools/rl/environments/observation.h>

#include <gtest/gtest.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
using REACHER_SPEC = rlt::rl::environments::reacher::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environments::Reacher<REACHER_SPEC>;

TEST(RL_TOOLS_RL_ENVIRONMENTS_REACHER, BASIC_STEP){
    DEVICE device;
    ENVIRONMENT env;
    ENVIRONMENT::Parameters parameters;
    ENVIRONMENT::State state, next_state;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    rlt::sample_initial_state(device, env, parameters, state, rng);

    EXPECT_GE(state.x, -1.0f);
    EXPECT_LE(state.x, 1.0f);
    EXPECT_GE(state.y, -1.0f);
    EXPECT_LE(state.y, 1.0f);
    EXPECT_GE(state.target_x, -1.0f);
    EXPECT_LE(state.target_x, 1.0f);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 2, false>> action;
    rlt::set(action, 0, 0, 1.0f);
    rlt::set(action, 0, 1, 0.0f);

    rlt::step(device, env, parameters, state, action, next_state, rng);

    T expected_dx = ENVIRONMENT::Parameters::MAX_VELOCITY * ENVIRONMENT::Parameters::DT;
    T expected_x = rlt::rl::environments::reacher::clip(state.x + expected_dx, -ENVIRONMENT::Parameters::ARENA_SIZE, ENVIRONMENT::Parameters::ARENA_SIZE);
    EXPECT_NEAR(next_state.x, expected_x, 1e-5f);
    EXPECT_NEAR(next_state.y, state.y, 1e-5f);
    EXPECT_EQ(next_state.target_x, state.target_x);
    EXPECT_EQ(next_state.target_y, state.target_y);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_REACHER, OBSERVE_DENSE){
    DEVICE device;
    ENVIRONMENT env;
    ENVIRONMENT::Parameters parameters;
    ENVIRONMENT::State state;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    state.x = 0.1f;
    state.y = 0.2f;
    state.target_x = 0.3f;
    state.target_y = 0.4f;

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4, false>> observation;
    rlt::observe(device, env, parameters, state, rlt::rl::environments::reacher::ObservationDense<TI>{}, observation, rng);

    EXPECT_NEAR(rlt::get(observation, 0, 0), 0.1f, 1e-6f);
    EXPECT_NEAR(rlt::get(observation, 0, 1), 0.2f, 1e-6f);
    EXPECT_NEAR(rlt::get(observation, 0, 2), 0.3f, 1e-6f);
    EXPECT_NEAR(rlt::get(observation, 0, 3), 0.4f, 1e-6f);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_REACHER, REWARD){
    DEVICE device;
    ENVIRONMENT env;
    ENVIRONMENT::Parameters parameters;
    ENVIRONMENT::State state, next_state;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 2, false>> action;
    rlt::set(action, 0, 0, 0.0f);
    rlt::set(action, 0, 1, 0.0f);

    next_state.x = 0.0f;
    next_state.y = 0.0f;
    next_state.target_x = 0.3f;
    next_state.target_y = 0.4f;

    T r = rlt::reward(device, env, parameters, state, action, next_state, rng);
    T expected_distance = std::sqrt(0.3f * 0.3f + 0.4f * 0.4f);
    EXPECT_NEAR(r, -expected_distance, 1e-5f);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_REACHER, TERMINATED){
    DEVICE device;
    ENVIRONMENT env;
    ENVIRONMENT::Parameters parameters;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    ENVIRONMENT::State state;
    state.x = 0.0f;
    state.y = 0.0f;
    state.target_x = 1.0f;
    state.target_y = 1.0f;
    EXPECT_FALSE(rlt::terminated(device, env, parameters, state, rng));

    state.target_x = 0.01f;
    state.target_y = 0.01f;
    EXPECT_TRUE(rlt::terminated(device, env, parameters, state, rng));
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_REACHER, OBSERVATION_SHAPE_TRAIT){
    using OBS = ENVIRONMENT::Observation;
    using SHAPE = OBS::SHAPE;
    static_assert(rlt::length(SHAPE{}) == 1, "Dense observation should have rank 1");
    static_assert(rlt::get<0>(SHAPE{}) == 4, "Dense observation dim should be 4");
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_REACHER, IMAGE_OBSERVATION_SHAPE_TRAIT){
    using IMAGE_OBS = rlt::rl::environments::reacher::ObservationImage<TI, 32, 32>;
    using SHAPE = IMAGE_OBS::SHAPE;
    static_assert(rlt::length(SHAPE{}) == 3, "Image observation should have rank 3");
    static_assert(rlt::get<0>(SHAPE{}) == 32, "Height should be 32");
    static_assert(rlt::get<1>(SHAPE{}) == 32, "Width should be 32");
    static_assert(rlt::get<2>(SHAPE{}) == 3, "Channels should be 3");
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_REACHER, COMPOSE_TRAIT){
    using IMAGE_OBS = rlt::rl::environments::reacher::ObservationImage<TI, 32, 32>;
    using DENSE_OBS = rlt::rl::environments::reacher::ObservationDense<TI>;
    using COMPOSED = rlt::rl::environments::observation::Compose<IMAGE_OBS, DENSE_OBS>;
    static_assert(rlt::rl::environments::observation::is_compose_v<COMPOSED>, "Compose should be detected");
    static_assert(!rlt::rl::environments::observation::is_compose_v<DENSE_OBS>, "Dense should not be detected as compose");
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_REACHER, ABS_DIFF){
    DEVICE device;
    ENVIRONMENT::State s1, s2;
    s1.x = 1.0f; s1.y = 2.0f; s1.target_x = 3.0f; s1.target_y = 4.0f;
    s2.x = 1.5f; s2.y = 2.5f; s2.target_x = 3.5f; s2.target_y = 4.5f;
    T diff = rlt::abs_diff(device, s1, s2);
    EXPECT_NEAR(diff, 2.0f, 1e-6f);
}
