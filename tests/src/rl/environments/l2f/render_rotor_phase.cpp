#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = double;
using TI = typename DEVICE::index_t;

constexpr TI ACTION_HISTORY_LENGTH = 4;

template <typename T_STATE_TYPE>
struct STATIC_PARAMETERS_TEMPLATE{
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    static constexpr bool CLOSED_FORM = false;

    using STATE_TYPE = T_STATE_TYPE;

    using OBSERVATION_TYPE =
        l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
        l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
        l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
        l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI,
        l2f::observation::ActionHistory<l2f::observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
    using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
    using BASE = l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI>;
    using PARAMETERS = typename BASE::PARAMETERS_TYPE;
    static constexpr auto PARAMETER_VALUES = BASE::nominal_parameters;
    static constexpr T STATE_LIMIT_POSITION_X = 100000;
    static constexpr T STATE_LIMIT_POSITION_Y = 100000;
    static constexpr T STATE_LIMIT_POSITION_Z = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
};

using STATE_PLAIN = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, false, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, l2f::StateBase<l2f::StateSpecification<T, TI>>>>>>>>;
using STATE_RENDER = l2f::StateRenderRotorPhase<l2f::StateSpecification<T, TI, STATE_PLAIN>>;

using ENV_PLAIN = rlt::rl::environments::Multirotor<l2f::Specification<T, TI, STATIC_PARAMETERS_TEMPLATE<STATE_PLAIN>>>;
using ENV_RENDER = rlt::rl::environments::Multirotor<l2f::Specification<T, TI, STATIC_PARAMETERS_TEMPLATE<STATE_RENDER>>>;

constexpr T TWO_PI = 2 * M_PI;

TEST(L2F_RENDER_ROTOR_PHASE, INITIAL_STATE_ZERO){
    DEVICE device;
    ENV_RENDER env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    typename ENV_RENDER::Parameters parameters;
    rlt::initial_parameters(device, env, parameters);
    typename ENV_RENDER::State state;
    rlt::initial_state(device, env, parameters, state);
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        EXPECT_EQ(state.rotor_phase[rotor_i], (T)0);
    }
    RNG rng;
    rlt::init(device, rng, 0);
    rlt::sample_initial_state(device, env, parameters, state, rng);
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        EXPECT_EQ(state.rotor_phase[rotor_i], (T)0);
    }
    rlt::free(device, env);
}

TEST(L2F_RENDER_ROTOR_PHASE, PHASE_INTEGRATION_AND_WRAP){
    DEVICE device;
    ENV_RENDER env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    typename ENV_RENDER::Parameters parameters;
    rlt::initial_parameters(device, env, parameters);
    RNG rng;
    rlt::init(device, rng, 0);
    typename ENV_RENDER::State state, next_state;
    rlt::sample_initial_state(device, env, parameters, state, rng);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV_RENDER::ACTION_DIM, false>> action;
    for(TI i = 0; i < ENV_RENDER::ACTION_DIM; i++){
        rlt::set(action, 0, i, (T)0.5);
    }

    for(TI step_i = 0; step_i < 200; step_i++){
        rlt::step(device, env, parameters, state, action, next_state, rng);
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            T expected = state.rotor_phase[rotor_i] + next_state.rpm[rotor_i] * (TWO_PI / (T)60) * parameters.integration.dt;
            expected = expected - std::floor(expected / TWO_PI) * TWO_PI;
            EXPECT_EQ(next_state.rotor_phase[rotor_i], expected);
            EXPECT_GE(next_state.rotor_phase[rotor_i], (T)0);
            EXPECT_LT(next_state.rotor_phase[rotor_i], TWO_PI);
        }
        state = next_state;
    }
    bool phase_advanced = false;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        phase_advanced = phase_advanced || state.rotor_phase[rotor_i] != (T)0;
    }
    EXPECT_TRUE(phase_advanced);
    rlt::free(device, env);
}

TEST(L2F_RENDER_ROTOR_PHASE, JSON_ROUND_TRIP){
    DEVICE device;
    ENV_RENDER env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    typename ENV_RENDER::Parameters parameters;
    rlt::initial_parameters(device, env, parameters);
    typename ENV_RENDER::State state{}, restored{};
    rlt::initial_state(device, env, parameters, state);
    state.rotor_phase[0] = 0.1;
    state.rotor_phase[1] = 1.2;
    state.rotor_phase[2] = 2.3;
    state.rotor_phase[3] = 3.4;
    std::string json_string = rlt::json(device, env, parameters, state);
    EXPECT_NE(json_string.find("rotor_phase"), std::string::npos);
    rlt::from_json(device, env, parameters, json_string, restored);
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        EXPECT_NEAR(restored.rotor_phase[rotor_i], state.rotor_phase[rotor_i], 1e-5);
    }
    rlt::free(device, env);
}

// StateRender* contract: stripping the component leaves the underlying chain (and the RNG
// stream) bit-exact
TEST(L2F_RENDER_ROTOR_PHASE, STRIP_INVARIANCE){
    DEVICE device;
    ENV_PLAIN env_plain;
    ENV_RENDER env_render;
    rlt::malloc(device, env_plain);
    rlt::malloc(device, env_render);
    rlt::init(device, env_plain);
    rlt::init(device, env_render);
    typename ENV_PLAIN::Parameters parameters_plain;
    typename ENV_RENDER::Parameters parameters_render;
    RNG rng_plain, rng_render, rng_action;
    rlt::init(device, rng_plain, 1337);
    rlt::init(device, rng_render, 1337);
    rlt::init(device, rng_action, 42);
    rlt::sample_initial_parameters(device, env_plain, parameters_plain, rng_plain);
    rlt::sample_initial_parameters(device, env_render, parameters_render, rng_render);
    typename ENV_PLAIN::State state_plain{}, next_state_plain{};
    typename ENV_RENDER::State state_render{}, next_state_render{};
    rlt::sample_initial_state(device, env_plain, parameters_plain, state_plain, rng_plain);
    rlt::sample_initial_state(device, env_render, parameters_render, state_render, rng_render);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV_PLAIN::ACTION_DIM, false>> action;
    for(TI step_i = 0; step_i < 100; step_i++){
        for(TI i = 0; i < ENV_PLAIN::ACTION_DIM; i++){
            rlt::set(action, 0, i, rlt::random::uniform_real_distribution(device.random, (T)-1, (T)1, rng_action));
        }
        rlt::step(device, env_plain, parameters_plain, state_plain, action, next_state_plain, rng_plain);
        rlt::step(device, env_render, parameters_render, state_render, action, next_state_render, rng_render);
        T reward_plain = rlt::reward(device, env_plain, parameters_plain, state_plain, action, next_state_plain, rng_plain);
        T reward_render = rlt::reward(device, env_render, parameters_render, state_render, action, next_state_render, rng_render);
        EXPECT_EQ(reward_plain, reward_render);
        bool terminated_plain = rlt::terminated(device, env_plain, parameters_plain, next_state_plain, rng_plain);
        bool terminated_render = rlt::terminated(device, env_render, parameters_render, next_state_render, rng_render);
        EXPECT_EQ(terminated_plain, terminated_render);
        ASSERT_EQ(std::memcmp(&static_cast<const STATE_PLAIN&>(next_state_render), &next_state_plain, sizeof(STATE_PLAIN)), 0) << "diverged at step " << step_i;
        state_plain = next_state_plain;
        state_render = next_state_render;
    }
    rlt::free(device, env_plain);
    rlt::free(device, env_render);
}
