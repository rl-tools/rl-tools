#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

namespace rlt = rl_tools;
using rlt::prologue;
using rlt::epilogue;
using rlt::reset;
namespace l2f = rlt::rl::environments::l2f;
namespace on_policy_runner = rlt::rl::components::on_policy_runner;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
static const std::string SCENE_PATH = "";
#endif

namespace test_hyperdrone_episodes {
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

    struct DYNAMICS_STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;
        static constexpr TI EPISODE_STEP_LIMIT = test_hyperdrone_episodes::EPISODE_STEP_LIMIT;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
        using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr auto dynamics = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)0.01};
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, REWARD_FUNCTION{}, {}, {}, {}};
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
        static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{dynamics, integration, mdp}, disturbances};
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

    struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS> {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        using SHADING = rlt::rendering::raytracing::Low;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI NUMBER_OF_ENVIRONMENTS = 2;
    using ENVIRONMENT = rlt::rl::environments::hyperdrone::MultiEnvironment<WORLD, NUMBER_OF_ENVIRONMENTS>;
    constexpr TI INSTANCES = ENVIRONMENT::INSTANCES;
    using POLICY_STATE = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1>>>;
}

using namespace test_hyperdrone_episodes;

static std::string scene_directory(){
    std::string directory = std::filesystem::temp_directory_path() / "rl_tools_hyperdrone_episodes_scenes";
    std::filesystem::create_directories(directory);
    for (const char* name : {"scene_0.glb", "scene_1.glb"}) {
        std::filesystem::path link = std::filesystem::path(directory) / name;
        if (!std::filesystem::exists(link)) {
            std::filesystem::create_symlink(SCENE_PATH, link);
        }
    }
    return directory;
}

template <TI STEPS, TI STEP_LIMIT = 3>
struct Harness {
    using RUNNER_SPEC = on_policy_runner::Specification<rlt::numeric_types::Policy<T>, ENVIRONMENT, POLICY_STATE, ENVIRONMENT::Observation, ENVIRONMENT::ObservationPrivileged, T, T, STEP_LIMIT>;
    using RUNNER = rlt::rl::components::OnPolicyRunner<RUNNER_SPEC>;
    using BUFFER = on_policy_runner::Buffer<RUNNER_SPEC>;
    using DATASET_SPEC = on_policy_runner::DatasetSpecification<RUNNER_SPEC, STEPS>;
    using DATASET = on_policy_runner::Dataset<DATASET_SPEC>;

    DEVICE& device;
    ENVIRONMENT& env;
    RNG rng;
    RUNNER runner;
    BUFFER buffer;
    DATASET dataset;

    Harness(DEVICE& device, ENVIRONMENT& env, TI seed): device(device), env(env){
        rlt::malloc(device, rng);
        rlt::init(device, rng, seed);
        rlt::malloc(device, runner);
        rlt::malloc(device, buffer);
        rlt::malloc(device, dataset);
        rlt::set_all(device, buffer.actions, (T)0);
        rlt::init(device, runner, env, rng);
        prologue(device, dataset, runner, env, rng);
    }
    ~Harness(){
        rlt::free(device, dataset);
        rlt::free(device, buffer);
        rlt::free(device, runner);
        rlt::free(device, rng);
    }
    void step(TI step_i){
        epilogue(device, dataset, runner, buffer, env, rng, step_i);
    }
    template <typename MASK_SPEC>
    void reset(const rlt::Tensor<MASK_SPEC>& mask){
        rlt::reset(device, runner, env, mask, rng);
        prologue(device, dataset, runner, env, rng);
    }
    void reset(){
        rlt::reset(device, runner, env, rng);
        prologue(device, dataset, runner, env, rng);
    }
};

struct Fixture: ::testing::Test {
    DEVICE device;
    ENVIRONMENT* env = nullptr;
    void SetUp() override {
        if(SCENE_PATH.empty()){
            GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
        }
        rlt::init(device);
        env = new ENVIRONMENT;
        rlt::malloc(device, *env);
        rlt::init(device, *env, rlt::rendering::datasets::procthor::GLB{scene_directory(), {}});
    }
    void TearDown() override {
        if(env != nullptr){
            rlt::free(device, *env);
            delete env;
        }
    }
};

TEST_F(Fixture, TIME_LIMIT_AND_DATASET_FLAGS){
    constexpr TI STEPS = 8;
    constexpr TI STEP_LIMIT = 3;
    Harness<STEPS> harness(device, *env, 1337);
    TI truncations = 0;
    for(TI step_i = 0; step_i < STEPS; step_i++){
        harness.step(step_i);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const TI pos = step_i * INSTANCES + instance_i;
            const bool terminated = rlt::get(device, harness.buffer.terminated, instance_i);
            const bool truncated = rlt::get(harness.dataset.truncated, pos, 0) > (T)0.5;
            ASSERT_EQ(rlt::get(harness.dataset.terminated, pos, 0) > (T)0.5, terminated);
            ASSERT_EQ(rlt::get(harness.dataset.all_reset, pos + INSTANCES, 0) > (T)0.5, truncated);
            ASSERT_EQ(rlt::get(device, harness.runner.reset, instance_i), truncated);
            ASSERT_TRUE(!terminated || truncated);
            if(truncated){
                truncations++;
                ASSERT_EQ(rlt::get(device, harness.runner.episode_step, instance_i), (TI)0);
            }
            else{
                ASSERT_GT(rlt::get(device, harness.runner.episode_step, instance_i), (TI)0);
            }
        }
    }
    EXPECT_GE(truncations, 2 * INSTANCES);
}

TEST_F(Fixture, NO_LIMIT){
    constexpr TI STEPS = 5;
    Harness<STEPS, 0> harness(device, *env, 1337);
    for(TI step_i = 0; step_i < STEPS; step_i++){
        harness.step(step_i);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const TI pos = step_i * INSTANCES + instance_i;
            ASSERT_EQ(rlt::get(harness.dataset.truncated, pos, 0) > (T)0.5, rlt::get(device, harness.buffer.terminated, instance_i));
        }
    }
}

TEST_F(Fixture, EXPLICIT_RESET_IS_IMMEDIATE_AND_RECORDED_BY_PROLOGUE){
    constexpr TI STEPS = 6;
    Harness<STEPS, 0> harness(device, *env, 42);
    harness.step(0);
    harness.step(1);
    harness.reset();
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        ASSERT_TRUE(rlt::get(device, harness.runner.reset, instance_i));
        ASSERT_EQ(rlt::get(device, harness.runner.episode_step, instance_i), (TI)0);
        ASSERT_EQ(rlt::get(harness.dataset.reset, instance_i, 0), (T)1);
    }

    harness.step(2);
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> mask;
    rlt::malloc(device, mask);
    rlt::set_all(device, mask, false);
    rlt::set(device, mask, true, 0);
    harness.reset(mask);
    ASSERT_TRUE(rlt::get(device, harness.runner.reset, 0));
    ASSERT_EQ(rlt::get(harness.dataset.reset, 0, 0), (T)1);
    for(TI instance_i = 1; instance_i < INSTANCES; instance_i++){
        ASSERT_FALSE(rlt::get(device, harness.runner.reset, instance_i));
        ASSERT_EQ(rlt::get(device, harness.runner.episode_step, instance_i), (TI)1);
        ASSERT_EQ(rlt::get(harness.dataset.reset, instance_i, 0), (T)0);
    }

    Harness<2, 1> pending(device, *env, 7);
    pending.step(0);
    pending.reset(mask);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        ASSERT_TRUE(rlt::get(device, pending.runner.reset, instance_i));
        ASSERT_EQ(rlt::get(pending.dataset.reset, instance_i, 0), (T)1);
    }
    rlt::free(device, mask);
}

TEST_F(Fixture, DETERMINISM){
    constexpr TI STEPS = 6;
    typename WORLD::State states[2][STEPS][INSTANCES];
    TI lengths[2][STEPS][INSTANCES];
    T rewards[2][STEPS][INSTANCES];
    for(TI run_i = 0; run_i < 2; run_i++){
        Harness<STEPS, 2> harness(device, *env, 99);
        for(TI step_i = 0; step_i < STEPS; step_i++){
            harness.step(step_i);
            for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
                states[run_i][step_i][instance_i] = rlt::get(device, harness.runner.states, instance_i);
                lengths[run_i][step_i][instance_i] = rlt::get(device, harness.runner.episode_step, instance_i);
                rewards[run_i][step_i][instance_i] = rlt::get(harness.dataset.rewards, step_i * INSTANCES + instance_i, 0);
            }
        }
    }
    for(TI step_i = 0; step_i < STEPS; step_i++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const auto& a = states[0][step_i][instance_i];
            const auto& b = states[1][step_i][instance_i];
            for(TI dim = 0; dim < 3; dim++){
                ASSERT_EQ(a.position[dim], b.position[dim]);
                ASSERT_EQ(a.linear_velocity[dim], b.linear_velocity[dim]);
                ASSERT_EQ(a.angular_velocity[dim], b.angular_velocity[dim]);
                ASSERT_EQ(a.force[dim], b.force[dim]);
                ASSERT_EQ(a.torque[dim], b.torque[dim]);
            }
            for(TI dim = 0; dim < 4; dim++){
                ASSERT_EQ(a.orientation[dim], b.orientation[dim]);
                ASSERT_EQ(a.last_action[dim], b.last_action[dim]);
                ASSERT_EQ(a.rpm[dim], b.rpm[dim]);
                for(TI history = 0; history < WORLD::State::HISTORY_LENGTH; history++){
                    ASSERT_EQ(a.action_history[history][dim], b.action_history[history][dim]);
                }
            }
            ASSERT_EQ(a.rotor_history_step, b.rotor_history_step);
            ASSERT_EQ(lengths[0][step_i][instance_i], lengths[1][step_i][instance_i]);
            ASSERT_FLOAT_EQ(rewards[0][step_i][instance_i], rewards[1][step_i][instance_i]);
        }
    }
}
