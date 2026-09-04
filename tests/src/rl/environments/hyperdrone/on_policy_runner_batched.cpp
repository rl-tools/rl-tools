#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>

namespace rlt = rl_tools;
using rlt::prologue;
using rlt::interlude;
using rlt::epilogue;
namespace l2f = rlt::rl::environments::l2f;
namespace on_policy_runner = rlt::rl::components::on_policy_runner;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
static const std::string SCENE_PATH = "";
#endif

namespace test_hyperdrone_on_policy_runner_batched {
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

    struct DYNAMICS_STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;
        static constexpr TI EPISODE_STEP_LIMIT = test_hyperdrone_on_policy_runner_batched::EPISODE_STEP_LIMIT;
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
        static constexpr TI CAM_WIDTH = 16;
        static constexpr TI CAM_HEIGHT = 16;
        using SHADING = rlt::rendering::raytracing::Low;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI NUMBER_OF_ENVIRONMENTS = 2;
    using ENVIRONMENT = rlt::rl::environments::hyperdrone::MultiEnvironment<WORLD, NUMBER_OF_ENVIRONMENTS>;
    constexpr TI INSTANCES = ENVIRONMENT::INSTANCES;
    constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
    constexpr TI OBSERVATION_DIM = ENVIRONMENT::Observation::DIM;
    constexpr TI OBSERVATION_PRIVILEGED_DIM = ENVIRONMENT::ObservationPrivileged::DIM;
    static_assert(OBSERVATION_DIM != OBSERVATION_PRIVILEGED_DIM, "the fixture exercises asymmetric observations (frames for the actor, dynamics for the critic)");

    constexpr TI STEPS = 7;
    constexpr TI STEP_LIMIT = 3;

    using FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using FLATTEN = rlt::nn::layers::flatten::BindConfiguration<FLATTEN_CONFIG>;
    using ACTOR_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, ACTION_DIM, 2, 16, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR_MLP = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<ACTOR_CONFIG>;
    using ACTOR_MODULE_CHAIN = rlt::nn_models::sequential::Module<FLATTEN, rlt::nn_models::sequential::Module<ACTOR_MLP>>;
    using ACTOR_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<typename ENVIRONMENT::Observation::SHAPE, INSTANCES>, 1>;
    using ACTOR = rlt::nn_models::sequential::Build<rlt::nn::capability::Forward<>, ACTOR_MODULE_CHAIN, ACTOR_INPUT_SHAPE>;
    using ACTOR_BUFFERS = typename ACTOR::template Buffer<>;
    using POLICY_STATE = typename ACTOR::template State<>;

    using RUNNER_SPEC = on_policy_runner::Specification<TYPE_POLICY, ENVIRONMENT, POLICY_STATE>;
    using RUNNER = rlt::rl::components::OnPolicyRunner<RUNNER_SPEC>;
    using RUNNER_BUFFER = on_policy_runner::Buffer<RUNNER_SPEC>;
    using DATASET_SPEC = on_policy_runner::DatasetSpecification<RUNNER_SPEC, STEPS>;
    using DATASET = on_policy_runner::Dataset<DATASET_SPEC>;
}

using namespace test_hyperdrone_on_policy_runner_batched;

static std::string scene_directory(){
    std::string directory = std::filesystem::temp_directory_path() / "rl_tools_hyperdrone_on_policy_runner_batched_scenes";
    std::filesystem::create_directories(directory);
    for (const char* name : {"scene_0.glb", "scene_1.glb"}) {
        std::filesystem::path link = std::filesystem::path(directory) / name;
        if (!std::filesystem::exists(link)) {
            std::filesystem::create_symlink(SCENE_PATH, link);
        }
    }
    return directory;
}

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

struct Rollout {
    DEVICE& device;
    RNG rng;
    ACTOR actor;
    ACTOR_BUFFERS actor_buffers;
    RUNNER runner;
    RUNNER_BUFFER runner_buffer;
    DATASET dataset;
    Rollout(DEVICE& device, ENVIRONMENT& env, TI seed): device(device){
        rlt::malloc(device, rng);
        rlt::init(device, rng, seed);
        rlt::malloc(device, actor);
        rlt::malloc(device, actor_buffers);
        rlt::init_weights(device, actor, rng);
        rlt::malloc(device, runner);
        rlt::malloc(device, runner_buffer);
        rlt::malloc(device, dataset);
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            env.environments[environment_i].history_step = 0;
        }
        rlt::init(device, runner, env, rng);
        runner.episode_step_limit = STEP_LIMIT;
    }
    ~Rollout(){
        rlt::free(device, actor);
        rlt::free(device, actor_buffers);
        rlt::free(device, runner);
        rlt::free(device, runner_buffer);
        rlt::free(device, dataset);
        rlt::free(device, rng);
    }
};

// the phases driven by hand (the way custom loops use them)
TEST_F(Fixture, PHASES){
    Rollout rollout(device, *env, 1337);
    auto& runner = rollout.runner;
    auto& dataset = rollout.dataset;
    prologue(device, dataset, runner, *env, rollout.rng);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        ASSERT_EQ(rlt::get(dataset.reset, instance_i, 0), (T)1) << "the first rollout starts with a reset of every instance";
    }
    for(TI step_i = 0; step_i < STEPS; step_i++){
        interlude(device, dataset, runner, rollout.runner_buffer, rollout.actor, rollout.actor_buffers, rollout.rng, step_i);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const TI pos = step_i * INSTANCES + instance_i;
            for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                ASSERT_FLOAT_EQ(rlt::get(device, rollout.runner_buffer.actions, instance_i, action_i), rlt::get(dataset.actions, pos, action_i)) << "the step actions are the sampled dataset actions";
            }
        }
        epilogue(device, dataset, runner, rollout.runner_buffer, *env, rollout.rng, step_i);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const TI pos = step_i * INSTANCES + instance_i;
            const T truncated = rlt::get(dataset.truncated, pos, 0);
            const T terminated = rlt::get(dataset.terminated, pos, 0);
            ASSERT_TRUE(truncated == (T)0 || truncated == (T)1);
            ASSERT_LE(terminated, truncated) << "terminated implies truncated";
            ASSERT_EQ(rlt::get(dataset.all_reset, pos + INSTANCES, 0), truncated) << "the reset of the next row is the truncation of this one";
            ASSERT_EQ(rlt::get(device, runner.reset, instance_i) ? (T)1 : (T)0, truncated) << "the epilogue applies the reset immediately";
            ASSERT_EQ(rlt::get(device, runner.episode_step, instance_i) == 0, truncated == (T)1) << "a reset instance starts a fresh episode";
            ASSERT_LT(rlt::get(device, runner.episode_step, instance_i), STEP_LIMIT) << "the time limit truncates in time";
        }
    }
    T action_deviation = 0;
    for(TI pos = 0; pos < DATASET::STEPS_TOTAL; pos++){
        ASSERT_TRUE(std::isfinite(rlt::get(dataset.action_log_probs, pos, 0)));
        ASSERT_TRUE(std::isfinite(rlt::get(dataset.rewards, pos, 0)));
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            action_deviation += std::abs(rlt::get(dataset.actions, pos, action_i) - rlt::get(dataset.actions_mean, pos, action_i));
        }
    }
    EXPECT_GT(action_deviation, (T)0) << "the actions are sampled around the means";

    // the final row is the current observation (observe is pure, so it can be re-taken)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, OBSERVATION_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, OBSERVATION_PRIVILEGED_DIM>>> observations_privileged;
    rlt::malloc(device, observations);
    rlt::malloc(device, observations_privileged);
    rlt::observe(device, *env, runner.env_parameters, runner.states, typename ENVIRONMENT::Observation{}, observations, rollout.rng);
    rlt::observe(device, *env, runner.env_parameters, runner.states, typename ENVIRONMENT::ObservationPrivileged{}, observations_privileged, rollout.rng);
    T observation_energy = 0;
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        const TI row = STEPS * INSTANCES + instance_i;
        for(TI dim_i = 0; dim_i < OBSERVATION_DIM; dim_i++){
            ASSERT_EQ(rlt::get(device, dataset.all_observations, row, dim_i), rlt::get(device, observations, instance_i, dim_i)) << "instance " << instance_i << " dim " << dim_i;
            observation_energy += std::abs(rlt::get(device, observations, instance_i, dim_i));
        }
        for(TI dim_i = 0; dim_i < OBSERVATION_PRIVILEGED_DIM; dim_i++){
            ASSERT_EQ(rlt::get(device, dataset.all_observations_privileged, row, dim_i), rlt::get(device, observations_privileged, instance_i, dim_i)) << "instance " << instance_i << " dim " << dim_i;
        }
    }
    EXPECT_GT(observation_energy, (T)0) << "the frames are rendered";
    rlt::free(device, observations);
    rlt::free(device, observations_privileged);

    on_policy_runner::EpisodeStatistics<T, TI> statistics;
    rlt::summarize(device, dataset, runner, statistics);
    EXPECT_GE(statistics.finished, 2 * INSTANCES) << "with a limit of 3 every instance finishes at least twice in 7 steps";
    EXPECT_EQ(statistics.time_limit + statistics.terminated, statistics.finished);
    EXPECT_LE(statistics.mean_length, (T)STEP_LIMIT);
    EXPECT_GT(statistics.mean_length, (T)0);
}

TEST_F(Fixture, COLLECT_DETERMINISM){
    Rollout rollout_a(device, *env, 7);
    rlt::collect(device, rollout_a.dataset, rollout_a.runner, rollout_a.runner_buffer, *env, rollout_a.actor, rollout_a.actor_buffers, rollout_a.rng);
    rlt::collect(device, rollout_a.dataset, rollout_a.runner, rollout_a.runner_buffer, *env, rollout_a.actor, rollout_a.actor_buffers, rollout_a.rng);
    Rollout rollout_b(device, *env, 7);
    rlt::collect(device, rollout_b.dataset, rollout_b.runner, rollout_b.runner_buffer, *env, rollout_b.actor, rollout_b.actor_buffers, rollout_b.rng);
    rlt::collect(device, rollout_b.dataset, rollout_b.runner, rollout_b.runner_buffer, *env, rollout_b.actor, rollout_b.actor_buffers, rollout_b.rng);
    EXPECT_EQ(rollout_a.runner.step, 2 * INSTANCES * STEPS);
    // only the fields a rollout writes (values/advantages are the trainer's)
    EXPECT_EQ(rlt::abs_diff(device, rollout_a.dataset.all_observations, rollout_b.dataset.all_observations), (T)0);
    EXPECT_EQ(rlt::abs_diff(device, rollout_a.dataset.all_observations_privileged, rollout_b.dataset.all_observations_privileged), (T)0);
    EXPECT_EQ(rlt::abs_diff(device, rollout_a.dataset.actions_mean, rollout_b.dataset.actions_mean), (T)0);
    EXPECT_EQ(rlt::abs_diff(device, rollout_a.dataset.actions, rollout_b.dataset.actions), (T)0);
    EXPECT_EQ(rlt::abs_diff(device, rollout_a.dataset.action_log_probs, rollout_b.dataset.action_log_probs), (T)0);
    EXPECT_EQ(rlt::abs_diff(device, rollout_a.dataset.rewards, rollout_b.dataset.rewards), (T)0);
    EXPECT_EQ(rlt::abs_diff(device, rollout_a.dataset.terminated, rollout_b.dataset.terminated), (T)0);
    EXPECT_EQ(rlt::abs_diff(device, rollout_a.dataset.truncated, rollout_b.dataset.truncated), (T)0);
    EXPECT_EQ(rlt::abs_diff(device, rollout_a.dataset.all_reset, rollout_b.dataset.all_reset), (T)0);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_EQ(rlt::get(rollout_a.dataset.reset, instance_i, 0), rlt::get(device, rollout_a.runner.reset, instance_i) ? (T)1 : (T)0) << "consecutive rollouts chain: the second rollout's first reset column is the state the first one left";
    }
}
