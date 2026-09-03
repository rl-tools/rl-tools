#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/components/episodes/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <string>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;
namespace episodes = rlt::rl::components::episodes;

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
    struct EPISODES_SPEC: episodes::Specification<ENVIRONMENT> {};
    struct SYNCHRONIZED_EPISODES_SPEC: episodes::Specification<ENVIRONMENT> {
        static constexpr bool SYNCHRONIZED = true;
    };
    using END_REASON = episodes::EndReason;
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

// host reference of the bookkeeping rules; the component under test must reproduce it exactly
struct Reference {
    TI episode_step[INSTANCES] = {};
    bool truncated[INSTANCES] = {};
    bool forced[INSTANCES] = {};
    END_REASON reason[INSTANCES] = {};
    T episode_return[INSTANCES] = {};
    bool reset[INSTANCES] = {};
    bool finished[INSTANCES] = {};
    TI finished_length[INSTANCES] = {};
    T finished_return[INSTANCES] = {};
    END_REASON finished_reason[INSTANCES] = {};
    Reference(){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            truncated[instance_i] = true;
        }
    }
    void begin(bool synchronized){
        bool any_due = false;
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            any_due = any_due || truncated[instance_i] || forced[instance_i];
        }
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const bool due = synchronized ? any_due : (truncated[instance_i] || forced[instance_i]);
            finished[instance_i] = due && episode_step[instance_i] > 0;
            reset[instance_i] = due;
            finished_length[instance_i] = finished[instance_i] ? episode_step[instance_i] : 0;
            finished_return[instance_i] = finished[instance_i] ? episode_return[instance_i] : (T)0;
            finished_reason[instance_i] = finished[instance_i] ? reason[instance_i] : END_REASON::NONE;
            if(due){
                episode_step[instance_i] = 0;
                episode_return[instance_i] = 0;
                reason[instance_i] = END_REASON::NONE;
                truncated[instance_i] = false;
                forced[instance_i] = false;
            }
        }
    }
    void end(const bool terminated[INSTANCES], const T rewards[INSTANCES], TI step_limit){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            episode_step[instance_i]++;
            episode_return[instance_i] += rewards[instance_i];
            const bool time_limit = step_limit > 0 && episode_step[instance_i] >= step_limit;
            truncated[instance_i] = terminated[instance_i] || time_limit;
            if(truncated[instance_i]){
                reason[instance_i] = terminated[instance_i] ? END_REASON::TERMINATED : END_REASON::TIME_LIMIT;
            }
        }
    }
    void force(const bool mask[INSTANCES]){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            if(mask[instance_i]){
                forced[instance_i] = true;
                if(!truncated[instance_i] && episode_step[instance_i] > 0){
                    reason[instance_i] = END_REASON::FORCED;
                }
            }
        }
    }
};

template <typename EPISODES_SPEC_TYPE, TI STEPS>
struct Harness {
    using EPISODES = episodes::Episodes<EPISODES_SPEC_TYPE>;
    using LOG = episodes::Log<EPISODES_SPEC_TYPE, STEPS>;
    DEVICE& device;
    ENVIRONMENT& env;
    RNG rng;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES>>> rewards;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> mask;
    EPISODES episodes;
    LOG log;
    Reference reference;
    Harness(DEVICE& device, ENVIRONMENT& env, TI seed): device(device), env(env){
        rlt::malloc(device, rng);
        rlt::init(device, rng, seed);
        rlt::malloc(device, parameters);
        rlt::malloc(device, states);
        rlt::malloc(device, next_states);
        rlt::malloc(device, actions);
        rlt::malloc(device, rewards);
        rlt::malloc(device, mask);
        rlt::malloc(device, episodes);
        rlt::malloc(device, log);
        rlt::set_all(device, actions, (T)0);
        rlt::init(device, episodes);
        rlt::init(device, log);
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            env.environments[environment_i].history_step = 0;
        }
    }
    ~Harness(){
        rlt::free(device, parameters);
        rlt::free(device, states);
        rlt::free(device, next_states);
        rlt::free(device, actions);
        rlt::free(device, rewards);
        rlt::free(device, mask);
        rlt::free(device, episodes);
        rlt::free(device, log);
        rlt::free(device, rng);
    }
    void begin(TI step_i){
        reference.begin(EPISODES_SPEC_TYPE::SYNCHRONIZED);
        rlt::begin_step(device, env, episodes, parameters, states, rng);
        rlt::record(device, log, episodes, step_i);
        rlt::render(device, env, parameters, states, episodes.reset);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            ASSERT_EQ(rlt::get(device, episodes.reset, instance_i), reference.reset[instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(rlt::get(device, episodes.episode_step, instance_i), reference.episode_step[instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(rlt::get(device, episodes.truncated, instance_i), reference.truncated[instance_i]);
            ASSERT_EQ(rlt::get(device, episodes.forced, instance_i), reference.forced[instance_i]);
            ASSERT_EQ(rlt::get(device, episodes.end_reason, instance_i), reference.reason[instance_i]);
            ASSERT_EQ(rlt::get(device, episodes.finished, instance_i), reference.finished[instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(rlt::get(device, log.finished, step_i, instance_i), reference.finished[instance_i]);
            ASSERT_EQ(rlt::get(device, log.finished_length, step_i, instance_i), reference.finished_length[instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(rlt::get(device, log.finished_return, step_i, instance_i), reference.finished_return[instance_i]);
            ASSERT_EQ(rlt::get(device, log.finished_reason, step_i, instance_i), reference.finished_reason[instance_i]);
        }
    }
    void end(TI step_i){
        rlt::step(device, env, parameters, states, actions, next_states, rng);
        rlt::reward(device, env, parameters, states, actions, next_states, rewards, rng);
        rlt::copy(device, device, next_states, states);
        rlt::end_step(device, env, episodes, parameters, states, rewards, rng);
        bool terminated[INSTANCES];
        T reward_values[INSTANCES];
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            terminated[instance_i] = rlt::get(device, episodes.terminated, instance_i);
            reward_values[instance_i] = rlt::get(device, rewards, instance_i);
        }
        reference.end(terminated, reward_values, episodes.step_limit);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            ASSERT_EQ(rlt::get(device, episodes.episode_step, instance_i), reference.episode_step[instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(rlt::get(device, episodes.truncated, instance_i), reference.truncated[instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(rlt::get(device, episodes.end_reason, instance_i), reference.reason[instance_i]);
            ASSERT_FLOAT_EQ(rlt::get(device, episodes.episode_return, instance_i), reference.episode_return[instance_i]);
            ASSERT_TRUE(!terminated[instance_i] || rlt::get(device, episodes.truncated, instance_i)) << "terminated must imply truncated";
        }
    }
    void force(const bool force_mask[INSTANCES]){
        bool all = true;
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            rlt::set(device, mask, force_mask[instance_i], instance_i);
            all = all && force_mask[instance_i];
        }
        if(all){
            rlt::force_reset(device, episodes);
        } else {
            rlt::force_reset(device, episodes, mask);
        }
        reference.force(force_mask);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            ASSERT_EQ(rlt::get(device, episodes.forced, instance_i), reference.forced[instance_i]);
            ASSERT_EQ(rlt::get(device, episodes.end_reason, instance_i), reference.reason[instance_i]);
        }
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

TEST_F(Fixture, TIME_LIMIT_AND_FLAGS){
    constexpr TI STEPS = 8;
    constexpr TI STEP_LIMIT = 3;
    Harness<EPISODES_SPEC, STEPS> harness(device, *env, 1337);
    harness.episodes.step_limit = STEP_LIMIT;
    TI truncations = 0;
    for(TI step_i = 0; step_i < STEPS; step_i++){
        harness.begin(step_i);
        if(step_i == 0){
            for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
                ASSERT_TRUE(rlt::get(device, harness.episodes.reset, instance_i)) << "every instance starts with a reset";
            }
        }
        harness.end(step_i);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const bool truncated = rlt::get(device, harness.episodes.truncated, instance_i);
            truncations += truncated ? 1 : 0;
            if(!rlt::get(device, harness.episodes.terminated, instance_i)){
                ASSERT_EQ(truncated, rlt::get(device, harness.episodes.episode_step, instance_i) == STEP_LIMIT) << "a time limit truncates exactly at the limit";
            }
        }
    }
    EXPECT_GE(truncations, 2 * INSTANCES) << "with a limit of 3 every instance truncates at least twice in 8 steps";
}

TEST_F(Fixture, NO_LIMIT){
    constexpr TI STEPS = 5;
    Harness<EPISODES_SPEC, STEPS> harness(device, *env, 1337);
    harness.episodes.step_limit = 0;
    for(TI step_i = 0; step_i < STEPS; step_i++){
        harness.begin(step_i);
        harness.end(step_i);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            ASSERT_EQ(rlt::get(device, harness.episodes.truncated, instance_i), rlt::get(device, harness.episodes.terminated, instance_i)) << "without a limit only terminations truncate";
        }
    }
}

TEST_F(Fixture, FORCED){
    constexpr TI STEPS = 6;
    Harness<EPISODES_SPEC, STEPS> harness(device, *env, 42);
    harness.episodes.step_limit = 0;
    for(TI step_i = 0; step_i < 2; step_i++){
        harness.begin(step_i);
        harness.end(step_i);
    }
    bool all[INSTANCES];
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        all[instance_i] = true;
    }
    harness.force(all);
    harness.begin(2);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        ASSERT_TRUE(rlt::get(device, harness.episodes.reset, instance_i));
        ASSERT_TRUE(rlt::get(device, harness.log.finished, 2, instance_i));
        ASSERT_EQ(rlt::get(device, harness.log.finished_length, 2, instance_i), (TI)2);
        const END_REASON reason = rlt::get(device, harness.log.finished_reason, 2, instance_i);
        ASSERT_TRUE(reason == END_REASON::FORCED || reason == END_REASON::TERMINATED);
    }
    harness.end(2);
    harness.begin(3);
    harness.end(3);
    bool first_only[INSTANCES] = {};
    first_only[0] = true;
    harness.force(first_only);
    harness.begin(4);
    ASSERT_TRUE(rlt::get(device, harness.episodes.reset, 0));
    for(TI instance_i = 1; instance_i < INSTANCES; instance_i++){
        ASSERT_EQ(rlt::get(device, harness.episodes.reset, instance_i), harness.reference.reset[instance_i]);
    }
    harness.end(4);
}

TEST_F(Fixture, SYNCHRONIZED){
    constexpr TI STEPS = 4;
    Harness<SYNCHRONIZED_EPISODES_SPEC, STEPS> harness(device, *env, 42);
    harness.episodes.step_limit = 0;
    for(TI step_i = 0; step_i < 2; step_i++){
        harness.begin(step_i);
        harness.end(step_i);
    }
    bool first_only[INSTANCES] = {};
    first_only[0] = true;
    harness.force(first_only);
    harness.begin(2);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        ASSERT_TRUE(rlt::get(device, harness.episodes.reset, instance_i)) << "synchronized: one due instance resets all";
    }
    harness.end(2);
    harness.begin(3);
    harness.end(3);
}

TEST_F(Fixture, SUMMARY){
    constexpr TI STEPS = 7;
    constexpr TI STEP_LIMIT = 3;
    Harness<EPISODES_SPEC, STEPS> harness(device, *env, 7);
    harness.episodes.step_limit = STEP_LIMIT;
    TI expected_finished = 0, expected_time_limit = 0, expected_terminated = 0, expected_in_progress = 0;
    T expected_length_sum = 0, expected_return_sum = 0, expected_in_progress_length_sum = 0;
    for(TI step_i = 0; step_i < STEPS; step_i++){
        harness.begin(step_i);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            if(harness.reference.finished[instance_i]){
                expected_finished++;
                expected_length_sum += (T)harness.reference.finished_length[instance_i];
                expected_return_sum += harness.reference.finished_return[instance_i];
                expected_time_limit += harness.reference.finished_reason[instance_i] == END_REASON::TIME_LIMIT ? 1 : 0;
                expected_terminated += harness.reference.finished_reason[instance_i] == END_REASON::TERMINATED ? 1 : 0;
            }
        }
        harness.end(step_i);
    }
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        if(harness.reference.episode_step[instance_i] > 0){
            expected_in_progress++;
            expected_in_progress_length_sum += (T)harness.reference.episode_step[instance_i];
        }
    }
    episodes::Statistics<T, TI> statistics;
    rlt::summarize(device, harness.log, harness.episodes, statistics);
    EXPECT_EQ(statistics.finished, expected_finished);
    EXPECT_GE(statistics.finished, 2 * INSTANCES);
    EXPECT_EQ(statistics.time_limit, expected_time_limit);
    EXPECT_EQ(statistics.terminated, expected_terminated);
    EXPECT_EQ(statistics.forced, 0);
    EXPECT_EQ(statistics.in_progress, expected_in_progress);
    EXPECT_FLOAT_EQ(statistics.length_sum, expected_length_sum);
    EXPECT_FLOAT_EQ(statistics.return_sum, expected_return_sum);
    EXPECT_FLOAT_EQ(statistics.in_progress_length_sum, expected_in_progress_length_sum);
    EXPECT_FLOAT_EQ(statistics.mean_length, expected_length_sum / (T)expected_finished);
    EXPECT_FLOAT_EQ(statistics.terminated_share, (T)expected_terminated / (T)expected_finished);
}

TEST_F(Fixture, DETERMINISM){
    constexpr TI STEPS = 6;
    typename WORLD::State states_a[STEPS][INSTANCES];
    typename WORLD::State states_b[STEPS][INSTANCES];
    TI lengths_a[STEPS][INSTANCES];
    TI lengths_b[STEPS][INSTANCES];
    for(TI run_i = 0; run_i < 2; run_i++){
        Harness<EPISODES_SPEC, STEPS> harness(device, *env, 99);
        harness.episodes.step_limit = 2;
        for(TI step_i = 0; step_i < STEPS; step_i++){
            harness.begin(step_i);
            harness.end(step_i);
            for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
                (run_i == 0 ? states_a : states_b)[step_i][instance_i] = rlt::get(device, harness.states, instance_i);
                (run_i == 0 ? lengths_a : lengths_b)[step_i][instance_i] = rlt::get(device, harness.log.finished_length, step_i, instance_i);
            }
        }
    }
    for(TI step_i = 0; step_i < STEPS; step_i++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            ASSERT_EQ(std::memcmp(&states_a[step_i][instance_i], &states_b[step_i][instance_i], sizeof(typename WORLD::State)), 0);
            ASSERT_EQ(lengths_a[step_i][instance_i], lengths_b[step_i][instance_i]);
        }
    }
}

// the on-policy dataset ingests the flags through the runner's batched record verbs
TEST_F(Fixture, DATASET_RECORD){
    constexpr TI STEPS = 4;
    using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<rlt::numeric_types::Policy<T>, ENVIRONMENT, bool>;
    using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, STEPS>;
    using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;
    DATASET dataset;
    rlt::malloc(device, dataset);
    Harness<EPISODES_SPEC, STEPS> harness(device, *env, 5);
    harness.episodes.step_limit = 2;
    for(TI step_i = 0; step_i < STEPS; step_i++){
        harness.begin(step_i);
        if(step_i == 0){
            rlt::record_reset(device, dataset, harness.episodes.reset);
            for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
                ASSERT_EQ(rlt::get(dataset.reset, instance_i, 0), (T)1) << "all instances reset at the start";
            }
        }
        harness.end(step_i);
        rlt::record_step(device, dataset, step_i, harness.rewards, harness.episodes.terminated, harness.episodes.truncated);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const TI pos = step_i * INSTANCES + instance_i;
            const bool truncated = rlt::get(device, harness.episodes.truncated, instance_i);
            ASSERT_FLOAT_EQ(rlt::get(dataset.rewards, pos, 0), rlt::get(device, harness.rewards, instance_i));
            ASSERT_EQ(rlt::get(dataset.terminated, pos, 0), rlt::get(device, harness.episodes.terminated, instance_i) ? (T)1 : (T)0);
            ASSERT_EQ(rlt::get(dataset.truncated, pos, 0), truncated ? (T)1 : (T)0);
            ASSERT_EQ(rlt::get(dataset.all_reset, pos + INSTANCES, 0), truncated ? (T)1 : (T)0);
            if(step_i + 1 < STEPS){
                ASSERT_EQ(rlt::get(dataset.reset, pos + INSTANCES, 0), truncated ? (T)1 : (T)0) << "reset is truncation delayed by one step";
            }
        }
    }
    rlt::free(device, dataset);
}
