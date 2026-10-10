#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
static const std::string SCENE_PATH = "";
#endif

namespace test_hyperdrone_world {
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

    struct DYNAMICS_STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;
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
        static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = 0.05;
        static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = 0.01;
        static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = 0.05;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.25;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI NUMBER_OF_ENVIRONMENTS = 2;
    using ENVIRONMENT = rlt::rl::environments::hyperdrone::MultiEnvironment<WORLD, NUMBER_OF_ENVIRONMENTS>;
    constexpr TI INSTANCES = ENVIRONMENT::INSTANCES;
    static_assert(INSTANCES == NUMBER_OF_ENVIRONMENTS * WORLD_SPEC::INSTANCES_PER_ENVIRONMENT);
}

using namespace test_hyperdrone_world;

static std::string scene_directory(){
    // a directory containing only scene GLBs (the dataset enumerates *.glb): symlink the test
    // scene twice so the two members get distinct scene-set entries (deduped in the library)
    std::string directory = std::filesystem::temp_directory_path() / "rl_tools_hyperdrone_world_scenes";
    std::filesystem::create_directories(directory);
    for (const char* name : {"scene_0.glb", "scene_1.glb"}) {
        std::filesystem::path link = std::filesystem::path(directory) / name;
        if (!std::filesystem::exists(link)) {
            std::filesystem::create_symlink(SCENE_PATH, link);
        }
    }
    return directory;
}

template <TI N>
struct Tensors {
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, N>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, N>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, N>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, WORLD::OBSERVATION_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N>>> rewards;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, N>>> terminated_flags;
    void allocate(DEVICE& device){
        rlt::malloc(device, parameters);
        rlt::malloc(device, states);
        rlt::malloc(device, next_states);
        rlt::malloc(device, reset_mask);
        rlt::malloc(device, actions);
        rlt::malloc(device, observations);
        rlt::malloc(device, rewards);
        rlt::malloc(device, terminated_flags);
    }
    void deallocate(DEVICE& device){
        rlt::free(device, parameters);
        rlt::free(device, states);
        rlt::free(device, next_states);
        rlt::free(device, reset_mask);
        rlt::free(device, actions);
        rlt::free(device, observations);
        rlt::free(device, rewards);
        rlt::free(device, terminated_flags);
    }
};

constexpr TI ROLLOUT_STEPS = 5;
struct Rollout {
    typename WORLD::State states[ROLLOUT_STEPS][INSTANCES];
    T observations[ROLLOUT_STEPS][INSTANCES][WORLD::OBSERVATION_DIM];
    T rewards[ROLLOUT_STEPS][INSTANCES];
};

static void rollout(DEVICE& device, ENVIRONMENT& env, Rollout& record, TI seed){
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    Tensors<INSTANCES> tensors;
    tensors.allocate(device);
    rlt::set_all(device, tensors.reset_mask, true);
    rlt::set_all(device, tensors.actions, (T)0);
    rlt::sample_initial_parameters(device, env, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, env, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    for(TI step_i = 0; step_i < ROLLOUT_STEPS; step_i++){
        rlt::render(device, env, tensors.parameters, tensors.states, tensors.reset_mask);
        rlt::observe(device, env, tensors.parameters, tensors.states, typename WORLD::Observation{}, tensors.observations, rng);
        rlt::step(device, env, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
        rlt::reward(device, env, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, tensors.rewards, rng);
        rlt::terminated(device, env, tensors.parameters, tensors.states, tensors.terminated_flags, rng);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            record.states[step_i][instance_i] = rlt::get(device, tensors.next_states, instance_i);
            record.rewards[step_i][instance_i] = rlt::get(device, tensors.rewards, instance_i);
            for(TI dim_i = 0; dim_i < WORLD::OBSERVATION_DIM; dim_i++){
                record.observations[step_i][instance_i][dim_i] = rlt::get(device, tensors.observations, instance_i, dim_i);
            }
        }
        rlt::copy(device, device, tensors.next_states, tensors.states);
        rlt::set_all(device, tensors.reset_mask, false);
    }
    tensors.deallocate(device);
    rlt::free(device, rng);
}

// the nominal camera set on the World before init becomes its single drone's
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, CAMERA_PARAMETER_SAMPLING){
    using namespace rlt;
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    init(device);
    WORLD world;
    typename WORLD::SharedContext shared;
    malloc(device, shared.library);
    malloc(device, world);
    world.parameters.scene_translation[0] = 3;
    world.parameters.scene_yaw_cos = 0;
    world.parameters.scene_yaw_sin = 1;
    world.parameters.agents[0].cameras[0].fov = 68;
    world.parameters.agents[0].cameras[0].mount.pose.position[0] = 0.1;
    world.parameters.agents[0].cameras[0].mount.pose.position[1] = -0.2;
    world.parameters.agents[0].cameras[0].mount.pose.position[2] = 0.3;
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    init(device, world, shared, dataset, corpus, 0, 1, 0);
    ASSERT_EQ(shared.num_drones, 1);
    EXPECT_FLOAT_EQ(shared.drones[0].parameters.cameras[0].fov, 68);
    RNG rng;
    malloc(device, rng);
    init(device, rng, 2);
    for(TI sample = 0; sample < 128; sample++){
        WORLD::Parameters parameters;
        sample_initial_parameters(device, world, parameters, rng);
        EXPECT_EQ(parameters.agents[0].drone, 0);
        EXPECT_FLOAT_EQ(parameters.scene_translation[0], 3);
        EXPECT_FLOAT_EQ(parameters.scene_yaw_cos, 0);
        EXPECT_FLOAT_EQ(parameters.scene_yaw_sin, 1);
        EXPECT_GE(parameters.agents[0].cameras[0].fov, world.parameters.agents[0].cameras[0].fov - WORLD_SPEC::CAMERA_FOV_RANDOMIZATION_RANGE);
        EXPECT_LE(parameters.agents[0].cameras[0].fov, world.parameters.agents[0].cameras[0].fov + WORLD_SPEC::CAMERA_FOV_RANDOMIZATION_RANGE);
        EXPECT_GE(parameters.brightness_scale, 1 - WORLD_SPEC::BRIGHTNESS_RANDOMIZATION_RANGE);
        EXPECT_LE(parameters.brightness_scale, 1 + WORLD_SPEC::BRIGHTNESS_RANDOMIZATION_RANGE);
        const auto& mount = parameters.agents[0].cameras[0].mount;
        T forward[3], up[3];
        rlt::rl::environments::hyperdrone::mount_axes(device, mount, forward, up);
        T forward_norm = 0, up_norm = 0, dot = 0;
        for(TI axis = 0; axis < 3; axis++){
            EXPECT_NEAR(mount.pose.position[axis], world.parameters.agents[0].cameras[0].mount.pose.position[axis], WORLD_SPEC::CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE + 1e-6);
            forward_norm += forward[axis] * forward[axis];
            up_norm += up[axis] * up[axis];
            dot += forward[axis] * up[axis];
        }
        EXPECT_NEAR(forward_norm, 1, 1e-5);
        EXPECT_NEAR(up_norm, 1, 1e-5);
        EXPECT_NEAR(dot, 0, 1e-5);
    }
    free(device, rng);
    free(device, world);
    free(device, shared.library);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, SEEDED_ROLLOUT_DETERMINISM){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    ENVIRONMENT env;
    rlt::malloc(device, env);
    rlt::init(device, env, rlt::rendering::datasets::procthor::GLB{scene_directory(), {}});
    for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
        EXPECT_EQ(env.environments[environment_i].slots.size(), 1);
        EXPECT_EQ(env.environments[environment_i].slots[0].corpus_index, environment_i);
        EXPECT_GT(env.environments[environment_i].slots[0].annotations.num_positions, 0);
    }
    auto* record_a = new Rollout;
    auto* record_b = new Rollout;
    rollout(device, env, *record_a, 1337);
    // reset the per-World frame cursors so the second run starts from the same state
    for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
        env.environments[environment_i].history_step = 0;
    }
    rollout(device, env, *record_b, 1337);
    bool nonzero_observation = false;
    for(TI step_i = 0; step_i < ROLLOUT_STEPS; step_i++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            ASSERT_EQ(std::memcmp(&record_a->states[step_i][instance_i], &record_b->states[step_i][instance_i], sizeof(typename WORLD::State)), 0);
            ASSERT_EQ(record_a->rewards[step_i][instance_i], record_b->rewards[step_i][instance_i]);
            for(TI dim_i = 0; dim_i < WORLD::OBSERVATION_DIM; dim_i++){
                ASSERT_EQ(record_a->observations[step_i][instance_i][dim_i], record_b->observations[step_i][instance_i][dim_i]) << "step " << step_i << " instance " << instance_i << " dim " << dim_i;
                nonzero_observation = nonzero_observation || record_a->observations[step_i][instance_i][dim_i] != (T)0;
            }
        }
    }
    EXPECT_TRUE(nonzero_observation) << "the rendered observations should not be all black";
    delete record_a;
    delete record_b;
    rlt::free(device, env);
}

// the World render path must reproduce a manual composition of the same pose math and renderer
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, RENDER_PATH_EQUIVALENCE){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    using SINGLE_WORLD = WORLD;
    SINGLE_WORLD world;
    typename SINGLE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    Tensors<WORLD_SPEC::INSTANCES_PER_ENVIRONMENT> tensors;
    tensors.allocate(device);
    rlt::set_all(device, tensors.reset_mask, true);
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::Observation{}, tensors.observations, rng);

    // manual composition of the same inputs
    constexpr TI NUM_CAMERAS = WORLD_SPEC::INSTANCES_PER_ENVIRONMENT;
    const T aspect = (T)WORLD_SPEC::CAM_WIDTH / (T)WORLD_SPEC::CAM_HEIGHT;
    std::vector<rlt::rendering::raytracing::Camera<T>> manual_cameras(NUM_CAMERAS);
    for(TI instance_i = 0; instance_i < NUM_CAMERAS; instance_i++){
        typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, instance_i);
        typename WORLD::State state = rlt::get(device, tensors.states, instance_i);
        manual_cameras[instance_i] = rlt::rl::environments::hyperdrone::make_camera<DEVICE, T>(device, instance_parameters.agents[0].cameras[0], state.orientation, state.position, aspect, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin);
    }
    rlt::Tensor<typename WORLD::PREV_CAMERAS_SPEC> camera_alias;
    camera_alias._data = manual_cameras.data();
    auto& renderer = rlt::get_renderer(device, world);
    rlt::copy(device, renderer.device, camera_alias, rlt::cameras(device, renderer));
    rlt::render(device, renderer);
    constexpr TI CAM_PIXELS = WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT;
    std::vector<float> manual_observation(NUM_CAMERAS * CAM_PIXELS * 3);
    {
        rlt::Tensor<rlt::tensor::Specification<float, TI, typename decltype(renderer.observation)::SPEC::SHAPE>> observation_alias;
        observation_alias._data = manual_observation.data();
        rlt::copy(renderer.device, device, renderer.observation, observation_alias);
    }
    for(TI instance_i = 0; instance_i < NUM_CAMERAS; instance_i++){
        typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, instance_i);
        const float scale = (float)instance_parameters.brightness_scale;
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            for(TI channel_i = 0; channel_i < 3; channel_i++){
                float value = manual_observation[(instance_i * CAM_PIXELS + pixel_i) * 3 + channel_i] * scale;
                value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
                T world_value = rlt::get(device, tensors.observations, instance_i, pixel_i * WORLD::IMAGE_CHANNELS + channel_i);
                ASSERT_EQ(world_value, (T)value) << "instance " << instance_i << " pixel " << pixel_i << " channel " << channel_i;
            }
        }
    }

    tensors.deallocate(device);
    rlt::free(device, world);
    rlt::free(device, shared.library);
    rlt::free(device, rng);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, SCENE_ROTATION){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    WORLD world;
    typename WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH, SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 2, 0);
    EXPECT_EQ(world.active_slot, 0);
    rlt::rotate_scene(device, world);
    EXPECT_EQ(world.active_slot, 1);
    EXPECT_EQ(world.slots[world.active_slot].corpus_index, 1);
    rlt::select_scene(device, world, (TI)0);
    EXPECT_EQ(world.active_slot, 0);
    rlt::select_scene(device, world, (TI)1);
    EXPECT_EQ(world.active_slot, 1);
    EXPECT_TRUE(world.render_pending);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    Tensors<WORLD_SPEC::INSTANCES_PER_ENVIRONMENT> tensors;
    tensors.allocate(device);
    rlt::set_all(device, tensors.reset_mask, true);
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::Observation{}, tensors.observations, rng);
    bool nonzero = false;
    for(TI dim_i = 0; dim_i < WORLD::OBSERVATION_DIM; dim_i++){
        nonzero = nonzero || rlt::get(device, tensors.observations, 0, dim_i) != (T)0;
    }
    EXPECT_TRUE(nonzero);
    rlt::rotate_scene(device, world);
    EXPECT_EQ(world.active_slot, 0);

    tensors.deallocate(device);
    rlt::free(device, world);
    rlt::free(device, shared.library);
    rlt::free(device, rng);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace test_hyperdrone_acquisition {
    template <TI HISTORY, bool FLOW = false>
    struct Specification: WORLD_SPEC {
        static constexpr TI HISTORY_LENGTH = HISTORY;
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr TI MOTION_BLUR_SAMPLES = 4;
        static constexpr bool OUTPUT_FLOW = FLOW;
    };

    template <TI HISTORY, bool FLOW = false>
    std::vector<T> rollout(bool read_every_step){
        using namespace rlt;
        using WORLD_TYPE = rl::environments::hyperdrone::World<Specification<HISTORY, FLOW>>;
        constexpr TI COUNT = WORLD_TYPE::INSTANCES;
        DEVICE device;
        RNG rng;
        WORLD_TYPE world;
        typename WORLD_TYPE::SharedContext shared;
        Tensor<tensor::Specification<typename WORLD_TYPE::Parameters, TI, tensor::Shape<TI, COUNT>>> parameters;
        Tensor<tensor::Specification<typename WORLD_TYPE::State, TI, tensor::Shape<TI, COUNT>>> states, next_states;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, COUNT>>> reset_mask;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, COUNT, WORLD_TYPE::ACTION_DIM>>> actions;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, COUNT, WORLD_TYPE::OBSERVATION_DIM>>> observations;
        init(device);
        malloc(device, rng);
        init(device, rng, 913);
        malloc(device, shared.library);
        malloc(device, world);
        malloc(device, parameters);
        malloc(device, states);
        malloc(device, next_states);
        malloc(device, reset_mask);
        malloc(device, actions);
        malloc(device, observations);
        rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
        typename decltype(dataset)::Corpus corpus;
        rendering::datasets::procthor::enumerate(device, dataset, corpus);
        init(device, world, shared, dataset, corpus, 0, 1, 0);
        set_all(device, reset_mask, true);
        sample_initial_parameters(device, world, parameters, reset_mask, rng);
        sample_initial_state(device, world, parameters, states, reset_mask, rng);
        set_all(device, actions, T(0.1));
        set_all(device, reset_mask, false);
        for(TI index = 0; index < 8; index++){
            if(index == 3){
                set(device, reset_mask, true, TI(0));
                sample_initial_parameters(device, world, parameters, reset_mask, rng);
                sample_initial_state(device, world, parameters, states, reset_mask, rng);
                set_all(device, reset_mask, false);
            }
            step(device, world, parameters, states, actions, next_states, rng);
            if constexpr (FLOW){
                EXPECT_EQ(world.flow_scene_version, world.scene_version);
            }
            copy(device, device, next_states, states);
            if(read_every_step){
                observe(device, world, parameters, states, typename WORLD_TYPE::Observation{}, observations, rng);
                std::vector<T> before(COUNT * WORLD_TYPE::OBSERVATION_DIM);
                for(TI row = 0; row < COUNT; row++){
                    for(TI column = 0; column < WORLD_TYPE::OBSERVATION_DIM; column++){
                        before[row * WORLD_TYPE::OBSERVATION_DIM + column] = get(device, observations, row, column);
                    }
                }
                render(device, world, parameters, states, reset_mask);
                observe(device, world, parameters, states, typename WORLD_TYPE::Observation{}, observations, rng);
                for(TI row = 0; row < COUNT; row++){
                    for(TI column = 0; column < WORLD_TYPE::OBSERVATION_DIM; column++){
                        EXPECT_EQ(get(device, observations, row, column), before[row * WORLD_TYPE::OBSERVATION_DIM + column]);
                    }
                }
            }
        }
        EXPECT_EQ(world.history_step, 9);
        observe(device, world, parameters, states, typename WORLD_TYPE::Observation{}, observations, rng);
        std::vector<T> result;
        for(TI row = 0; row < COUNT; row++){
            for(TI column = 0; column < WORLD_TYPE::OBSERVATION_DIM; column++){
                result.push_back(get(device, observations, row, column));
            }
            const auto& state = get_ref(device, states, row);
            for(TI axis = 0; axis < 3; axis++){
                result.push_back(state.position[axis]);
                result.push_back(state.linear_velocity[axis]);
            }
        }
        if constexpr (FLOW){
            using RENDERER = typename WORLD_TYPE::RENDERER;
            std::vector<float> flow(RENDERER::FLOW_TENSOR_SPEC::SIZE);
            std::vector<uint32_t> status(RENDERER::FLOW_STATUS_TENSOR_SPEC::SIZE);
            Tensor<typename RENDERER::FLOW_TENSOR_SPEC> flow_alias;
            Tensor<typename RENDERER::FLOW_STATUS_TENSOR_SPEC> status_alias;
            flow_alias._data = flow.data();
            status_alias._data = status.data();
            auto& renderer = get_renderer(device, world);
            copy(renderer.device, device, flow_buffer(device, renderer), flow_alias);
            copy(renderer.device, device, flow_status_buffer(device, renderer), status_alias);
            result.insert(result.end(), flow.begin(), flow.end());
            for(const auto value : status){
                result.push_back(static_cast<T>(value));
            }
            for(TI row = 0; row < COUNT; row++){
                const TI start = get(device, world.episode_start, row);
                EXPECT_EQ(start, row == 0 ? 3 : 0);
                result.push_back(static_cast<T>(start));
            }
        }
        free(device, observations);
        free(device, actions);
        free(device, reset_mask);
        free(device, next_states);
        free(device, states);
        free(device, parameters);
        free(device, world);
        free(device, shared.library);
        free(device, rng);
        return result;
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, ACQUISITION_READ_ORDER_AND_HISTORY){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    EXPECT_EQ(test_hyperdrone_acquisition::rollout<1>(false), test_hyperdrone_acquisition::rollout<1>(true));
    EXPECT_EQ(test_hyperdrone_acquisition::rollout<3>(false), test_hyperdrone_acquisition::rollout<3>(true));
    EXPECT_EQ((test_hyperdrone_acquisition::rollout<1, true>(false)), (test_hyperdrone_acquisition::rollout<1, true>(true)));
    EXPECT_EQ((test_hyperdrone_acquisition::rollout<3, true>(false)), (test_hyperdrone_acquisition::rollout<3, true>(true)));
}
