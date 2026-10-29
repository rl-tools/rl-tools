#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>

#include "../../../utils/utils.h"
#include "camera_sampling_test.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace rro = rlt::hyperlux::outputs;
using rlt::output;
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
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI INSTANCES_PER_WORLD = 2;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        using SHADING = rlt::hyperlux::Low;
        static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = 0.05;
        static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = 0.01;
        static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = 0.05;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.25;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI N_WORLDS = 2;
    using ENVIRONMENT = rlt::rl::environments::HyperDrone<WORLD, N_WORLDS>;
    constexpr TI INSTANCES = ENVIRONMENT::INSTANCES;
    static_assert(INSTANCES == N_WORLDS * WORLD_SPEC::INSTANCES_PER_WORLD);

    struct MOUNTS_SPEC: WORLD_SPEC {
        using OBSERVATION = rlt::rl::environments::hyperdrone::ImageObservation<0, 1, 2>;
        static constexpr TI N_AGENTS = 3;
        static constexpr TI N_CAMERA_AGENTS = 2;
        static constexpr bool SELF_VISIBLE = true;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 32;
        using CAMERA_OUTPUTS = rlt::rl::environments::hyperdrone::CameraOutputs<
            rro::List<rro::RGB<rro::format::Float32>, rro::Depth<>, rro::Probes>,
            rro::List<rro::RGB<rro::format::Float32>, rro::Normals<>, rro::Probes>,
            rro::List<rro::RGB<rro::format::Float32>, rro::Depth<>, rro::Probes>>;
    };
    using MOUNTS_WORLD = rlt::rl::environments::hyperdrone::World<MOUNTS_SPEC>;
    using MOUNTS_RENDERER_SPEC = typename MOUNTS_WORLD::RENDERER_SPEC;
    using EXPECTED_OUTPUTS = rro::List<
        rro::RGB<rro::format::Float32>,
        rro::Cameras<rro::cameras::Every<3u, 0u, 2u>, rro::Depth<>>,
        rro::Probes,
        rro::Cameras<rro::cameras::Every<3u, 1u>, rro::Normals<>>>;
    static_assert(rlt::utils::typing::is_same_v<typename MOUNTS_RENDERER_SPEC::OUTPUTS, EXPECTED_OUTPUTS>);
    static_assert(MOUNTS_WORLD::N_CAMERAS_PER_AGENT == 3 && MOUNTS_WORLD::N_VIEWS == 6);
    static_assert(sizeof(rlt::rl::environments::hyperdrone::DroneParameters<MOUNTS_SPEC>{}.cameras) == 3 * sizeof(rlt::rl::environments::hyperdrone::Camera<T>));
    static_assert(rro::TensorSpecification<MOUNTS_RENDERER_SPEC, rro::Depth<>>::SHAPE::template GET<0> == 8);
    static_assert(rro::TensorSpecification<MOUNTS_RENDERER_SPEC, rro::Normals<>>::SHAPE::template GET<0> == 4);
    static_assert(rro::TensorSpecification<MOUNTS_RENDERER_SPEC, rro::RGB<rro::format::Float32>>::SHAPE::template GET<0> == 12);
    static_assert(!rro::Layout<MOUNTS_RENDERER_SPEC>::ALL_CAMERAS);
    struct SINGLE_MOUNT_SPEC: MOUNTS_SPEC {
        using OBSERVATION = rlt::rl::environments::hyperdrone::ImageObservation<0>;
        using CAMERA_OUTPUTS = typename WORLD_SPEC::CAMERA_OUTPUTS;
    };
    using SINGLE_MOUNT_WORLD = rlt::rl::environments::hyperdrone::World<SINGLE_MOUNT_SPEC>;
    static_assert(SINGLE_MOUNT_WORLD::N_CAMERAS_PER_AGENT == 1);
    static_assert(SINGLE_MOUNT_WORLD::RENDERER_SPEC::NUM_CAMERAS == 4);
    static_assert(rro::Layout<typename SINGLE_MOUNT_WORLD::RENDERER_SPEC>::ALL_CAMERAS);
}

using namespace test_hyperdrone_world;

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, AUXILIARY_CAMERAS_SAMPLE_INDEPENDENTLY_OF_POLICY_STREAM) {
    using rlt::init;
    DEVICE device;
    RNG rng;
    init(device, rng, 991);
    EXPECT_TRUE(camera_sampling_test::independent_auxiliary_sampling<WORLD_SPEC>(device, rng));
    EXPECT_TRUE(camera_sampling_test::unobserved_preserves_sampling<WORLD_SPEC>(device, rng));
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, AUXILIARY_CAMERAS_PRESERVE_PORTABLE_RANDOM_STREAMS) {
    using rlt::init;
    using rlt::malloc;
    using rlt::free;
    DEVICE device;
    rlt::devices::generic::random::PortableState scalar;
    init(device, scalar, 991);
    EXPECT_TRUE(camera_sampling_test::independent_auxiliary_sampling<WORLD_SPEC>(device, scalar));
    EXPECT_TRUE(camera_sampling_test::unobserved_preserves_sampling<WORLD_SPEC>(device, scalar));
    init(device, scalar, 0x55555556u);
    EXPECT_NE(scalar.state, 0u);
    const auto initialized = scalar.state;
    rlt::random::uniform_real_distribution(device.random, (T)0, (T)1, scalar);
    EXPECT_NE(scalar.state, initialized);
    rlt::devices::generic::random::ArrayENGINE<rlt::devices::generic::random::ArraySpecification<TI, 4>> array, reference;
    malloc(device, array);
    malloc(device, reference);
    init(device, array, 991);
    init(device, reference, 991);
    EXPECT_TRUE(camera_sampling_test::array_preserves_sampling<WORLD_SPEC>(device, array, reference));
    free(device, reference);
    free(device, array);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, CAMERA_OUTPUT_LAYOUT) {
    DEVICE device;
    unsigned int table[MOUNTS_RENDERER_SPEC::NUM_CAMERAS * rro::CAMERA_TABLE_STRIDE];
    using rlt::fill_camera_table;
    fill_camera_table<MOUNTS_RENDERER_SPEC>(device, table);
    for(TI camera = 0; camera < MOUNTS_RENDERER_SPEC::NUM_CAMERAS; camera++) {
        const auto* entry = table + camera * rro::CAMERA_TABLE_STRIDE;
        const bool depth = camera % 3 != 1;
        const unsigned int expected_mask = rro::bit(rro::Slot::RGB_FLOAT32) | rro::bit(rro::Slot::PROBES)
            | rro::bit(depth ? rro::Slot::DEPTH_FLOAT32 : rro::Slot::NORMALS_FLOAT32);
        EXPECT_EQ(entry[0], expected_mask);
        EXPECT_EQ(entry[1 + (unsigned int)rro::Slot::RGB_FLOAT32], camera);
        EXPECT_EQ(entry[1 + (unsigned int)rro::Slot::PROBES], camera);
        EXPECT_EQ(entry[1 + (unsigned int)rro::Slot::DEPTH_FLOAT32], depth ? (camera / 3) * 2 + (camera % 3 == 2 ? 1 : 0) : rro::NO_ROW);
        EXPECT_EQ(entry[1 + (unsigned int)rro::Slot::NORMALS_FLOAT32], depth ? rro::NO_ROW : camera / 3);
    }
}

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

template <TI N, typename WORLD_TYPE = WORLD>
struct Tensors {
    rlt::Tensor<rlt::tensor::Specification<typename WORLD_TYPE::Parameters, TI, rlt::tensor::Shape<TI, N>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD_TYPE::State, TI, rlt::tensor::Shape<TI, N>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, N>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, WORLD_TYPE::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N, WORLD_TYPE::OBSERVATION_DIM>>> observations;
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
    rlt::hyperlux::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::hyperlux::datasets::procthor::enumerate(device, dataset, corpus);
    init(device, world, shared, dataset, corpus, 0, 1, 0);
    ASSERT_EQ(shared.num_drones, 1);
    EXPECT_FLOAT_EQ(shared.drones[0].parameters.cameras[0].fov, 68);
    RNG rng;
    malloc(device, rng);
    init(device, rng, 2);
    Tensor<tensor::Specification<WORLD::Parameters, TI, tensor::Shape<TI, WORLD::INSTANCES>, false>> parameters_batch;
    Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, WORLD::INSTANCES>, false>> reset_mask;
    set_all(device, reset_mask, true);
    for(TI sample = 0; sample < 128; sample++){
        sample_initial_parameters(device, world, parameters_batch, reset_mask, rng);
        const auto& parameters = get_ref(device, parameters_batch, (TI)0);
        EXPECT_EQ(parameters.agents[0].drone, 0);
        EXPECT_NEAR(parameters.scene_yaw_cos * parameters.scene_yaw_cos + parameters.scene_yaw_sin * parameters.scene_yaw_sin, 1, 1e-5);
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
    rlt::init(device, env, rlt::hyperlux::datasets::procthor::GLB{scene_directory(), {}});
    for(TI world_i = 0; world_i < N_WORLDS; world_i++){
        EXPECT_EQ(env.worlds[world_i].slots.size(), 1);
        EXPECT_EQ(env.worlds[world_i].slots[0].corpus_index, world_i);
        EXPECT_GT(env.worlds[world_i].slots[0].annotations.num_positions, 0);
    }
    auto* record_a = new Rollout;
    auto* record_b = new Rollout;
    rollout(device, env, *record_a, 1337);
    // reset the per-World frame cursors so the second run starts from the same state
    for(TI world_i = 0; world_i < N_WORLDS; world_i++){
        env.worlds[world_i].history_step = 0;
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
    rlt::hyperlux::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::hyperlux::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    Tensors<WORLD_SPEC::INSTANCES_PER_WORLD> tensors;
    tensors.allocate(device);
    rlt::set_all(device, tensors.reset_mask, true);
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::Observation{}, tensors.observations, rng);

    // manual composition of the same inputs
    constexpr TI NUM_CAMERAS = WORLD_SPEC::INSTANCES_PER_WORLD;
    const T aspect = (T)WORLD_SPEC::CAM_WIDTH / (T)WORLD_SPEC::CAM_HEIGHT;
    std::vector<rlt::hyperlux::Camera<T>> manual_cameras(NUM_CAMERAS);
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
        rlt::Tensor<rro::TensorSpecification<typename WORLD::RENDERER_SPEC, rro::RGB<rro::format::Float32>>> observation_alias;
        observation_alias._data = manual_observation.data();
        rlt::copy(renderer.device, device, output(device, renderer, rro::RGB<rro::format::Float32>{}), observation_alias);
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
    rlt::hyperlux::datasets::procthor::GLB dataset{{}, {SCENE_PATH, SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::hyperlux::datasets::procthor::enumerate(device, dataset, corpus);
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
    Tensors<WORLD_SPEC::INSTANCES_PER_WORLD> tensors;
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
    template <TI HISTORY, bool FLOW = false, bool IMAGE = true>
    struct Specification: WORLD_SPEC {
        using OBSERVATION = rlt::utils::typing::conditional_t<IMAGE, rlt::rl::environments::hyperdrone::ImageObservation<0>, rlt::rl::environments::hyperdrone::NoObservation>;
        static constexpr TI HISTORY_LENGTH = HISTORY;
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr TI MOTION_BLUR_SAMPLES = 4;
        using CAMERA_OUTPUTS = rlt::rl::environments::hyperdrone::CameraOutputs<typename rro::Concat<
            rro::If<(IMAGE), rro::RGB<rro::format::Float32>>,
            rro::If<(FLOW), rro::Flow<>>,
            rro::If<(FLOW), rro::FlowStatus<>>,
            rro::List<rro::Probes>
        >::type>;
    };

    struct Checkpoint {
        TI history_step, scene_version;
        std::vector<T> observation, history, dynamics, flow;
        std::vector<TI> episode_start, rotor_history_step;
        std::vector<uint32_t> flow_status;
    };

    void expect_equal(const Checkpoint& a, const Checkpoint& b){
        EXPECT_EQ(a.history_step, b.history_step);
        EXPECT_EQ(a.scene_version, b.scene_version);
        EXPECT_EQ(a.observation, b.observation);
        EXPECT_EQ(a.history, b.history);
        EXPECT_EQ(a.dynamics, b.dynamics);
        EXPECT_EQ(a.flow, b.flow);
        EXPECT_EQ(a.flow_status, b.flow_status);
        EXPECT_EQ(a.episode_start, b.episode_start);
        EXPECT_EQ(a.rotor_history_step, b.rotor_history_step);
    }

    template <TI HISTORY, bool FLOW = false, bool IMAGE = true, bool LAUNCH = false>
    std::vector<Checkpoint> rollout(bool read_every_step){
        using namespace rlt;
        using WORLD_TYPE = rl::environments::hyperdrone::World<Specification<HISTORY, FLOW, IMAGE>>;
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
        hyperlux::datasets::procthor::GLB dataset{{}, {SCENE_PATH, SCENE_PATH}};
        typename decltype(dataset)::Corpus corpus;
        hyperlux::datasets::procthor::enumerate(device, dataset, corpus);
        init(device, world, shared, dataset, corpus, 0, 2, 0);
        EXPECT_EQ(data(world.prev_cameras) != nullptr, IMAGE);
        EXPECT_EQ(data(world.camera_staging_open) != nullptr, IMAGE);
        EXPECT_EQ(data(world.camera_staging_previous) != nullptr, IMAGE || FLOW);
        if constexpr (IMAGE){
            set_all(device, world.history, T(-1));
        }
        set_all(device, reset_mask, true);
        sample_initial_parameters(device, world, parameters, reset_mask, rng);
        sample_initial_state(device, world, parameters, states, reset_mask, rng);
        set_all(device, actions, T(0.1));
        set_all(device, reset_mask, false);
        TI expected_episode_start[COUNT] = {};
        const auto acquire_frame = [&](){
            if constexpr (IMAGE){
                observe(device, world, parameters, states, typename WORLD_TYPE::Observation{}, observations, rng);
            }
            else if constexpr (LAUNCH){
                render_launch(device, world, parameters, states, reset_mask);
            }
            else {
                render(device, world, parameters, states, reset_mask);
            }
        };
        const auto checkpoint = [&](){
            acquire_frame();
            EXPECT_FALSE(world.render_pending);
            Checkpoint result;
            result.history_step = world.history_step;
            result.scene_version = world.scene_version;
            if constexpr (IMAGE){
                for(TI slot = 0; slot < HISTORY; slot++){
                    for(TI camera = 0; camera < COUNT * WORLD_TYPE::N_VIEWS; camera++){
                        for(TI column = 0; column < WORLD_TYPE::FRAME_DIM; column++){
                            const T value = get(device, world.history, slot, camera, column);
                            EXPECT_GE(value, T(0));
                            EXPECT_LE(value, T(1));
                            result.history.push_back(value);
                        }
                    }
                }
            }
            for(TI row = 0; row < COUNT; row++){
                if constexpr (IMAGE){
                    for(TI column = 0; column < WORLD_TYPE::OBSERVATION_DIM; column++){
                        result.observation.push_back(get(device, observations, row, column));
                    }
                }
                const auto& state = get_ref(device, states, row);
                result.dynamics.insert(result.dynamics.end(), state.position, state.position + 3);
                result.dynamics.insert(result.dynamics.end(), state.orientation, state.orientation + 4);
                result.dynamics.insert(result.dynamics.end(), state.linear_velocity, state.linear_velocity + 3);
                result.dynamics.insert(result.dynamics.end(), state.angular_velocity, state.angular_velocity + 3);
                result.dynamics.insert(result.dynamics.end(), state.last_action, state.last_action + 4);
                result.dynamics.insert(result.dynamics.end(), state.force, state.force + 3);
                result.dynamics.insert(result.dynamics.end(), state.torque, state.torque + 3);
                result.dynamics.insert(result.dynamics.end(), state.rpm, state.rpm + 4);
                for(TI slot = 0; slot < DYNAMICS_STATIC_PARAMETERS::ACTION_HISTORY_LENGTH; slot++){
                    result.dynamics.insert(result.dynamics.end(), state.action_history[slot], state.action_history[slot] + 4);
                }
                result.rotor_history_step.push_back(state.rotor_history_step);
                const TI episode_start = get(device, world.episode_start, row);
                EXPECT_EQ(episode_start, expected_episode_start[row]);
                result.episode_start.push_back(episode_start);
            }
            if constexpr (FLOW){
                EXPECT_EQ(world.flow_scene_version, world.scene_version);
                auto& renderer = get_renderer(device, world);
                using FLOW_SPEC = rro::TensorSpecification<typename WORLD_TYPE::RENDERER_SPEC, rro::Flow<>>;
                using STATUS_SPEC = rro::TensorSpecification<typename WORLD_TYPE::RENDERER_SPEC, rro::FlowStatus<>>;
                result.flow.resize(FLOW_SPEC::SIZE);
                result.flow_status.resize(STATUS_SPEC::SIZE);
                Tensor<FLOW_SPEC> flow_alias;
                Tensor<STATUS_SPEC> status_alias;
                flow_alias._data = result.flow.data();
                status_alias._data = result.flow_status.data();
                copy(renderer.device, device, output(device, renderer, rro::Flow<>{}), flow_alias);
                copy(renderer.device, device, output(device, renderer, rro::FlowStatus<>{}), status_alias);
            }
            return result;
        };
        std::vector<Checkpoint> result;
        for(TI index = 0; index < 12; index++){
            SCOPED_TRACE(index);
            if(index == 3 || index == 6 || index == 9){
                if(index == 6){
                    select_scene(device, world, TI(1));
                    EXPECT_EQ(world.active_slot, 1);
                    set_all(device, reset_mask, true);
                }
                else{
                    set(device, reset_mask, true, index == 3 ? TI(0) : TI(1));
                }
                for(TI row = 0; row < COUNT; row++){
                    if(get(device, reset_mask, row)){
                        expected_episode_start[row] = index;
                    }
                }
                sample_initial_parameters(device, world, parameters, reset_mask, rng);
                sample_initial_state(device, world, parameters, states, reset_mask, rng);
                set_all(device, reset_mask, false);
            }
            step(device, world, parameters, states, actions, next_states, rng);
            copy(device, device, next_states, states);
            EXPECT_EQ(world.history_step, index + 2);
            const bool acquire = index == 1 || index == 4 || index == 6 || index == 10 || index == 11;
            if(read_every_step || acquire){
                acquire_frame();
            }
            if(acquire){
                result.push_back(checkpoint());
                render(device, world, parameters, states, reset_mask);
                expect_equal(result.back(), checkpoint());
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

    template <TI HISTORY, bool FLOW = false, bool IMAGE = true>
    void compare_acquisition(){
        SCOPED_TRACE(HISTORY);
        SCOPED_TRACE(FLOW);
        const auto deferred = rollout<HISTORY, FLOW, IMAGE>(false);
        const auto eager = rollout<HISTORY, FLOW, IMAGE>(true);
        ASSERT_EQ(deferred.size(), 5);
        ASSERT_EQ(deferred.size(), eager.size());
        for(size_t index = 0; index < deferred.size(); index++){
            SCOPED_TRACE(index);
            expect_equal(deferred[index], eager[index]);
        }
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, ACQUISITION_READ_ORDER_AND_HISTORY){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    test_hyperdrone_acquisition::compare_acquisition<1>();
    test_hyperdrone_acquisition::compare_acquisition<3>();
    test_hyperdrone_acquisition::compare_acquisition<1, true>();
    test_hyperdrone_acquisition::compare_acquisition<3, true>();
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, OUTPUTS_WITHOUT_IMAGE_HISTORY){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    using namespace test_hyperdrone_acquisition;
    using WORLD_TYPE = rlt::rl::environments::hyperdrone::World<Specification<3, true, false>>;
    static_assert(!WORLD_TYPE::HAS_IMAGE_OBSERVATION);
    static_assert(rlt::utils::typing::is_same_v<decltype(WORLD_TYPE{}.history), rlt::rl::environments::hyperdrone::world::NoImageHistory>);
    const auto without_image = rollout<3, true, false>(false);
    compare_acquisition<3, true, false>();
    compare_acquisition<3, false, false>();
    const auto launched = rollout<3, true, false, true>(false);
    ASSERT_EQ(launched.size(), without_image.size());
    for(size_t index = 0; index < launched.size(); index++){
        expect_equal(launched[index], without_image[index]);
    }
}

namespace test_sparse_observation {
    struct Specification: WORLD_SPEC {
        static constexpr TI CAM_WIDTH = 9;
        static constexpr TI CAM_HEIGHT = 7;
        using CAMERA_OUTPUTS = rlt::rl::environments::hyperdrone::CameraOutputs<rro::List<rro::RGB<rro::format::Float32>>, rro::List<rro::Depth<>>, rro::List<rro::RGB<rro::format::Float32>>>;
        using OBSERVATION = rlt::rl::environments::hyperdrone::ImageObservation<2>;
    };
    using World = rlt::rl::environments::hyperdrone::World<Specification>;
    static_assert(World::N_VIEWS == 3 && World::N_OBSERVATION_VIEWS == 1);
    static_assert(World::RENDERER_CONFIG::INTERNAL_PROBES);
    static_assert(World::observation_rgb_row(0) == 1 && World::observation_rgb_row(1) == 3);
    struct ReorderedSpecification: Specification {
        using OBSERVATION = rlt::rl::environments::hyperdrone::ImageObservation<2, 0>;
    };
    using ReorderedWorld = rlt::rl::environments::hyperdrone::World<ReorderedSpecification>;
    static_assert(ReorderedWorld::observation_rgb_row(0) == 1 && ReorderedWorld::observation_rgb_row(1) == 0 && ReorderedWorld::observation_rgb_row(2) == 3);
    struct MultiObservation: MOUNTS_SPEC {
        using OBSERVATION = rlt::rl::environments::hyperdrone::ImageObservation<1>;
    };
    using MultiWorld = rlt::rl::environments::hyperdrone::World<MultiObservation>;
    static_assert(MultiWorld::N_OBSERVATION_VIEWS == 2);
    static_assert(MultiWorld::observation_camera(0) == 1 && MultiWorld::observation_camera(1) == 4 && MultiWorld::observation_camera(2) == 7);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD, SPARSE_OBSERVATION_AND_INTERNAL_PROBES) {
    using namespace rlt;
    using SparseWorld = test_sparse_observation::World;
    if(SCENE_PATH.empty()) {
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    init(device);
    SparseWorld world;
    typename SparseWorld::SharedContext shared;
    malloc(device, shared.library);
    malloc(device, world);
    world.parameters.agents[0].cameras[2].fov = 100;
    hyperlux::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    hyperlux::datasets::procthor::enumerate(device, dataset, corpus);
    init(device, world, shared, dataset, corpus, 0, 1, 0);
    RNG rng;
    malloc(device, rng);
    init(device, rng, 42);
    Tensors<SparseWorld::INSTANCES, SparseWorld> tensors;
    tensors.allocate(device);
    set_all(device, tensors.reset_mask, true);
    sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    observe(device, world, tensors.parameters, tensors.states, typename SparseWorld::Observation{}, tensors.observations, rng);
    auto& renderer = get_renderer(device, world);
    using RGBSpec = rro::TensorSpecification<SparseWorld::RENDERER_SPEC, rro::RGB<rro::format::Float32>>;
    Tensor<RGBSpec> rgb;
    malloc(device, rgb);
    copy(renderer.device, device, output(device, renderer, rro::RGB<rro::format::Float32>{}), rgb);
    for(TI instance = 0; instance < SparseWorld::INSTANCES; instance++) {
        const float scale = get_ref(device, tensors.parameters, instance).brightness_scale;
        for(TI element = 0; element < SparseWorld::FRAME_DIM; element++) {
            const float raw = data(rgb)[(instance * 2 + 1) * SparseWorld::FRAME_DIM + element] * scale;
            const float expected = raw < 0 ? 0 : (raw > 1 ? 1 : raw);
            EXPECT_FLOAT_EQ(get(device, tensors.observations, instance, element), expected);
        }
    }
    free(device, rgb);
    tensors.deallocate(device);
    free(device, rng);
    free(device, world);
    free(device, shared.library);
}
