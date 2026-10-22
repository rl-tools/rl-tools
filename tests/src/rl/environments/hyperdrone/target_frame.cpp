#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cstring>
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

namespace test_target_frame {
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
        static constexpr TI HISTORY_LENGTH = 4;
        using SHADING = rlt::hyperlux::Low;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.25;
    };
    using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD> {
        static constexpr TI IMAGE_STACK_N = 2;
        static constexpr TI IMAGE_STACK_STRIDE = 1;
        static constexpr TI PAD_CHANNELS_TO = 8;
        static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = 0.1;
        static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0.25;
        static constexpr TI TARGET_BANK_REFRESH_STEPS = 3;
    };
    using WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI CAM_PIXELS = WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT;
    constexpr TI IMAGE_CHANNELS = BASE_WORLD::IMAGE_CHANNELS;
    constexpr TI FRAME_DIM = BASE_WORLD::FRAME_DIM;
    // stack(2)*3 + target*3 = 9 logical channels, padded to 16
    static_assert(WORLD::OBSERVATION_CHANNELS == 16);
}

using namespace test_target_frame;

template <typename WORLD_TYPE>
struct TensorsFor {
    rlt::Tensor<rlt::tensor::Specification<typename WORLD_TYPE::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD_TYPE::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD_TYPE::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD_TYPE::OBSERVATION_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES>>> rewards;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> terminated_flags;
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

using Tensors = TensorsFor<WORLD>;

namespace test_target_auxiliary {
    namespace out = rlt::hyperlux::outputs;
    struct Specification: WORLD_SPEC {
        using CAMERA_OUTPUTS = rlt::rl::environments::hyperdrone::CameraOutputs<out::List<out::RGB<>>, out::List<out::RGB<out::format::Float32>, out::Flow<>, out::FlowStatus<>>, out::List<out::RGB<out::format::Float32>, out::Depth<>>>;
        using OBSERVATION = rlt::rl::environments::hyperdrone::ImageObservation<1>;
    };
    using Base = rlt::rl::environments::hyperdrone::World<Specification>;
    struct Task: TASK_SPEC {
        using NEXT_WORLD = Base;
    };
    using World = rlt::rl::environments::hyperdrone::tasks::target_frame::World<Task>;
    static_assert(World::N_VIEWS == 3 && World::N_OBSERVATION_VIEWS == 1 && World::TARGET_CAMERA == 1);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME, SELECTED_TARGET_CAMERA_WITH_AUXILIARY_VIEWS) {
    using namespace rlt;
    using SparseWorld = test_target_auxiliary::World;
    namespace out = hyperlux::outputs;
    if(SCENE_PATH.empty()) {
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    init(device);
    SparseWorld world;
    typename SparseWorld::SharedContext shared;
    malloc(device, shared.library);
    malloc(device, world);
    world.parameters.agents[0].cameras[0].mount.pose.position[0] = -1;
    world.parameters.agents[0].cameras[1].fov = 95;
    world.parameters.agents[0].cameras[2].fov = 110;
    hyperlux::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    hyperlux::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);
    RNG rng;
    malloc(device, rng);
    init(device, rng, 1337);
    TensorsFor<SparseWorld> tensors;
    tensors.allocate(device);
    set_all(device, tensors.reset_mask, true);
    sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    observe(device, world, tensors.parameters, tensors.states, typename SparseWorld::Observation{}, tensors.observations, rng);
    auto& renderer = get_renderer(device, world);
    Tensor<out::TensorSpecification<typename SparseWorld::RENDERER_SPEC, out::RGB<out::format::Float32>>> rgb;
    malloc(device, rgb);
    copy(renderer.device, device, output(device, renderer, out::RGB<out::format::Float32>{}), rgb);
    for(TI instance = 0; instance < INSTANCES; instance++) {
        const float scale = get_ref(device, tensors.parameters, instance).brightness_scale;
        for(TI pixel = 0; pixel < CAM_PIXELS; pixel++) {
            for(TI channel = 0; channel < 3; channel++) {
                const float raw = data(rgb)[(instance * 2 * CAM_PIXELS + pixel) * 3 + channel] * scale;
                const float expected = raw < 0 ? 0 : (raw > 1 ? 1 : raw);
                EXPECT_FLOAT_EQ(get(device, tensors.observations, instance, pixel * SparseWorld::OBSERVATION_CHANNELS + channel), expected);
                EXPECT_FLOAT_EQ(get(device, tensors.observations, instance, pixel * SparseWorld::OBSERVATION_CHANNELS + TASK_SPEC::IMAGE_STACK_N * 3 + channel), get(device, world.target_frames, instance, pixel * 3 + channel));
            }
        }
    }
    free(device, rgb);
    tensors.deallocate(device);
    free(device, rng);
    rlt::free(device, world);
    free(device, shared.library);
}

static T stack_channel(DEVICE& device, Tensors& tensors, TI instance_i, TI pixel_i, TI frame_i, TI channel_i){
    return rlt::get(device, tensors.observations, instance_i, pixel_i * WORLD::OBSERVATION_CHANNELS + frame_i * IMAGE_CHANNELS + channel_i);
}
static T target_channel(DEVICE& device, Tensors& tensors, TI instance_i, TI pixel_i, TI channel_i){
    return rlt::get(device, tensors.observations, instance_i, pixel_i * WORLD::OBSERVATION_CHANNELS + TASK_SPEC::IMAGE_STACK_N * IMAGE_CHANNELS + channel_i);
}
static std::vector<float> target_rows(DEVICE& device, WORLD& world){
    std::vector<float> rows(INSTANCES * FRAME_DIM);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI element_i = 0; element_i < FRAME_DIM; element_i++){
            rows[instance_i * FRAME_DIM + element_i] = rlt::get(device, world.target_frames, instance_i, element_i);
        }
    }
    return rows;
}
static std::vector<float> bank_rows(DEVICE& device, WORLD& world){
    std::vector<float> rows(INSTANCES * FRAME_DIM);
    for(TI setup_i = 0; setup_i < INSTANCES; setup_i++){
        for(TI element_i = 0; element_i < FRAME_DIM; element_i++){
            rows[setup_i * FRAME_DIM + element_i] = rlt::get(device, world.bank_frames, setup_i, element_i);
        }
    }
    return rows;
}
// a resetting instance adopts a setup whole: its target row is that setup's bank frame and its
// view parameters are the setup's
static void expect_adopted(DEVICE& device, WORLD& world, Tensors& tensors, TI instance_i){
    const TI index = rlt::get(device, world.setup_index, instance_i);
    ASSERT_LT(index, INSTANCES);
    const auto& setup = rlt::get_ref(device, world.setups, index);
    const typename WORLD::Parameters parameters = rlt::get(device, tensors.parameters, instance_i);
    EXPECT_EQ(parameters.agents[0].cameras[0].fov, setup.agents[0].cameras[0].fov);
    EXPECT_EQ(parameters.brightness_scale, setup.brightness_scale);
    EXPECT_EQ(parameters.scene_translation[0], setup.scene_translation[0]);
    EXPECT_EQ(parameters.scene_yaw_cos, setup.scene_yaw_cos);
    EXPECT_EQ(parameters.target_roll, setup.target_roll);
    EXPECT_EQ(parameters.brightness_mismatch, setup.brightness_mismatch);
    for(TI element_i = 0; element_i < FRAME_DIM; element_i++){
        ASSERT_EQ(rlt::get(device, world.target_frames, instance_i, element_i), rlt::get(device, world.bank_frames, index, element_i)) << "instance " << instance_i << " element " << element_i;
    }
}

template<bool RAW_FIRST>
void bank_and_stack_semantics(){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    rlt::hyperlux::datasets::procthor::GLB dataset{{}, {SCENE_PATH, SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::hyperlux::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 2, 0);
    EXPECT_TRUE(rlt::rl::environments::hyperdrone::tasks::target_frame::_bank_due(world));

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    Tensors tensors;
    tensors.allocate(device);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, BASE_WORLD::OBSERVATION_DIM>>> raw_observations;
    rlt::malloc(device, raw_observations);
    rlt::set_all(device, world.target_frames, (T)-123);
    auto render_observation = [&](){
        rlt::request_render(device, world, tensors.reset_mask);
        if constexpr(RAW_FIRST){
            rlt::observe(device, world, tensors.parameters, tensors.states, typename BASE_WORLD::Observation{}, raw_observations, rng);
        }
        else{
            rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
        }
        const TI history_step = world.history_step;
        rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::Observation{}, tensors.observations, rng);
        EXPECT_EQ(world.history_step, history_step);
    };
    rlt::set_all(device, tensors.reset_mask, true);
    rlt::set_all(device, tensors.actions, (T)0);
    // the first reset fills the bank and the resetting instances adopt its setups in order, so
    // a full reset gives every instance its own
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    EXPECT_EQ(world.bank_scene_version, world.scene_version);
    EXPECT_EQ(world.bank_step, (TI)0);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_EQ(rlt::get(device, world.setup_index, instance_i), instance_i);
        expect_adopted(device, world, tensors, instance_i);
    }
    EXPECT_EQ(rlt::get(device, world.bank_cursor, (TI)0), INSTANCES);
    {
        // the task component was sampled into the setups
        const auto& setup = rlt::get_ref(device, world.setups, 0);
        EXPECT_NE(setup.target_roll, (T)0);
        EXPECT_NE(setup.brightness_mismatch, (T)1);
    }

    // step 0: the observation carries the adopted target, the stack clamps to the episode start
    render_observation();
    std::vector<T> target_block_step0(INSTANCES * CAM_PIXELS * IMAGE_CHANNELS);
    std::vector<T> frame0_step0(INSTANCES * CAM_PIXELS * IMAGE_CHANNELS);
    bool target_nonzero = false;
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            for(TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++){
                T target_value = target_channel(device, tensors, instance_i, pixel_i, channel_i);
                target_block_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i] = target_value;
                frame0_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i] = stack_channel(device, tensors, instance_i, pixel_i, 0, channel_i);
                ASSERT_EQ(target_value, rlt::get(device, world.target_frames, instance_i, pixel_i * IMAGE_CHANNELS + channel_i));
                target_nonzero = target_nonzero || target_value != (T)0;
                EXPECT_EQ(stack_channel(device, tensors, instance_i, pixel_i, 0, channel_i), stack_channel(device, tensors, instance_i, pixel_i, 1, channel_i));
                for(TI pad_i = (TASK_SPEC::IMAGE_STACK_N + 1) * IMAGE_CHANNELS; pad_i < WORLD::OBSERVATION_CHANNELS; pad_i++){
                    EXPECT_EQ(rlt::get(device, tensors.observations, instance_i, pixel_i * WORLD::OBSERVATION_CHANNELS + pad_i), (T)0);
                }
            }
        }
    }
    EXPECT_TRUE(target_nonzero);

    // step 1: no reset, drone moved — the target block stays byte-identical, the stack shifts
    rlt::set_all(device, tensors.reset_mask, false);
    rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
    rlt::copy(device, device, tensors.next_states, tensors.states);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename WORLD::State state = rlt::get(device, tensors.states, instance_i);
        state.position[0] += (T)0.2;
        state.position[2] += (T)0.1;
        rlt::set(device, tensors.states, state, instance_i);
    }
    render_observation();
    bool student_changed = false;
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            for(TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++){
                ASSERT_EQ(target_channel(device, tensors, instance_i, pixel_i, channel_i), target_block_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i]) << "target frame not cached across steps";
                ASSERT_EQ(stack_channel(device, tensors, instance_i, pixel_i, 1, channel_i), frame0_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i]);
                student_changed = student_changed || stack_channel(device, tensors, instance_i, pixel_i, 0, channel_i) != frame0_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i];
            }
        }
    }
    EXPECT_TRUE(student_changed) << "the student frame should change when the drone moves";

    // step 2: reset instance 0 only — it adopts a setup from the bank, instance 1's row stays
    const std::vector<float> rows_before = target_rows(device, world);
    rlt::set(device, tensors.reset_mask, true, (TI)0);
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    expect_adopted(device, world, tensors, 0);
    EXPECT_EQ(rlt::get(device, world.bank_cursor, (TI)0), INSTANCES + 1);
    render_observation();
    for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
        for(TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++){
            ASSERT_EQ(target_channel(device, tensors, 1, pixel_i, channel_i), target_block_step0[(1 * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i]) << "unreset instance's target must be untouched";
        }
    }

    // the bank refreshes every TARGET_BANK_REFRESH_STEPS frames without touching running episodes
    const std::vector<float> bank_before = bank_rows(device, world);
    rlt::set_all(device, tensors.reset_mask, false);
    for(TI step_i = 0; step_i < TASK_SPEC::TARGET_BANK_REFRESH_STEPS; step_i++){
        rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
        rlt::copy(device, device, tensors.next_states, tensors.states);
        rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    }
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    EXPECT_EQ(world.bank_step, world.history_step);
    EXPECT_EQ(rlt::get(device, world.bank_cursor, (TI)0), (TI)0);
    EXPECT_NE(bank_rows(device, world), bank_before) << "the refresh should draw new setups";
    {
        const std::vector<float> rows_after = target_rows(device, world);
        for(TI element_i = 0; element_i < FRAME_DIM; element_i++){
            ASSERT_EQ(rows_after[FRAME_DIM + element_i], rows_before[FRAME_DIM + element_i]) << "a refresh must leave running episodes' targets alone";
        }
    }

    // a scene change invalidates the bank, re-selecting the same scene included
    rlt::rotate_scene(device, world);
    EXPECT_EQ(world.active_slot, (TI)1);
    EXPECT_TRUE(rlt::rl::environments::hyperdrone::tasks::target_frame::_bank_due(world));
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    EXPECT_EQ(world.bank_scene_version, world.scene_version);
    rlt::select_scene(device, world, world.active_slot);
    EXPECT_TRUE(rlt::rl::environments::hyperdrone::tasks::target_frame::_bank_due(world));

    // the unextended verbs fall through to the base through deduction-from-derived
    rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
    rlt::reward(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, tensors.rewards, rng);
    rlt::terminated(device, world, tensors.parameters, tensors.states, tensors.terminated_flags, rng);

    rlt::free(device, raw_observations);
    tensors.deallocate(device);
    rlt::free(device, world);
    rlt::free(device, shared.library);
    rlt::free(device, rng);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME, BANK_AND_STACK_SEMANTICS){
    bank_and_stack_semantics<false>();
}
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME, RAW_OBSERVATION_KEEPS_TARGET){
    bank_and_stack_semantics<true>();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
