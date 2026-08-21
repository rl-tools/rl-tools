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
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        static constexpr TI HISTORY_LENGTH = 4;
        using SHADING = rlt::rendering::raytracing::Low;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.25;
    };
    using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD> {
        static constexpr TI IMAGE_STACK_N = 2;
        static constexpr TI IMAGE_STACK_STRIDE = 1;
        static constexpr TI PAD_CHANNELS_TO = 8;
        static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = 0.1;
        static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0.25;
    };
    using WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI CAM_PIXELS = WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT;
    constexpr TI IMAGE_CHANNELS = BASE_WORLD::IMAGE_CHANNELS;
    // stack(2)*3 + target*3 = 9 logical channels, padded to 16
    static_assert(WORLD::OBSERVATION_CHANNELS == 16);
}

using namespace test_target_frame;

struct Tensors {
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
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

static T stack_channel(DEVICE& device, Tensors& tensors, TI instance_i, TI pixel_i, TI frame_i, TI channel_i){
    return rlt::get(device, tensors.observations, instance_i, pixel_i * WORLD::OBSERVATION_CHANNELS + frame_i * IMAGE_CHANNELS + channel_i);
}
static T target_channel(DEVICE& device, Tensors& tensors, TI instance_i, TI pixel_i, TI channel_i){
    return rlt::get(device, tensors.observations, instance_i, pixel_i * WORLD::OBSERVATION_CHANNELS + TASK_SPEC::IMAGE_STACK_N * IMAGE_CHANNELS + channel_i);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME, CACHE_AND_STACK_SEMANTICS){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    shared.scene_set.paths = {SCENE_PATH};
    rlt::init(device, world, shared, 0, 1, 0);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    Tensors tensors;
    tensors.allocate(device);
    rlt::set_all(device, tensors.reset_mask, true);
    rlt::set_all(device, tensors.actions, (T)0);
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    {
        // the task parameters component was sampled
        typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, 0);
        EXPECT_NE(instance_parameters.target_roll, (T)0);
        EXPECT_NE(instance_parameters.brightness_mismatch, (T)1);
    }

    // step 0: everything reset — target rendered and cached, stack clamps to the episode start
    rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::Observation{}, tensors.observations, rng);
    std::vector<T> target_block_step0(INSTANCES * CAM_PIXELS * IMAGE_CHANNELS);
    std::vector<T> frame0_step0(INSTANCES * CAM_PIXELS * IMAGE_CHANNELS);
    bool target_nonzero = false;
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            for(TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++){
                T target_value = target_channel(device, tensors, instance_i, pixel_i, channel_i);
                target_block_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i] = target_value;
                frame0_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i] = stack_channel(device, tensors, instance_i, pixel_i, 0, channel_i);
                target_nonzero = target_nonzero || target_value != (T)0;
                // stack frame 1 clamps to the episode start = frame 0
                EXPECT_EQ(stack_channel(device, tensors, instance_i, pixel_i, 0, channel_i), stack_channel(device, tensors, instance_i, pixel_i, 1, channel_i));
                // padded channels are zero
                for(TI pad_i = (TASK_SPEC::IMAGE_STACK_N + 1) * IMAGE_CHANNELS; pad_i < WORLD::OBSERVATION_CHANNELS; pad_i++){
                    EXPECT_EQ(rlt::get(device, tensors.observations, instance_i, pixel_i * WORLD::OBSERVATION_CHANNELS + pad_i), (T)0);
                }
            }
        }
    }
    EXPECT_TRUE(target_nonzero);

    // step 1: no reset, drone moved — the target block stays byte-identical (cache), the stack
    // shifts (frame 1 = previous frame 0)
    rlt::set_all(device, tensors.reset_mask, false);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename WORLD::State state = rlt::get(device, tensors.states, instance_i);
        state.position[0] += (T)0.2;
        state.position[2] += (T)0.1;
        rlt::set(device, tensors.states, state, instance_i);
    }
    rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::Observation{}, tensors.observations, rng);
    bool student_changed = false;
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            for(TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++){
                ASSERT_EQ(target_channel(device, tensors, instance_i, pixel_i, channel_i), target_block_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i]) << "target frame not cached across steps";
                // frame 1 (one step back) equals the previous step's frame 0
                ASSERT_EQ(stack_channel(device, tensors, instance_i, pixel_i, 1, channel_i), frame0_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i]);
                student_changed = student_changed || stack_channel(device, tensors, instance_i, pixel_i, 0, channel_i) != frame0_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i];
            }
        }
    }
    EXPECT_TRUE(student_changed) << "the student frame should change when the drone moves";

    // step 2: reset instance 0 only — its target refreshes (new episode, resampled parameters),
    // instance 1's target row stays identical
    rlt::set(device, tensors.reset_mask, true, (TI)0);
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::Observation{}, tensors.observations, rng);
    bool instance0_target_changed = false;
    for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
        for(TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++){
            instance0_target_changed = instance0_target_changed || target_channel(device, tensors, 0, pixel_i, channel_i) != target_block_step0[(0 * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i];
            ASSERT_EQ(target_channel(device, tensors, 1, pixel_i, channel_i), target_block_step0[(1 * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i]) << "unreset instance's target must be untouched";
        }
    }
    EXPECT_TRUE(instance0_target_changed);

    // the unextended verbs fall through to the base through deduction-from-derived
    rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
    rlt::reward(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, tensors.rewards, rng);
    rlt::terminated(device, world, tensors.parameters, tensors.states, tensors.terminated_flags, rng);

    tensors.deallocate(device);
    rlt::free(device, world);
    rlt::free(device, shared.library);
    rlt::free(device, rng);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
