#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
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
static const std::string DRONE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/x500.glb";
#else
static const std::string SCENE_PATH = "";
static const std::string DRONE_PATH = "";
#endif

namespace test_self_visible {
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
        using STATE_PLAIN = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
        using STATE_TYPE = l2f::StateRenderRotorPhase<l2f::StateSpecification<T, TI, STATE_PLAIN>>;
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
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 8;
        static constexpr bool SELF_VISIBLE = true;
        static constexpr bool OUTPUT_SEGMENTATION = true;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI OBS_DIM = WORLD::OBSERVATION_DIM;
    constexpr TI CAM_PIXELS = WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT;
    constexpr TI MAX_SLOTS = WORLD_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE;
    constexpr uint32_t MISS = 0xFFFFFFFFu;
    static_assert(rlt::rl::environments::hyperdrone::world::HasRotorPhase<typename WORLD::State>::VALUE);
}

using namespace test_self_visible;

// external third-person mount 1.2m behind and slightly above the body, pitched down onto it —
// guarantees the drone geometry is in frame regardless of the sampled scene placement
static void arrange(DEVICE& device,
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>>& parameters,
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>>& states, T rotor_phase){
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename WORLD::Parameters instance_parameters = rlt::get(device, parameters, instance_i);
        typename WORLD::State state = rlt::get(device, states, instance_i);
        instance_parameters.scene_yaw_cos = 1;
        instance_parameters.scene_yaw_sin = 0;
        instance_parameters.camera_mount.offset_body[0] = (T)-1.2;
        instance_parameters.camera_mount.offset_body[1] = 0;
        instance_parameters.camera_mount.offset_body[2] = (T)0.4;
        instance_parameters.camera_mount.forward_body[0] = 1;
        instance_parameters.camera_mount.forward_body[1] = 0;
        instance_parameters.camera_mount.forward_body[2] = (T)-0.3;
        state.orientation[0] = 1;
        state.orientation[1] = 0;
        state.orientation[2] = 0;
        state.orientation[3] = 0;
        state.position[0] = 0;
        state.position[1] = 0;
        state.position[2] = 0;
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            state.rotor_phase[rotor_i] = rotor_phase;
        }
        rlt::set(device, parameters, instance_parameters, instance_i);
        rlt::set(device, states, state, instance_i);
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SELF_VISIBLE, OWN_DRONE_AND_ARTICULATION){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    WORLD world;
    typename WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    world.drone_asset_path = DRONE_PATH;
    shared.scene_set.paths = {SCENE_PATH};
    rlt::init(device, world, shared, 0, 1, 0);

    ASSERT_EQ(world.entity_kinds.size(), 1);
    EXPECT_GT(world.drone_rig.num_props, 0);
    ASSERT_EQ(world.slots[0].entity_placements.size(), INSTANCES);
    EXPECT_GT(world.slots[0].entity_placements[0].num_parts, 1);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, OBS_DIM>>> observations;
    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, reset_mask);
    rlt::malloc(device, observations);
    rlt::set_all(device, reset_mask, true);
    rlt::sample_initial_parameters(device, world, parameters, reset_mask, rng);
    rlt::sample_initial_state(device, world, parameters, states, reset_mask, rng);
    arrange(device, parameters, states, (T)0);
    rlt::render(device, world, parameters, states, reset_mask);
    rlt::observe(device, world, parameters, states, typename WORLD::Observation{}, observations, rng);
    std::vector<T> frame_phase_zero(INSTANCES * OBS_DIM);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI dim = 0; dim < OBS_DIM; dim++){
            frame_phase_zero[instance_i * OBS_DIM + dim] = rlt::get(device, observations, instance_i, dim);
        }
    }

    // each camera sees its own drone's overlay ids and no foreign ones
    {
        const uint32_t S = (uint32_t)shared.library.scenes.front().instances.size();
        std::vector<uint32_t> segmentation(INSTANCES * CAM_PIXELS);
        rlt::Tensor<rlt::tensor::Specification<uint32_t, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD_SPEC::CAM_HEIGHT, WORLD_SPEC::CAM_WIDTH>>> alias;
        alias._data = segmentation.data();
        rlt::copy(world.renderer.device, device, rlt::segmentation_buffer(device, world.renderer), alias);
        for(TI camera_i = 0; camera_i < INSTANCES; camera_i++){
            const uint32_t own_first = S + (uint32_t)(camera_i * MAX_SLOTS);
            const uint32_t own_end = own_first + (uint32_t)MAX_SLOTS;
            bool own_drone_seen = false;
            for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
                const uint32_t id = segmentation[camera_i * CAM_PIXELS + pixel_i];
                if(id == MISS || id < S){
                    continue;
                }
                EXPECT_GE(id, own_first) << "camera " << camera_i << " sees a foreign drone id " << id;
                EXPECT_LT(id, own_end) << "camera " << camera_i << " sees a foreign drone id " << id;
                own_drone_seen = true;
            }
            EXPECT_TRUE(own_drone_seen) << "camera " << camera_i << " should see its own drone";
        }
    }

    // spinning the props (rotor phase) must change the rendered frame; the pose production is
    // live per render, driven by StateRenderRotorPhase
    arrange(device, parameters, states, (T)(M_PI / 2));
    rlt::set_all(device, reset_mask, false);
    rlt::render(device, world, parameters, states, reset_mask);
    rlt::observe(device, world, parameters, states, typename WORLD::Observation{}, observations, rng);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        bool changed = false;
        for(TI dim = 0; dim < OBS_DIM; dim++){
            changed = changed || rlt::get(device, observations, instance_i, dim) != frame_phase_zero[instance_i * OBS_DIM + dim];
        }
        EXPECT_TRUE(changed) << "instance " << instance_i << ": prop articulation should be visible";
    }

    rlt::free(device, parameters);
    rlt::free(device, states);
    rlt::free(device, reset_mask);
    rlt::free(device, observations);
    rlt::free(device, rng);
    rlt::free(device, world);
    rlt::free(device, shared.library);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
