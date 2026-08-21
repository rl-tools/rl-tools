#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cstdint>
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
static const std::string GATE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/x500.glb";
#else
static const std::string SCENE_PATH = "";
static const std::string GATE_PATH = "";
#endif

namespace test_segmentation_determinism {
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
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 8;
        static constexpr bool OUTPUT_SEGMENTATION = true;
    };
    using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::moving_gate::Specification<BASE_WORLD> {};
    using WORLD = rlt::rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI CAM_PIXELS = WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT;
    constexpr TI MAX_SLOTS = WORLD_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE;
    constexpr uint32_t MISS = 0xFFFFFFFFu;
}

using namespace test_segmentation_determinism;

struct Capture {
    std::vector<uint32_t> segmentation;
    TI scene_instances;
};

// one full lifecycle: fresh shared context + world, seeded sampling, deterministic gate
// placement (2m ahead of each instance's own camera), one render, segmentation copied out
static Capture run(DEVICE& device){
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    world.gate_asset_path = GATE_PATH;
    shared.scene_set.paths = {SCENE_PATH};
    rlt::init(device, world, shared, 0, 1, 0);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, reset_mask);
    rlt::set_all(device, reset_mask, true);
    rlt::sample_initial_parameters(device, world, parameters, reset_mask, rng);
    rlt::sample_initial_state(device, world, parameters, states, reset_mask, rng);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename WORLD::Parameters instance_parameters = rlt::get(device, parameters, instance_i);
        typename WORLD::State state = rlt::get(device, states, instance_i);
        instance_parameters.scene_yaw_cos = 1;
        instance_parameters.scene_yaw_sin = 0;
        instance_parameters.gate_orientation[0] = 1;
        instance_parameters.gate_orientation[1] = 0;
        instance_parameters.gate_orientation[2] = 0;
        instance_parameters.gate_orientation[3] = 0;
        for(TI dim = 0; dim < 3; dim++){
            instance_parameters.gate_center[dim] = instance_parameters.scene_translation[dim];
        }
        instance_parameters.gate_center[0] += 2;
        instance_parameters.gate_amplitude = 0;
        state.orientation[0] = 1;
        state.orientation[1] = 0;
        state.orientation[2] = 0;
        state.orientation[3] = 0;
        state.position[0] = 0;
        state.position[1] = 0;
        state.position[2] = 0;
        state.gate_phase = 0;
        rlt::set(device, parameters, instance_parameters, instance_i);
        rlt::set(device, states, state, instance_i);
    }
    rlt::render(device, world, parameters, states, reset_mask);

    Capture capture;
    capture.scene_instances = (TI)shared.library.scenes.front().instances.size();
    capture.segmentation.resize(INSTANCES * CAM_PIXELS);
    {
        rlt::Tensor<rlt::tensor::Specification<uint32_t, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD_SPEC::CAM_HEIGHT, WORLD_SPEC::CAM_WIDTH>>> alias;
        alias._data = capture.segmentation.data();
        rlt::copy(world.renderer.device, device, rlt::segmentation_buffer(device, world.renderer), alias);
    }

    rlt::free(device, parameters);
    rlt::free(device, states);
    rlt::free(device, reset_mask);
    rlt::free(device, rng);
    rlt::free(device, world);
    rlt::free(device, shared.library);
    return capture;
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SEGMENTATION, ID_LAYOUT_AND_DETERMINISM){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    Capture first = run(device);
    Capture second = run(device);

    // identical scene + spec + verb sequence must yield identical ids across runs
    ASSERT_EQ(first.segmentation, second.segmentation);
    ASSERT_EQ(first.scene_instances, second.scene_instances);

    // the pinned global layout: scene instances occupy [0, S); overlay o's slot s sits at
    // S + o*MAX_OVERLAY_INSTANCES + s, with one overlay per instance in instance order — the
    // same formula every backend must realize, so passing per backend pins cross-backend identity
    const uint32_t S = (uint32_t)first.scene_instances;
    for(TI camera_i = 0; camera_i < INSTANCES; camera_i++){
        const uint32_t own_first = S + (uint32_t)(camera_i * MAX_SLOTS);
        const uint32_t own_end = own_first + (uint32_t)MAX_SLOTS;
        bool own_gate_seen = false;
        bool scene_seen = false;
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            const uint32_t id = first.segmentation[camera_i * CAM_PIXELS + pixel_i];
            if(id == MISS){
                continue;
            }
            if(id < S){
                scene_seen = true;
                continue;
            }
            EXPECT_GE(id, own_first) << "camera " << camera_i << " sees a foreign overlay id " << id;
            EXPECT_LT(id, own_end) << "camera " << camera_i << " sees a foreign overlay id " << id;
            own_gate_seen = true;
        }
        EXPECT_TRUE(scene_seen) << "camera " << camera_i << " should see the scene";
        EXPECT_TRUE(own_gate_seen) << "camera " << camera_i << " should see its own gate 2m ahead";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
