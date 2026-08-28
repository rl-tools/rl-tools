#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
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
static const std::string GATE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/x500.glb";
#else
static const std::string SCENE_PATH = "";
static const std::string GATE_PATH = "";
#endif

namespace test_moving_gate {
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
    };
    using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::moving_gate::Specification<BASE_WORLD> {
        static constexpr T GATE_PASS_REWARD = 10;
    };
    using WORLD = rlt::rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI OBS_DIM = BASE_WORLD::OBSERVATION_DIM;
}

using namespace test_moving_gate;

struct Tensors {
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, OBS_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM_PRIVILEGED>>> observations_privileged;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES>>> rewards;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> terminated_flags;
    void allocate(DEVICE& device){
        rlt::malloc(device, parameters);
        rlt::malloc(device, states);
        rlt::malloc(device, next_states);
        rlt::malloc(device, reset_mask);
        rlt::malloc(device, actions);
        rlt::malloc(device, observations);
        rlt::malloc(device, observations_privileged);
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
        rlt::free(device, observations_privileged);
        rlt::free(device, rewards);
        rlt::free(device, terminated_flags);
    }
};

// deterministic scene placement: identity yaw, drone at the sampled indoor position, gate 2m
// ahead of instance 0's forward-looking camera; instance 1's gate is parked far away
static void arrange(DEVICE& device, Tensors& tensors){
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, instance_i);
        typename WORLD::State state = rlt::get(device, tensors.states, instance_i);
        instance_parameters.scene_yaw_cos = 1;
        instance_parameters.scene_yaw_sin = 0;
        instance_parameters.gate_orientation[0] = 1;
        instance_parameters.gate_orientation[1] = 0;
        instance_parameters.gate_orientation[2] = 0;
        instance_parameters.gate_orientation[3] = 0;
        state.orientation[0] = 1;
        state.orientation[1] = 0;
        state.orientation[2] = 0;
        state.orientation[3] = 0;
        state.position[0] = 0;
        state.position[1] = 0;
        state.position[2] = 0;
        if(instance_i == 0){
            for(TI dim = 0; dim < 3; dim++){
                instance_parameters.gate_center[dim] = instance_parameters.scene_translation[dim];
            }
            instance_parameters.gate_center[0] += 2;
            instance_parameters.gate_amplitude = (T)0.5;
        } else {
            instance_parameters.gate_center[0] = 1000;
            instance_parameters.gate_center[1] = 1000;
            instance_parameters.gate_center[2] = -1000;
            instance_parameters.gate_amplitude = 0;
        }
        state.gate_phase = 0;
        rlt::set(device, tensors.parameters, instance_parameters, instance_i);
        rlt::set(device, tensors.states, state, instance_i);
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_MOVING_GATE, ENTITY_MECHANISM_AND_TASK){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    world.gate_asset_path = GATE_PATH;
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);

    // entity bookkeeping: one kind, spawned per instance in pinned order
    ASSERT_EQ(world.entity_kinds.size(), 1);
    ASSERT_EQ(world.slots[0].entity_placements.size(), INSTANCES);
    EXPECT_GT(world.slots[0].entity_placements[0].num_parts, 0);

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
        // the task state/parameters components were sampled
        typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, 0);
        EXPECT_GE(instance_parameters.gate_frequency, TASK_SPEC::GATE_FREQUENCY_MIN);
        EXPECT_LE(instance_parameters.gate_frequency, TASK_SPEC::GATE_FREQUENCY_MAX);
        typename WORLD::State state = rlt::get(device, tensors.states, 0);
        EXPECT_FALSE(state.gate_passed);
    }
    arrange(device, tensors);

    // the gate is visible to its own instance's camera and only to it
    rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device, world, tensors.parameters, tensors.states, typename BASE_WORLD::Observation{}, tensors.observations, rng);
    std::vector<T> frame_gate_near(INSTANCES * OBS_DIM);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI dim = 0; dim < OBS_DIM; dim++){
            frame_gate_near[instance_i * OBS_DIM + dim] = rlt::get(device, tensors.observations, instance_i, dim);
        }
    }
    // park instance 0's gate far away too and re-render
    {
        typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, (TI)0);
        instance_parameters.gate_center[0] = 1000;
        instance_parameters.gate_center[2] = -1000;
        rlt::set(device, tensors.parameters, instance_parameters, (TI)0);
    }
    rlt::set_all(device, tensors.reset_mask, false);
    rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device, world, tensors.parameters, tensors.states, typename BASE_WORLD::Observation{}, tensors.observations, rng);
    bool instance0_changed = false;
    bool instance1_changed = false;
    for(TI dim = 0; dim < OBS_DIM; dim++){
        instance0_changed = instance0_changed || rlt::get(device, tensors.observations, (TI)0, dim) != frame_gate_near[0 * OBS_DIM + dim];
        instance1_changed = instance1_changed || rlt::get(device, tensors.observations, (TI)1, dim) != frame_gate_near[1 * OBS_DIM + dim];
    }
    EXPECT_TRUE(instance0_changed) << "the gate should be visible in its own instance's camera";
    EXPECT_FALSE(instance1_changed) << "per-camera attachment must isolate instances";

    // gate motion state: phase advances, plane crossing sets passed within the aperture
    {
        typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, (TI)0);
        instance_parameters.gate_center[0] = instance_parameters.scene_translation[0] + (T)0.05;
        instance_parameters.gate_center[1] = instance_parameters.scene_translation[1];
        instance_parameters.gate_center[2] = instance_parameters.scene_translation[2];
        instance_parameters.gate_amplitude = 0;
        instance_parameters.gate_aperture_radius = (T)10;  // pass regardless of lateral drift
        rlt::set(device, tensors.parameters, instance_parameters, (TI)0);
        typename WORLD::State state = rlt::get(device, tensors.states, (TI)0);
        state.linear_velocity[0] = (T)20;  // crosses the plane 5cm ahead within one 10ms step
        rlt::set(device, tensors.states, state, (TI)0);
    }
    rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
    rlt::reward(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, tensors.rewards, rng);
    {
        typename WORLD::State before = rlt::get(device, tensors.states, (TI)0);
        typename WORLD::State after = rlt::get(device, tensors.next_states, (TI)0);
        EXPECT_NE(after.gate_phase, before.gate_phase) << "gate phase should advance with the step";
        EXPECT_TRUE(after.gate_passed) << "crossing within the aperture should set gate_passed";
        EXPECT_FALSE(after.gate_crashed);
        typename WORLD::State after_1 = rlt::get(device, tensors.next_states, (TI)1);
        EXPECT_FALSE(after_1.gate_passed);
    }

    // crossing outside the aperture crashes and terminates
    {
        typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, (TI)0);
        instance_parameters.gate_aperture_radius = (T)1e-6;
        rlt::set(device, tensors.parameters, instance_parameters, (TI)0);
        typename WORLD::State state = rlt::get(device, tensors.states, (TI)0);
        state.gate_passed = false;
        state.gate_crashed = false;
        state.position[0] = 0;
        state.position[1] = (T)0.5;  // off-axis
        state.linear_velocity[0] = (T)20;
        rlt::set(device, tensors.states, state, (TI)0);
    }
    rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
    {
        typename WORLD::State after = rlt::get(device, tensors.next_states, (TI)0);
        EXPECT_TRUE(after.gate_crashed);
        rlt::copy(device, device, tensors.next_states, tensors.states);
        rlt::terminated(device, world, tensors.parameters, tensors.states, tensors.terminated_flags, rng);
        EXPECT_TRUE(rlt::get(device, tensors.terminated_flags, (TI)0));
        EXPECT_FALSE(rlt::get(device, tensors.terminated_flags, (TI)1));
    }

    // privileged observation carries the gate state (sin/cos phase in the last two entries)
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::ObservationPrivileged{}, tensors.observations_privileged, rng);
    {
        typename WORLD::State state = rlt::get(device, tensors.states, (TI)0);
        constexpr TI BASE_DIM = BASE_WORLD::OBSERVATION_DIM_PRIVILEGED;
        EXPECT_NEAR(rlt::get(device, tensors.observations_privileged, (TI)0, BASE_DIM + 6), std::sin(state.gate_phase), 1e-5);
        EXPECT_NEAR(rlt::get(device, tensors.observations_privileged, (TI)0, BASE_DIM + 7), std::cos(state.gate_phase), 1e-5);
    }

    tensors.deallocate(device);
    rlt::free(device, world);
    rlt::free(device, shared.library);
    rlt::free(device, rng);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
