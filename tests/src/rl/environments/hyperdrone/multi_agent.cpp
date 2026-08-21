#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <sstream>
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

namespace test_multi_agent {
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
        static constexpr TI N_AGENTS = 2;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        using SHADING = rlt::rendering::raytracing::Low;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 16;
        static constexpr bool SELF_VISIBLE = true;
        static constexpr bool OUTPUT_SEGMENTATION = true;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI N_AGENTS = WORLD::N_AGENTS;
    constexpr TI OBS_DIM = WORLD::OBSERVATION_DIM;
    constexpr TI PER_AGENT_OBS_PRIVILEGED = WORLD::DYNAMICS_ENV::Observation::DIM;
    constexpr TI CAM_PIXELS = WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT;
    constexpr TI MAX_SLOTS = WORLD_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE;
    constexpr uint32_t MISS = 0xFFFFFFFFu;
    static_assert(WORLD::ACTION_DIM == N_AGENTS * WORLD::DYNAMICS_ENV::ACTION_DIM);
    static_assert(WORLD::OBSERVATION_DIM == N_AGENTS * WORLD::FRAME_DIM);
    static_assert(WORLD::OBSERVATION_DIM_PRIVILEGED == N_AGENTS * PER_AGENT_OBS_PRIVILEGED);
}

using namespace test_multi_agent;

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

// identity yaw; agent 0 at the instance origin with a slightly forward FPV mount, agent 1
// parked 0.7m ahead: within the indoor-position clearance radius, so no scene geometry can
// occlude it in agent 0's field of view
static void arrange(DEVICE& device, Tensors& tensors){
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, instance_i);
        typename WORLD::State state = rlt::get(device, tensors.states, instance_i);
        instance_parameters.scene_yaw_cos = 1;
        instance_parameters.scene_yaw_sin = 0;
        instance_parameters.camera_mount.offset_body[0] = (T)0.2;
        instance_parameters.camera_mount.offset_body[1] = 0;
        instance_parameters.camera_mount.offset_body[2] = (T)0.05;
        for(TI agent_i = 0; agent_i < N_AGENTS; agent_i++){
            auto& agent_state = state.agent_states[agent_i];
            agent_state.orientation[0] = 1;
            agent_state.orientation[1] = 0;
            agent_state.orientation[2] = 0;
            agent_state.orientation[3] = 0;
            agent_state.position[0] = agent_i == 0 ? (T)0 : (T)0.7;
            agent_state.position[1] = 0;
            agent_state.position[2] = 0;
            for(TI dim = 0; dim < 3; dim++){
                agent_state.linear_velocity[dim] = 0;
                agent_state.angular_velocity[dim] = 0;
            }
            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                agent_state.rotor_phase[rotor_i] = 0;
            }
        }
        rlt::set(device, tensors.parameters, instance_parameters, instance_i);
        rlt::set(device, tensors.states, state, instance_i);
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_MULTI_AGENT, AGENTS_STEP_SEE_AND_COLLIDE){
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

    // one drone entity kind per agent, spawned per instance
    ASSERT_EQ(world.entity_kinds.size(), N_AGENTS);
    ASSERT_EQ(world.slots[0].entity_placements.size(), INSTANCES * N_AGENTS);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    Tensors tensors;
    tensors.allocate(device);
    rlt::set_all(device, tensors.reset_mask, true);
    rlt::set_all(device, tensors.actions, (T)0);
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    arrange(device, tensors);

    // agent 0's camera sees agent 1's drone; instance isolation holds per camera
    rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    {
        const uint32_t S = (uint32_t)shared.library.scenes.front().instances.size();
        constexpr TI NUM_CAMERAS = INSTANCES * N_AGENTS;
        std::vector<uint32_t> segmentation(NUM_CAMERAS * CAM_PIXELS);
        rlt::Tensor<rlt::tensor::Specification<uint32_t, TI, rlt::tensor::Shape<TI, NUM_CAMERAS, WORLD_SPEC::CAM_HEIGHT, WORLD_SPEC::CAM_WIDTH>>> alias;
        alias._data = segmentation.data();
        rlt::copy(world.renderer.device, device, rlt::segmentation_buffer(device, world.renderer), alias);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const auto& other_placement = world.slots[0].entity_placements[instance_i * N_AGENTS + world.drone_entity_kind_index + 1];
            const uint32_t other_first = S + (uint32_t)(instance_i * MAX_SLOTS + other_placement.first_slot);
            const uint32_t other_end = other_first + (uint32_t)other_placement.num_parts;
            const uint32_t own_range_first = S + (uint32_t)(instance_i * MAX_SLOTS);
            const uint32_t own_range_end = own_range_first + (uint32_t)MAX_SLOTS;
            const TI camera_i = instance_i * N_AGENTS;  // agent 0's camera
            bool other_agent_seen = false;
            for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
                const uint32_t id = segmentation[camera_i * CAM_PIXELS + pixel_i];
                if(id == MISS || id < S){
                    continue;
                }
                EXPECT_GE(id, own_range_first) << "camera of instance " << instance_i << " sees a foreign overlay id " << id;
                EXPECT_LT(id, own_range_end) << "camera of instance " << instance_i << " sees a foreign overlay id " << id;
                other_agent_seen = other_agent_seen || (id >= other_first && id < other_end);
            }
            if(!other_agent_seen){
                std::map<uint32_t, TI> histogram;
                for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
                    histogram[segmentation[camera_i * CAM_PIXELS + pixel_i]]++;
                }
                std::ostringstream ids;
                for(const auto& [id, count] : histogram){
                    ids << id << "x" << count << " ";
                }
                ADD_FAILURE() << "agent 0 of instance " << instance_i << " should see agent 1's drone ahead (expected ids [" << other_first << "," << other_end << ")); camera sees: " << ids.str();
            }
        }
    }
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::Observation{}, tensors.observations, rng);

    // agents step independently: velocity on agent 0 only moves agent 0's slice
    {
        typename WORLD::State state = rlt::get(device, tensors.states, (TI)0);
        state.agent_states[0].linear_velocity[0] = (T)1;
        rlt::set(device, tensors.states, state, (TI)0);
    }
    rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
    rlt::reward(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, tensors.rewards, rng);
    {
        typename WORLD::State after = rlt::get(device, tensors.next_states, (TI)0);
        EXPECT_GT(after.agent_states[0].position[0], (T)0.005) << "agent 0 should have moved forward";
        EXPECT_NEAR(after.agent_states[1].position[0], (T)0.7, (T)0.005) << "agent 1 should have stayed near its parked position";
    }

    // privileged observation is the per-agent concatenation (position leads the chain)
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::ObservationPrivileged{}, tensors.observations_privileged, rng);
    {
        EXPECT_NEAR(rlt::get(device, tensors.observations_privileged, (TI)0, (TI)0), (T)0, (T)1e-4);
        EXPECT_NEAR(rlt::get(device, tensors.observations_privileged, (TI)0, PER_AGENT_OBS_PRIVILEGED + 0), (T)0.7, (T)1e-4);
    }

    // proximity below the collision distance terminates the instance
    {
        typename WORLD::State state = rlt::get(device, tensors.states, (TI)0);
        state.agent_states[1].position[0] = (T)0.05;
        rlt::set(device, tensors.states, state, (TI)0);
    }
    rlt::terminated(device, world, tensors.parameters, tensors.states, tensors.terminated_flags, rng);
    EXPECT_TRUE(rlt::get(device, tensors.terminated_flags, (TI)0));
    EXPECT_FALSE(rlt::get(device, tensors.terminated_flags, (TI)1));

    tensors.deallocate(device);
    rlt::free(device, rng);
    rlt::free(device, world);
    rlt::free(device, shared.library);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
