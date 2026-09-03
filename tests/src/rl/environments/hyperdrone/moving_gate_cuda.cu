#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cuda.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GPU_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_GPU_SPEC>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
static const std::string GATE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/x500.glb";
#else
static const std::string SCENE_PATH = "";
static const std::string GATE_PATH = "";
#endif

namespace test_moving_gate_cuda {
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

using namespace test_moving_gate_cuda;

template <typename T_DEVICE>
struct Tensors {
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, OBS_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM_PRIVILEGED>>> observations_privileged;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES>>> rewards;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> terminated_flags;
    void allocate(T_DEVICE& device){
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
    void deallocate(T_DEVICE& device){
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

static bool cuda_available(){
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

// deterministic scene placement mirroring the CPU test: identity yaw + gate orientation, drone
// at the origin of its instance frame, gate 2m ahead of instance 0; instance 1's gate parked far
static void arrange(DEVICE& device, Tensors<DEVICE>& host){
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename WORLD::Parameters instance_parameters = rlt::get(device, host.parameters, instance_i);
        typename WORLD::State state = rlt::get(device, host.states, instance_i);
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
        rlt::set(device, host.parameters, instance_parameters, instance_i);
        rlt::set(device, host.states, state, instance_i);
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_MOVING_GATE_CUDA, DEVICE_ENTITY_MECHANISM_AND_TASK){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    if(!cuda_available()){
        GTEST_SKIP() << "no CUDA device available";
    }
    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device);
    rlt::init(device_gpu);
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device_gpu, world);
    world.gate_asset_path = GATE_PATH;
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device_gpu, world, shared, dataset, corpus, 0, 1, 0);

    RNG_GPU rng;
    rlt::malloc(device_gpu, rng);
    rlt::init(device_gpu, rng, 1337);
    Tensors<DEVICE_GPU> tensors;
    tensors.allocate(device_gpu);
    Tensors<DEVICE> host;
    host.allocate(device);

    rlt::set_all(device_gpu, tensors.reset_mask, true);
    rlt::set_all(device_gpu, tensors.actions, (T)0);
    rlt::sample_initial_parameters(device_gpu, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device_gpu, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, tensors.parameters, host.parameters);
    rlt::copy(device_gpu, device, tensors.states, host.states);
    {
        // the gate components were sampled by the device kernel
        typename WORLD::Parameters instance_parameters = rlt::get(device, host.parameters, (TI)0);
        EXPECT_GE(instance_parameters.gate_frequency, TASK_SPEC::GATE_FREQUENCY_MIN);
        EXPECT_LE(instance_parameters.gate_frequency, TASK_SPEC::GATE_FREQUENCY_MAX);
        typename WORLD::State state = rlt::get(device, host.states, (TI)0);
        EXPECT_FALSE(state.gate_passed);
        EXPECT_FALSE(state.gate_crashed);
    }
    arrange(device, host);
    rlt::copy(device, device_gpu, host.parameters, tensors.parameters);
    rlt::copy(device, device_gpu, host.states, tensors.states);

    // the gate is visible to its own instance's camera and only to it
    rlt::render(device_gpu, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device_gpu, world, tensors.parameters, tensors.states, typename BASE_WORLD::Observation{}, tensors.observations, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, tensors.observations, host.observations);
    std::vector<T> frame_gate_near(INSTANCES * OBS_DIM);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI dim = 0; dim < OBS_DIM; dim++){
            frame_gate_near[instance_i * OBS_DIM + dim] = rlt::get(device, host.observations, instance_i, dim);
        }
    }
    {
        typename WORLD::Parameters instance_parameters = rlt::get(device, host.parameters, (TI)0);
        instance_parameters.gate_center[0] = 1000;
        instance_parameters.gate_center[2] = -1000;
        rlt::set(device, host.parameters, instance_parameters, (TI)0);
    }
    rlt::copy(device, device_gpu, host.parameters, tensors.parameters);
    rlt::set_all(device_gpu, tensors.reset_mask, false);
    rlt::render(device_gpu, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device_gpu, world, tensors.parameters, tensors.states, typename BASE_WORLD::Observation{}, tensors.observations, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, tensors.observations, host.observations);
    bool instance0_changed = false;
    bool instance1_changed = false;
    for(TI dim = 0; dim < OBS_DIM; dim++){
        instance0_changed = instance0_changed || rlt::get(device, host.observations, (TI)0, dim) != frame_gate_near[0 * OBS_DIM + dim];
        instance1_changed = instance1_changed || rlt::get(device, host.observations, (TI)1, dim) != frame_gate_near[1 * OBS_DIM + dim];
    }
    EXPECT_TRUE(instance0_changed) << "the gate should be visible in its own instance's camera";
    EXPECT_FALSE(instance1_changed) << "per-camera attachment must isolate instances";

    // plane crossing within the aperture sets gate_passed and pays the pass reward, device-side
    {
        typename WORLD::Parameters instance_parameters = rlt::get(device, host.parameters, (TI)0);
        instance_parameters.gate_center[0] = instance_parameters.scene_translation[0] + (T)0.05;
        instance_parameters.gate_center[1] = instance_parameters.scene_translation[1];
        instance_parameters.gate_center[2] = instance_parameters.scene_translation[2];
        instance_parameters.gate_amplitude = 0;
        instance_parameters.gate_aperture_radius = (T)10;
        rlt::set(device, host.parameters, instance_parameters, (TI)0);
        typename WORLD::State state = rlt::get(device, host.states, (TI)0);
        state.linear_velocity[0] = (T)20;
        rlt::set(device, host.states, state, (TI)0);
    }
    rlt::copy(device, device_gpu, host.parameters, tensors.parameters);
    rlt::copy(device, device_gpu, host.states, tensors.states);
    rlt::step(device_gpu, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
    rlt::reward(device_gpu, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, tensors.rewards, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, tensors.next_states, host.next_states);
    rlt::copy(device_gpu, device, tensors.rewards, host.rewards);
    {
        typename WORLD::State before = rlt::get(device, host.states, (TI)0);
        typename WORLD::State after = rlt::get(device, host.next_states, (TI)0);
        EXPECT_NE(after.gate_phase, before.gate_phase) << "gate phase should advance with the step";
        EXPECT_TRUE(after.gate_passed);
        EXPECT_FALSE(after.gate_crashed);
        typename WORLD::State after_1 = rlt::get(device, host.next_states, (TI)1);
        EXPECT_FALSE(after_1.gate_passed);
        EXPECT_GT(rlt::get(device, host.rewards, (TI)0), rlt::get(device, host.rewards, (TI)1) + (T)5) << "the pass transition should pay the gate bonus";
    }

    // crossing outside the aperture crashes and terminates
    {
        typename WORLD::Parameters instance_parameters = rlt::get(device, host.parameters, (TI)0);
        instance_parameters.gate_aperture_radius = (T)1e-6;
        rlt::set(device, host.parameters, instance_parameters, (TI)0);
        typename WORLD::State state = rlt::get(device, host.states, (TI)0);
        state.gate_passed = false;
        state.gate_crashed = false;
        state.position[0] = 0;
        state.position[1] = (T)0.5;
        state.linear_velocity[0] = (T)20;
        rlt::set(device, host.states, state, (TI)0);
    }
    rlt::copy(device, device_gpu, host.parameters, tensors.parameters);
    rlt::copy(device, device_gpu, host.states, tensors.states);
    rlt::step(device_gpu, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, tensors.next_states, host.next_states);
    {
        typename WORLD::State after = rlt::get(device, host.next_states, (TI)0);
        EXPECT_TRUE(after.gate_crashed);
        rlt::copy(device_gpu, device_gpu, tensors.next_states, tensors.states);
        rlt::terminated(device_gpu, world, tensors.parameters, tensors.states, tensors.terminated_flags, rng);
        cudaDeviceSynchronize();
        rlt::copy(device_gpu, device, tensors.terminated_flags, host.terminated_flags);
        EXPECT_TRUE(rlt::get(device, host.terminated_flags, (TI)0));
        EXPECT_FALSE(rlt::get(device, host.terminated_flags, (TI)1));
    }

    // privileged observation carries the gate state (sin/cos phase in the last two entries)
    rlt::observe(device_gpu, world, tensors.parameters, tensors.states, typename WORLD::ObservationPrivileged{}, tensors.observations_privileged, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, tensors.observations_privileged, host.observations_privileged);
    rlt::copy(device_gpu, device, tensors.states, host.states);
    {
        typename WORLD::State state = rlt::get(device, host.states, (TI)0);
        constexpr TI BASE_DIM = BASE_WORLD::OBSERVATION_DIM_PRIVILEGED;
        EXPECT_NEAR(rlt::get(device, host.observations_privileged, (TI)0, BASE_DIM + 6), std::sin(state.gate_phase), 1e-5);
        EXPECT_NEAR(rlt::get(device, host.observations_privileged, (TI)0, BASE_DIM + 7), std::cos(state.gate_phase), 1e-5);
    }

    tensors.deallocate(device_gpu);
    host.deallocate(device);
    rlt::free(device_gpu, world);
    rlt::free(device, shared.library);
    rlt::free(device_gpu, rng);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
