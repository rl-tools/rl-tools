#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cuda.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace rro = rlt::rendering::raytracing::outputs;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GPU_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_GPU_SPEC>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
static const std::string DRONE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/x500.glb";
#else
static const std::string SCENE_PATH = "";
static const std::string DRONE_PATH = "";
#endif

namespace test_self_visible_cuda {
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
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        using SHADING = rlt::rendering::raytracing::Low;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 8;
        static constexpr bool SELF_VISIBLE = true;
        using OUTPUTS = rro::List<rro::RGB<rro::format::Float32>, rro::Segmentation<rro::format::UInt32, rro::labels::Instance>, rro::Probes>;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI OBS_DIM = WORLD::OBSERVATION_DIM;
    constexpr TI CAM_PIXELS = WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT;
    constexpr TI MAX_SLOTS = WORLD_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE;
    constexpr uint32_t MISS = 0xFFFFFFFFu;
    static_assert(rlt::rl::environments::hyperdrone::world::HasRotorPhase<typename WORLD::State>::VALUE);
}

using namespace test_self_visible_cuda;

static bool cuda_available(){
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

static void arrange(DEVICE& device,
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>>& parameters,
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>>& states, T rotor_phase){
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename WORLD::Parameters instance_parameters = rlt::get(device, parameters, instance_i);
        typename WORLD::State state = rlt::get(device, states, instance_i);
        instance_parameters.scene_yaw_cos = 1;
        instance_parameters.scene_yaw_sin = 0;
        instance_parameters.scene_translation[0] = 0;
        instance_parameters.scene_translation[1] = 0;
        instance_parameters.scene_translation[2] = 100;
        using rlt::rotor_wheelbase;
        const T wheelbase = rotor_wheelbase(device, instance_parameters.agents[0].dynamics.dynamics.rotor_positions, WORLD::DYNAMICS_ENV::ACTION_DIM);
        instance_parameters.agents[0].cameras[0].mount.pose.position[0] = (T)-2 * wheelbase;
        instance_parameters.agents[0].cameras[0].mount.pose.position[1] = 0;
        instance_parameters.agents[0].cameras[0].mount.pose.position[2] = (T)0.75 * wheelbase;
        const T half_pitch = std::atan2((T)0.75, (T)2) / 2;
        instance_parameters.agents[0].cameras[0].mount.pose.orientation[0] = std::cos(half_pitch);
        instance_parameters.agents[0].cameras[0].mount.pose.orientation[1] = 0;
        instance_parameters.agents[0].cameras[0].mount.pose.orientation[2] = std::sin(half_pitch);
        instance_parameters.agents[0].cameras[0].mount.pose.orientation[3] = 0;
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

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SELF_VISIBLE_CUDA, OWN_DRONE_AND_ARTICULATION){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    if(!cuda_available()){
        GTEST_SKIP() << "no CUDA device available";
    }
    DEVICE_GPU device_gpu;
    auto& device = device_gpu.rendering;
    rlt::init(device);
    rlt::init(device_gpu);
    WORLD world;
    typename WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device_gpu, world);
    shared.drones.push_back({DRONE_PATH, {}});
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device_gpu, world, shared, dataset, corpus, 0, 1, 0);

    ASSERT_TRUE(world.entity_kinds.empty());
    ASSERT_EQ(shared.drone_bodies.size(), 1);
    const auto& body = shared.drone_bodies[0];
    EXPECT_EQ(body.rig.num_coordinates, 4);
    EXPECT_GT(body.rig.num_parts, 1);

    RNG_GPU rng;
    rlt::malloc(device_gpu, rng);
    rlt::init(device_gpu, rng, 1337);
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters, parameters_gpu;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, states_gpu;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, OBS_DIM>>> observations, observations_gpu;
    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, observations);
    rlt::malloc(device_gpu, parameters_gpu);
    rlt::malloc(device_gpu, states_gpu);
    rlt::malloc(device_gpu, reset_mask);
    rlt::malloc(device_gpu, observations_gpu);
    rlt::set_all(device_gpu, reset_mask, true);
    rlt::sample_initial_parameters(device_gpu, world, parameters_gpu, reset_mask, rng);
    rlt::sample_initial_state(device_gpu, world, parameters_gpu, states_gpu, reset_mask, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, parameters_gpu, parameters);
    rlt::copy(device_gpu, device, states_gpu, states);
    arrange(device, parameters, states, (T)0);
    rlt::copy(device, device_gpu, parameters, parameters_gpu);
    rlt::copy(device, device_gpu, states, states_gpu);
    rlt::render(device_gpu, world, parameters_gpu, states_gpu, reset_mask);
    rlt::observe(device_gpu, world, parameters_gpu, states_gpu, typename WORLD::Observation{}, observations_gpu, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, observations_gpu, observations);
    std::vector<T> frame_phase_zero(INSTANCES * OBS_DIM);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI dim = 0; dim < OBS_DIM; dim++){
            frame_phase_zero[instance_i * OBS_DIM + dim] = rlt::get(device, observations, instance_i, dim);
        }
    }

    // the rows the kernel wrote into the renderer equal the host rig evaluation of the same state
    {
        using DRONE_RIG_SPEC = typename decltype(body.rig)::SPEC;
        auto& renderer = rlt::get_renderer(device, world);
        std::vector<float> rows(INSTANCES * MAX_SLOTS * 12);
        rlt::Tensor<typename decltype(renderer.transforms)::SPEC> alias;
        alias._data = rows.data();
        rlt::copy(renderer.device, device, rlt::transforms(device, renderer), alias);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const typename WORLD::Parameters instance_parameters = rlt::get(device, parameters, instance_i);
            const typename WORLD::State state = rlt::get(device, states, instance_i);
            float base[12];
            rlt::rl::environments::hyperdrone::rig::make_body_transform(device, state.orientation, state.position, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin, base);
            float coordinates_open[WORLD::DYNAMICS_ENV::ACTION_DIM], coordinates_close[WORLD::DYNAMICS_ENV::ACTION_DIM];
            rlt::rl::environments::hyperdrone::_drone_coordinates<DEVICE, WORLD_SPEC>(device, instance_parameters, (TI)0, state, coordinates_open, coordinates_close);
            rlt::rendering::rig::Buffer<DRONE_RIG_SPEC> buffer;
            float poses[DRONE_RIG_SPEC::MAX_PARTS][12];
            using rlt::rotor_scale;
            const float scale = rotor_scale(device, body.rig, instance_parameters.agents[0].dynamics.dynamics.rotor_positions, (TI)WORLD::DYNAMICS_ENV::ACTION_DIM);
            rlt::evaluate(device, body.rig, base, coordinates_close, buffer, &poses[0][0], scale);
            for(TI part = 0; part < body.rig.num_parts; part++){
                for(TI element = 0; element < 12; element++){
                    EXPECT_NEAR(rows[(instance_i * MAX_SLOTS + world.drone_slot_offset + part) * 12 + element], poses[part][element], 1e-4) << "instance " << instance_i << " part " << part << " element " << element;
                }
            }
        }
    }

    // each camera sees its own drone's overlay ids and no foreign ones
    {
        const uint32_t S = (uint32_t)shared.library.scenes.front().instances.size();
        std::vector<uint32_t> segmentation(INSTANCES * CAM_PIXELS);
        rlt::Tensor<rlt::tensor::Specification<uint32_t, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD_SPEC::CAM_HEIGHT, WORLD_SPEC::CAM_WIDTH>>> alias;
        alias._data = segmentation.data();
        auto& renderer = rlt::get_renderer(device, world);
        rlt::copy(renderer.device, device, rlt::output(device, renderer, rro::Segmentation<>{}), alias);
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
    rlt::copy(device, device_gpu, parameters, parameters_gpu);
    rlt::copy(device, device_gpu, states, states_gpu);
    rlt::set_all(device_gpu, reset_mask, false);
    rlt::request_render(device_gpu, world);
    rlt::render(device_gpu, world, parameters_gpu, states_gpu, reset_mask);
    rlt::observe(device_gpu, world, parameters_gpu, states_gpu, typename WORLD::Observation{}, observations_gpu, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, observations_gpu, observations);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        bool changed = false;
        for(TI dim = 0; dim < OBS_DIM; dim++){
            changed = changed || rlt::get(device, observations, instance_i, dim) != frame_phase_zero[instance_i * OBS_DIM + dim];
        }
        EXPECT_TRUE(changed) << "instance " << instance_i << ": prop articulation should be visible";
    }

    rlt::free(device, parameters);
    rlt::free(device, states);
    rlt::free(device, observations);
    rlt::free(device_gpu, parameters_gpu);
    rlt::free(device_gpu, states_gpu);
    rlt::free(device_gpu, reset_mask);
    rlt::free(device_gpu, observations_gpu);
    rlt::free(device_gpu, rng);
    rlt::free(device_gpu, world);
    rlt::free(device, shared.library);
}

__global__ void spin_kernel(long long cycles){
    const long long start = clock64();
    while(clock64() - start < cycles){
    }
}

namespace test_self_visible_cuda {
    struct FAN_OUT_WORLD_SPEC: WORLD_SPEC {
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr TI MOTION_BLUR_SAMPLES = 4;
    };
    using FAN_OUT_MEMBER = rlt::rl::environments::hyperdrone::World<FAN_OUT_WORLD_SPEC>;
    using FAN_OUT_MULTI = rlt::rl::environments::hyperdrone::MultiEnvironment<FAN_OUT_MEMBER, 3>;
    struct FanOutRollout {
        FAN_OUT_MULTI env;
        RNG_GPU rng;
        rlt::Tensor<rlt::tensor::Specification<typename FAN_OUT_MULTI::Parameters, TI, rlt::tensor::Shape<TI, FAN_OUT_MULTI::INSTANCES>>> parameters;
        rlt::Tensor<rlt::tensor::Specification<typename FAN_OUT_MULTI::State, TI, rlt::tensor::Shape<TI, FAN_OUT_MULTI::INSTANCES>>> states, next_states;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, FAN_OUT_MULTI::INSTANCES, FAN_OUT_MEMBER::OBSERVATION_DIM>>> observations;
    };
}

// a MultiEnvironment renders and observes its members concurrently on their renderer streams; the
// frames must equal rendering the same members one after another through the World verbs
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SELF_VISIBLE_CUDA, MULTI_ENVIRONMENT_FAN_OUT){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    if(!cuda_available()){
        GTEST_SKIP() << "no CUDA device available";
    }
    constexpr TI N = FAN_OUT_MULTI::NUMBER_OF_ENVIRONMENTS;
    constexpr TI M = FAN_OUT_MULTI::INSTANCES_PER_ENVIRONMENT;
    constexpr TI TOTAL = FAN_OUT_MULTI::INSTANCES;
    constexpr TI DIM = FAN_OUT_MEMBER::OBSERVATION_DIM;
    DEVICE_GPU device_gpu;
    auto& device = device_gpu.rendering;
    rlt::init(device);
    rlt::init(device_gpu);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH, SCENE_PATH, SCENE_PATH}};
    auto fanned_owner = std::make_unique<FanOutRollout>();
    auto memberwise_owner = std::make_unique<FanOutRollout>();
    FanOutRollout& fanned = *fanned_owner;
    FanOutRollout& memberwise = *memberwise_owner;
    for(FanOutRollout* rollout : {&fanned, &memberwise}){
        rlt::malloc(device_gpu, rollout->env);
        rollout->env.shared.drones.push_back({DRONE_PATH, {}});
        rlt::init(device_gpu, rollout->env, dataset);
        rlt::malloc(device_gpu, rollout->rng);
        rlt::init(device_gpu, rollout->rng, 7);
        rlt::malloc(device_gpu, rollout->parameters);
        rlt::malloc(device_gpu, rollout->states);
        rlt::malloc(device_gpu, rollout->next_states);
        rlt::malloc(device_gpu, rollout->observations);
    }
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, TOTAL>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL, FAN_OUT_MULTI::ACTION_DIM>>> actions, actions_host;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TOTAL, DIM>>> fanned_host, memberwise_host;
    rlt::malloc(device_gpu, reset_mask);
    rlt::malloc(device_gpu, actions);
    rlt::malloc(device, actions_host);
    rlt::malloc(device, fanned_host);
    rlt::malloc(device, memberwise_host);
    rlt::set_all(device_gpu, reset_mask, true);
    for(FanOutRollout* rollout : {&fanned, &memberwise}){
        rlt::sample_initial_parameters(device_gpu, rollout->env, rollout->parameters, reset_mask, rollout->rng);
        rlt::sample_initial_state(device_gpu, rollout->env, rollout->parameters, rollout->states, reset_mask, rollout->rng);
    }
    constexpr long long SPIN_CYCLES = 20000000;
    for(TI step_i = 0; step_i < 4; step_i++){
        // a member stream busy beyond the delayed step below: the observations are read on
        // device.stream, which must join it (each member in turn)
        spin_kernel<<<1, 1, 0, rlt::stream(device_gpu, rlt::get_renderer(device_gpu, fanned.env.environments[step_i % N]))>>>(3 * SPIN_CYCLES);
        rlt::render(device_gpu, fanned.env, fanned.parameters, fanned.states, reset_mask);
        rlt::observe(device_gpu, fanned.env, fanned.parameters, fanned.states, typename FAN_OUT_MULTI::Observation{}, fanned.observations, fanned.rng);
        for(TI environment_i = 0; environment_i < N; environment_i++){
            auto parameters_block = rlt::view_range(device_gpu, memberwise.parameters, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto states_block = rlt::view_range(device_gpu, memberwise.states, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto reset_block = rlt::view_range(device_gpu, reset_mask, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto observations_block = rlt::view_range(device_gpu, memberwise.observations, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            rlt::render(device_gpu, memberwise.env.environments[environment_i], parameters_block, states_block, reset_block);
            rlt::observe(device_gpu, memberwise.env.environments[environment_i], parameters_block, states_block, typename FAN_OUT_MEMBER::Observation{}, observations_block, memberwise.rng);
        }
        rlt::copy(device_gpu, device, fanned.observations, fanned_host);
        rlt::copy(device_gpu, device, memberwise.observations, memberwise_host);
        TI mismatches = 0;
        T total = 0;
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            for(TI dim = 0; dim < DIM; dim++){
                mismatches += rlt::get(device, fanned_host, instance_i, dim) != rlt::get(device, memberwise_host, instance_i, dim);
                total += rlt::get(device, fanned_host, instance_i, dim);
            }
        }
        EXPECT_EQ(mismatches, 0) << "step " << step_i;
        EXPECT_GT(total, 0) << "step " << step_i;
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            for(TI action_i = 0; action_i < FAN_OUT_MULTI::ACTION_DIM; action_i++){
                rlt::set(device, actions_host, (T)(((instance_i * 7 + action_i * 3 + step_i * 5) % 11) / 5.0 - 1), instance_i, action_i);
            }
        }
        rlt::copy(device, device_gpu, actions_host, actions);
        // a delayed step: the next render must fork it into the member streams
        spin_kernel<<<1, 1, 0, device_gpu.stream>>>(SPIN_CYCLES);
        for(FanOutRollout* rollout : {&fanned, &memberwise}){
            rlt::step(device_gpu, rollout->env, rollout->parameters, rollout->states, actions, rollout->next_states, rollout->rng);
            rlt::copy(device_gpu, device_gpu, rollout->next_states, rollout->states);
        }
        rlt::set_all(device_gpu, reset_mask, false);
    }
    for(FanOutRollout* rollout : {&fanned, &memberwise}){
        rlt::free(device_gpu, rollout->parameters);
        rlt::free(device_gpu, rollout->states);
        rlt::free(device_gpu, rollout->next_states);
        rlt::free(device_gpu, rollout->observations);
        rlt::free(device_gpu, rollout->rng);
        rlt::free(device_gpu, rollout->env);
    }
    rlt::free(device_gpu, reset_mask);
    rlt::free(device_gpu, actions);
    rlt::free(device, actions_host);
    rlt::free(device, fanned_host);
    rlt::free(device, memberwise_host);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
