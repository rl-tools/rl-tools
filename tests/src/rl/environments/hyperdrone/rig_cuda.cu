// host/device rig parity: the same StateRenderRotorPhase rollout rendered through the host
// set_transform_pair(rig) verbs and through the device slab producer must be pixel-identical
#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1

#include <rl_tools/operations/cuda.h>
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/hyperdrone/pose.h>
#include <rl_tools/rl/environments/hyperdrone/rig/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;

using T = float;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GPU = rlt::devices::DefaultCUDA;
using TI = typename DEVICE::index_t;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
static const std::string DRONE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/x500.glb";
#else
static const std::string SCENE_PATH = "";
static const std::string DRONE_PATH = "";
#endif

static constexpr TI CAM_WIDTH = 128;
static constexpr TI CAM_HEIGHT = 96;
static constexpr TI NUM_CAMERAS = 2;
static constexpr TI ONBOARD_CAMERA = 0;
static constexpr TI THIRD_PERSON_CAMERA = 1;

struct CONFIG: rlt::rendering::raytracing::config::Default<T, TI>{
    static constexpr TI CAM_WIDTH = ::CAM_WIDTH, CAM_HEIGHT = ::CAM_HEIGHT, NUM_CAMERAS = ::NUM_CAMERAS, NUM_PROBES = 1;
    using SHADING = rlt::rendering::raytracing::High;
    static constexpr bool OUTPUT_RGB = true;
    static constexpr TI NUM_OVERLAYS = 1;
    static constexpr TI MAX_OVERLAY_INSTANCES = 8;
    static constexpr TI MAX_OVERLAYS_PER_CAMERA = 1;
    static constexpr bool ENABLE_MOTION_BLUR = true;
    static constexpr TI MOTION_BLUR_SAMPLES = 4;
    static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = true;
};
using SPEC = rlt::rendering::raytracing::Specification<CONFIG>;
using Renderer = rlt::rendering::raytracing::Renderer<SPEC>;

namespace test_hyperdrone_rig_cuda {
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI SIMULATION_FREQUENCY = 100;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

    static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::x500;

    struct STATIC_PARAMETERS {
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
        static constexpr auto dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)1 / (T)SIMULATION_FREQUENCY};
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
    using ENV = rlt::rl::environments::Multirotor<l2f::Specification<T, TI, STATIC_PARAMETERS>>;
}

using RIG = rlt::rl::environments::hyperdrone::rig::Rotorcraft<T, TI, 4>;
static constexpr TI TOTAL_SLOTS = SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;

struct KernelParams {
    RIG rig;
    TI first_slot;
    float body_open[12];
    float body_close[12];
    T phase_open[4];
    T phase_close[4];
};

template <typename T_DEVICE>
__global__ void produce_transforms_kernel(T_DEVICE device, KernelParams params, float* transforms_pair){
    if(blockIdx.x != 0 || threadIdx.x != 0){
        return;
    }
    rlt::set_transform_pair(device, transforms_pair, TOTAL_SLOTS, params.first_slot, params.rig, params.body_open, params.body_close, params.phase_open, params.phase_close);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RIG_CUDA, HOST_DEVICE_RENDER_PARITY){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    int device_count = 0;
    if(cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0){
        GTEST_SKIP() << "CUDA device unavailable";
    }

    DEVICE device;
    rlt::init(device);
    DEVICE_GPU device_gpu;

    rlt::rendering::Bundle<T> bundle;
    ASSERT_TRUE((rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, bundle, SCENE_PATH)));
    rlt::rendering::raytracing::ObjectAssembly drone_assembly;
    ASSERT_TRUE((rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, drone_assembly, DRONE_PATH)));

    RIG rig;
    ASSERT_TRUE(rlt::init(device, rig, drone_assembly));
    ASSERT_EQ(rig.num_props, 4);

    rlt::rendering::raytracing::AssetPool pool;
    const auto drone_asset = rlt::add(device, pool, drone_assembly);

    Renderer renderer;
    rlt::malloc(device, renderer);
    rlt::init(device, renderer, bundle, pool);
    rlt::attach(device, renderer, ONBOARD_CAMERA, rlt::rendering::raytracing::OverlayIndex{0});
    rlt::attach(device, renderer, THIRD_PERSON_CAMERA, rlt::rendering::raytracing::OverlayIndex{0});
    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const auto placement = rlt::spawn(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, drone_asset, identity);

    // shutter pair from an actual StateRenderRotorPhase rollout
    using ENV = test_hyperdrone_rig_cuda::ENV;
    ENV env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    typename ENV::Parameters parameters;
    rlt::initial_parameters(device, env, parameters);
    RNG rng;
    rlt::init(device, rng, 0);
    typename ENV::State state, prev_state;
    rlt::sample_initial_state(device, env, parameters, state, rng);
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM, false>> action;
    for(TI i = 0; i < ENV::ACTION_DIM; i++){
        rlt::set(action, 0, i, (T)0.0);
    }
    for(TI step_i = 0; step_i < 50; step_i++){
        prev_state = state;
        typename ENV::State next_state;
        rlt::step(device, env, parameters, state, action, next_state, rng);
        state = next_state;
    }

    const T scene_translation[3] = {-5.0, -5.2, 1.55};
    KernelParams params{};
    params.rig = rig;
    params.first_slot = (TI)placement.first_slot;
    rlt::rl::environments::hyperdrone::rig::make_body_transform(device, prev_state.orientation, prev_state.position, scene_translation, (T)1, (T)0, params.body_open);
    rlt::rl::environments::hyperdrone::rig::make_body_transform(device, state.orientation, state.position, scene_translation, (T)1, (T)0, params.body_close);
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        params.phase_open[rotor_i] = prev_state.rotor_phase[rotor_i];
        params.phase_close[rotor_i] = state.rotor_phase[rotor_i];
    }

    // cameras (host-produced in both paths; the rig producer is the parity target)
    const T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
    const T body_position[3] = {
        state.position[0] + scene_translation[0],
        state.position[1] + scene_translation[1],
        state.position[2] + scene_translation[2]
    };
    rlt::rendering::raytracing::Camera<T> cameras[NUM_CAMERAS];
    {
        rlt::rl::environments::hyperdrone::CameraMount<T> mount;
        mount.offset_body[0] = 0.10;
        mount.offset_body[2] = 0.05;
        cameras[ONBOARD_CAMERA] = rlt::rl::environments::hyperdrone::make_camera<DEVICE, T>(device, mount, (T)1.1, state.orientation, state.position, aspect, scene_translation, (T)1, (T)0);
        const T tripod[3] = {body_position[0] - 1.5f, body_position[1] - 1.2f, body_position[2] + 0.6f};
        const T up[3] = {0, 0, 1};
        cameras[THIRD_PERSON_CAMERA] = rlt::make_camera_data(tripod, body_position, up, (T)0.9, aspect);
    }
    rlt::Tensor<typename decltype(renderer.cameras)::SPEC> camera_alias;
    camera_alias._data = cameras;
    rlt::copy(device, renderer.device, camera_alias, rlt::cameras_open(device, renderer));
    rlt::copy(device, renderer.device, camera_alias, rlt::cameras_close(device, renderer));

    constexpr size_t CAM_PIXELS = static_cast<size_t>(CAM_WIDTH) * static_cast<size_t>(CAM_HEIGHT);
    std::vector<uint32_t> frame_host_path(NUM_CAMERAS * CAM_PIXELS);
    std::vector<uint32_t> frame_device_path(NUM_CAMERAS * CAM_PIXELS);

    // host path: verbs
    rlt::set_transform_pair(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, placement, rig, params.body_open, params.body_close, params.phase_open, params.phase_close);
    rlt::update(device, renderer);
    rlt::render(device, renderer);
    rlt::synchronize(device, renderer);
    {
        rlt::Tensor<typename decltype(renderer.frame_buffer)::SPEC> frame_alias;
        frame_alias._data = frame_host_path.data();
        rlt::copy(renderer.device, device, rlt::frame_buffer(device, renderer), frame_alias);
    }

    // device path: kernel producer into the transforms_pair slab
    cudaStream_t render_stream = rlt::stream(device, renderer);
    float* transforms_pair = rlt::data(rlt::transforms_pair(device, renderer));
    produce_transforms_kernel<<<1, 1, 0, render_stream>>>(device_gpu, params, transforms_pair);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    rlt::expand_motion_transforms_launch(device, renderer);
    rlt::update_launch(device, renderer);
    rlt::render_launch(device, renderer);
    rlt::render_sync(device, renderer);
    {
        rlt::Tensor<typename decltype(renderer.frame_buffer)::SPEC> frame_alias;
        frame_alias._data = frame_device_path.data();
        rlt::copy(renderer.device, device, rlt::frame_buffer(device, renderer), frame_alias);
    }

    // the host producers compile under gcc, the device producers under nvcc: trig/contraction
    // ULP differences in the spin/slerp math can move a prop-blur edge by one 8-bit quantization
    // step, so the moving case is gated at quantization-level agreement; the static case below
    // pins the plumbing bit-exactly
    size_t mismatches = 0;
    int max_channel_delta = 0;
    for(size_t pixel_i = 0; pixel_i < frame_host_path.size(); pixel_i++){
        if(frame_host_path[pixel_i] != frame_device_path[pixel_i]){
            mismatches++;
            for(int channel_i = 0; channel_i < 4; channel_i++){
                int a = (frame_host_path[pixel_i] >> (channel_i * 8)) & 0xff;
                int b = (frame_device_path[pixel_i] >> (channel_i * 8)) & 0xff;
                int delta = a > b ? a - b : b - a;
                if(delta > max_channel_delta) max_channel_delta = delta;
            }
        }
    }
    EXPECT_LE(mismatches, frame_host_path.size() / 1000) << "moving-case mismatches beyond isolated silhouette flips";
    EXPECT_LE(max_channel_delta, (255 + SPEC::MOTION_BLUR_SAMPLES - 1) / SPEC::MOTION_BLUR_SAMPLES) << "moving-case delta beyond a single blur-sample flip";

    // static case: identity body rotation, zero phases, identical open/close pair -> every
    // producer and slerp intermediate is exact on both sides, so the full plumbing (slab
    // layout, expand, update_launch) must be pixel-identical
    {
        KernelParams static_params = params;
        const T identity_orientation[4] = {1, 0, 0, 0};
        rlt::rl::environments::hyperdrone::rig::make_body_transform(device, identity_orientation, state.position, scene_translation, (T)1, (T)0, static_params.body_close);
        for(TI element_i = 0; element_i < 12; element_i++){
            static_params.body_open[element_i] = static_params.body_close[element_i];
        }
        for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
            static_params.phase_open[rotor_i] = 0;
            static_params.phase_close[rotor_i] = 0;
        }
        rlt::set_transform_pair(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, placement, rig, static_params.body_open, static_params.body_close, static_params.phase_open, static_params.phase_close);
        rlt::update(device, renderer);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        std::vector<uint32_t> static_host(NUM_CAMERAS * CAM_PIXELS);
        {
            rlt::Tensor<typename decltype(renderer.frame_buffer)::SPEC> frame_alias;
            frame_alias._data = static_host.data();
            rlt::copy(renderer.device, device, rlt::frame_buffer(device, renderer), frame_alias);
        }
        produce_transforms_kernel<<<1, 1, 0, render_stream>>>(device_gpu, static_params, transforms_pair);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        rlt::expand_motion_transforms_launch(device, renderer);
        rlt::update_launch(device, renderer);
        rlt::render_launch(device, renderer);
        rlt::render_sync(device, renderer);
        std::vector<uint32_t> static_device(NUM_CAMERAS * CAM_PIXELS);
        {
            rlt::Tensor<typename decltype(renderer.frame_buffer)::SPEC> frame_alias;
            frame_alias._data = static_device.data();
            rlt::copy(renderer.device, device, rlt::frame_buffer(device, renderer), frame_alias);
        }
        size_t static_mismatches = 0;
        for(size_t pixel_i = 0; pixel_i < static_host.size(); pixel_i++){
            if(static_host[pixel_i] != static_device[pixel_i]){
                static_mismatches++;
            }
        }
        EXPECT_EQ(static_mismatches, 0) << "of " << static_host.size() << " pixels";
    }

    // the drone must actually be in frame: the FPV camera sees its own props
    size_t host_vs_static_scene = 0;
    {
        // re-render with the drone parked far below the scene: any pixel that changes belonged
        // to the drone
        float far_away[12] = {1,0,0,0, 0,1,0,-1000, 0,0,1,-1000};
        rlt::set_transform_pair(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, placement, far_away, far_away);
        rlt::update(device, renderer);
        rlt::render(device, renderer);
        rlt::synchronize(device, renderer);
        std::vector<uint32_t> frame_no_drone(NUM_CAMERAS * CAM_PIXELS);
        rlt::Tensor<typename decltype(renderer.frame_buffer)::SPEC> frame_alias;
        frame_alias._data = frame_no_drone.data();
        rlt::copy(renderer.device, device, rlt::frame_buffer(device, renderer), frame_alias);
        for(size_t pixel_i = 0; pixel_i < frame_host_path.size(); pixel_i++){
            if(frame_host_path[pixel_i] != frame_no_drone[pixel_i]){
                host_vs_static_scene++;
            }
        }
    }
    EXPECT_GT(host_vs_static_scene, 100) << "drone not visible in the rendered frames";

    rlt::free(device, env);
    rlt::free(device, renderer);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
