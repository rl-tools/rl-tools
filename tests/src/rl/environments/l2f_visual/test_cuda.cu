#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cuda.h>

#include <gtest/gtest.h>

#include <cmath>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;
using DEVICE_TAG = rlt::devices::cuda::TAG<DEVICE_GPU, true>;
using T = float;
using TI = typename DEVICE::index_t;

namespace test_l2f_visual_cuda {
    namespace l2f = rlt::rl::environments::l2f;
    namespace obs = l2f::observation;

    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI SIMULATION_FREQUENCY = 100;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

    static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie;

    struct STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;

        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
        using OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI,
                obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
                obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
                obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI>>>>>>>>;
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
}

constexpr TI NUM_ENVS = 3;
constexpr TI CAM_WIDTH = 80;
constexpr TI CAM_HEIGHT = 50;
constexpr TI NUM_PROBES = 1;

using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, test_l2f_visual_cuda::STATIC_PARAMETERS, NUM_ENVS, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES>;
using ENV = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;
using CAMERA = rlt::rendering::raytracing::Camera<T>;

__global__ void build_l2f_visual_cameras(
    DEVICE_TAG device,
    typename ENV::Parameters* params,
    typename ENV::State* states,
    CAMERA* cameras,
    CAMERA* target_cameras
){
    TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
    if(env_i >= NUM_ENVS){
        return;
    }
    T scene_translation[3] = {0, 0, 0};
    T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
    cameras[env_i] = rlt::rl::environments::l2f_visual::cuda::make_camera_for_state<DEVICE_TAG, VISUAL_SPEC>(
        device, params[env_i], states[env_i], aspect, scene_translation, (T)1, (T)0
    );
    target_cameras[env_i] = rlt::rl::environments::l2f_visual::cuda::make_target_camera<DEVICE_TAG, VISUAL_SPEC>(
        device, params[env_i], aspect, scene_translation, (T)1, (T)0
    );
}

T norm3(const T v[3]){
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void fill_inputs(typename ENV::Parameters params[NUM_ENVS], typename ENV::State states[NUM_ENVS], T fov_small, T fov_large){
    for(TI env_i = 0; env_i < NUM_ENVS; env_i++){
        params[env_i].fov = env_i == 1 ? fov_large : fov_small;
        states[env_i].position[0] = (T)1;
        states[env_i].position[1] = (T)2;
        states[env_i].position[2] = (T)3;
        states[env_i].orientation[0] = (T)1;
        states[env_i].orientation[1] = (T)0;
        states[env_i].orientation[2] = (T)0;
        states[env_i].orientation[3] = (T)0;
    }
}

void expect_camera_equal(const CAMERA& a, const CAMERA& b){
    for(TI i = 0; i < 3; i++){
        EXPECT_NEAR(a.pos[i], b.pos[i], (T)1e-6);
        EXPECT_NEAR(a.dir_00[i], b.dir_00[i], (T)1e-6);
        EXPECT_NEAR(a.dir_du[i], b.dir_du[i], (T)1e-6);
        EXPECT_NEAR(a.dir_dv[i], b.dir_dv[i], (T)1e-6);
    }
}

void expect_fov_effect(const CAMERA cameras[NUM_ENVS], const CAMERA target_cameras[NUM_ENVS], T fov_small, T fov_large){
    const T expected_ratio = std::tan(fov_large / (T)2) / std::tan(fov_small / (T)2);
    EXPECT_NEAR(norm3(cameras[1].dir_du) / norm3(cameras[0].dir_du), expected_ratio, (T)1e-5);
    EXPECT_NEAR(norm3(cameras[1].dir_dv) / norm3(cameras[0].dir_dv), expected_ratio, (T)1e-5);
    EXPECT_NEAR(norm3(target_cameras[1].dir_du) / norm3(target_cameras[0].dir_du), expected_ratio, (T)1e-5);
    EXPECT_NEAR(norm3(target_cameras[1].dir_dv) / norm3(target_cameras[0].dir_dv), expected_ratio, (T)1e-5);

    for(TI i = 0; i < 3; i++){
        EXPECT_NEAR(cameras[0].pos[i], cameras[1].pos[i], (T)1e-6);
        EXPECT_NEAR(target_cameras[0].pos[i], target_cameras[1].pos[i], (T)1e-6);
    }
    expect_camera_equal(cameras[0], cameras[2]);
    expect_camera_equal(target_cameras[0], target_cameras[2]);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_CUDA, CAMERA_HELPERS_USE_PER_ENV_FOV_HOST) {
    typename ENV::Parameters params[NUM_ENVS]{};
    typename ENV::State states[NUM_ENVS]{};
    CAMERA cameras[NUM_ENVS];
    CAMERA target_cameras[NUM_ENVS];
    const T fov_small = (T)1.0;
    const T fov_large = (T)1.2;
    fill_inputs(params, states, fov_small, fov_large);

    DEVICE device;
    T scene_translation[3] = {0, 0, 0};
    T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
    for(TI env_i = 0; env_i < NUM_ENVS; env_i++){
        cameras[env_i] = rlt::rl::environments::l2f_visual::cuda::make_camera_for_state<DEVICE, VISUAL_SPEC>(
            device, params[env_i], states[env_i], aspect, scene_translation, (T)1, (T)0
        );
        target_cameras[env_i] = rlt::rl::environments::l2f_visual::cuda::make_target_camera<DEVICE, VISUAL_SPEC>(
            device, params[env_i], aspect, scene_translation, (T)1, (T)0
        );
    }

    expect_fov_effect(cameras, target_cameras, fov_small, fov_large);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL_CUDA, CAMERA_HELPERS_USE_PER_ENV_FOV_DEVICE) {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if(status != cudaSuccess || device_count == 0){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    status = cudaSetDevice(0);
    if(status != cudaSuccess){
        GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }

    typename ENV::Parameters params_host[NUM_ENVS]{};
    typename ENV::State states_host[NUM_ENVS]{};
    const T fov_small = (T)1.0;
    const T fov_large = (T)1.2;
    fill_inputs(params_host, states_host, fov_small, fov_large);

    typename ENV::Parameters* params_device = nullptr;
    typename ENV::State* states_device = nullptr;
    CAMERA* cameras_device = nullptr;
    CAMERA* target_cameras_device = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(&params_device, NUM_ENVS * sizeof(typename ENV::Parameters)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&states_device, NUM_ENVS * sizeof(typename ENV::State)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&cameras_device, NUM_ENVS * sizeof(CAMERA)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&target_cameras_device, NUM_ENVS * sizeof(CAMERA)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(params_device, params_host, NUM_ENVS * sizeof(typename ENV::Parameters), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(states_device, states_host, NUM_ENVS * sizeof(typename ENV::State), cudaMemcpyHostToDevice));

    DEVICE_TAG device{};
    build_l2f_visual_cameras<<<1, NUM_ENVS>>>(device, params_device, states_device, cameras_device, target_cameras_device);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());
    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    CAMERA cameras_host[NUM_ENVS];
    CAMERA target_cameras_host[NUM_ENVS];
    ASSERT_EQ(cudaSuccess, cudaMemcpy(cameras_host, cameras_device, NUM_ENVS * sizeof(CAMERA), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(target_cameras_host, target_cameras_device, NUM_ENVS * sizeof(CAMERA), cudaMemcpyDeviceToHost));

    expect_fov_effect(cameras_host, target_cameras_host, fov_small, fov_large);

    cudaFree(params_device);
    cudaFree(states_device);
    cudaFree(cameras_device);
    cudaFree(target_cameras_device);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
