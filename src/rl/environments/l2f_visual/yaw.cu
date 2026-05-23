#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_cuda.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_cuda.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cuda.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/unflatten/operations_generic.h>
#include <rl_tools/nn/layers/unflatten/operations_cuda.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/gru/helper_operations_cuda.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_cuda.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cuda.h>

#include <rl_tools/nn/loss_functions/mse/operations_generic.h>
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>

#include "../../../../src/nn_models/port_checkpoint/raptor/policy.h"

#include <rl_tools/utils/extrack/operations_cpu.h>
#include <rl_tools/utils/zlib/operations_cpu.h>

#include <rl_tools/persist/backends/tar/operations_cpu.h>
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
#include <rl_tools/persist/backends/hdf5/hdf5.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#endif
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/flatten/persist.h>
#include <rl_tools/nn/layers/unflatten/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>
#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/matrix/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/conv2d/persist_code.h>
#include <rl_tools/nn/layers/gru/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/flatten/persist_code.h>
#include <rl_tools/nn/layers/unflatten/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/parallel/persist_code.h>

#include <cuda_bf16.h>

#include <array>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <vector>
#include <numeric>
#include <cstring>
#include <mutex>
#include <string>
#include <filesystem>
#include <fstream>

namespace rlt = rl_tools;

// Uncomment to feed the student CNN all-black visual observations (stacked frames + target).
// Rendering still runs; only the tensor handed to the policy is zeroed. Use to verify the
// training pipeline and that the policy can learn yaw from the non-visual state branch alone.
// #define RL_TOOLS_L2F_VISUAL_IMITATION_BLIND_TRAINING

// =========================================================================
// Device types
// =========================================================================
#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
using LOGGER = rlt::devices::logging::CPU_TENSORBOARD<>;
#else
using LOGGER = rlt::devices::logging::CPU;
#endif
using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<float,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Parameter, __nv_bfloat16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Activation, __nv_bfloat16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Gradient, __nv_bfloat16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, float>>;
using T_ACTIVATION = TYPE_POLICY::GET<rlt::numeric_types::categories::Activation>;
using T_GRADIENT = TYPE_POLICY::GET<rlt::numeric_types::categories::Gradient>;
using TEACHER_TYPE_POLICY = rlt::numeric_types::Policy<float>;
using TI = typename DEVICE::index_t;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;

// =========================================================================
// L2F dynamics configuration
// =========================================================================
namespace l2f = rlt::rl::environments::l2f;
namespace obs = l2f::observation;

using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
static constexpr TI SIMULATION_FREQUENCY = 100;
static constexpr TI EPISODE_STEP_LIMIT = 50;
static constexpr T INIT_ORIENTATION_CURRICULUM_START_DEG = static_cast<T>(0);
static constexpr T INIT_ORIENTATION_CURRICULUM_FULL_DEG = static_cast<T>(90);
static constexpr T INIT_ORIENTATION_CURRICULUM_STEP_DEG = static_cast<T>(1);
static constexpr T INIT_ORIENTATION_CURRICULUM_FRONTIER_RATIO = static_cast<T>(0.25);
static constexpr T INIT_ORIENTATION_CURRICULUM_FRONTIER_MIN_DEG = static_cast<T>(2.5);
static constexpr T INIT_ORIENTATION_CURRICULUM_FRONTIER_MAX_DEG = static_cast<T>(5);
using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
struct DOMAIN_RANDOMIZATION_OPTIONS {
    static constexpr bool THRUST_TO_WEIGHT = true;
    static constexpr bool MASS = false;
    static constexpr bool TORQUE_TO_INERTIA = false;
    static constexpr bool MASS_SIZE_DEVIATION = false;
    static constexpr bool ROTOR_TORQUE_CONSTANT = false;
    static constexpr bool DISTURBANCE_FORCE = false;
    static constexpr bool ROTOR_TIME_CONSTANT = false;
};
using PARAMETERS_BASE = l2f::ParametersBase<PARAMETERS_SPEC>;
using PARAMETERS_IMU = l2f::ParametersIMU<l2f::ParametersSpecification<T, TI, PARAMETERS_BASE>>;
using PARAMETERS_DISTURBANCES = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, PARAMETERS_IMU>>;
using PARAMETERS_TYPE = l2f::ParametersDomainRandomization<l2f::ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, PARAMETERS_DISTURBANCES>>;

static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie_openmv;

static constexpr REWARD_FUNCTION reward_function = {
    false, 0.10, 1.00, -100.00, 10.00, 0.00, 1.00, 0.05, 1.50, 0.00, 0.00,
    {0.00, 0.00, 0.00, 0.00}, {1.50, 1.50, 1.50, 1.50}, 0.00
};
static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
    0.0, 0.0, INIT_ORIENTATION_CURRICULUM_FULL_DEG/static_cast<T>(180)*rlt::math::PI<T>, 0.0, 1.0, true, -1, +1,
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    true, 1.0, 0, 10, 35, 10000, 50000,
};
static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::Integration integration = {
    static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)
};
static constexpr typename PARAMETERS_TYPE::MDP mdp = { init, reward_function, {}, {}, termination };
static constexpr T MAX_THRUST_PER_ROTOR = dynamics.rotor_thrust_coefficients[0][0]
                                        + dynamics.rotor_thrust_coefficients[0][1]
                                        + dynamics.rotor_thrust_coefficients[0][2];
static constexpr T MAX_THRUST = static_cast<T>(PARAMETERS_SPEC::N) * MAX_THRUST_PER_ROTOR;
static constexpr T ROTOR_ARM = dynamics.rotor_positions[0][0] < 0 ? -dynamics.rotor_positions[0][0] : dynamics.rotor_positions[0][0];
static constexpr T DISTURBANCE_FRACTION = static_cast<T>(0.0);
static constexpr T DISTURBANCE_FORCE_STD = DISTURBANCE_FRACTION * MAX_THRUST;
static constexpr T DISTURBANCE_TORQUE_STD = DISTURBANCE_FRACTION * MAX_THRUST * ROTOR_ARM;
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
    {0, DISTURBANCE_FORCE_STD},
    {0, DISTURBANCE_TORQUE_STD}
};
static constexpr typename PARAMETERS_TYPE::IMU imu = {
    {static_cast<T>(0.02), static_cast<T>(0), static_cast<T>(0)}
};
static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
    1.7, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static constexpr PARAMETERS_TYPE nominal_parameters = { {{{dynamics, integration, mdp}, imu}, disturbances}, domain_randomization };


// =========================================================================
// Environment static parameters
// =========================================================================
static constexpr TI ACTION_HISTORY_LENGTH = 8;

struct STATIC_PARAMETERS {
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI CLOSED_FORM = false;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_BASE_LA = l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE>>;
    using STATE_BASE_LAH = l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, STATE_BASE_LA>>;
    using STATE_BASE_GB = l2f::StateGyroBias<l2f::StateGyroBiasSpecification<T, TI, STATE_BASE_LAH>>;
    using STATE_BASE_MAHONY = l2f::StateMahony<l2f::StateMahonySpecification<T, TI, STATE_BASE_GB>>;
    using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE_MAHONY>>>>>>;
    using OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI,
            obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
            obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
            obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI,
            obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
    using OBSERVATION_TYPE_PRIVILEGED = obs::Position<obs::PositionSpecificationPrivileged<T, TI,
            obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecificationPrivileged<T, TI,
            obs::LinearVelocity<obs::LinearVelocitySpecificationPrivileged<T, TI,
            obs::AngularVelocity<obs::AngularVelocitySpecificationPrivileged<T, TI,
            obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
    using PARAMETERS = PARAMETERS_TYPE;
    static constexpr auto PARAMETER_VALUES = nominal_parameters;
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

// =========================================================================
// Visual environment specification
// =========================================================================
static constexpr TI N_TRAIN_SCENES = 25;
static constexpr TI N_VALIDATION_SCENES = 5;
static constexpr TI N_TOTAL_SCENES = N_TRAIN_SCENES + N_VALIDATION_SCENES;
static constexpr TI N_ACTIVE_SCENES = 2;
static constexpr TI N_ENVIRONMENTS_PER_SCENE = 64;
static constexpr TI N_ENVIRONMENTS = N_ACTIVE_SCENES * N_ENVIRONMENTS_PER_SCENE;
static_assert(N_ACTIVE_SCENES <= N_TRAIN_SCENES, "Active training scenes must fit within the training split");
static constexpr TI CAM_WIDTH = 80;
static constexpr TI CAM_HEIGHT = 50;
static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
static constexpr TI NUM_PROBES = 64;
static constexpr T CAMERA_FOV = static_cast<T>(79.6) / static_cast<T>(180) * rlt::math::PI<T>;
static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = static_cast<T>(5.0) / static_cast<T>(180) * rlt::math::PI<T>;
static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X = static_cast<T>(0.01);
static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y = static_cast<T>(0.01);
static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z = static_cast<T>(0.01);
static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X = static_cast<T>(5.0) / static_cast<T>(180) * rlt::math::PI<T>;
static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y = static_cast<T>(5.0) / static_cast<T>(180) * rlt::math::PI<T>;
static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z = static_cast<T>(5.0) / static_cast<T>(180) * rlt::math::PI<T>;
static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = static_cast<T>(10.0) / static_cast<T>(180) * rlt::math::PI<T>;
static_assert(TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE >= static_cast<T>(0), "Invalid l2f_visual yaw target frame roll/pitch randomization range");

constexpr bool HIGH_FIDELITY_SHADING = true;
static constexpr bool RENDER_ENABLE_MOTION_BLUR = true;
static constexpr TI RENDER_MOTION_BLUR_SAMPLES = 2;
static constexpr bool RENDER_ENABLE_ANTI_ALIASING = true;
static constexpr TI RENDER_ANTI_ALIASING_GRID_SIZE = 2;
static constexpr T RENDER_SHUTTER_FRACTION_MIN = static_cast<T>(0.25);
static constexpr T RENDER_SHUTTER_FRACTION_MAX = static_cast<T>(1);
static_assert(RENDER_SHUTTER_FRACTION_MIN >= static_cast<T>(0) && RENDER_SHUTTER_FRACTION_MIN <= RENDER_SHUTTER_FRACTION_MAX && RENDER_SHUTTER_FRACTION_MAX <= static_cast<T>(1), "Invalid l2f_visual yaw shutter fraction range");
using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, N_ENVIRONMENTS_PER_SCENE, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES, HIGH_FIDELITY_SHADING, RENDER_ENABLE_MOTION_BLUR, RENDER_MOTION_BLUR_SAMPLES, RENDER_ENABLE_ANTI_ALIASING, RENDER_ANTI_ALIASING_GRID_SIZE>;
using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;
using CAMERA_DATA = rlt::rendering::raytracing::CameraData<T>;
static constexpr bool RENDER_MOTION_BLUR_ACTIVE = ENVIRONMENT::SPEC::RENDERER_SPEC::ENABLE_MOTION_BLUR;
static constexpr bool RENDER_ANTI_ALIASING_ACTIVE = ENVIRONMENT::SPEC::RENDERER_SPEC::ENABLE_ANTI_ALIASING;

// =========================================================================
// RAPTOR teacher (CPU only)
// =========================================================================
static constexpr TI RAPTOR_HIDDEN_DIM = 16;
using RAPTOR_OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI,
        obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
        obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
        obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI,
        obs::ActionHistory<obs::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
static constexpr TI RAPTOR_OBS_DIM = RAPTOR_OBSERVATION_TYPE::DIM;

using RAPTOR_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<TEACHER_TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
using RAPTOR_DENSE1 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE1_CONFIG>;
using RAPTOR_GRU_CONFIG = rlt::nn::layers::gru::Configuration<TEACHER_TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM>;
using RAPTOR_GRU = rlt::nn::layers::gru::BindConfiguration<RAPTOR_GRU_CONFIG>;
using RAPTOR_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<TEACHER_TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
using RAPTOR_DENSE2 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE2_CONFIG>;

using RAPTOR_MODULE = rlt::nn_models::sequential::Module<RAPTOR_DENSE1, RAPTOR_GRU, RAPTOR_DENSE2>;
using RAPTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, RAPTOR_OBS_DIM>;
using RAPTOR_CAPABILITY = rlt::nn::capability::Forward<true, false>;
using RAPTOR_MODEL = rlt::nn_models::sequential::Build<RAPTOR_CAPABILITY, RAPTOR_MODULE, RAPTOR_INPUT_SHAPE>;

// =========================================================================
// Student CNN (GPU)
// =========================================================================
static constexpr TI ACTOR_HIDDEN_DIM = 64;
static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
static constexpr TI TARGET_DIM = 2;
static constexpr TI YAW_METRIC_ABS_ERROR_RAD = 0;
static constexpr TI YAW_METRIC_MSE_RAD = 1;
static constexpr TI YAW_METRIC_OUTPUT_NORM = 2;
static constexpr TI YAW_METRIC_NULL_MSE_LOSS = 3;
static constexpr TI YAW_METRIC_NULL_MSE_RAD = 4;
static constexpr TI YAW_NUM_METRICS = 5;
static constexpr T YAW_R2_EPS = static_cast<T>(1e-8);
static constexpr TI INDOOR_POSITION_DIM = 3;
static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
static constexpr TI BATCH_SIZE = 512;
static constexpr TI STEPS_PER_ENV = 50;
static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * N_ENVIRONMENTS;
static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
static constexpr TI NUM_EPOCHS = 1000000;
static constexpr TI TEACHER_FORCING_EPOCHS = 0;
static constexpr T TEACHER_FORCING_FRACTION = 1.0;
static constexpr T EFFECTIVE_TEACHER_FORCING_FRACTION = TEACHER_FORCING_FRACTION;
static constexpr TI N_TRAIN_PASSES = 4;
static constexpr TI VIDEO_CADENCE = 10;
static constexpr TI CHECKPOINT_CADENCE = 1000;
static constexpr TI VALIDATION_CADENCE = 10;
static constexpr bool EXPORT_CHECKPOINT_TAR = false;
static constexpr bool EXPORT_CHECKPOINT_CODE = false;
static constexpr TI N_EXAMPLES = 512;
static constexpr TI REDUCED_BATCH_SIZE = 2;
static_assert(REDUCED_BATCH_SIZE <= N_EXAMPLES);
static constexpr T OBSERVATION_NOISE_STD = 0.00;
static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.5;
static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0.25;
static constexpr TI ENV_GRID_SIDE = 8; // sqrt(N_ENVIRONMENTS_PER_SCENE)
static constexpr TI SCENE_GRID_COLS = N_ACTIVE_SCENES;
static constexpr TI SCENE_GRID_ROWS = (N_ACTIVE_SCENES + SCENE_GRID_COLS - 1) / SCENE_GRID_COLS;
static_assert(ENV_GRID_SIDE * ENV_GRID_SIDE == N_ENVIRONMENTS_PER_SCENE, "N_ENVIRONMENTS_PER_SCENE must be a perfect square for per-scene video mosaic");

static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");

// =========================================================================
// Frame stacking configuration
// =========================================================================
static constexpr TI FRAME_STACK_N = 1;
static constexpr TI FRAME_STACK_STRIDE = 10;
static constexpr TI FRAME_STACK_HISTORY_LENGTH = FRAME_STACK_STRIDE * (FRAME_STACK_N - 1) + 1;
static constexpr TI STACKED_IMG_C = ENVIRONMENT::Observation::CHANNELS * FRAME_STACK_N;
static constexpr TI STACKED_OBS_DIM = ENVIRONMENT::Observation::HEIGHT * ENVIRONMENT::Observation::WIDTH * STACKED_IMG_C;
static constexpr TI COMBINED_IMG_C_LOGICAL = STACKED_IMG_C + ENVIRONMENT::Observation::CHANNELS;
// Pad up to next multiple of 8 so cuDNN uses the tensor-core fast path for Conv1 without inserting an NHWC layout-padding reformat kernel.
static constexpr TI COMBINED_IMG_C = (COMBINED_IMG_C_LOGICAL + 7) & ~((TI)7);
static constexpr TI COMBINED_OBS_DIM = ENVIRONMENT::Observation::HEIGHT * ENVIRONMENT::Observation::WIDTH * COMBINED_IMG_C;
static constexpr TI VALIDATION_BATCH_SIZE = N_ENVIRONMENTS_PER_SCENE;
static constexpr TI N_VALIDATION_YAW_BINS = 11;

// =========================================================================
// Trajectory recording
// =========================================================================
static constexpr TI TRAJECTORY_SAVE_INTERVAL = 10; // save every N epochs
static constexpr TI TRAJECTORY_NUM_ENVS = 10;
static constexpr TI TRAJECTORY_MAX_EPISODES = 10;

struct TrajectoryStep {
    typename ENVIRONMENT::State state;
    T actions[ENVIRONMENT::ACTION_DIM];
    T reward;
    bool terminated;
};
struct CompletedEpisode {
    typename ENVIRONMENT::Parameters parameters;
    std::vector<TrajectoryStep> steps;
};
struct EpisodeRecorder {
    std::vector<TrajectoryStep> current_episode;
    typename ENVIRONMENT::Parameters parameters_snapshot;
    bool episode_started = false;
};

std::string trajectory_episodes_to_json(DEVICE& device, ENVIRONMENT& env, const std::vector<CompletedEpisode>& episodes, T dt){
    if(episodes.empty()) return "[]";
    std::string json = "[";
    for(TI ep_i = 0; ep_i < episodes.size(); ep_i++){
        auto& episode = episodes[ep_i];
        auto& parameters = episode.parameters;
        json += "{\"parameters\": " + rlt::json(device, env, parameters) + ",\n";
        json += "\"trajectory\": [";
        for(TI step_i = 0; step_i < episode.steps.size(); step_i++){
            auto& s = episode.steps[step_i];
            json += "{\"state\":" + rlt::json(device, env, parameters, s.state) + ",";
            json += "\"action\":[";
            for(TI a = 0; a < ENVIRONMENT::ACTION_DIM; a++){
                json += std::to_string(s.actions[a]);
                if(a < ENVIRONMENT::ACTION_DIM - 1) json += ",";
            }
            json += "],";
            json += "\"dt\":" + std::to_string(dt) + ",";
            json += "\"reward\":" + std::to_string(s.reward) + ",";
            json += "\"terminated\":" + (s.terminated ? std::string("true") : std::string("false"));
            json += "}";
            if(step_i < episode.steps.size() - 1) json += ",";
        }
        json += "]}";
        if(ep_i < episodes.size() - 1) json += ",";
    }
    json += "]";
    return json;
}

// =========================================================================
// GPU collection kernels
// =========================================================================
namespace imitation_kernels{

    __device__ CAMERA_DATA interpolate_camera(const CAMERA_DATA& from, const CAMERA_DATA& to, T alpha){
        CAMERA_DATA out;
        for(TI i = 0; i < 3; i++){
            out.pos[i] = from.pos[i] + (to.pos[i] - from.pos[i]) * alpha;
            out.dir_00[i] = from.dir_00[i] + (to.dir_00[i] - from.dir_00[i]) * alpha;
            out.dir_du[i] = from.dir_du[i] + (to.dir_du[i] - from.dir_du[i]) * alpha;
            out.dir_dv[i] = from.dir_dv[i] + (to.dir_dv[i] - from.dir_dv[i]) * alpha;
        }
        return out;
    }

    __device__ void yaw_target_from_state(const typename ENVIRONMENT::State& state, T& yaw_cos, T& yaw_sin){
        const T qw = state.orientation[0];
        const T qx = state.orientation[1];
        const T qy = state.orientation[2];
        const T qz = state.orientation[3];
        yaw_cos = static_cast<T>(1) - static_cast<T>(2) * (qy * qy + qz * qz);
        yaw_sin = static_cast<T>(2) * (qx * qy + qw * qz);
        T norm = sqrtf(yaw_cos * yaw_cos + yaw_sin * yaw_sin);
        if(norm > static_cast<T>(1e-6)){
            yaw_cos /= norm;
            yaw_sin /= norm;
        } else {
            yaw_cos = static_cast<T>(1);
            yaw_sin = static_cast<T>(0);
        }
    }

    __device__ void write_yaw_target(const typename ENVIRONMENT::State& state, T_ACTIVATION* target_ptr){
        T yaw_cos;
        T yaw_sin;
        yaw_target_from_state(state, yaw_cos, yaw_sin);
        target_ptr[0] = static_cast<T_ACTIVATION>(yaw_cos);
        target_ptr[1] = static_cast<T_ACTIVATION>(yaw_sin);
    }

    template<bool ENABLE_MOTION_BLUR, typename RENDERER>
    void set_active_scene_camera_open_buffer(void*& camera_open_buffer, RENDERER* renderer){
        if constexpr(ENABLE_MOTION_BLUR){
            camera_open_buffer = (void*)owlBufferGetPointer((OWLBuffer)renderer->backend.owl_cameras_open_buffer, 0);
        }
    }

    template<typename DEVICE, typename RNG>
    __global__
    void prologue_kernel(
        DEVICE device,
        ENVIRONMENT* envs, typename ENVIRONMENT::Parameters* env_params, typename ENVIRONMENT::State* states,
        bool* terminated_flags, TI* episode_step_arr, bool* teacher_forcing_arr,
        T* episode_return_arr, bool* needs_reset_flags,
        T* shutter_fraction_arr,
        T* episode_lengths_log, T* episode_tf_log, T* episode_terminated_log,
        T teacher_forcing_fraction, bool full_teacher_forcing,
        T* teacher_obs_ptr,
        T* raptor_gru_state_ptr, T* raptor_gru_initial_hidden_ptr, TI* raptor_gru_step_ptr,
        TI* episode_start_step,
        T* brightness_scale_arr,
        T* target_brightness_scale_arr,
        T* target_frame_roll_arr,
        T* target_frame_pitch_arr,
        T* scene_translation_arr,
        T* scene_yaw_arr,
        T* scene_yaw_cos_arr,
        T* scene_yaw_sin_arr,
        T* indoor_positions_ptr, TI* num_indoor_positions_ptr, TI* env_scene_ptr, TI max_indoor_pos,
        RNG rng, TI step_i, TI episode_step_limit, T init_orientation_max_rad
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& rng_state = rl_tools::get(rng.states, 0, env_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        bool need_reset = terminated_flags[env_i] || episode_step_arr[env_i] >= episode_step_limit;
        needs_reset_flags[env_i] = need_reset;
        if(need_reset){
            if(episode_step_arr[env_i] > 0){
                episode_lengths_log[env_i] = (T)episode_step_arr[env_i];
                episode_tf_log[env_i] = teacher_forcing_arr[env_i] ? (T)1 : (T)0;
                episode_terminated_log[env_i] = terminated_flags[env_i] ? (T)1 : (T)0;
            } else {
                episode_lengths_log[env_i] = (T)-1;
                episode_tf_log[env_i] = (T)0;
                episode_terminated_log[env_i] = (T)-1;
            }
            rl_tools::sample_initial_parameters(device, env, params, rng_state);
            params.dynamics.mdp.init.max_angle = init_orientation_max_rad;
            rl_tools::sample_initial_state(device, env, params, state, rng_state);
            TI scene_idx = env_scene_ptr[env_i];
            TI num_pos = num_indoor_positions_ptr[scene_idx];
            TI pos_idx = rl_tools::random::uniform_int_distribution(device.random, (TI)0, num_pos - 1, rng_state);
            T* pos = indoor_positions_ptr + (scene_idx * max_indoor_pos + pos_idx) * INDOOR_POSITION_DIM;
            scene_translation_arr[env_i * 3 + 0] = pos[0];
            scene_translation_arr[env_i * 3 + 1] = pos[1];
            scene_translation_arr[env_i * 3 + 2] = pos[2];
            T scene_yaw = rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)(2.0 * 3.14159265358979323846), rng_state);
            scene_yaw_arr[env_i] = scene_yaw;
            scene_yaw_cos_arr[env_i] = rl_tools::math::cos(device.math, scene_yaw);
            scene_yaw_sin_arr[env_i] = rl_tools::math::sin(device.math, scene_yaw);
            episode_step_arr[env_i] = 0;
            terminated_flags[env_i] = false;
            episode_return_arr[env_i] = (T)0;
            teacher_forcing_arr[env_i] = full_teacher_forcing || rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, rng_state) < teacher_forcing_fraction;
            brightness_scale_arr[env_i] = (T)1 + (rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, rng_state) * (T)2 - (T)1) * BRIGHTNESS_RANDOMIZATION_RANGE;
            if constexpr(TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE > static_cast<T>(0)){
                T mismatch = (T)1 + (rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, rng_state) * (T)2 - (T)1) * TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE;
                target_brightness_scale_arr[env_i] = brightness_scale_arr[env_i] * mismatch;
            } else {
                target_brightness_scale_arr[env_i] = brightness_scale_arr[env_i];
            }
            if constexpr(RENDER_MOTION_BLUR_ACTIVE){
                shutter_fraction_arr[env_i] = rl_tools::random::uniform_real_distribution(device.random, RENDER_SHUTTER_FRACTION_MIN, RENDER_SHUTTER_FRACTION_MAX, rng_state);
            }
            if constexpr(TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE > static_cast<T>(0)){
                target_frame_roll_arr[env_i] = rl_tools::random::uniform_real_distribution(device.random, -TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, rng_state);
                target_frame_pitch_arr[env_i] = rl_tools::random::uniform_real_distribution(device.random, -TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE, rng_state);
            } else {
                target_frame_roll_arr[env_i] = (T)0;
                target_frame_pitch_arr[env_i] = (T)0;
            }
            for(TI h = 0; h < RAPTOR_HIDDEN_DIM; h++){
                raptor_gru_state_ptr[env_i * RAPTOR_HIDDEN_DIM + h] = raptor_gru_initial_hidden_ptr[h];
            }
            raptor_gru_step_ptr[env_i] = 0;
            episode_start_step[env_i] = step_i;
        } else {
            episode_lengths_log[env_i] = (T)-1;
            episode_tf_log[env_i] = (T)0;
            episode_terminated_log[env_i] = (T)-1;
        }
        {
            rlt::Matrix<rlt::matrix::Specification<T, TI, 1, RAPTOR_OBS_DIM, true, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> obs_mat;
            obs_mat._data = teacher_obs_ptr + env_i * RAPTOR_OBS_DIM;
            rl_tools::observe(device, env.dynamics, params.dynamics, state, RAPTOR_OBSERVATION_TYPE{}, obs_mat, rng_state);
        }
    }

    template<typename DEVICE, typename RNG>
    __global__
    void epilogue_kernel(
        DEVICE device,
        ENVIRONMENT* envs, typename ENVIRONMENT::Parameters* env_params, typename ENVIRONMENT::State* states,
        bool* terminated_flags, TI* episode_step_arr, T* episode_return_arr,
        T* teacher_actions_ptr, T_ACTIVATION* all_targets_ptr,
        RNG rng, TI step_i
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& rng_state = rl_tools::get(rng.states, 0, env_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        TI pos = step_i * N_ENVIRONMENTS + env_i;
        write_yaw_target(state, all_targets_ptr + pos * TARGET_DIM);
        T action_arr[ACTION_DIM];
        for(TI a = 0; a < ACTION_DIM; a++){
            action_arr[a] = teacher_actions_ptr[env_i * ACTION_DIM + a];
        }
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_DIM, true, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> action_matrix;
        action_matrix._data = action_arr;
        typename ENVIRONMENT::State next_state;
        rl_tools::step(device, env, params, state, action_matrix, next_state, rng_state);
        terminated_flags[env_i] = rl_tools::terminated(device, env, params, next_state, rng_state);
        state = next_state;
        episode_step_arr[env_i]++;
    }

    template<bool ENABLE_MOTION_BLUR, typename DEVICE>
    __global__
    void make_cameras_kernel(
        DEVICE device,
        typename ENVIRONMENT::Parameters* env_params, typename ENVIRONMENT::State* states,
        CAMERA_DATA* gpu_cameras,
        CAMERA_DATA* gpu_cameras_open,
        CAMERA_DATA* gpu_prev_cameras,
        const bool* needs_reset_flags,
        const T* shutter_fraction_arr,
        TI step_i,
        T aspect,
        T* scene_translation_arr, T* scene_yaw_cos_arr, T* scene_yaw_sin_arr
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        auto& state = states[env_i];
        const auto& params = env_params[env_i];
        CAMERA_DATA close_camera = rlt::rl::environments::l2f_visual::cuda::make_camera_for_state<DEVICE, VISUAL_SPEC>(
            device, params, state, aspect,
            scene_translation_arr + env_i * 3,
            scene_yaw_cos_arr[env_i], scene_yaw_sin_arr[env_i]
        );
        gpu_cameras[env_i] = close_camera;
        if constexpr(ENABLE_MOTION_BLUR){
            CAMERA_DATA open_camera = close_camera;
            if(step_i > 0 && !needs_reset_flags[env_i]){
                T shutter_fraction = shutter_fraction_arr[env_i];
                open_camera = interpolate_camera(close_camera, gpu_prev_cameras[env_i], shutter_fraction);
            }
            gpu_cameras_open[env_i] = open_camera;
            gpu_prev_cameras[env_i] = close_camera;
        }
    }

    template<typename DEVICE>
    __global__
    void make_target_cameras_kernel(
        DEVICE device,
        typename ENVIRONMENT::Parameters* env_params,
        rlt::rendering::raytracing::CameraData<T>* target_cameras,
        T aspect,
        const T* target_frame_roll_arr,
        const T* target_frame_pitch_arr,
        T* scene_translation_arr, T* scene_yaw_cos_arr, T* scene_yaw_sin_arr
    ){
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) return;
        const auto& params = env_params[env_i];
        target_cameras[env_i] = rlt::rl::environments::l2f_visual::cuda::make_target_camera<DEVICE, VISUAL_SPEC>(
            device, params, aspect,
            scene_translation_arr + env_i * 3,
            scene_yaw_cos_arr[env_i], scene_yaw_sin_arr[env_i],
            target_frame_roll_arr[env_i], target_frame_pitch_arr[env_i]
        );
    }

    __global__
    void mse_batch_loss_kernel(const T_ACTIVATION* student_output_ptr, const T_ACTIVATION* target_ptr, T* loss_out, TI n_elements){
        if(blockIdx.x == 0 && threadIdx.x == 0){
            T acc = 0;
            for(TI i = 0; i < n_elements; i++){
                T diff = static_cast<T>(student_output_ptr[i]) - static_cast<T>(target_ptr[i]);
                acc += diff * diff;
            }
            loss_out[0] = n_elements > 0 ? acc / static_cast<T>(n_elements) : static_cast<T>(0);
        }
    }

    __global__
    void yaw_batch_metrics_kernel(const T_ACTIVATION* student_output_ptr, const T_ACTIVATION* target_ptr, T* metrics_out, TI batch_size){
        if(blockIdx.x == 0 && threadIdx.x == 0){
            T abs_error = 0;
            T squared_error = 0;
            T output_norm = 0;
            T null_mse_loss = 0;
            T null_squared_error = 0;
            for(TI sample_i = 0; sample_i < batch_size; sample_i++){
                const TI base = sample_i * TARGET_DIM;
                T pred_cos = static_cast<T>(student_output_ptr[base + 0]);
                T pred_sin = static_cast<T>(student_output_ptr[base + 1]);
                const T target_cos = static_cast<T>(target_ptr[base + 0]);
                const T target_sin = static_cast<T>(target_ptr[base + 1]);
                T pred_norm = sqrtf(pred_cos * pred_cos + pred_sin * pred_sin);
                output_norm += pred_norm;
                if(pred_norm > static_cast<T>(1e-6)){
                    pred_cos /= pred_norm;
                    pred_sin /= pred_norm;
                } else {
                    pred_cos = static_cast<T>(1);
                    pred_sin = static_cast<T>(0);
                }
                T error = atan2f(pred_sin * target_cos - pred_cos * target_sin,
                                  pred_cos * target_cos + pred_sin * target_sin);
                T null_diff_cos = static_cast<T>(1) - target_cos;
                T null_diff_sin = -target_sin;
                T null_error = atan2f(-target_sin, target_cos);
                abs_error += fabsf(error);
                squared_error += error * error;
                null_mse_loss += (null_diff_cos * null_diff_cos + null_diff_sin * null_diff_sin) / static_cast<T>(TARGET_DIM);
                null_squared_error += null_error * null_error;
            }
            metrics_out[YAW_METRIC_ABS_ERROR_RAD] = batch_size > 0 ? abs_error / static_cast<T>(batch_size) : static_cast<T>(0);
            metrics_out[YAW_METRIC_MSE_RAD] = batch_size > 0 ? squared_error / static_cast<T>(batch_size) : static_cast<T>(0);
            metrics_out[YAW_METRIC_OUTPUT_NORM] = batch_size > 0 ? output_norm / static_cast<T>(batch_size) : static_cast<T>(0);
            metrics_out[YAW_METRIC_NULL_MSE_LOSS] = batch_size > 0 ? null_mse_loss / static_cast<T>(batch_size) : static_cast<T>(0);
            metrics_out[YAW_METRIC_NULL_MSE_RAD] = batch_size > 0 ? null_squared_error / static_cast<T>(batch_size) : static_cast<T>(0);
        }
    }

    __global__
    void reduce_episode_stats_kernel(
        const T* episode_lengths_log,
        const T* episode_tf_log,
        const T* episode_terminated_log,
        const TI* episode_step_arr,
        const bool* teacher_forcing_arr,
        T* stats_out
    ){
        if(blockIdx.x == 0 && threadIdx.x == 0){
            T tf_length_sum = 0;
            T tf_episode_count = 0;
            T student_length_sum = 0;
            T student_episode_count = 0;
            T terminated_count = 0;
            T complete_length_sum = 0;
            T complete_episode_count = 0;
            for(TI pos = 0; pos < STEPS_TOTAL; pos++){
                T episode_length = episode_lengths_log[pos];
                if(episode_length >= (T)0){
                    if(episode_tf_log[pos] > (T)0.5){
                        tf_length_sum += episode_length;
                        tf_episode_count += (T)1;
                    } else {
                        student_length_sum += episode_length;
                        student_episode_count += (T)1;
                    }
                    if(episode_terminated_log[pos] > (T)0.5){
                        terminated_count += (T)1;
                    }
                    complete_length_sum += episode_length;
                    complete_episode_count += (T)1;
                }
            }
            for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                TI episode_step = episode_step_arr[env_i];
                if(episode_step > 0){
                    if(teacher_forcing_arr[env_i]){
                        tf_length_sum += (T)episode_step;
                        tf_episode_count += (T)1;
                    } else {
                        student_length_sum += (T)episode_step;
                        student_episode_count += (T)1;
                    }
                }
            }
            stats_out[0] = tf_length_sum;
            stats_out[1] = tf_episode_count;
            stats_out[2] = student_length_sum;
            stats_out[3] = student_episode_count;
            stats_out[4] = terminated_count;
            stats_out[5] = complete_length_sum;
            stats_out[6] = complete_episode_count;
        }
    }

    template<typename DEVICE, typename MODEL_SPEC, typename INPUT_SPEC, typename STATE_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE>
    __global__
    void raptor_evaluate_step_kernel(
        DEVICE device,
        const rlt::nn_models::sequential::ModuleForward<MODEL_SPEC> model,
        rlt::Tensor<INPUT_SPEC> input,
        rlt::nn_models::sequential::ModuleState<STATE_SPEC> state,
        rlt::Tensor<OUTPUT_SPEC> output,
        rlt::nn_models::sequential::ModuleBuffer<BUFFER_SPEC> buffers,
        RNG rng,
        rlt::Mode<MODE> mode
    ){
        rlt::_evaluate_step(device, model, input, state, state.content_state, output, buffers, buffers.content_buffer, rng, mode);
    }
}

struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
    // static constexpr T ALPHA = 1e-1;
    // static constexpr T EPSILON = 1e-5;
    // static constexpr T EPSILON_SQRT = 1e-5;
};

template<typename CAPABILITY, typename T_TYPE_POLICY = TYPE_POLICY>
struct StudentActor{
    static constexpr TI STEPS = 1;
    static constexpr TI FORWARD_BATCH_SIZE = BATCH_SIZE;

    static constexpr TI IMG_H = ENVIRONMENT::Observation::HEIGHT;
    static constexpr TI IMG_W = ENVIRONMENT::Observation::WIDTH;
    static constexpr TI IMG_C = ENVIRONMENT::Observation::CHANNELS;

    using IMAGE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;

    static constexpr TI HIDDEN_DIM_MULTIPLIER = 2;
    using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 16*HIDDEN_DIM_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
    using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 32*HIDDEN_DIM_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
    using CONV3_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 64*HIDDEN_DIM_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV3 = rlt::nn::layers::conv2d::BindConfiguration<CONV3_CONFIG>;
    using CONV4_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, TI, 128*HIDDEN_DIM_MULTIPLIER, 3, 3, 2, 2, 1, 1, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV4 = rlt::nn::layers::conv2d::BindConfiguration<CONV4_CONFIG>;
    using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, TI>;
    using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
    using IMAGE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using IMAGE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<IMAGE_DENSE_EMBED_CONFIG>;
    using IMAGE_BRANCH = rlt::nn_models::sequential::Module<CONV1, CONV2, CONV3, CONV4, OUTPUT_FLATTEN, IMAGE_DENSE_EMBED>;

    using HEAD_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using HEAD_DENSE1 = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE1_CONFIG>;
    using HEAD_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using HEAD_DENSE2 = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE2_CONFIG>;
    using HEAD_DENSE_OUT_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, TARGET_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD_DENSE_OUT = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE_OUT_CONFIG>;
    using SEQUENTIAL_HEAD = rlt::nn_models::sequential::Module<HEAD_DENSE1, HEAD_DENSE2, HEAD_DENSE_OUT>;
    using BRANCH_IMAGE = rlt::nn_models::parallel::Branch<IMAGE_BRANCH, IMAGE_INPUT_SHAPE>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, SEQUENTIAL_HEAD, BRANCH_IMAGE>;
};

using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, true>;
using STUDENT_TYPE = typename StudentActor<CAPABILITY_ADAM>::MODEL;
using STUDENT_BUFFERS = typename STUDENT_TYPE::Buffer<true>;
using CAPABILITY_FORWARD_CPU = rlt::nn::capability::Forward<true>;
using CPU_STUDENT_TYPE = typename StudentActor<CAPABILITY_FORWARD_CPU, TEACHER_TYPE_POLICY>::MODEL;
using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, ADAM_PARAMETERS, true>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;

static constexpr TI IMG_H = ENVIRONMENT::Observation::HEIGHT;
static constexpr TI IMG_W = ENVIRONMENT::Observation::WIDTH;
static constexpr TI IMG_C = ENVIRONMENT::Observation::CHANNELS;

// =========================================================================
// Main
// =========================================================================
// Parse hex hash string (40 chars) into 20 bytes
static bool parse_hex_hash(const char* hex, unsigned char* out, unsigned len){
    for(unsigned i = 0; i < len; i++){
        unsigned byte = 0;
        for(int nibble = 0; nibble < 2; nibble++){
            char c = hex[i * 2 + nibble];
            if(c >= '0' && c <= '9') byte = (byte << 4) | (c - '0');
            else if(c >= 'a' && c <= 'f') byte = (byte << 4) | (c - 'a' + 10);
            else if(c >= 'A' && c <= 'F') byte = (byte << 4) | (c - 'A' + 10);
            else return false;
        }
        out[i] = static_cast<unsigned char>(byte);
    }
    return true;
}

template<typename DEVICE, typename OBS_SPEC>
void render_target_observation(
    DEVICE& device,
    ENVIRONMENT& env,
    const typename ENVIRONMENT::Parameters& parameters,
    rlt::Matrix<OBS_SPEC>& observation
){
    static_assert(OBS_SPEC::ROWS == 1);
    static_assert(OBS_SPEC::COLS == OBSERVATION_DIM);

    typename ENVIRONMENT::State target_state = {};
    target_state.orientation[0] = (T)1;
    target_state.orientation[1] = (T)0;
    target_state.orientation[2] = (T)0;
    target_state.orientation[3] = (T)0;
    auto camera = rlt::rl::environments::l2f_visual::make_camera_for_state(device, env, parameters, target_state);

    for(TI camera_i = 0; camera_i < N_ENVIRONMENTS_PER_SCENE; camera_i++){
        rlt::set(device, env.renderer->cameras, camera, camera_i);
    }
    rlt::set_cameras(device, *env.renderer, env.renderer->cameras);
    rlt::render(device, *env.renderer);
    rlt::read_frame_buffer(device, *env.renderer, env.renderer->frame_buffer);

    constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    const uint32_t* fb_data = rlt::data(env.renderer->frame_buffer);
    for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
        const uint32_t rgba = fb_data[pixel_i];
        rlt::set(observation, 0, pixel_i * 3 + 0, static_cast<T>((rgba >>  0) & 0xFF) / static_cast<T>(255));
        rlt::set(observation, 0, pixel_i * 3 + 1, static_cast<T>((rgba >>  8) & 0xFF) / static_cast<T>(255));
        rlt::set(observation, 0, pixel_i * 3 + 2, static_cast<T>((rgba >> 16) & 0xFF) / static_cast<T>(255));
    }
}

template <bool APPLY_BRIGHTNESS>
__global__ void scatter_pixel_to_float_kernel(
    const uint32_t* __restrict__ fb, float* __restrict__ output,
    const float* __restrict__ brightness_scales,
    int base_env, int n_envs,
    int pixels_per_camera, int obs_dim
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_envs * pixels_per_camera;
    if(tid >= total) return;
    int local_env = tid / pixels_per_camera;
    int pixel_idx = tid % pixels_per_camera;
    int fb_pixel = local_env * pixels_per_camera + pixel_idx;
    uint32_t rgba = fb[fb_pixel];
    float r = static_cast<float>((rgba >>  0) & 0xFF) / 255.0f;
    float g = static_cast<float>((rgba >>  8) & 0xFF) / 255.0f;
    float b = static_cast<float>((rgba >> 16) & 0xFF) / 255.0f;
    int global_env = base_env + local_env;
    if constexpr(APPLY_BRIGHTNESS){
        float scale = brightness_scales[global_env];
        r = fminf(fmaxf(r * scale, 0.0f), 1.0f);
        g = fminf(fmaxf(g * scale, 0.0f), 1.0f);
        b = fminf(fmaxf(b * scale, 0.0f), 1.0f);
    }
    int out_base = global_env * obs_dim + pixel_idx * 3;
    output[out_base + 0] = r;
    output[out_base + 1] = g;
    output[out_base + 2] = b;
}

__global__ void brightness_apply_kernel(float* __restrict__ obs, const float* __restrict__ brightness_scales, int num_envs, int obs_dim){
    int env_i = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(env_i >= num_envs || idx >= obs_dim) return;
    float scale = brightness_scales[env_i];
    int global_idx = env_i * obs_dim + idx;
    obs[global_idx] = fminf(fmaxf(obs[global_idx] * scale, 0.0f), 1.0f);
}

__global__ void observation_noise_kernel(float* __restrict__ obs, int n, float std, unsigned long long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    curandState rng_state;
    curand_init(seed, idx, 0, &rng_state);
    float val = obs[idx] + curand_normal(&rng_state) * std;
    obs[idx] = fminf(fmaxf(val, 0.0f), 1.0f);
}

template<typename T_OUT>
__global__ void build_frame_stacked_with_target_from_history_kernel(
    const float* __restrict__ history_obs,
    const float* __restrict__ target_obs,
    const TI* __restrict__ episode_start_step,
    TI step_i,
    T_OUT* __restrict__ combined_out,
    int obs_dim, int img_c, int n_frames, int frame_stride, int combined_img_c, int combined_obs_dim, int num_envs
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx >= num_envs * combined_obs_dim) return;
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_BLIND_TRAINING
    combined_out[global_idx] = (T_OUT)0.0f;
#else
    int sample = global_idx / combined_obs_dim;
    int offset = global_idx % combined_obs_dim;
    int pixel = offset / combined_img_c;
    int frame_channel = offset % combined_img_c;
    int logical_channels = n_frames * img_c + img_c;
    if(frame_channel < n_frames * img_c){
        int frame = frame_channel / img_c;
        int channel = frame_channel % img_c;
        TI episode_start = episode_start_step[sample];
        TI back = static_cast<TI>(frame) * static_cast<TI>(frame_stride);
        TI desired_step = step_i >= back ? step_i - back : episode_start;
        if(desired_step < episode_start){
            desired_step = episode_start;
        }
        TI history_slot = desired_step % FRAME_STACK_HISTORY_LENGTH;
        TI src_row = history_slot * num_envs + sample;
        combined_out[global_idx] = (T_OUT)history_obs[src_row * obs_dim + pixel * img_c + channel];
    } else if(frame_channel < logical_channels) {
        int channel = frame_channel - n_frames * img_c;
        combined_out[global_idx] = (T_OUT)target_obs[sample * obs_dim + pixel * img_c + channel];
    } else {
        combined_out[global_idx] = (T_OUT)0.0f;
    }
#endif
}

template<typename T_OUT>
__global__ void build_validation_combined_with_target_kernel(
    const float* __restrict__ obs,
    const float* __restrict__ target_obs,
    T_OUT* __restrict__ combined_out,
    int obs_dim, int img_c, int n_frames, int combined_img_c, int combined_obs_dim
){
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx >= VALIDATION_BATCH_SIZE * combined_obs_dim) return;
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_BLIND_TRAINING
    combined_out[global_idx] = (T_OUT)0.0f;
#else
    int sample = global_idx / combined_obs_dim;
    int offset = global_idx % combined_obs_dim;
    int pixel = offset / combined_img_c;
    int frame_channel = offset % combined_img_c;
    int logical_channels = n_frames * img_c + img_c;
    if(frame_channel < n_frames * img_c){
        int channel = frame_channel % img_c;
        combined_out[global_idx] = (T_OUT)obs[sample * obs_dim + pixel * img_c + channel];
    } else if(frame_channel < logical_channels) {
        int channel = frame_channel - n_frames * img_c;
        combined_out[global_idx] = (T_OUT)target_obs[sample * obs_dim + pixel * img_c + channel];
    } else {
        combined_out[global_idx] = (T_OUT)0.0f;
    }
#endif
}

int main(int argc, char** argv){
#ifdef RL_TOOLS_DEBUG_CUDA_CHECK
#define CUDA_CHECK(msg) { cudaError_t e = cudaGetLastError(); if(e != cudaSuccess){ std::cerr << "CUDA ERROR [" << msg << "]: " << cudaGetErrorString(e) << std::endl; return 1; } e = cudaDeviceSynchronize(); if(e != cudaSuccess){ std::cerr << "CUDA SYNC ERROR [" << msg << "]: " << cudaGetErrorString(e) << std::endl; return 1; } }
#else
#define CUDA_CHECK(msg) ((void)0)
#endif
    TI seed = 0;
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <scene_directory or scene.glb> [seed]" << std::endl;
        return 1;
    }
    if(argc > 2){
        seed = std::atoi(argv[2]);
    }

    static constexpr TI EPOCHS_PER_SCENE = 50;
    static constexpr TI RENDER_TIMING_SAMPLE_PERIOD = 16;

    std::vector<std::string> scene_paths;
    std::mt19937 scene_rng(seed);
    const char* scene_arg = argv[1];
    if(std::filesystem::is_directory(scene_arg)){
        std::vector<std::string> all_glbs;
        for(auto& entry : std::filesystem::directory_iterator(scene_arg)){
            if(entry.path().extension() == ".glb"){
                all_glbs.push_back(entry.path().string());
            }
        }
        std::sort(all_glbs.begin(), all_glbs.end(), [](const std::string& a, const std::string& b){
            auto extract_number = [](const std::string& path) -> int {
                auto filename = std::filesystem::path(path).stem().string();
                auto pos = filename.rfind('-');
                if(pos != std::string::npos){
                    try { return std::stoi(filename.substr(pos + 1)); } catch(...) {}
                }
                return 0;
            };
            return extract_number(a) < extract_number(b);
        });

        if(static_cast<TI>(all_glbs.size()) < N_TOTAL_SCENES){
            std::cerr << "Need at least " << N_TOTAL_SCENES << " GLB scenes, found " << all_glbs.size() << std::endl;
            return 1;
        }
        for(TI i = 0; i < N_TOTAL_SCENES; i++){
            scene_paths.push_back(all_glbs[i]);
        }
        std::cout << "Selected " << scene_paths.size() << " scenes from " << scene_arg
                  << " (" << N_TRAIN_SCENES << " train, " << N_VALIDATION_SCENES << " validation)" << std::endl;
        for(TI i = 0; i < scene_paths.size(); i++){
            std::cout << "  [" << i << "] "
                      << (i < N_TRAIN_SCENES ? "train " : "valid ")
                      << std::filesystem::path(scene_paths[i]).filename().string() << std::endl;
        }
    } else {
        scene_paths.resize(N_TOTAL_SCENES, scene_arg);
        std::cout << "Replicating single scene across " << N_TOTAL_SCENES << " renderers: " << scene_arg << std::endl;
    }
    if(scene_paths.size() != N_TOTAL_SCENES){
        std::cerr << "Expected exactly " << N_TOTAL_SCENES << " scenes, got " << scene_paths.size() << std::endl;
        return 1;
    }

    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device);

    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = "l2f_visual_yaw_cuda";
    rlt::init(device, extrack_config, extrack_paths, seed);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);

    // =========================================================================
    // RAPTOR teacher (CPU)
    // =========================================================================
    RAPTOR_MODEL raptor;
    typename RAPTOR_MODEL::Buffer<true> raptor_buffer;
    typename RAPTOR_MODEL::State<true> raptor_state;
    rlt::malloc(device, raptor);
    rlt::malloc(device, raptor_buffer);
    rlt::malloc(device, raptor_state);
    rlt::copy(device, device, rl_tools::checkpoint::actor::module, raptor);
    rlt::reset(device, raptor, raptor_state, rng);

    // RAPTOR input/output tensors (CPU)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, RAPTOR_OBS_DIM>>> teacher_obs;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTION_DIM>>> teacher_actions;
    rlt::malloc(device, teacher_obs);
    rlt::malloc(device, teacher_actions);

    // All teacher actions for entire epoch (CPU)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, ACTION_DIM>>> cpu_all_teacher_actions;
    rlt::malloc(device, cpu_all_teacher_actions);

    // =========================================================================
    // Student (CPU copy for warmup, float for CPU compatibility)
    // =========================================================================
    using CPU_STUDENT_INIT_TYPE = typename StudentActor<CAPABILITY_ADAM, TEACHER_TYPE_POLICY>::MODEL;
    CPU_STUDENT_INIT_TYPE student_cpu;
    rlt::malloc(device, student_cpu);
    rlt::init_weights(device, student_cpu, rng);

    // =========================================================================
    // Environment setup (N renderers for N scenes)
    // =========================================================================
    using RENDERER_TYPE = rlt::rendering::raytracing::Renderer<typename ENVIRONMENT::SPEC::RENDERER_SPEC>;
    using SCENE_TYPE = rlt::rendering::raytracing::scene::procthor::Scene<typename ENVIRONMENT::SPEC::SCENE_SPEC>;

    ENVIRONMENT envs[N_ENVIRONMENTS];
    typename ENVIRONMENT::Parameters env_parameters[N_ENVIRONMENTS];

    std::array<RENDERER_TYPE*, N_TOTAL_SCENES> renderers{};
    std::array<SCENE_TYPE*, N_TOTAL_SCENES> scenes{};

    // Load all scenes
    {
        ENVIRONMENT loader_env;
        rlt::malloc(device, loader_env);
        for(TI s = 0; s < N_TOTAL_SCENES; s++){
            std::cout << "Loading scene [" << s << "]: " << std::filesystem::path(scene_paths[s]).filename().string() << std::flush;
            if(s > 0){
                loader_env.renderer = new RENDERER_TYPE{};
                rlt::malloc(device, *loader_env.renderer);
                loader_env.owns_renderer = true;
                loader_env.scene = new SCENE_TYPE{};
            }
            loader_env.scene_path = scene_paths[s].c_str();
            loader_env.renderer_initialized = false;
            rlt::init(device, loader_env);
            TI num_pos = loader_env.scene->num_indoor_positions;
            std::cout << " — " << num_pos << " indoor positions" << std::endl;
            if(num_pos == 0){
                std::cerr << "Scene has no valid positions with 1m clearance: " << scene_paths[s] << std::endl;
                return 1;
            }
            renderers[s] = loader_env.renderer;
            scenes[s] = loader_env.scene;
            loader_env.renderer = nullptr;
            loader_env.scene = nullptr;
            loader_env.owns_renderer = false;
        }
        std::cout << "Loaded " << N_TOTAL_SCENES << " scenes" << std::endl;
    }

    // GPU buffers for per-scene indoor positions
    static constexpr TI MAX_INDOOR_POS = 256;
    T* gpu_indoor_positions = nullptr;
    TI* gpu_num_indoor_positions = nullptr;
    TI* gpu_env_scene = nullptr;
    cudaMalloc(&gpu_indoor_positions, N_TOTAL_SCENES * MAX_INDOOR_POS * INDOOR_POSITION_DIM * sizeof(T));
    cudaMalloc(&gpu_num_indoor_positions, N_TOTAL_SCENES * sizeof(TI));
    cudaMalloc(&gpu_env_scene, N_ENVIRONMENTS * sizeof(TI));
    {
        std::vector<T> all_positions(N_TOTAL_SCENES * MAX_INDOOR_POS * INDOOR_POSITION_DIM, 0);
        std::vector<TI> all_counts(N_TOTAL_SCENES);
        for(TI s = 0; s < N_TOTAL_SCENES; s++){
            all_counts[s] = scenes[s]->num_indoor_positions;
            for(TI i = 0; i < all_counts[s]; i++){
                TI base = (s * MAX_INDOOR_POS + i) * INDOOR_POSITION_DIM;
                all_positions[base + 0] = scenes[s]->indoor_positions[i].position[0];
                all_positions[base + 1] = scenes[s]->indoor_positions[i].position[1];
                all_positions[base + 2] = scenes[s]->indoor_positions[i].position[2];
            }
        }
        cudaMemcpy(gpu_indoor_positions, all_positions.data(), all_positions.size() * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_num_indoor_positions, all_counts.data(), N_TOTAL_SCENES * sizeof(TI), cudaMemcpyHostToDevice);
    }

    std::array<TI, N_ACTIVE_SCENES> active_scene_indices{};
    std::vector<TI> scene_permutation(N_TRAIN_SCENES);
    std::iota(scene_permutation.begin(), scene_permutation.end(), 0);
    auto upload_active_scenes = [&](){
        std::array<TI, N_ENVIRONMENTS> env_scene{};
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            TI actual_scene_i = active_scene_indices[active_scene_i];
            std::cout << "  Active scene slot " << active_scene_i << " -> scene " << actual_scene_i << std::endl;
            for(TI local_env_i = 0; local_env_i < N_ENVIRONMENTS_PER_SCENE; local_env_i++){
                env_scene[active_scene_i * N_ENVIRONMENTS_PER_SCENE + local_env_i] = actual_scene_i;
            }
        }
        cudaMemcpy(gpu_env_scene, env_scene.data(), N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
    };
    for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
        active_scene_indices[active_scene_i] = active_scene_i;
    }
    upload_active_scenes();
    for(TI scene_i = 0; scene_i < N_TOTAL_SCENES; scene_i++){
        rlt::set_cameras(device, *renderers[scene_i], renderers[scene_i]->cameras);
    }

    // Set up envs (use first renderer for env0 reference)
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        TI scene_i = env_i / N_ENVIRONMENTS_PER_SCENE;
        rlt::malloc(device, envs[env_i].dynamics);
        envs[env_i].renderer = renderers[scene_i];
        envs[env_i].scene = scenes[scene_i];
        envs[env_i].owns_renderer = false;
        envs[env_i].renderer_initialized = true;
        envs[env_i].use_target_mode = true;
        envs[env_i].parameters.fov = CAMERA_FOV;
        envs[env_i].parameters.camera_randomization.fov_range = CAMERA_FOV_RANDOMIZATION_RANGE;
        envs[env_i].parameters.camera_randomization.offset_body_range[0] = CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X;
        envs[env_i].parameters.camera_randomization.offset_body_range[1] = CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y;
        envs[env_i].parameters.camera_randomization.offset_body_range[2] = CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z;
        envs[env_i].parameters.camera_randomization.rotation_body_range[0] = CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X;
        envs[env_i].parameters.camera_randomization.rotation_body_range[1] = CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y;
        envs[env_i].parameters.camera_randomization.rotation_body_range[2] = CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z;
        rlt::initial_parameters(device, envs[env_i], env_parameters[env_i]);
        env_parameters[env_i].scene_translation[0] = 0;
        env_parameters[env_i].scene_translation[1] = 0;
        env_parameters[env_i].scene_translation[2] = 0;
    }
    auto& env0 = envs[0];

    {
        std::string ui = rlt::get_ui(device, envs[0].dynamics);
        if(!ui.empty()){
            std::filesystem::create_directories(extrack_paths.seed);
            std::ofstream ui_file(extrack_paths.seed / "ui.esm.js");
            ui_file << ui;
        }
    }

    EpisodeRecorder episode_recorders[TRAJECTORY_NUM_ENVS];
    std::vector<CompletedEpisode> completed_episodes;
    std::vector<typename ENVIRONMENT::State> cpu_prestep_state_buf(TRAJECTORY_NUM_ENVS);
    std::vector<uint8_t> cpu_needs_reset_buf(TRAJECTORY_NUM_ENVS);
    std::vector<uint8_t> cpu_terminated_buf(TRAJECTORY_NUM_ENVS);
    std::vector<T> cpu_rollout_action_buf(TRAJECTORY_NUM_ENVS * ACTION_DIM);
    std::vector<typename ENVIRONMENT::Parameters> cpu_params_snapshot_buf(TRAJECTORY_NUM_ENVS);
    T simulation_dt = static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY);
    TI global_step = 0;

    // =========================================================================
    // GPU init
    // =========================================================================
    rlt::init(device_gpu);
    RNG_GPU rng_gpu;
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_gpu, rng_gpu, seed);

    STUDENT_TYPE student_gpu;
    STUDENT_BUFFERS student_buffers;
    OPTIMIZER optimizer_gpu;
    rlt::malloc(device_gpu, student_gpu);
    rlt::malloc(device_gpu, student_buffers);
    rlt::malloc(device_gpu, optimizer_gpu);
    rlt::copy(device, device_gpu, student_cpu, student_gpu);
    rlt::init(device_gpu, optimizer_gpu);
    rlt::reset_optimizer_state(device_gpu, optimizer_gpu, student_gpu);
    // GPU RAPTOR teacher
    RAPTOR_MODEL raptor_gpu;
    typename RAPTOR_MODEL::Buffer<true> raptor_buffer_gpu;
    typename RAPTOR_MODEL::State<true> raptor_state_gpu;
    rlt::malloc(device_gpu, raptor_gpu);
    rlt::malloc(device_gpu, raptor_buffer_gpu);
    rlt::malloc(device_gpu, raptor_state_gpu);
    rlt::copy(device, device_gpu, raptor, raptor_gpu);
    rlt::reset(device_gpu, raptor_gpu, raptor_state_gpu, rng_gpu);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, RAPTOR_OBS_DIM>>> gpu_teacher_obs;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTION_DIM>>> gpu_teacher_actions_step;
    rlt::malloc(device_gpu, gpu_teacher_obs);
    rlt::malloc(device_gpu, gpu_teacher_actions_step);

    // GPU cameras
    CAMERA_DATA* gpu_cameras = nullptr;
    cudaMalloc(&gpu_cameras, N_ENVIRONMENTS * sizeof(CAMERA_DATA));
    CAMERA_DATA* gpu_cameras_open = nullptr;
    CAMERA_DATA* gpu_prev_cameras = nullptr;
    if constexpr(RENDER_MOTION_BLUR_ACTIVE){
        cudaMalloc(&gpu_cameras_open, N_ENVIRONMENTS * sizeof(CAMERA_DATA));
        cudaMalloc(&gpu_prev_cameras, N_ENVIRONMENTS * sizeof(CAMERA_DATA));
    }
    CAMERA_DATA* gpu_target_cameras = nullptr;
    cudaMalloc(&gpu_target_cameras, N_ENVIRONMENTS * sizeof(CAMERA_DATA));
    CAMERA_DATA* gpu_validation_cameras = nullptr;
    CAMERA_DATA* gpu_validation_target_cameras = nullptr;
    cudaMalloc(&gpu_validation_cameras, VALIDATION_BATCH_SIZE * sizeof(CAMERA_DATA));
    cudaMalloc(&gpu_validation_target_cameras, VALIDATION_BATCH_SIZE * sizeof(CAMERA_DATA));

    // Event for cross-stream synchronization (make_cameras on device_gpu.stream → optix_stream)
    cudaEvent_t cameras_ready_event;
    cudaEventCreateWithFlags(&cameras_ready_event, cudaEventDisableTiming);
    cudaEvent_t target_cameras_ready_event;
    cudaEventCreateWithFlags(&target_cameras_ready_event, cudaEventDisableTiming);
    cudaEvent_t validation_cameras_ready_event;
    cudaEventCreateWithFlags(&validation_cameras_ready_event, cudaEventDisableTiming);
    cudaEvent_t validation_target_cameras_ready_event;
    cudaEventCreateWithFlags(&validation_target_cameras_ready_event, cudaEventDisableTiming);
    cudaEvent_t validation_render_scatter_done_event;
    cudaEventCreateWithFlags(&validation_render_scatter_done_event, cudaEventDisableTiming);
    cudaEvent_t validation_target_render_scatter_done_event;
    cudaEventCreateWithFlags(&validation_target_render_scatter_done_event, cudaEventDisableTiming);
    std::array<cudaEvent_t, N_ACTIVE_SCENES> render_scatter_done_events;
    std::array<cudaEvent_t, N_ACTIVE_SCENES> target_render_scatter_done_events;
    for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
        cudaEventCreateWithFlags(&render_scatter_done_events[active_scene_i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&target_render_scatter_done_events[active_scene_i], cudaEventDisableTiming);
    }
    std::vector<cudaEvent_t> render_pass_start_events(STEPS_PER_ENV * N_ACTIVE_SCENES);
    std::vector<cudaEvent_t> render_pass_stop_events(STEPS_PER_ENV * N_ACTIVE_SCENES);
    std::vector<cudaEvent_t> target_render_pass_start_events(STEPS_PER_ENV * N_ACTIVE_SCENES);
    std::vector<cudaEvent_t> target_render_pass_stop_events(STEPS_PER_ENV * N_ACTIVE_SCENES);
    for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            TI event_i = step_i * N_ACTIVE_SCENES + active_scene_i;
            cudaEventCreate(&render_pass_start_events[event_i]);
            cudaEventCreate(&render_pass_stop_events[event_i]);
            cudaEventCreate(&target_render_pass_start_events[event_i]);
            cudaEventCreate(&target_render_pass_stop_events[event_i]);
        }
    }
    TI max_train_timing_calls = N_TRAIN_PASSES * N_BATCHES;
    TI max_logged_loss_calls = N_BATCHES;
    std::vector<cudaEvent_t> train_forward_start_events(max_train_timing_calls);
    std::vector<cudaEvent_t> train_forward_stop_events(max_train_timing_calls);
    std::vector<cudaEvent_t> train_backward_start_events(max_train_timing_calls);
    std::vector<cudaEvent_t> train_backward_stop_events(max_train_timing_calls);
    std::vector<cudaEvent_t> train_update_start_events(max_train_timing_calls);
    std::vector<cudaEvent_t> train_update_stop_events(max_train_timing_calls);
    for(TI call_i = 0; call_i < max_train_timing_calls; call_i++){
        cudaEventCreate(&train_forward_start_events[call_i]);
        cudaEventCreate(&train_forward_stop_events[call_i]);
        cudaEventCreate(&train_backward_start_events[call_i]);
        cudaEventCreate(&train_backward_stop_events[call_i]);
        cudaEventCreate(&train_update_start_events[call_i]);
        cudaEventCreate(&train_update_stop_events[call_i]);
    }
    T* gpu_logged_batch_losses = nullptr;
    cudaMalloc(&gpu_logged_batch_losses, max_logged_loss_calls * sizeof(T));
    std::vector<T> cpu_logged_batch_losses(max_logged_loss_calls);
    T* gpu_logged_yaw_metrics = nullptr;
    cudaMalloc(&gpu_logged_yaw_metrics, max_logged_loss_calls * YAW_NUM_METRICS * sizeof(T));
    std::vector<T> cpu_logged_yaw_metrics(max_logged_loss_calls * YAW_NUM_METRICS);
    T* gpu_validation_metrics = nullptr;
    cudaMalloc(&gpu_validation_metrics, YAW_NUM_METRICS * sizeof(T));
    std::array<T, YAW_NUM_METRICS> cpu_validation_metrics{};
    T* gpu_epoch_episode_stats = nullptr;
    cudaMalloc(&gpu_epoch_episode_stats, 7 * sizeof(T));
    std::array<T, 7> cpu_epoch_episode_stats{};

    // GPU tensors
    static constexpr TI GPU_OBS_ROWS = STEPS_TOTAL + BATCH_SIZE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, OBSERVATION_DIM>>> gpu_all_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, OBSERVATION_DIM>>> gpu_all_target_observations;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, TARGET_DIM>>> gpu_all_targets;
    rlt::Matrix<rlt::matrix::Specification<T_GRADIENT, TI, BATCH_SIZE, TARGET_DIM>> gpu_d_action_train;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, TARGET_DIM>>> gpu_student_output_train;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, VALIDATION_BATCH_SIZE, OBSERVATION_DIM>>> gpu_validation_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, VALIDATION_BATCH_SIZE, OBSERVATION_DIM>>> gpu_validation_target_observations;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>>> gpu_validation_combined_observations;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, BATCH_SIZE, TARGET_DIM>>> gpu_validation_targets;
    rlt::malloc(device_gpu, gpu_all_observations);
    rlt::malloc(device_gpu, gpu_all_target_observations);
    rlt::malloc(device_gpu, gpu_all_targets);
    rlt::malloc(device_gpu, gpu_d_action_train);
    rlt::malloc(device_gpu, gpu_student_output_train);
    rlt::malloc(device_gpu, gpu_validation_observations);
    rlt::malloc(device_gpu, gpu_validation_target_observations);
    rlt::malloc(device_gpu, gpu_validation_combined_observations);
    rlt::malloc(device_gpu, gpu_validation_targets);

    static constexpr TI FRAME_STACK_HISTORY_ROWS = FRAME_STACK_HISTORY_LENGTH * N_ENVIRONMENTS;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, FRAME_STACK_HISTORY_ROWS, OBSERVATION_DIM>>> gpu_frame_stack_history;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, COMBINED_OBS_DIM>>> gpu_all_combined_observations;
    rlt::malloc(device_gpu, gpu_frame_stack_history);
    {
        size_t combined_observations_bytes = static_cast<size_t>(STEPS_TOTAL) * static_cast<size_t>(COMBINED_OBS_DIM) * sizeof(T_ACTIVATION);
        cudaError_t allocation_status = cudaMalloc((void**)rlt::data_pointer(gpu_all_combined_observations), combined_observations_bytes);
        if(allocation_status != cudaSuccess){
            std::cerr << "Failed to allocate gpu_all_combined_observations: " << cudaGetErrorString(allocation_status) << std::endl;
            return 1;
        }
    }
    TI* gpu_episode_start_step = nullptr;
    cudaMalloc(&gpu_episode_start_step, N_ENVIRONMENTS * sizeof(TI));
    cudaMemset(gpu_episode_start_step, 0, N_ENVIRONMENTS * sizeof(TI));

    // =========================================================================
    // GPU-resident environment state
    // =========================================================================
    ENVIRONMENT* gpu_envs_arr = nullptr;
    typename ENVIRONMENT::Parameters* gpu_params_arr = nullptr;
    typename ENVIRONMENT::State* gpu_states_arr = nullptr;
    typename ENVIRONMENT::Parameters* gpu_validation_params_arr = nullptr;
    typename ENVIRONMENT::State* gpu_validation_states_arr = nullptr;
    bool* gpu_terminated_arr = nullptr;
    TI* gpu_episode_step_arr = nullptr;
    bool* gpu_teacher_forcing_arr = nullptr;
    T* gpu_episode_return_arr = nullptr;
    bool* gpu_needs_reset = nullptr;
    T* gpu_episode_lengths_log = nullptr;
    T* gpu_episode_tf_log = nullptr;
    T* gpu_episode_terminated_log = nullptr;
    T* gpu_brightness_scale_arr = nullptr;
    T* gpu_target_brightness_scale_arr = nullptr;
    T* gpu_target_frame_roll_arr = nullptr;
    T* gpu_target_frame_pitch_arr = nullptr;
    T* gpu_shutter_fraction_arr = nullptr;
    T* gpu_scene_translation_arr = nullptr;
    T* gpu_scene_yaw_arr = nullptr;
    T* gpu_scene_yaw_cos_arr = nullptr;
    T* gpu_scene_yaw_sin_arr = nullptr;
    T* gpu_validation_scene_translation_arr = nullptr;
    T* gpu_validation_scene_yaw_cos_arr = nullptr;
    T* gpu_validation_scene_yaw_sin_arr = nullptr;
    T* gpu_validation_target_frame_roll_arr = nullptr;
    T* gpu_validation_target_frame_pitch_arr = nullptr;
    cudaMalloc(&gpu_envs_arr, N_ENVIRONMENTS * sizeof(ENVIRONMENT));
    cudaMalloc(&gpu_params_arr, N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::Parameters));
    cudaMalloc(&gpu_states_arr, N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::State));
    cudaMalloc(&gpu_validation_params_arr, VALIDATION_BATCH_SIZE * sizeof(typename ENVIRONMENT::Parameters));
    cudaMalloc(&gpu_validation_states_arr, VALIDATION_BATCH_SIZE * sizeof(typename ENVIRONMENT::State));
    cudaMalloc(&gpu_terminated_arr, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_step_arr, N_ENVIRONMENTS * sizeof(TI));
    cudaMalloc(&gpu_teacher_forcing_arr, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_return_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_needs_reset, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_lengths_log, STEPS_TOTAL * sizeof(T));
    cudaMalloc(&gpu_episode_tf_log, STEPS_TOTAL * sizeof(T));
    cudaMalloc(&gpu_episode_terminated_log, STEPS_TOTAL * sizeof(T));
    cudaMalloc(&gpu_brightness_scale_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_target_brightness_scale_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_target_frame_roll_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_target_frame_pitch_arr, N_ENVIRONMENTS * sizeof(T));
    if constexpr(RENDER_MOTION_BLUR_ACTIVE){
        cudaMalloc(&gpu_shutter_fraction_arr, N_ENVIRONMENTS * sizeof(T));
    }
    cudaMalloc(&gpu_scene_translation_arr, N_ENVIRONMENTS * 3 * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_cos_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_sin_arr, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_validation_scene_translation_arr, VALIDATION_BATCH_SIZE * 3 * sizeof(T));
    cudaMalloc(&gpu_validation_scene_yaw_cos_arr, VALIDATION_BATCH_SIZE * sizeof(T));
    cudaMalloc(&gpu_validation_scene_yaw_sin_arr, VALIDATION_BATCH_SIZE * sizeof(T));
    cudaMalloc(&gpu_validation_target_frame_roll_arr, VALIDATION_BATCH_SIZE * sizeof(T));
    cudaMalloc(&gpu_validation_target_frame_pitch_arr, VALIDATION_BATCH_SIZE * sizeof(T));
    {
        std::vector<T> ones(N_ENVIRONMENTS, (T)1);
        std::vector<T> zeros(N_ENVIRONMENTS, (T)0);
        cudaMemcpy(gpu_brightness_scale_arr, ones.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_target_brightness_scale_arr, ones.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_target_frame_roll_arr, zeros.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_target_frame_pitch_arr, zeros.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        if constexpr(RENDER_MOTION_BLUR_ACTIVE){
            std::vector<T> shutter_fractions(N_ENVIRONMENTS, RENDER_SHUTTER_FRACTION_MAX);
            cudaMemcpy(gpu_shutter_fraction_arr, shutter_fractions.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        }
        std::vector<T> scene_translations(N_ENVIRONMENTS * 3, (T)0);
        cudaMemcpy(gpu_scene_translation_arr, scene_translations.data(), N_ENVIRONMENTS * 3 * sizeof(T), cudaMemcpyHostToDevice);
        std::vector<T> scene_yaws(N_ENVIRONMENTS, (T)0);
        cudaMemcpy(gpu_scene_yaw_arr, scene_yaws.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_scene_yaw_cos_arr, ones.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_scene_yaw_sin_arr, zeros.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
    }
    typename ENVIRONMENT::State cpu_states_for_cameras[N_ENVIRONMENTS];
    cudaMemcpy(gpu_envs_arr, envs, N_ENVIRONMENTS * sizeof(ENVIRONMENT), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_params_arr, env_parameters, N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::Parameters), cudaMemcpyHostToDevice);
    {
        bool init_terminated[N_ENVIRONMENTS];
        TI init_step[N_ENVIRONMENTS];
        T init_return[N_ENVIRONMENTS];
        bool init_teacher_forcing[N_ENVIRONMENTS];
        for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
            init_terminated[env_i] = true;
            init_step[env_i] = 0;
            init_return[env_i] = (T)0;
            init_teacher_forcing[env_i] = true;
        }
        cudaMemcpy(gpu_terminated_arr, init_terminated, N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_step_arr, init_step, N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_episode_return_arr, init_return, N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_teacher_forcing_arr, init_teacher_forcing, N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
    }
    T episode_length_sum_tf = 0;
    TI episode_count_tf = 0;
    T episode_length_sum_student = 0;
    TI episode_count_student = 0;

    struct ValidationSummary {
        T yaw_abs_error_rad = 0;
        T yaw_mse_rad = 0;
        T yaw_output_norm = 0;
        T yaw_null_mse_loss = 0;
        T yaw_null_mse_rad = 0;
        T yaw_angle_r2_uniform = 0;
        T frontier_yaw_mse_rad = 0;
        T frontier_min_deg = 0;
        T frontier_max_deg = 0;
        std::array<T, N_VALIDATION_YAW_BINS> bin_yaw_mse_rad{};
    };
    const std::array<T, N_VALIDATION_YAW_BINS> validation_yaw_bins = {
        -static_cast<T>(2)*rlt::math::PI<T> / static_cast<T>(180),
        -static_cast<T>(4)*rlt::math::PI<T> / static_cast<T>(180),
        -static_cast<T>(8)*rlt::math::PI<T> / static_cast<T>(180),
        -static_cast<T>(16)*rlt::math::PI<T> / static_cast<T>(180),
        -static_cast<T>(32)*rlt::math::PI<T> / static_cast<T>(180),
        static_cast<T>(0),
        static_cast<T>(2)*rlt::math::PI<T> / static_cast<T>(180),
        static_cast<T>(4)*rlt::math::PI<T> / static_cast<T>(180),
        static_cast<T>(8)*rlt::math::PI<T> / static_cast<T>(180),
        static_cast<T>(16)*rlt::math::PI<T> / static_cast<T>(180),
        static_cast<T>(32)*rlt::math::PI<T> / static_cast<T>(180)
    };
    const std::array<const char*, N_VALIDATION_YAW_BINS> validation_yaw_bin_names = {
        "m02", "m04", "m08", "m16", "m32", "p00", "p02", "p04", "p08", "p16", "p32"
    };
    typename ENVIRONMENT::Parameters validation_base_parameters;
    rlt::initial_parameters(device, envs[0], validation_base_parameters);
    std::vector<typename ENVIRONMENT::Parameters> cpu_validation_params(VALIDATION_BATCH_SIZE);
    std::vector<typename ENVIRONMENT::State> cpu_validation_states(VALIDATION_BATCH_SIZE);
    std::vector<T> cpu_validation_scene_translation(VALIDATION_BATCH_SIZE * 3);
    std::vector<T> cpu_validation_scene_yaw_cos(VALIDATION_BATCH_SIZE);
    std::vector<T> cpu_validation_scene_yaw_sin(VALIDATION_BATCH_SIZE);
    std::vector<T> cpu_validation_target_frame_roll(VALIDATION_BATCH_SIZE, static_cast<T>(0));
    std::vector<T> cpu_validation_target_frame_pitch(VALIDATION_BATCH_SIZE, static_cast<T>(0));
    std::vector<T_ACTIVATION> cpu_validation_targets(VALIDATION_BATCH_SIZE * TARGET_DIM);

    auto run_validation = [&](T frontier_min_deg, T frontier_max_deg) -> ValidationSummary {
        ValidationSummary summary;
        summary.frontier_min_deg = frontier_min_deg;
        summary.frontier_max_deg = frontier_max_deg;
        std::array<T, N_VALIDATION_YAW_BINS> bin_weight{};
        T total_weight = static_cast<T>(0);
        T frontier_weight = static_cast<T>(0);
        constexpr TI VALIDATION_BLOCKSIZE = 32;
        constexpr TI VALIDATION_N_BLOCKS = (VALIDATION_BATCH_SIZE + VALIDATION_BLOCKSIZE - 1) / VALIDATION_BLOCKSIZE;
        dim3 validation_grid(VALIDATION_N_BLOCKS);
        dim3 validation_block(VALIDATION_BLOCKSIZE);
        rlt::devices::cuda::TAG<DEVICE_GPU, true> tag_device{};
        const T cam_aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
        auto evaluate_validation_batch = [&](RENDERER_TYPE* renderer, cudaStream_t optix_stream, void* camera_buffer, void* camera_open_buffer, const uint32_t* fb_ptr) -> std::array<T, YAW_NUM_METRICS> {
            cudaMemcpy(gpu_validation_params_arr, cpu_validation_params.data(), VALIDATION_BATCH_SIZE * sizeof(typename ENVIRONMENT::Parameters), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_validation_states_arr, cpu_validation_states.data(), VALIDATION_BATCH_SIZE * sizeof(typename ENVIRONMENT::State), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_validation_scene_translation_arr, cpu_validation_scene_translation.data(), VALIDATION_BATCH_SIZE * 3 * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_validation_scene_yaw_cos_arr, cpu_validation_scene_yaw_cos.data(), VALIDATION_BATCH_SIZE * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_validation_scene_yaw_sin_arr, cpu_validation_scene_yaw_sin.data(), VALIDATION_BATCH_SIZE * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_validation_target_frame_roll_arr, cpu_validation_target_frame_roll.data(), VALIDATION_BATCH_SIZE * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_validation_target_frame_pitch_arr, cpu_validation_target_frame_pitch.data(), VALIDATION_BATCH_SIZE * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemset(rlt::data(gpu_validation_targets), 0, BATCH_SIZE * TARGET_DIM * sizeof(T_ACTIVATION));
            cudaMemcpy(rlt::data(gpu_validation_targets), cpu_validation_targets.data(), cpu_validation_targets.size() * sizeof(T_ACTIVATION), cudaMemcpyHostToDevice);

            imitation_kernels::make_cameras_kernel<false><<<validation_grid, validation_block, 0, device_gpu.stream>>>(
                tag_device, gpu_validation_params_arr, gpu_validation_states_arr,
                gpu_validation_cameras,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                static_cast<TI>(0),
                cam_aspect,
                gpu_validation_scene_translation_arr, gpu_validation_scene_yaw_cos_arr, gpu_validation_scene_yaw_sin_arr);
            CUDA_CHECK("validation make_cameras_kernel");
            cudaEventRecord(validation_cameras_ready_event, device_gpu.stream);
            cudaStreamWaitEvent(optix_stream, validation_cameras_ready_event, 0);
            if constexpr(RENDER_MOTION_BLUR_ACTIVE){
                cudaMemcpyAsync(camera_open_buffer, gpu_validation_cameras, VALIDATION_BATCH_SIZE * sizeof(CAMERA_DATA), cudaMemcpyDeviceToDevice, optix_stream);
            }
            cudaMemcpyAsync(camera_buffer, gpu_validation_cameras, VALIDATION_BATCH_SIZE * sizeof(CAMERA_DATA), cudaMemcpyDeviceToDevice, optix_stream);
            rlt::render_rgb_only_launch(device, *renderer);
            {
                int total_scatter = VALIDATION_BATCH_SIZE * CAM_PIXELS;
                int pf_block = 256;
                int pf_grid = (total_scatter + pf_block - 1) / pf_block;
                scatter_pixel_to_float_kernel<false><<<pf_grid, pf_block, 0, optix_stream>>>(
                    fb_ptr, rlt::data(gpu_validation_observations), nullptr, 0, VALIDATION_BATCH_SIZE, CAM_PIXELS, OBSERVATION_DIM);
            }
            cudaEventRecord(validation_render_scatter_done_event, optix_stream);
            cudaStreamWaitEvent(device_gpu.stream, validation_render_scatter_done_event, 0);

            imitation_kernels::make_target_cameras_kernel<<<validation_grid, validation_block, 0, device_gpu.stream>>>(
                tag_device, gpu_validation_params_arr, gpu_validation_target_cameras,
                cam_aspect,
                gpu_validation_target_frame_roll_arr, gpu_validation_target_frame_pitch_arr,
                gpu_validation_scene_translation_arr, gpu_validation_scene_yaw_cos_arr, gpu_validation_scene_yaw_sin_arr);
            CUDA_CHECK("validation make_target_cameras_kernel");
            cudaEventRecord(validation_target_cameras_ready_event, device_gpu.stream);
            cudaStreamWaitEvent(optix_stream, validation_target_cameras_ready_event, 0);
            if constexpr(RENDER_MOTION_BLUR_ACTIVE){
                cudaMemcpyAsync(camera_open_buffer, gpu_validation_target_cameras, VALIDATION_BATCH_SIZE * sizeof(CAMERA_DATA), cudaMemcpyDeviceToDevice, optix_stream);
            }
            cudaMemcpyAsync(camera_buffer, gpu_validation_target_cameras, VALIDATION_BATCH_SIZE * sizeof(CAMERA_DATA), cudaMemcpyDeviceToDevice, optix_stream);
            rlt::render_rgb_only_launch(device, *renderer);
            {
                int total_scatter = VALIDATION_BATCH_SIZE * CAM_PIXELS;
                int pf_block = 256;
                int pf_grid = (total_scatter + pf_block - 1) / pf_block;
                scatter_pixel_to_float_kernel<false><<<pf_grid, pf_block, 0, optix_stream>>>(
                    fb_ptr, rlt::data(gpu_validation_target_observations), nullptr, 0, VALIDATION_BATCH_SIZE, CAM_PIXELS, OBSERVATION_DIM);
            }
            cudaEventRecord(validation_target_render_scatter_done_event, optix_stream);
            cudaStreamWaitEvent(device_gpu.stream, validation_target_render_scatter_done_event, 0);

            cudaMemset(rlt::data(gpu_validation_combined_observations), 0, static_cast<size_t>(BATCH_SIZE) * static_cast<size_t>(COMBINED_OBS_DIM) * sizeof(T_ACTIVATION));
            build_validation_combined_with_target_kernel<<<(VALIDATION_BATCH_SIZE * COMBINED_OBS_DIM + 255) / 256, 256, 0, device_gpu.stream>>>(
                rlt::data(gpu_validation_observations),
                rlt::data(gpu_validation_target_observations),
                rlt::data(gpu_validation_combined_observations),
                OBSERVATION_DIM, IMG_C, FRAME_STACK_N, COMBINED_IMG_C, COMBINED_OBS_DIM);
            CUDA_CHECK("validation build_combined");
            {
                auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_validation_combined_observations);
                rlt::forward(device_gpu, student_gpu, inputs, gpu_student_output_train, student_buffers, rng_gpu);
            }
            CUDA_CHECK("validation forward");
            imitation_kernels::yaw_batch_metrics_kernel<<<1, 1, 0, device_gpu.stream>>>(
                rlt::data(gpu_student_output_train),
                rlt::data(gpu_validation_targets),
                gpu_validation_metrics,
                VALIDATION_BATCH_SIZE);
            cudaMemcpy(cpu_validation_metrics.data(), gpu_validation_metrics, YAW_NUM_METRICS * sizeof(T), cudaMemcpyDeviceToHost);
            return cpu_validation_metrics;
        };
        for(TI validation_scene_i = 0; validation_scene_i < N_VALIDATION_SCENES; validation_scene_i++){
            const TI scene_i = N_TRAIN_SCENES + validation_scene_i;
            auto* renderer = renderers[scene_i];
            auto* scene = scenes[scene_i];
            OWLParams rgb_lp = (OWLParams)renderer->backend.launch_params;
            cudaStream_t optix_stream = (cudaStream_t)owlParamsGetCudaStream(rgb_lp, 0);
            void* camera_buffer = (void*)owlBufferGetPointer((OWLBuffer)renderer->backend.owl_cameras_buffer, 0);
            void* camera_open_buffer = nullptr;
            imitation_kernels::set_active_scene_camera_open_buffer<RENDER_MOTION_BLUR_ACTIVE>(camera_open_buffer, renderer);
            const uint32_t* fb_ptr = rlt::get_framebuffer_device_ptr(device, *renderer);
            for(TI yaw_bin_i = 0; yaw_bin_i < N_VALIDATION_YAW_BINS; yaw_bin_i++){
                const T yaw = validation_yaw_bins[yaw_bin_i];
                const T yaw_half = yaw / static_cast<T>(2);
                for(TI sample_i = 0; sample_i < VALIDATION_BATCH_SIZE; sample_i++){
                    cpu_validation_params[sample_i] = validation_base_parameters;
                    cpu_validation_states[sample_i] = {};
                    cpu_validation_states[sample_i].orientation[0] = std::cos(yaw_half);
                    cpu_validation_states[sample_i].orientation[1] = static_cast<T>(0);
                    cpu_validation_states[sample_i].orientation[2] = static_cast<T>(0);
                    cpu_validation_states[sample_i].orientation[3] = std::sin(yaw_half);
                    const TI pos_i = sample_i % scene->num_indoor_positions;
                    const auto& indoor_position = scene->indoor_positions[pos_i];
                    cpu_validation_scene_translation[sample_i * 3 + 0] = indoor_position.position[0];
                    cpu_validation_scene_translation[sample_i * 3 + 1] = indoor_position.position[1];
                    cpu_validation_scene_translation[sample_i * 3 + 2] = indoor_position.position[2];
                    const T scene_yaw = static_cast<T>(2) * rlt::math::PI<T>
                        * static_cast<T>((sample_i + validation_scene_i * 7 + yaw_bin_i * 13) % VALIDATION_BATCH_SIZE)
                        / static_cast<T>(VALIDATION_BATCH_SIZE);
                    cpu_validation_scene_yaw_cos[sample_i] = std::cos(scene_yaw);
                    cpu_validation_scene_yaw_sin[sample_i] = std::sin(scene_yaw);
                    cpu_validation_targets[sample_i * TARGET_DIM + 0] = static_cast<T_ACTIVATION>(std::cos(yaw));
                    cpu_validation_targets[sample_i * TARGET_DIM + 1] = static_cast<T_ACTIVATION>(std::sin(yaw));
                }
                auto metrics = evaluate_validation_batch(renderer, optix_stream, camera_buffer, camera_open_buffer, fb_ptr);
                const T weight = static_cast<T>(VALIDATION_BATCH_SIZE);
                summary.yaw_abs_error_rad += metrics[YAW_METRIC_ABS_ERROR_RAD] * weight;
                summary.yaw_mse_rad += metrics[YAW_METRIC_MSE_RAD] * weight;
                summary.yaw_output_norm += metrics[YAW_METRIC_OUTPUT_NORM] * weight;
                summary.yaw_null_mse_loss += metrics[YAW_METRIC_NULL_MSE_LOSS] * weight;
                summary.yaw_null_mse_rad += metrics[YAW_METRIC_NULL_MSE_RAD] * weight;
                summary.bin_yaw_mse_rad[yaw_bin_i] += metrics[YAW_METRIC_MSE_RAD] * weight;
                bin_weight[yaw_bin_i] += weight;
                total_weight += weight;
            }
            {
                const T frontier_min_rad = frontier_min_deg / static_cast<T>(180) * rlt::math::PI<T>;
                const T frontier_max_rad = frontier_max_deg / static_cast<T>(180) * rlt::math::PI<T>;
                for(TI sample_i = 0; sample_i < VALIDATION_BATCH_SIZE; sample_i++){
                    const T alpha = (static_cast<T>(sample_i / 2) + static_cast<T>(0.5)) / static_cast<T>(VALIDATION_BATCH_SIZE / 2);
                    const T yaw_abs = frontier_min_rad + (frontier_max_rad - frontier_min_rad) * alpha;
                    const T yaw = (sample_i % 2 == 0) ? -yaw_abs : yaw_abs;
                    const T yaw_half = yaw / static_cast<T>(2);
                    cpu_validation_params[sample_i] = validation_base_parameters;
                    cpu_validation_states[sample_i] = {};
                    cpu_validation_states[sample_i].orientation[0] = std::cos(yaw_half);
                    cpu_validation_states[sample_i].orientation[1] = static_cast<T>(0);
                    cpu_validation_states[sample_i].orientation[2] = static_cast<T>(0);
                    cpu_validation_states[sample_i].orientation[3] = std::sin(yaw_half);
                    const TI pos_i = sample_i % scene->num_indoor_positions;
                    const auto& indoor_position = scene->indoor_positions[pos_i];
                    cpu_validation_scene_translation[sample_i * 3 + 0] = indoor_position.position[0];
                    cpu_validation_scene_translation[sample_i * 3 + 1] = indoor_position.position[1];
                    cpu_validation_scene_translation[sample_i * 3 + 2] = indoor_position.position[2];
                    const T scene_yaw = static_cast<T>(2) * rlt::math::PI<T>
                        * static_cast<T>((sample_i + validation_scene_i * 17 + 19) % VALIDATION_BATCH_SIZE)
                        / static_cast<T>(VALIDATION_BATCH_SIZE);
                    cpu_validation_scene_yaw_cos[sample_i] = std::cos(scene_yaw);
                    cpu_validation_scene_yaw_sin[sample_i] = std::sin(scene_yaw);
                    cpu_validation_targets[sample_i * TARGET_DIM + 0] = static_cast<T_ACTIVATION>(std::cos(yaw));
                    cpu_validation_targets[sample_i * TARGET_DIM + 1] = static_cast<T_ACTIVATION>(std::sin(yaw));
                }
                auto metrics = evaluate_validation_batch(renderer, optix_stream, camera_buffer, camera_open_buffer, fb_ptr);
                const T weight = static_cast<T>(VALIDATION_BATCH_SIZE);
                summary.frontier_yaw_mse_rad += metrics[YAW_METRIC_MSE_RAD] * weight;
                frontier_weight += weight;
            }
        }
        if(total_weight > static_cast<T>(0)){
            summary.yaw_abs_error_rad /= total_weight;
            summary.yaw_mse_rad /= total_weight;
            summary.yaw_output_norm /= total_weight;
            summary.yaw_null_mse_loss /= total_weight;
            summary.yaw_null_mse_rad /= total_weight;
            summary.yaw_angle_r2_uniform = static_cast<T>(1) - summary.yaw_mse_rad / (summary.yaw_null_mse_rad > YAW_R2_EPS ? summary.yaw_null_mse_rad : YAW_R2_EPS);
        }
        for(TI yaw_bin_i = 0; yaw_bin_i < N_VALIDATION_YAW_BINS; yaw_bin_i++){
            if(bin_weight[yaw_bin_i] > static_cast<T>(0)){
                summary.bin_yaw_mse_rad[yaw_bin_i] /= bin_weight[yaw_bin_i];
            }
        }
        if(frontier_weight > static_cast<T>(0)){
            summary.frontier_yaw_mse_rad /= frontier_weight;
        }
        return summary;
    };

    // =========================================================================
    // Training loop
    // =========================================================================
    std::cout << "Starting yaw prediction training (visual L2F, RAPTOR rollout, CUDA)" << std::endl;
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_BLIND_TRAINING
    std::cout << "  [BLIND_TRAINING] enabled - visual observations zeroed at kernel level" << std::endl;
#endif
    std::cout << "  N_ENVIRONMENTS: " << N_ENVIRONMENTS << std::endl;
    std::cout << "  N_TRAIN_SCENES: " << N_TRAIN_SCENES << std::endl;
    std::cout << "  N_VALIDATION_SCENES: " << N_VALIDATION_SCENES << std::endl;
    std::cout << "  VALIDATION_CADENCE: " << VALIDATION_CADENCE << std::endl;
    std::cout << "  N_VALIDATION_YAW_BINS: " << N_VALIDATION_YAW_BINS << std::endl;
    std::cout << "  STEPS_PER_ENV: " << STEPS_PER_ENV << std::endl;
    std::cout << "  BATCH_SIZE: " << BATCH_SIZE << std::endl;
    std::cout << "  N_BATCHES: " << N_BATCHES << std::endl;
    std::cout << "  N_TRAIN_PASSES: " << N_TRAIN_PASSES << std::endl;
    std::cout << "  OBSERVATION_DIM (image): " << OBSERVATION_DIM << std::endl;
    std::cout << "  RAPTOR_OBS_DIM: " << RAPTOR_OBS_DIM << std::endl;
    std::cout << "  TARGET_DIM: " << TARGET_DIM << std::endl;
    std::cout << "  [YAW] targets are cos/sin yaw offsets from the target frame" << std::endl;
    std::cout << "  RAPTOR_TEACHER_FORCING_EPOCHS: " << TEACHER_FORCING_EPOCHS << std::endl;
    std::cout << "  RAPTOR_TEACHER_FORCING_FRACTION: " << EFFECTIVE_TEACHER_FORCING_FRACTION << std::endl;
    std::cout << "  FRAME_STACK_N: " << FRAME_STACK_N << std::endl;
    std::cout << "  FRAME_STACK_STRIDE: " << FRAME_STACK_STRIDE << " (" << (FRAME_STACK_STRIDE > 0 ? SIMULATION_FREQUENCY / FRAME_STACK_STRIDE : SIMULATION_FREQUENCY) << " Hz)" << std::endl;
    std::cout << "  STACKED_IMG_C: " << STACKED_IMG_C << std::endl;
    std::cout << "  COMBINED_IMG_C: " << COMBINED_IMG_C << std::endl;
    std::cout << "  RENDER_AA: " << (RENDER_ANTI_ALIASING_ACTIVE ? "on" : "off") << " grid=" << (RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : (TI)1) << std::endl;
    std::cout << "  RENDER_MOTION_BLUR: " << (RENDER_MOTION_BLUR_ACTIVE ? "on" : "off") << " samples=" << (RENDER_MOTION_BLUR_ACTIVE ? RENDER_MOTION_BLUR_SAMPLES : (TI)1) << " shutter=[" << RENDER_SHUTTER_FRACTION_MIN << ", " << RENDER_SHUTTER_FRACTION_MAX << "]" << std::endl;
    std::cout << "  TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE: "
              << TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE << " rad ("
              << TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE * static_cast<T>(180) / rlt::math::PI<T> << " deg)" << std::endl;
    std::cout << "  TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE: " << TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE << std::endl;
    std::cout << "  INIT_ORIENTATION_CURRICULUM: start=" << INIT_ORIENTATION_CURRICULUM_START_DEG
              << " deg step=" << INIT_ORIENTATION_CURRICULUM_STEP_DEG
              << " deg full=" << INIT_ORIENTATION_CURRICULUM_FULL_DEG
              << " deg frontier_threshold=clamp("
              << INIT_ORIENTATION_CURRICULUM_FRONTIER_RATIO
              << " * init_deg, "
              << INIT_ORIENTATION_CURRICULUM_FRONTIER_MIN_DEG
              << ", "
              << INIT_ORIENTATION_CURRICULUM_FRONTIER_MAX_DEG
              << ") deg" << std::endl;

    using NO_AUTO_RESET_MODE = rlt::Mode<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Default<>>>;
    NO_AUTO_RESET_MODE no_auto_reset_mode;

    auto training_start = std::chrono::high_resolution_clock::now();

    // Video recording buffers
    static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    static constexpr TI MOSAIC_W = SCENE_GRID_COLS * ENV_GRID_SIDE * CAM_WIDTH * 2;
    static constexpr TI MOSAIC_H = SCENE_GRID_ROWS * ENV_GRID_SIDE * CAM_HEIGHT;
    std::vector<uint32_t> video_pixel_buffer(N_ENVIRONMENTS * CAM_PIXELS);
    std::vector<uint8_t> mosaic_frame(MOSAIC_W * MOSAIC_H * 3);

    auto curriculum_step_limit = [](TI epoch) -> TI {
//        if(epoch < 100) return 50;
//        if(epoch < 1000) return 100;
//        if(epoch < 3000) return 200;
        return 500;
    };
    auto curriculum_frontier_threshold_deg = [](T init_orientation_max_deg) -> T {
        T threshold = INIT_ORIENTATION_CURRICULUM_FRONTIER_RATIO * init_orientation_max_deg;
        threshold = std::max(INIT_ORIENTATION_CURRICULUM_FRONTIER_MIN_DEG, threshold);
        threshold = std::min(INIT_ORIENTATION_CURRICULUM_FRONTIER_MAX_DEG, threshold);
        return threshold;
    };

    T current_init_orientation_max_deg = INIT_ORIENTATION_CURRICULUM_START_DEG;
    T current_init_orientation_max_rad = current_init_orientation_max_deg / static_cast<T>(180) * rlt::math::PI<T>;
    bool init_orientation_curriculum_full = current_init_orientation_max_deg >= INIT_ORIENTATION_CURRICULUM_FULL_DEG;

    for(TI epoch_i = 0; epoch_i < NUM_EPOCHS; epoch_i++){
        TI current_episode_step_limit = curriculum_step_limit(epoch_i);
        T epoch_init_orientation_max_deg = current_init_orientation_max_deg;
        T epoch_init_orientation_max_rad = current_init_orientation_max_rad;
        bool epoch_init_orientation_full = init_orientation_curriculum_full;
        T epoch_next_init_orientation_max_deg = epoch_init_orientation_full
            ? INIT_ORIENTATION_CURRICULUM_FULL_DEG
            : std::min(epoch_init_orientation_max_deg + INIT_ORIENTATION_CURRICULUM_STEP_DEG, INIT_ORIENTATION_CURRICULUM_FULL_DEG);
        T epoch_frontier_min_deg = epoch_init_orientation_full
            ? std::max(INIT_ORIENTATION_CURRICULUM_START_DEG, INIT_ORIENTATION_CURRICULUM_FULL_DEG - INIT_ORIENTATION_CURRICULUM_STEP_DEG)
            : epoch_init_orientation_max_deg;
        T epoch_frontier_max_deg = epoch_init_orientation_full ? INIT_ORIENTATION_CURRICULUM_FULL_DEG : epoch_next_init_orientation_max_deg;
        T epoch_frontier_threshold_deg = curriculum_frontier_threshold_deg(epoch_init_orientation_max_deg);
        auto epoch_start = std::chrono::high_resolution_clock::now();
        std::array<RENDERER_TYPE*, N_ACTIVE_SCENES> active_renderers{};
        std::array<cudaStream_t, N_ACTIVE_SCENES> active_scene_render_streams{};
        std::array<void*, N_ACTIVE_SCENES> active_scene_camera_buffers{};
        std::array<void*, N_ACTIVE_SCENES> active_scene_camera_open_buffers{};
        std::array<const uint32_t*, N_ACTIVE_SCENES> active_scene_framebuffer_ptrs{};
        std::shuffle(scene_permutation.begin(), scene_permutation.end(), scene_rng);
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            active_scene_indices[active_scene_i] = scene_permutation[active_scene_i];
        }
        upload_active_scenes();
        for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
            TI active_scene_i = env_i / N_ENVIRONMENTS_PER_SCENE;
            TI actual_scene_i = active_scene_indices[active_scene_i];
            envs[env_i].renderer = renderers[actual_scene_i];
            envs[env_i].scene = scenes[actual_scene_i];
        }
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            TI actual_scene_i = active_scene_indices[active_scene_i];
            auto* renderer = renderers[actual_scene_i];
            active_renderers[active_scene_i] = renderer;
            OWLParams rgb_lp = (OWLParams)renderer->backend.launch_params;
            active_scene_render_streams[active_scene_i] = (cudaStream_t)owlParamsGetCudaStream(rgb_lp, 0);
            active_scene_camera_buffers[active_scene_i] = (void*)owlBufferGetPointer((OWLBuffer)renderer->backend.owl_cameras_buffer, 0);
            imitation_kernels::set_active_scene_camera_open_buffer<RENDER_MOTION_BLUR_ACTIVE>(active_scene_camera_open_buffers[active_scene_i], renderer);
            active_scene_framebuffer_ptrs[active_scene_i] = rlt::get_framebuffer_device_ptr(device, *renderer);
        }
        {
            std::vector<unsigned char> reset_terminated(N_ENVIRONMENTS, 1);
            std::vector<TI> reset_step(N_ENVIRONMENTS, 0);
            std::vector<T> reset_return(N_ENVIRONMENTS, (T)0);
            std::vector<unsigned char> reset_teacher_forcing(N_ENVIRONMENTS, 1);
            cudaMemcpy(gpu_terminated_arr, reset_terminated.data(), N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_episode_step_arr, reset_step.data(), N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_episode_return_arr, reset_return.data(), N_ENVIRONMENTS * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_teacher_forcing_arr, reset_teacher_forcing.data(), N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
        }
        bool full_teacher_forcing = true;
        bool record_video = (epoch_i % CHECKPOINT_CADENCE == 0);
        bool record_trajectories = (epoch_i % CHECKPOINT_CADENCE == 0);
        if(record_trajectories){
            for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                episode_recorders[env_i].current_episode.clear();
                episode_recorders[env_i].episode_started = false;
            }
            completed_episodes.clear();
        }
        TI epoch_end_step = global_step + STEPS_PER_ENV * N_ENVIRONMENTS;
        FILE* ffmpeg_pipe = nullptr;
        if(record_video){
            auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, epoch_end_step);
            auto video_path = step_folder / "video.mp4";
            char ffmpeg_cmd[1024];
            snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
                "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %lux%lu -framerate %lu -i - "
                "-c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast -loglevel warning %s",
                (unsigned long)MOSAIC_W, (unsigned long)MOSAIC_H, (unsigned long)SIMULATION_FREQUENCY, video_path.c_str());
            ffmpeg_pipe = popen(ffmpeg_cmd, "w");
            if(!ffmpeg_pipe){
                std::cerr << "Failed to open ffmpeg pipe for " << video_path << std::endl;
                record_video = false;
            }
        }

        // =================================================================
        // Data collection (GPU-resident)
        // =================================================================
        float epoch_render_time_ms = 0;
        float epoch_render_gpu_time_ms = 0;
        float epoch_train_forward_time_ms = 0;
        float epoch_train_backward_time_ms = 0;
        float epoch_train_update_time_ms = 0;
        TI epoch_train_forward_calls = 0;
        TI epoch_train_backward_calls = 0;
        TI epoch_train_update_calls = 0;
        {
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = (N_ENVIRONMENTS + BLOCKSIZE - 1) / BLOCKSIZE;
            dim3 grid(N_BLOCKS);
            dim3 block(BLOCKSIZE);
            rlt::devices::cuda::TAG<DEVICE_GPU, true> tag_device{};
            auto& raptor_gru_layer = rlt::nn_models::sequential::layer<1>(raptor_gpu);
            auto& raptor_gru_state_content = rlt::nn_models::sequential::content_state<1>(raptor_state_gpu.content_state);
            T cam_aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
            for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
                imitation_kernels::prologue_kernel<<<grid, block, 0, device_gpu.stream>>>(
                    tag_device, gpu_envs_arr, gpu_params_arr, gpu_states_arr,
                    gpu_terminated_arr, gpu_episode_step_arr, gpu_teacher_forcing_arr,
                    gpu_episode_return_arr, gpu_needs_reset,
                    gpu_shutter_fraction_arr,
                    gpu_episode_lengths_log + step_i * N_ENVIRONMENTS,
                    gpu_episode_tf_log + step_i * N_ENVIRONMENTS,
                    gpu_episode_terminated_log + step_i * N_ENVIRONMENTS,
                    EFFECTIVE_TEACHER_FORCING_FRACTION, full_teacher_forcing,
                    rlt::data(gpu_teacher_obs),
                    rlt::data(raptor_gru_state_content.state),
                    rlt::data(raptor_gru_layer.initial_hidden_state.parameters),
                    rlt::data(raptor_gru_state_content.step),
                    gpu_episode_start_step,
                    gpu_brightness_scale_arr,
                    gpu_target_brightness_scale_arr,
                    gpu_target_frame_roll_arr,
                    gpu_target_frame_pitch_arr,
                    gpu_scene_translation_arr,
                    gpu_scene_yaw_arr,
                    gpu_scene_yaw_cos_arr,
                    gpu_scene_yaw_sin_arr,
                    gpu_indoor_positions, gpu_num_indoor_positions, gpu_env_scene, MAX_INDOOR_POS,
                    rng_gpu, step_i, current_episode_step_limit, epoch_init_orientation_max_rad);
                CUDA_CHECK("prologue_kernel");
                if(record_trajectories){
                    cudaMemcpyAsync(cpu_prestep_state_buf.data(), gpu_states_arr, TRAJECTORY_NUM_ENVS * sizeof(typename ENVIRONMENT::State), cudaMemcpyDeviceToHost, device_gpu.stream);
                    cudaMemcpyAsync(cpu_needs_reset_buf.data(), gpu_needs_reset, TRAJECTORY_NUM_ENVS * sizeof(bool), cudaMemcpyDeviceToHost, device_gpu.stream);
                    cudaMemcpyAsync(cpu_params_snapshot_buf.data(), gpu_params_arr, TRAJECTORY_NUM_ENVS * sizeof(typename ENVIRONMENT::Parameters), cudaMemcpyDeviceToHost, device_gpu.stream);
                }
                auto render_start = std::chrono::high_resolution_clock::now();
                imitation_kernels::make_cameras_kernel<RENDER_MOTION_BLUR_ACTIVE><<<grid, block, 0, device_gpu.stream>>>(
                    tag_device, gpu_params_arr, gpu_states_arr,
                    gpu_cameras,
                    gpu_cameras_open,
                    gpu_prev_cameras,
                    gpu_needs_reset,
                    gpu_shutter_fraction_arr,
                    step_i,
                    cam_aspect,
                    gpu_scene_translation_arr, gpu_scene_yaw_cos_arr, gpu_scene_yaw_sin_arr);
                CUDA_CHECK("make_cameras_kernel");
                T* obs_ptr = rlt::data(gpu_all_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
                cudaEventRecord(cameras_ready_event, device_gpu.stream);
                for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                    TI event_i = step_i * N_ACTIVE_SCENES + active_scene_i;
                    auto& renderer = *active_renderers[active_scene_i];
                    constexpr TI n_envs_s = N_ENVIRONMENTS_PER_SCENE;
                    TI base_env = active_scene_i * N_ENVIRONMENTS_PER_SCENE;
                    cudaStream_t optix_stream = active_scene_render_streams[active_scene_i];
                    cudaStreamWaitEvent(optix_stream, cameras_ready_event, 0);
                    if constexpr(RENDER_MOTION_BLUR_ACTIVE){
                        cudaMemcpyAsync(
                            active_scene_camera_open_buffers[active_scene_i],
                            gpu_cameras_open + base_env,
                            n_envs_s * sizeof(CAMERA_DATA),
                            cudaMemcpyDeviceToDevice, optix_stream);
                    }
                    cudaMemcpyAsync(
                        active_scene_camera_buffers[active_scene_i],
                        gpu_cameras + base_env,
                        n_envs_s * sizeof(CAMERA_DATA),
                        cudaMemcpyDeviceToDevice, optix_stream);
                    int total_scatter = n_envs_s * CAM_PIXELS;
                    int pf_block = 256;
                    int pf_grid = (total_scatter + pf_block - 1) / pf_block;
                    if(step_i % RENDER_TIMING_SAMPLE_PERIOD == 0){
                        cudaEventRecord(render_pass_start_events[event_i], optix_stream);
                    }
                    rlt::render_rgb_only_launch(device, renderer);
                    if(step_i % RENDER_TIMING_SAMPLE_PERIOD == 0){
                        cudaEventRecord(render_pass_stop_events[event_i], optix_stream);
                    }
                    const uint32_t* fb_ptr = active_scene_framebuffer_ptrs[active_scene_i];
                    if constexpr(OBSERVATION_NOISE_STD == 0 && BRIGHTNESS_RANDOMIZATION_RANGE > 0){
                        scatter_pixel_to_float_kernel<true><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, obs_ptr, gpu_brightness_scale_arr, base_env, n_envs_s, CAM_PIXELS, OBSERVATION_DIM);
                    } else {
                        scatter_pixel_to_float_kernel<false><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, obs_ptr, nullptr, base_env, n_envs_s, CAM_PIXELS, OBSERVATION_DIM);
                    }
                    cudaEventRecord(render_scatter_done_events[active_scene_i], optix_stream);
                }
                rlt::evaluate_step(device_gpu, raptor_gpu, gpu_teacher_obs, raptor_state_gpu, gpu_teacher_actions_step, raptor_buffer_gpu, rng_gpu, no_auto_reset_mode);
                CUDA_CHECK("raptor evaluate_step");
                for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                    cudaStreamWaitEvent(device_gpu.stream, render_scatter_done_events[active_scene_i], 0);
                }
                CUDA_CHECK("multi-scene render");
                if constexpr(OBSERVATION_NOISE_STD > 0){
                    constexpr TI NOISE_N = N_ENVIRONMENTS * OBSERVATION_DIM;
                    int noise_block = 256;
                    int noise_grid = (NOISE_N + noise_block - 1) / noise_block;
                    unsigned long long noise_seed = seed + (unsigned long long)epoch_i * STEPS_PER_ENV + step_i;
                    observation_noise_kernel<<<noise_grid, noise_block, 0, device_gpu.stream>>>(obs_ptr, NOISE_N, OBSERVATION_NOISE_STD, noise_seed);
                    CUDA_CHECK("observation_noise_kernel");
                }
                if constexpr(OBSERVATION_NOISE_STD > 0 && BRIGHTNESS_RANDOMIZATION_RANGE > 0){
                    int br_block = 256;
                    dim3 br_grid((OBSERVATION_DIM + br_block - 1) / br_block, N_ENVIRONMENTS);
                    brightness_apply_kernel<<<br_grid, br_block, 0, device_gpu.stream>>>(obs_ptr, gpu_brightness_scale_arr, N_ENVIRONMENTS, OBSERVATION_DIM);
                    CUDA_CHECK("brightness_apply_kernel");
                }
                // Target frame rendering (second pass)
                imitation_kernels::make_target_cameras_kernel<<<grid, block, 0, device_gpu.stream>>>(
                    tag_device, gpu_params_arr, gpu_target_cameras,
                    cam_aspect,
                    gpu_target_frame_roll_arr, gpu_target_frame_pitch_arr,
                    gpu_scene_translation_arr, gpu_scene_yaw_cos_arr, gpu_scene_yaw_sin_arr);
                CUDA_CHECK("make_target_cameras_kernel");
                T* target_obs_ptr = rlt::data(gpu_all_target_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
                cudaEventRecord(target_cameras_ready_event, device_gpu.stream);
                for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                    TI event_i = step_i * N_ACTIVE_SCENES + active_scene_i;
                    auto& renderer = *active_renderers[active_scene_i];
                    constexpr TI n_envs_s = N_ENVIRONMENTS_PER_SCENE;
                    TI base_env = active_scene_i * N_ENVIRONMENTS_PER_SCENE;
                    cudaStream_t optix_stream = active_scene_render_streams[active_scene_i];
                    cudaStreamWaitEvent(optix_stream, target_cameras_ready_event, 0);
                    if constexpr(RENDER_MOTION_BLUR_ACTIVE){
                        cudaMemcpyAsync(
                            active_scene_camera_open_buffers[active_scene_i],
                            gpu_target_cameras + base_env,
                            n_envs_s * sizeof(CAMERA_DATA),
                            cudaMemcpyDeviceToDevice, optix_stream);
                    }
                    cudaMemcpyAsync(
                        active_scene_camera_buffers[active_scene_i],
                        gpu_target_cameras + base_env,
                        n_envs_s * sizeof(CAMERA_DATA),
                        cudaMemcpyDeviceToDevice, optix_stream);
                    int total_scatter = n_envs_s * CAM_PIXELS;
                    int pf_block = 256;
                    int pf_grid = (total_scatter + pf_block - 1) / pf_block;
                    if(step_i % RENDER_TIMING_SAMPLE_PERIOD == 0){
                        cudaEventRecord(target_render_pass_start_events[event_i], optix_stream);
                    }
                    rlt::render_rgb_only_launch(device, renderer);
                    if(step_i % RENDER_TIMING_SAMPLE_PERIOD == 0){
                        cudaEventRecord(target_render_pass_stop_events[event_i], optix_stream);
                    }
                    const uint32_t* fb_ptr = active_scene_framebuffer_ptrs[active_scene_i];
                    if constexpr(BRIGHTNESS_RANDOMIZATION_RANGE > 0 || TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE > 0){
                        scatter_pixel_to_float_kernel<true><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, target_obs_ptr, gpu_target_brightness_scale_arr, base_env, n_envs_s, CAM_PIXELS, OBSERVATION_DIM);
                    } else {
                        scatter_pixel_to_float_kernel<false><<<pf_grid, pf_block, 0, optix_stream>>>(fb_ptr, target_obs_ptr, nullptr, base_env, n_envs_s, CAM_PIXELS, OBSERVATION_DIM);
                    }
                    cudaEventRecord(target_render_scatter_done_events[active_scene_i], optix_stream);
                }
                {
                    TI history_slot = step_i % FRAME_STACK_HISTORY_LENGTH;
                    T* history_slot_ptr = rlt::data(gpu_frame_stack_history) + (TI)(history_slot * N_ENVIRONMENTS) * OBSERVATION_DIM;
                    cudaMemcpyAsync(history_slot_ptr, obs_ptr, N_ENVIRONMENTS * OBSERVATION_DIM * sizeof(T), cudaMemcpyDeviceToDevice, device_gpu.stream);
                }
                for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                    cudaStreamWaitEvent(device_gpu.stream, target_render_scatter_done_events[active_scene_i], 0);
                }
                {
                    int total_elements = N_ENVIRONMENTS * COMBINED_OBS_DIM;
                    build_frame_stacked_with_target_from_history_kernel<<<(total_elements + 255) / 256, 256, 0, device_gpu.stream>>>(
                        rlt::data(gpu_frame_stack_history),
                        target_obs_ptr,
                        gpu_episode_start_step,
                        step_i,
                        rlt::data(gpu_all_combined_observations) + (TI)(step_i * N_ENVIRONMENTS) * COMBINED_OBS_DIM,
                        OBSERVATION_DIM, IMG_C, FRAME_STACK_N, FRAME_STACK_STRIDE, COMBINED_IMG_C, COMBINED_OBS_DIM, N_ENVIRONMENTS
                    );
                }
                CUDA_CHECK("target render");
                auto render_end = std::chrono::high_resolution_clock::now();
                epoch_render_time_ms += std::chrono::duration<float, std::milli>(render_end - render_start).count();
                if(record_video && ffmpeg_pipe){
                    static constexpr TI VIDEO_OBS_SIZE = N_ENVIRONMENTS * OBSERVATION_DIM;
                    std::vector<float> cpu_obs(VIDEO_OBS_SIZE);
                    std::vector<float> cpu_target_obs(VIDEO_OBS_SIZE);
                    cudaStreamSynchronize(device_gpu.stream);
                    cudaMemcpy(cpu_obs.data(), obs_ptr, VIDEO_OBS_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(cpu_target_obs.data(), target_obs_ptr, VIDEO_OBS_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
                    for(TI scene_row = 0; scene_row < SCENE_GRID_ROWS; scene_row++){
                        for(TI scene_col = 0; scene_col < SCENE_GRID_COLS; scene_col++){
                            TI active_scene_i = scene_row * SCENE_GRID_COLS + scene_col;
                            if(active_scene_i >= N_ACTIVE_SCENES){
                                continue;
                            }
                            for(TI local_row = 0; local_row < ENV_GRID_SIDE; local_row++){
                                for(TI local_col = 0; local_col < ENV_GRID_SIDE; local_col++){
                                    TI local_env = local_row * ENV_GRID_SIDE + local_col;
                                    TI env_i = active_scene_i * N_ENVIRONMENTS_PER_SCENE + local_env;
                                    const float* env_obs = cpu_obs.data() + env_i * OBSERVATION_DIM;
                                    const float* env_target_obs = cpu_target_obs.data() + env_i * OBSERVATION_DIM;
                                    TI cell_x = (scene_col * ENV_GRID_SIDE + local_col) * CAM_WIDTH * 2;
                                    TI cell_y = (scene_row * ENV_GRID_SIDE + local_row) * CAM_HEIGHT;
                                    for(TI py = 0; py < CAM_HEIGHT; py++){
                                        for(TI px = 0; px < CAM_WIDTH; px++){
                                            TI pixel_i = py * CAM_WIDTH + px;
                                            TI mosaic_y = cell_y + py;
                                            TI target_x = cell_x + px;
                                            TI actual_x = cell_x + CAM_WIDTH + px;
                                            TI target_idx = (mosaic_y * MOSAIC_W + target_x) * 3;
                                            TI actual_idx = (mosaic_y * MOSAIC_W + actual_x) * 3;
                                            mosaic_frame[target_idx + 0] = static_cast<uint8_t>(std::clamp(env_target_obs[pixel_i * 3 + 0] * 255.0f, 0.0f, 255.0f));
                                            mosaic_frame[target_idx + 1] = static_cast<uint8_t>(std::clamp(env_target_obs[pixel_i * 3 + 1] * 255.0f, 0.0f, 255.0f));
                                            mosaic_frame[target_idx + 2] = static_cast<uint8_t>(std::clamp(env_target_obs[pixel_i * 3 + 2] * 255.0f, 0.0f, 255.0f));
                                            mosaic_frame[actual_idx + 0] = static_cast<uint8_t>(std::clamp(env_obs[pixel_i * 3 + 0] * 255.0f, 0.0f, 255.0f));
                                            mosaic_frame[actual_idx + 1] = static_cast<uint8_t>(std::clamp(env_obs[pixel_i * 3 + 1] * 255.0f, 0.0f, 255.0f));
                                            mosaic_frame[actual_idx + 2] = static_cast<uint8_t>(std::clamp(env_obs[pixel_i * 3 + 2] * 255.0f, 0.0f, 255.0f));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    fwrite(mosaic_frame.data(), 1, mosaic_frame.size(), ffmpeg_pipe);
                }
                {
                    imitation_kernels::epilogue_kernel<<<grid, block, 0, device_gpu.stream>>>(
                        tag_device, gpu_envs_arr, gpu_params_arr, gpu_states_arr,
                        gpu_terminated_arr, gpu_episode_step_arr,
                        gpu_episode_return_arr,
                        rlt::data(gpu_teacher_actions_step),
                        rlt::data(gpu_all_targets),
                        rng_gpu, step_i);
                }
                CUDA_CHECK("epilogue_kernel");
                if(record_trajectories){
                    cudaMemcpyAsync(cpu_rollout_action_buf.data(), rlt::data(gpu_teacher_actions_step), TRAJECTORY_NUM_ENVS * ACTION_DIM * sizeof(T), cudaMemcpyDeviceToHost, device_gpu.stream);
                    cudaMemcpyAsync(cpu_terminated_buf.data(), gpu_terminated_arr, TRAJECTORY_NUM_ENVS * sizeof(bool), cudaMemcpyDeviceToHost, device_gpu.stream);
                    cudaStreamSynchronize(device_gpu.stream);
                    for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                        auto& rec = episode_recorders[env_i];
                        bool needs_reset = cpu_needs_reset_buf[env_i] != 0;
                        if(needs_reset && rec.episode_started){
                            if(completed_episodes.size() < TRAJECTORY_MAX_EPISODES){
                                completed_episodes.push_back({rec.parameters_snapshot, std::move(rec.current_episode)});
                            }
                            rec.current_episode.clear();
                            rec.episode_started = false;
                        }
                        if(completed_episodes.size() >= TRAJECTORY_MAX_EPISODES) continue;
                        if(!rec.episode_started){
                            rec.parameters_snapshot = cpu_params_snapshot_buf[env_i];
                            rec.episode_started = true;
                        }
                        TrajectoryStep ts;
                        ts.state = cpu_prestep_state_buf[env_i];
                        for(TI a = 0; a < ENVIRONMENT::ACTION_DIM; a++){
                            ts.actions[a] = (T)cpu_rollout_action_buf[env_i * ACTION_DIM + a];
                        }
                        ts.reward = 0;
                        ts.terminated = cpu_terminated_buf[env_i] != 0;
                        rec.current_episode.push_back(ts);
                    }
                }
                global_step += N_ENVIRONMENTS;
            }
        }
        if(record_trajectories){
            for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                auto& rec = episode_recorders[env_i];
                if(rec.episode_started && !rec.current_episode.empty()){
                    if(completed_episodes.size() < TRAJECTORY_MAX_EPISODES){
                        completed_episodes.push_back({rec.parameters_snapshot, std::move(rec.current_episode)});
                    }
                    rec.current_episode.clear();
                    rec.episode_started = false;
                }
            }
        }
        if(ffmpeg_pipe){ pclose(ffmpeg_pipe); ffmpeg_pipe = nullptr; }

        // =================================================================
        // Training (GPU)
        // =================================================================
        T epoch_loss = 0;
        T epoch_loss_sum = 0;
        TI epoch_loss_count = 0;
        T epoch_yaw_abs_error_rad = 0;
        T epoch_yaw_mse_rad = 0;
        T epoch_yaw_output_norm = 0;
        T epoch_yaw_null_mse_loss = 0;
        T epoch_yaw_null_mse_rad = 0;

        for(TI pass = 0; pass < N_TRAIN_PASSES; pass++){
            // Shuffle batch order
            TI batch_order[N_BATCHES];
            for(TI i = 0; i < N_BATCHES; i++) batch_order[i] = i;
            for(TI i = N_BATCHES - 1; i > 0; i--){
                TI j = rlt::random::uniform_int_distribution(device.random, (TI)0, i, rng);
                std::swap(batch_order[i], batch_order[j]);
            }

            for(TI batch_idx = 0; batch_idx < N_BATCHES; batch_idx++){
                TI batch_i = batch_order[batch_idx];
                TI batch_offset = batch_i * BATCH_SIZE;

                rlt::zero_gradient(device_gpu, student_gpu);

                // Student forward on GPU
                using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
                auto gpu_combined_batch = rlt::view_range(device_gpu, gpu_all_combined_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_combined_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_combined_batch, ACTOR_INPUT_SHAPE{});
                TI train_forward_call_i = epoch_train_forward_calls;
                cudaEventRecord(train_forward_start_events[train_forward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_combined_batch_reshaped); rlt::forward(device_gpu, student_gpu, inputs, gpu_student_output_train, student_buffers, rng_gpu); }
                cudaEventRecord(train_forward_stop_events[train_forward_call_i], device_gpu.stream);
                epoch_train_forward_calls++;

                // MSE loss gradient
                auto student_output_matrix = rlt::matrix_view(device_gpu, gpu_student_output_train);
                auto target_batch_tensor = rlt::view_range(device_gpu, gpu_all_targets, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto target_batch = rlt::matrix_view(device_gpu, target_batch_tensor);
                rlt::nn::loss_functions::mse::gradient(device_gpu, student_output_matrix, target_batch, gpu_d_action_train, (T)0.5);

                // Compute loss for logging
                if(pass == 0){
                    imitation_kernels::mse_batch_loss_kernel<<<1, 1, 0, device_gpu.stream>>>(
                        rlt::data(gpu_student_output_train),
                        rlt::data(gpu_all_targets) + batch_offset * TARGET_DIM,
                        gpu_logged_batch_losses + epoch_loss_count,
                        BATCH_SIZE * TARGET_DIM);
                    imitation_kernels::yaw_batch_metrics_kernel<<<1, 1, 0, device_gpu.stream>>>(
                        rlt::data(gpu_student_output_train),
                        rlt::data(gpu_all_targets) + batch_offset * TARGET_DIM,
                        gpu_logged_yaw_metrics + epoch_loss_count * YAW_NUM_METRICS,
                        BATCH_SIZE);
                    epoch_loss_count++;
                }

                // Student backward + Adam step
                auto gpu_d_action_tensor = rlt::to_tensor(device_gpu, gpu_d_action_train);
                auto gpu_d_action_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, TARGET_DIM>{});
                TI train_backward_call_i = epoch_train_backward_calls;
                cudaEventRecord(train_backward_start_events[train_backward_call_i], device_gpu.stream);
                { auto inputs = rlt::nn_models::parallel::pack_inputs(gpu_combined_batch_reshaped); rlt::backward(device_gpu, student_gpu, inputs, gpu_d_action_reshaped, student_buffers); }
                cudaEventRecord(train_backward_stop_events[train_backward_call_i], device_gpu.stream);
                epoch_train_backward_calls++;
                TI train_update_call_i = epoch_train_update_calls;
                cudaEventRecord(train_update_start_events[train_update_call_i], device_gpu.stream);
                rlt::step(device_gpu, optimizer_gpu, student_gpu);
                cudaEventRecord(train_update_stop_events[train_update_call_i], device_gpu.stream);
                epoch_train_update_calls++;
            }
        }
        if(epoch_train_update_calls > 0){
            cudaEventSynchronize(train_update_stop_events[epoch_train_update_calls - 1]);
        } else if(epoch_train_backward_calls > 0){
            cudaEventSynchronize(train_backward_stop_events[epoch_train_backward_calls - 1]);
        } else if(epoch_train_forward_calls > 0){
            cudaEventSynchronize(train_forward_stop_events[epoch_train_forward_calls - 1]);
        }
        for(TI call_i = 0; call_i < epoch_train_forward_calls; call_i++){
            float train_forward_time_ms = 0;
            cudaEventElapsedTime(&train_forward_time_ms, train_forward_start_events[call_i], train_forward_stop_events[call_i]);
            epoch_train_forward_time_ms += train_forward_time_ms;
        }
        for(TI call_i = 0; call_i < epoch_train_backward_calls; call_i++){
            float train_backward_time_ms = 0;
            cudaEventElapsedTime(&train_backward_time_ms, train_backward_start_events[call_i], train_backward_stop_events[call_i]);
            epoch_train_backward_time_ms += train_backward_time_ms;
        }
        for(TI call_i = 0; call_i < epoch_train_update_calls; call_i++){
            float train_update_time_ms = 0;
            cudaEventElapsedTime(&train_update_time_ms, train_update_start_events[call_i], train_update_stop_events[call_i]);
            epoch_train_update_time_ms += train_update_time_ms;
        }
        if(epoch_loss_count > 0){
            cudaMemcpy(cpu_logged_batch_losses.data(), gpu_logged_batch_losses, epoch_loss_count * sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(cpu_logged_yaw_metrics.data(), gpu_logged_yaw_metrics, epoch_loss_count * YAW_NUM_METRICS * sizeof(T), cudaMemcpyDeviceToHost);
            for(TI loss_i = 0; loss_i < epoch_loss_count; loss_i++){
                epoch_loss_sum += cpu_logged_batch_losses[loss_i];
                epoch_yaw_abs_error_rad += cpu_logged_yaw_metrics[loss_i * YAW_NUM_METRICS + YAW_METRIC_ABS_ERROR_RAD];
                epoch_yaw_mse_rad += cpu_logged_yaw_metrics[loss_i * YAW_NUM_METRICS + YAW_METRIC_MSE_RAD];
                epoch_yaw_output_norm += cpu_logged_yaw_metrics[loss_i * YAW_NUM_METRICS + YAW_METRIC_OUTPUT_NORM];
                epoch_yaw_null_mse_loss += cpu_logged_yaw_metrics[loss_i * YAW_NUM_METRICS + YAW_METRIC_NULL_MSE_LOSS];
                epoch_yaw_null_mse_rad += cpu_logged_yaw_metrics[loss_i * YAW_NUM_METRICS + YAW_METRIC_NULL_MSE_RAD];
            }
        }
        epoch_loss = epoch_loss_count > 0 ? epoch_loss_sum / epoch_loss_count : (T)0;
        if(epoch_loss_count > 0){
            epoch_yaw_abs_error_rad /= static_cast<T>(epoch_loss_count);
            epoch_yaw_mse_rad /= static_cast<T>(epoch_loss_count);
            epoch_yaw_output_norm /= static_cast<T>(epoch_loss_count);
            epoch_yaw_null_mse_loss /= static_cast<T>(epoch_loss_count);
            epoch_yaw_null_mse_rad /= static_cast<T>(epoch_loss_count);
        }
        if(epoch_train_forward_calls == 0 && epoch_train_backward_calls == 0 && epoch_train_update_calls == 0){
            cudaStreamSynchronize(device_gpu.stream);
        }
        TI render_timing_samples = 0;
        for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i += RENDER_TIMING_SAMPLE_PERIOD){
            float render_pass_time_ms = 0;
            float target_render_pass_time_ms = 0;
            for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
                TI event_i = step_i * N_ACTIVE_SCENES + active_scene_i;
                float scene_render_time_ms = 0;
                cudaEventElapsedTime(&scene_render_time_ms, render_pass_start_events[event_i], render_pass_stop_events[event_i]);
                render_pass_time_ms = std::max(render_pass_time_ms, scene_render_time_ms);
                float target_scene_render_time_ms = 0;
                cudaEventElapsedTime(&target_scene_render_time_ms, target_render_pass_start_events[event_i], target_render_pass_stop_events[event_i]);
                target_render_pass_time_ms = std::max(target_render_pass_time_ms, target_scene_render_time_ms);
            }
            epoch_render_gpu_time_ms += render_pass_time_ms + target_render_pass_time_ms;
            render_timing_samples++;
        }
        if(render_timing_samples > 0){
            epoch_render_gpu_time_ms *= static_cast<float>(STEPS_PER_ENV) / static_cast<float>(render_timing_samples);
        }
        imitation_kernels::reduce_episode_stats_kernel<<<1, 1, 0, device_gpu.stream>>>(
            gpu_episode_lengths_log,
            gpu_episode_tf_log,
            gpu_episode_terminated_log,
            gpu_episode_step_arr,
            gpu_teacher_forcing_arr,
            gpu_epoch_episode_stats
        );
        cudaMemcpy(cpu_epoch_episode_stats.data(), gpu_epoch_episode_stats, cpu_epoch_episode_stats.size() * sizeof(T), cudaMemcpyDeviceToHost);
        episode_length_sum_tf = cpu_epoch_episode_stats[0];
        episode_count_tf = static_cast<TI>(cpu_epoch_episode_stats[1]);
        episode_length_sum_student = cpu_epoch_episode_stats[2];
        episode_count_student = static_cast<TI>(cpu_epoch_episode_stats[3]);
        TI episode_count_terminated = static_cast<TI>(cpu_epoch_episode_stats[4]);
        T complete_episode_length_sum = cpu_epoch_episode_stats[5];
        TI complete_episode_count = static_cast<TI>(cpu_epoch_episode_stats[6]);
        TI episode_count_started = episode_count_tf + episode_count_student;
        T episode_terminated_share = episode_count_started > 0 ? static_cast<T>(episode_count_terminated) / static_cast<T>(episode_count_started) : (T)0;
        T complete_terminated_share = complete_episode_count > 0 ? static_cast<T>(episode_count_terminated) / static_cast<T>(complete_episode_count) : (T)0;
        T complete_episode_length = complete_episode_count > 0 ? complete_episode_length_sum / static_cast<T>(complete_episode_count) : (T)0;
        bool run_validation_epoch = (epoch_i % VALIDATION_CADENCE == 0);
        ValidationSummary validation_summary;
        if(run_validation_epoch){
            validation_summary = run_validation(epoch_frontier_min_deg, epoch_frontier_max_deg);
        }

        // Logging
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> training_elapsed = now - training_start;
        std::chrono::duration<T> epoch_elapsed = now - epoch_start;
        T mean_episode_length_tf = episode_count_tf > 0 ? episode_length_sum_tf / episode_count_tf : 0;
        T mean_episode_length_student = episode_count_student > 0 ? episode_length_sum_student / episode_count_student : 0;
        TI episode_count = episode_count_tf + episode_count_student;
        T mean_episode_length = episode_count > 0 ? (episode_length_sum_tf + episode_length_sum_student) / episode_count : 0;
        T fps = epoch_elapsed.count() > 0 ? static_cast<T>(STEPS_TOTAL) / epoch_elapsed.count() : 0;
        T render_time_s = static_cast<T>(epoch_render_time_ms) / static_cast<T>(1000);
        T render_gpu_time_s = static_cast<T>(epoch_render_gpu_time_ms) / static_cast<T>(1000);
        T render_fps = render_time_s > 0 ? static_cast<T>(STEPS_PER_ENV * N_ENVIRONMENTS) / render_time_s : 0;
        T render_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * render_time_s / epoch_elapsed.count() : 0;
        T render_gpu_fps = render_gpu_time_s > 0 ? static_cast<T>(STEPS_PER_ENV * N_ENVIRONMENTS) / render_gpu_time_s : 0;
        T render_gpu_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * render_gpu_time_s / epoch_elapsed.count() : 0;
        T train_forward_time_s = static_cast<T>(epoch_train_forward_time_ms) / static_cast<T>(1000);
        T train_backward_time_s = static_cast<T>(epoch_train_backward_time_ms) / static_cast<T>(1000);
        T train_update_time_s = static_cast<T>(epoch_train_update_time_ms) / static_cast<T>(1000);
        T train_forward_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * train_forward_time_s / epoch_elapsed.count() : 0;
        T train_backward_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * train_backward_time_s / epoch_elapsed.count() : 0;
        T train_update_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * train_update_time_s / epoch_elapsed.count() : 0;
        T train_forward_avg_ms = epoch_train_forward_calls > 0 ? static_cast<T>(epoch_train_forward_time_ms) / static_cast<T>(epoch_train_forward_calls) : 0;
        T train_backward_avg_ms = epoch_train_backward_calls > 0 ? static_cast<T>(epoch_train_backward_time_ms) / static_cast<T>(epoch_train_backward_calls) : 0;
        T train_update_avg_ms = epoch_train_update_calls > 0 ? static_cast<T>(epoch_train_update_time_ms) / static_cast<T>(epoch_train_update_calls) : 0;
        T yaw_mse_r2_null = static_cast<T>(1) - epoch_loss / (epoch_yaw_null_mse_loss > YAW_R2_EPS ? epoch_yaw_null_mse_loss : YAW_R2_EPS);
        T yaw_angle_r2_null = static_cast<T>(1) - epoch_yaw_mse_rad / (epoch_yaw_null_mse_rad > YAW_R2_EPS ? epoch_yaw_null_mse_rad : YAW_R2_EPS);
        T frontier_yaw_rmse_deg = run_validation_epoch
            ? sqrtf(validation_summary.frontier_yaw_mse_rad) * static_cast<T>(180) / rlt::math::PI<T>
            : static_cast<T>(0);
        bool init_orientation_curriculum_advanced = false;
        if(run_validation_epoch && !init_orientation_curriculum_full && std::isfinite(frontier_yaw_rmse_deg) && frontier_yaw_rmse_deg <= epoch_frontier_threshold_deg){
            current_init_orientation_max_deg = epoch_next_init_orientation_max_deg;
            current_init_orientation_max_rad = current_init_orientation_max_deg / static_cast<T>(180) * rlt::math::PI<T>;
            init_orientation_curriculum_full = current_init_orientation_max_deg >= INIT_ORIENTATION_CURRICULUM_FULL_DEG;
            init_orientation_curriculum_advanced = true;
        }

        std::cout << (full_teacher_forcing ? "[TF] " : "[TF=" + std::to_string((int)(EFFECTIVE_TEACHER_FORCING_FRACTION * 100)) + "%] ")
                  << "Epoch: " << std::setw(5) << epoch_i
                  << " MSE: " << std::setw(10) << std::setprecision(6) << std::fixed << epoch_loss
                  << " mean_ep_len: " << std::setw(6) << std::setprecision(1) << mean_episode_length
                  << " ep_limit: " << std::setw(3) << current_episode_step_limit
                  << " init_deg: " << std::setw(4) << std::setprecision(0) << epoch_init_orientation_max_deg
                  << " episodes: " << std::setw(5) << episode_count
                  << " term_share: " << std::setw(5) << std::setprecision(2) << std::fixed << episode_terminated_share
                  << " fps: " << std::setw(7) << std::setprecision(0) << fps
                  << " render: " << std::setw(5) << std::setprecision(1) << render_time_s << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << render_pct << "%"
                  << " " << std::setw(7) << std::setprecision(0) << render_fps << " fps)"
                  << " render_gpu: " << std::setw(5) << std::setprecision(1) << render_gpu_time_s << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << render_gpu_pct << "%"
                  << " " << std::setw(7) << std::setprecision(0) << render_gpu_fps << " fps)"
                  << " fwd: " << std::setw(5) << std::setprecision(1) << train_forward_time_s << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << train_forward_pct << "% "
                  << std::setw(6) << std::setprecision(3) << train_forward_avg_ms << "ms)"
                  << " bwd: " << std::setw(5) << std::setprecision(1) << train_backward_time_s << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << train_backward_pct << "% "
                  << std::setw(6) << std::setprecision(3) << train_backward_avg_ms << "ms)"
                  << " upd: " << std::setw(5) << std::setprecision(1) << train_update_time_s << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << train_update_pct << "% "
                  << std::setw(6) << std::setprecision(3) << train_update_avg_ms << "ms)"
                  << " epoch_time: " << std::setw(6) << std::setprecision(1) << epoch_elapsed.count() << "s"
                  << " total: " << std::setw(8) << std::setprecision(1) << training_elapsed.count() << "s";
        std::cout << " yaw_abs_deg: " << std::setw(7) << std::setprecision(2) << std::fixed
                  << epoch_yaw_abs_error_rad * static_cast<T>(180) / rlt::math::PI<T>
                  << " yaw_rmse_deg: " << std::setw(7) << std::setprecision(2) << std::fixed
                  << sqrtf(epoch_yaw_mse_rad) * static_cast<T>(180) / rlt::math::PI<T>
                  << " yaw_norm: " << std::setw(6) << std::setprecision(3) << std::fixed << epoch_yaw_output_norm
                  << " yaw_r2_mse: " << std::setw(7) << std::setprecision(3) << std::fixed << yaw_mse_r2_null
                  << " yaw_r2_ang: " << std::setw(7) << std::setprecision(3) << std::fixed << yaw_angle_r2_null;
        if(init_orientation_curriculum_advanced){
            std::cout << " init_deg_next: " << std::setw(4) << std::setprecision(0) << current_init_orientation_max_deg;
        }
        if(run_validation_epoch){
            std::cout << " val_yaw_rmse_deg: " << std::setw(7) << std::setprecision(2) << std::fixed
                      << sqrtf(validation_summary.yaw_mse_rad) * static_cast<T>(180) / rlt::math::PI<T>
                      << " val_yaw_r2: " << std::setw(7) << std::setprecision(3) << std::fixed
                      << validation_summary.yaw_angle_r2_uniform
                      << " frontier_rmse_deg: " << std::setw(7) << std::setprecision(2) << std::fixed
                      << frontier_yaw_rmse_deg
                      << " frontier_thr_deg: " << std::setw(5) << std::setprecision(2) << std::fixed
                      << epoch_frontier_threshold_deg;
        }
        std::cout << std::endl;

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
        rlt::set_step(device, device.logger, epoch_i);
        rlt::add_scalar(device, device.logger, "training/mse_loss", epoch_loss);
        rlt::add_scalar(device, device.logger, "training/yaw_abs_error_rad", epoch_yaw_abs_error_rad);
        rlt::add_scalar(device, device.logger, "training/yaw_abs_error_deg", epoch_yaw_abs_error_rad * static_cast<T>(180) / rlt::math::PI<T>);
        rlt::add_scalar(device, device.logger, "training/yaw_mse_rad", epoch_yaw_mse_rad);
        rlt::add_scalar(device, device.logger, "training/yaw_rmse_deg", sqrtf(epoch_yaw_mse_rad) * static_cast<T>(180) / rlt::math::PI<T>);
        rlt::add_scalar(device, device.logger, "training/yaw_output_norm", epoch_yaw_output_norm);
        rlt::add_scalar(device, device.logger, "training/yaw_null_mse_loss", epoch_yaw_null_mse_loss);
        rlt::add_scalar(device, device.logger, "training/yaw_null_mse_rad", epoch_yaw_null_mse_rad);
        rlt::add_scalar(device, device.logger, "training/yaw_mse_r2_null", yaw_mse_r2_null);
        rlt::add_scalar(device, device.logger, "training/yaw_angle_r2_null", yaw_angle_r2_null);
        if(run_validation_epoch){
            rlt::add_scalar(device, device.logger, "validation/yaw_abs_error_rad", validation_summary.yaw_abs_error_rad);
            rlt::add_scalar(device, device.logger, "validation/yaw_abs_error_deg", validation_summary.yaw_abs_error_rad * static_cast<T>(180) / rlt::math::PI<T>);
            rlt::add_scalar(device, device.logger, "validation/yaw_mse_rad", validation_summary.yaw_mse_rad);
            rlt::add_scalar(device, device.logger, "validation/yaw_rmse_deg", sqrtf(validation_summary.yaw_mse_rad) * static_cast<T>(180) / rlt::math::PI<T>);
            rlt::add_scalar(device, device.logger, "validation/yaw_output_norm", validation_summary.yaw_output_norm);
            rlt::add_scalar(device, device.logger, "validation/yaw_null_mse_loss", validation_summary.yaw_null_mse_loss);
            rlt::add_scalar(device, device.logger, "validation/yaw_null_mse_rad", validation_summary.yaw_null_mse_rad);
            rlt::add_scalar(device, device.logger, "validation/yaw_angle_r2_uniform", validation_summary.yaw_angle_r2_uniform);
            rlt::add_scalar(device, device.logger, "validation/frontier_yaw_mse_rad", validation_summary.frontier_yaw_mse_rad);
            rlt::add_scalar(device, device.logger, "validation/frontier_yaw_rmse_deg", frontier_yaw_rmse_deg);
            rlt::add_scalar(device, device.logger, "validation/frontier_min_deg", validation_summary.frontier_min_deg);
            rlt::add_scalar(device, device.logger, "validation/frontier_max_deg", validation_summary.frontier_max_deg);
            for(TI yaw_bin_i = 0; yaw_bin_i < N_VALIDATION_YAW_BINS; yaw_bin_i++){
                std::string tag = std::string("validation/yaw_rmse_deg/bin_") + validation_yaw_bin_names[yaw_bin_i];
                rlt::add_scalar(device, device.logger, tag.c_str(), sqrtf(validation_summary.bin_yaw_mse_rad[yaw_bin_i]) * static_cast<T>(180) / rlt::math::PI<T>);
            }
        }
        rlt::add_scalar(device, device.logger, "training/episode_length", mean_episode_length);
        rlt::add_scalar(device, device.logger, "training/episodes", static_cast<T>(episode_count));
        if(episode_count_tf > 0){
            rlt::add_scalar(device, device.logger, "training/teacher/episode_length", mean_episode_length_tf);
            rlt::add_scalar(device, device.logger, "training/teacher/episodes", static_cast<T>(episode_count_tf));
        }
        if(episode_count_student > 0){
            rlt::add_scalar(device, device.logger, "training/student/episode_length", mean_episode_length_student);
            rlt::add_scalar(device, device.logger, "training/student/episodes", static_cast<T>(episode_count_student));
        }
        rlt::add_scalar(device, device.logger, "timing/fps", fps);
        rlt::add_scalar(device, device.logger, "timing/throughput_fps", fps);
        rlt::add_scalar(device, device.logger, "timing/render_time_s", render_time_s);
        rlt::add_scalar(device, device.logger, "timing/render_fps", render_fps);
        rlt::add_scalar(device, device.logger, "timing/render_pct", render_pct);
        rlt::add_scalar(device, device.logger, "timing/render_gpu_time_s", render_gpu_time_s);
        rlt::add_scalar(device, device.logger, "timing/render_gpu_fps", render_gpu_fps);
        rlt::add_scalar(device, device.logger, "timing/render_gpu_pct", render_gpu_pct);
        rlt::add_scalar(device, device.logger, "timing/model_forward_time_s", train_forward_time_s);
        rlt::add_scalar(device, device.logger, "timing/model_backward_time_s", train_backward_time_s);
        rlt::add_scalar(device, device.logger, "timing/model_update_time_s", train_update_time_s);
        rlt::add_scalar(device, device.logger, "timing/model_forward_pct", train_forward_pct);
        rlt::add_scalar(device, device.logger, "timing/model_backward_pct", train_backward_pct);
        rlt::add_scalar(device, device.logger, "timing/model_update_pct", train_update_pct);
        rlt::add_scalar(device, device.logger, "timing/model_forward_avg_ms", train_forward_avg_ms);
        rlt::add_scalar(device, device.logger, "timing/model_backward_avg_ms", train_backward_avg_ms);
        rlt::add_scalar(device, device.logger, "timing/model_update_avg_ms", train_update_avg_ms);
        rlt::add_scalar(device, device.logger, "timing/epoch_time_s", epoch_elapsed.count());
        rlt::add_scalar(device, device.logger, "timing/total_time_s", training_elapsed.count());
        rlt::add_scalar(device, device.logger, "training/teacher_forcing", full_teacher_forcing ? (T)1 : EFFECTIVE_TEACHER_FORCING_FRACTION);
        rlt::add_scalar(device, device.logger, "training/terminated_share", episode_terminated_share);
        rlt::add_scalar(device, device.logger, "training/terminated_episodes", static_cast<T>(episode_count_terminated));
        rlt::add_scalar(device, device.logger, "training/complete_terminated_share", complete_terminated_share);
        rlt::add_scalar(device, device.logger, "training/complete_episode_length", complete_episode_length);
        rlt::add_scalar(device, device.logger, "training/complete_episodes", static_cast<T>(complete_episode_count));
        rlt::add_scalar(device, device.logger, "curriculum/episode_step_limit", static_cast<T>(current_episode_step_limit));
        rlt::add_scalar(device, device.logger, "curriculum/init_orientation_max_deg", epoch_init_orientation_max_deg);
        rlt::add_scalar(device, device.logger, "curriculum/init_orientation_next_deg", epoch_next_init_orientation_max_deg);
        rlt::add_scalar(device, device.logger, "curriculum/init_orientation_stage", (epoch_init_orientation_max_deg - INIT_ORIENTATION_CURRICULUM_START_DEG) / INIT_ORIENTATION_CURRICULUM_STEP_DEG);
        rlt::add_scalar(device, device.logger, "curriculum/init_orientation_full", epoch_init_orientation_full ? static_cast<T>(1) : static_cast<T>(0));
        rlt::add_scalar(device, device.logger, "curriculum/init_orientation_advance", init_orientation_curriculum_advanced ? static_cast<T>(1) : static_cast<T>(0));
        rlt::add_scalar(device, device.logger, "curriculum/frontier_yaw_rmse_threshold_deg", epoch_frontier_threshold_deg);
        rlt::add_scalar(device, device.logger, "curriculum/init_orientation_step_deg", INIT_ORIENTATION_CURRICULUM_STEP_DEG);
        rlt::add_scalar(device, device.logger, "rendering/anti_aliasing_grid_size", RENDER_ANTI_ALIASING_ACTIVE ? static_cast<T>(RENDER_ANTI_ALIASING_GRID_SIZE) : static_cast<T>(1));
        rlt::add_scalar(device, device.logger, "rendering/motion_blur_samples", RENDER_MOTION_BLUR_ACTIVE ? static_cast<T>(RENDER_MOTION_BLUR_SAMPLES) : static_cast<T>(1));
        rlt::add_scalar(device, device.logger, "rendering/target_frame_roll_pitch_randomization_range", TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE);
        rlt::add_scalar(device, device.logger, "rendering/target_frame_brightness_mismatch_range", TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_offset_randomization_range_x", CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_offset_randomization_range_y", CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_offset_randomization_range_z", CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_rotation_randomization_range_x", CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_rotation_randomization_range_y", CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y);
        rlt::add_scalar(device, device.logger, "rendering/camera_mount_rotation_randomization_range_z", CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z);
        if constexpr(RENDER_MOTION_BLUR_ACTIVE){
            rlt::add_scalar(device, device.logger, "rendering/shutter_fraction_min", RENDER_SHUTTER_FRACTION_MIN);
            rlt::add_scalar(device, device.logger, "rendering/shutter_fraction_max", RENDER_SHUTTER_FRACTION_MAX);
        }
#endif

        if(epoch_i % CHECKPOINT_CADENCE == 0){
            auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, epoch_end_step);
            using EVAL_TYPE = typename CPU_STUDENT_TYPE::template CHANGE_BATCH_SIZE<TI, N_EXAMPLES>;
            EVAL_TYPE eval_student;
            rlt::malloc(device, eval_student);
            rlt::copy(device_gpu, device, student_gpu, eval_student);
            char fov_buf[32];
            std::snprintf(fov_buf, sizeof(fov_buf), "%.6g", (double)envs[0].parameters.fov);
            std::string image_obs_string = std::string("CameraRGBStackedWithTarget(") + fov_buf + ", "
                + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ", "
                + std::to_string(FRAME_STACK_STRIDE) + ", " + std::to_string(FRAME_STACK_N) + ")";
            std::string obs_string = image_obs_string;
            std::string rendering_string = std::string("{\"anti_aliasing\": ") + (RENDER_ANTI_ALIASING_ACTIVE ? "true" : "false")
                + ", \"anti_aliasing_grid_size\": " + std::to_string(RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : (TI)1)
                + ", \"motion_blur\": " + (RENDER_MOTION_BLUR_ACTIVE ? "true" : "false")
                + ", \"motion_blur_samples\": " + std::to_string(RENDER_MOTION_BLUR_ACTIVE ? RENDER_MOTION_BLUR_SAMPLES : (TI)1)
                + ", \"shutter_fraction_min\": " + std::to_string(RENDER_SHUTTER_FRACTION_MIN)
                + ", \"shutter_fraction_max\": " + std::to_string(RENDER_SHUTTER_FRACTION_MAX)
                + ", \"target_frame_roll_pitch_randomization_range\": " + std::to_string(TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE)
                + ", \"target_frame_brightness_mismatch_range\": " + std::to_string(TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE)
                + ", \"camera_mount_offset_randomization_range\": [" + std::to_string(CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X)
                + ", " + std::to_string(CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y)
                + ", " + std::to_string(CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z) + "]"
                + ", \"camera_mount_rotation_randomization_range\": [" + std::to_string(CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X)
                + ", " + std::to_string(CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y)
                + ", " + std::to_string(CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z) + "]}";
            std::string output_string = "YawOffsetCosSin";
            std::string meta = "{\"environment\": {\"name\": \"l2f_visual\", \"observation\": \"" + obs_string + "\", \"output\": \"" + output_string + "\", \"rendering\": " + rendering_string + "}}";
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, N_EXAMPLES, IMG_H, IMG_W, COMBINED_IMG_C>, true>> example_input_0_image;
            rlt::malloc(device, example_input_0_image);
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, N_EXAMPLES, TARGET_DIM>, true>> example_output;
            rlt::malloc(device, example_output);
            {
                {
                    static constexpr TI EXAMPLE_ROW_OFFSET = STEPS_TOTAL - N_EXAMPLES;
                    static_assert(EXAMPLE_ROW_OFFSET >= (FRAME_STACK_N - 1) * FRAME_STACK_STRIDE * N_ENVIRONMENTS,
                        "N_EXAMPLES too large: sampled window would include steps with clamped frame-stack history");
                    auto src_combined = rlt::view_range(device_gpu, gpu_all_combined_observations, EXAMPLE_ROW_OFFSET, rlt::tensor::ViewSpec<0, N_EXAMPLES>{});
                    auto dst_image_2d = rlt::reshape_row_major(device, example_input_0_image, rlt::tensor::Shape<TI, N_EXAMPLES, COMBINED_OBS_DIM>{});
                    rlt::copy(device_gpu, device, src_combined, dst_image_2d);
                }
                typename EVAL_TYPE::template Buffer<true> eval_buffer;
                rlt::malloc(device, eval_buffer);
                rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
                auto inputs = rlt::nn_models::parallel::pack_inputs(example_input_0_image);
                rlt::evaluate(device, eval_student, inputs, example_output, eval_buffer, rng, eval_mode);
                rlt::free(device, eval_buffer);
            }
            if constexpr(EXPORT_CHECKPOINT_TAR){
                std::filesystem::path checkpoint_path = step_folder / "checkpoint.tar";
                rlt::persist::backends::tar::Writer writer;
                rlt::persist::backends::tar::WriterGroup<rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> root_group{"", &writer};
                auto actor_group = rlt::create_group(device, root_group, "actor");
                rlt::set_attribute(device, actor_group, "checkpoint_name", step_folder.string().c_str());
                rlt::set_attribute(device, actor_group, "meta", meta.c_str());
                rlt::save(device, eval_student, actor_group);
                auto example_group = rlt::create_group(device, root_group, "example");
                auto inputs_group = rlt::create_group(device, example_group, "inputs");
                rlt::save(device, example_input_0_image, inputs_group, "0");
                auto outputs_group = rlt::create_group(device, example_group, "outputs");
                rlt::save(device, example_output, outputs_group, "0");
                rlt::persist::backends::tar::finalize(device, writer);
                std::ofstream f(checkpoint_path, std::ios::binary);
                f.write(writer.buffer.data(), writer.buffer.size());
            }
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
            auto save_hdf5 = [&](auto batch_size_tag){ // binary (hdf5)
                static constexpr TI BATCH_SIZE = decltype(batch_size_tag)::value;
                using SIZED_EVAL_TYPE = typename CPU_STUDENT_TYPE::template CHANGE_BATCH_SIZE<TI, BATCH_SIZE>;
                SIZED_EVAL_TYPE sized_eval_student;
                rlt::malloc(device, sized_eval_student);
                rlt::copy(device_gpu, device, student_gpu, sized_eval_student);
                std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
                std::filesystem::path checkpoint_path = step_folder / (std::string("checkpoint_") + std::to_string(BATCH_SIZE) + "examples.h5");
                rlt::persist::backends::hdf5::File root_file(checkpoint_path.string(), rlt::persist::backends::hdf5::Mode::WRITE);
                auto actor_group = rlt::create_group(device, root_file, "actor");
                rlt::set_attribute(device, actor_group, "checkpoint_name", step_folder.string().c_str());
                rlt::set_attribute(device, actor_group, "meta", meta.c_str());
                rlt::save(device, sized_eval_student, actor_group);
                auto example_group = rlt::create_group(device, root_file, "example");
                auto inputs_group = rlt::create_group(device, example_group, "inputs");
                auto example_input_0_image_view = rlt::view_range(device, example_input_0_image, (TI)0, rlt::tensor::ViewSpec<1, BATCH_SIZE>{});
                rlt::save(device, example_input_0_image_view, inputs_group, "0");
                auto outputs_group = rlt::create_group(device, example_group, "outputs");
                auto example_output_view = rlt::view_range(device, example_output, (TI)0, rlt::tensor::ViewSpec<1, BATCH_SIZE>{});
                rlt::save(device, example_output_view, outputs_group, "0");
                rlt::free(device, sized_eval_student);
            };
            save_hdf5(rlt::utils::typing::integral_constant<TI, REDUCED_BATCH_SIZE>{});
            save_hdf5(rlt::utils::typing::integral_constant<TI, N_EXAMPLES>{});
#endif
            if constexpr(EXPORT_CHECKPOINT_CODE){
                auto actor_weights = rlt::save_code(device, eval_student, std::string("rl_tools::checkpoint::actor"), true);
                std::stringstream output_ss;
                output_ss << actor_weights;
                output_ss << "\n" << "namespace rl_tools::checkpoint::example::inputs{";
                output_ss << "\n" << rlt::save_code(device, example_input_0_image, std::string("_0"), true);
                output_ss << "\n" << "}";
                output_ss << "\n" << "namespace rl_tools::checkpoint::example::outputs{";
                output_ss << "\n" << rlt::save_code(device, example_output, std::string("_0"), true);
                output_ss << "\n" << "}";
                output_ss << "\n" << "namespace rl_tools::checkpoint::meta{";
                output_ss << "\n" << "   " << "char name[] = \"" << step_folder.string() << "\";";
                output_ss << "\n" << "   " << "char commit_hash[] = \"" << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
                output_ss << "\n" << "   " << "char observation[] = \"" << obs_string << "\";";
                output_ss << "\n" << "}";
                std::string output_string = output_ss.str();
#ifdef RL_TOOLS_ENABLE_ZLIB
                {
                    std::filesystem::path checkpoint_code_path = step_folder / "checkpoint.h.gz";
                    std::vector<uint8_t> compressed;
                    rlt::compress_zlib(output_string, compressed);
                    std::ofstream f(checkpoint_code_path, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
                }
#endif
                {
                    std::filesystem::path checkpoint_code_path = step_folder / "checkpoint.h";
                    std::ofstream f(checkpoint_code_path);
                    f << output_string;
                }
            }
            rlt::free(device, example_input_0_image);
            rlt::free(device, example_output);
            rlt::free(device, eval_student);
            {
                std::string trajectories_json = trajectory_episodes_to_json(device, envs[0], completed_episodes, simulation_dt);
#ifdef RL_TOOLS_ENABLE_ZLIB
                std::vector<uint8_t> compressed;
                if(rlt::compress_zlib(trajectories_json, compressed)){
                    std::filesystem::path trajectories_path = step_folder / "trajectories.json.gz";
                    std::ofstream f(trajectories_path, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
                } else {
                    std::cerr << "Failed to compress trajectories" << std::endl;
                }
#else
                std::filesystem::path trajectories_path = step_folder / "trajectories.json";
                std::ofstream f(trajectories_path);
                f << trajectories_json;
#endif
            }
            std::cerr << "Checkpoint saved: " << std::filesystem::absolute(step_folder) << std::endl;
        }

        episode_length_sum_tf = 0;
        episode_count_tf = 0;
        episode_length_sum_student = 0;
        episode_count_student = 0;
    }

    std::cout << "Training finished." << std::endl;

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
    rlt::free(device, device.logger);
#endif

    // =========================================================================
    // Cleanup
    // =========================================================================
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        envs[env_i].renderer = nullptr;
        envs[env_i].scene = nullptr;
    }
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::free(device, envs[env_i]);
    }
    for(TI scene_i = 0; scene_i < N_TOTAL_SCENES; scene_i++){
        if(renderers[scene_i] != nullptr){
            rlt::free(device, *renderers[scene_i]);
            delete renderers[scene_i];
            renderers[scene_i] = nullptr;
        }
        if(scenes[scene_i] != nullptr){
            delete scenes[scene_i];
            scenes[scene_i] = nullptr;
        }
    }

    rlt::free(device, raptor);
    rlt::free(device, raptor_buffer);
    rlt::free(device, raptor_state);
    rlt::free(device, teacher_obs);
    rlt::free(device, teacher_actions);
    rlt::free(device, cpu_all_teacher_actions);
    rlt::free(device, student_cpu);
    rlt::free(device, rng);

    rlt::free(device_gpu, rng_gpu);
    rlt::free(device_gpu, gpu_teacher_actions_step);
    cudaFree(gpu_cameras);
    if constexpr(RENDER_MOTION_BLUR_ACTIVE){
        cudaFree(gpu_cameras_open);
        cudaFree(gpu_prev_cameras);
    }
    cudaFree(gpu_target_cameras);
    cudaFree(gpu_validation_cameras);
    cudaFree(gpu_validation_target_cameras);
    cudaEventDestroy(cameras_ready_event);
    cudaEventDestroy(target_cameras_ready_event);
    cudaEventDestroy(validation_cameras_ready_event);
    cudaEventDestroy(validation_target_cameras_ready_event);
    cudaEventDestroy(validation_render_scatter_done_event);
    cudaEventDestroy(validation_target_render_scatter_done_event);
    for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
        cudaEventDestroy(render_scatter_done_events[active_scene_i]);
        cudaEventDestroy(target_render_scatter_done_events[active_scene_i]);
    }
    for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
        for(TI active_scene_i = 0; active_scene_i < N_ACTIVE_SCENES; active_scene_i++){
            TI event_i = step_i * N_ACTIVE_SCENES + active_scene_i;
            cudaEventDestroy(render_pass_start_events[event_i]);
            cudaEventDestroy(render_pass_stop_events[event_i]);
            cudaEventDestroy(target_render_pass_start_events[event_i]);
            cudaEventDestroy(target_render_pass_stop_events[event_i]);
        }
    }
    for(TI call_i = 0; call_i < max_train_timing_calls; call_i++){
        cudaEventDestroy(train_forward_start_events[call_i]);
        cudaEventDestroy(train_forward_stop_events[call_i]);
        cudaEventDestroy(train_backward_start_events[call_i]);
        cudaEventDestroy(train_backward_stop_events[call_i]);
        cudaEventDestroy(train_update_start_events[call_i]);
        cudaEventDestroy(train_update_stop_events[call_i]);
    }
    cudaFree(gpu_logged_batch_losses);
    cudaFree(gpu_logged_yaw_metrics);
    cudaFree(gpu_validation_metrics);
    rlt::free(device_gpu, student_gpu);
    rlt::free(device_gpu, raptor_gpu);
    rlt::free(device_gpu, raptor_buffer_gpu);
    rlt::free(device_gpu, raptor_state_gpu);
    rlt::free(device_gpu, gpu_teacher_obs);
    rlt::free(device_gpu, student_buffers);
    rlt::free(device_gpu, optimizer_gpu);
    rlt::free(device_gpu, gpu_all_observations);
    rlt::free(device_gpu, gpu_all_target_observations);
    rlt::free(device_gpu, gpu_all_targets);
    rlt::free(device_gpu, gpu_d_action_train);
    rlt::free(device_gpu, gpu_student_output_train);
    rlt::free(device_gpu, gpu_validation_observations);
    rlt::free(device_gpu, gpu_validation_target_observations);
    rlt::free(device_gpu, gpu_validation_combined_observations);
    rlt::free(device_gpu, gpu_validation_targets);
    cudaFree(gpu_envs_arr);
    cudaFree(gpu_params_arr);
    cudaFree(gpu_states_arr);
    cudaFree(gpu_validation_params_arr);
    cudaFree(gpu_validation_states_arr);
    cudaFree(gpu_terminated_arr);
    cudaFree(gpu_episode_step_arr);
    cudaFree(gpu_teacher_forcing_arr);
    cudaFree(gpu_episode_return_arr);
    cudaFree(gpu_needs_reset);
    cudaFree(gpu_episode_lengths_log);
    cudaFree(gpu_episode_tf_log);
    cudaFree(gpu_episode_terminated_log);
    cudaFree(gpu_epoch_episode_stats);
    cudaFree(gpu_brightness_scale_arr);
    cudaFree(gpu_target_brightness_scale_arr);
    cudaFree(gpu_target_frame_roll_arr);
    cudaFree(gpu_target_frame_pitch_arr);
    if constexpr(RENDER_MOTION_BLUR_ACTIVE){
        cudaFree(gpu_shutter_fraction_arr);
    }
    cudaFree(gpu_scene_translation_arr);
    cudaFree(gpu_scene_yaw_arr);
    cudaFree(gpu_scene_yaw_cos_arr);
    cudaFree(gpu_scene_yaw_sin_arr);
    cudaFree(gpu_validation_scene_translation_arr);
    cudaFree(gpu_validation_scene_yaw_cos_arr);
    cudaFree(gpu_validation_scene_yaw_sin_arr);
    cudaFree(gpu_validation_target_frame_roll_arr);
    cudaFree(gpu_validation_target_frame_pitch_arr);
    cudaFree(gpu_indoor_positions);
    cudaFree(gpu_num_indoor_positions);
    cudaFree(gpu_env_scene);
    rlt::free(device_gpu, gpu_frame_stack_history);
    rlt::free(device_gpu, gpu_all_combined_observations);
    cudaFree(gpu_episode_start_step);

    return 0;
}
