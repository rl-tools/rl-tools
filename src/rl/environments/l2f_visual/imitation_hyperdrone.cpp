// Visual imitation learning on the hyperdrone target-frame task: the device-generic port of
// imitation_cuda.cu. The host DEVICE owns dataset enumeration, extrack, artifacts and the
// episode accounting; DEVICE_COMPUTE runs the environment verbs, the RAPTOR teacher, the
// student and its training. When this file is compiled as CUDA (OptiX backend) DEVICE_COMPUTE
// is the CUDA device and the type policy keeps the original bf16 storage; otherwise (Metal,
// Vulkan, WebGPU, generic, or a CPU-compute build on a CUDA machine) it is the host device with
// fp32 everywhere. All data movement goes through rl_tools::copy, there are no kernels here.
#ifdef __CUDACC__
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#define RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
#endif
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#endif
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/unflatten/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
#include <rl_tools/nn/layers/standardize/operations_cuda.h>
#include <rl_tools/nn/layers/conv2d/operations_cuda.h>
#include <rl_tools/nn/layers/dense/operations_cuda.h>
#include <rl_tools/nn/layers/unflatten/operations_cuda.h>
#include <rl_tools/nn/layers/gru/helper_operations_cuda.h>
#endif
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
#include <rl_tools/nn_models/parallel/operations_cuda.h>
#endif
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>
#endif

#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cuda.h>
#endif
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
#include <rl_tools/rl/components/on_policy_runner/operations_cuda.h>
#endif
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>

#include <rl_tools/nn/loss_functions/mse/operations_generic.h>
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>
#endif

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

#include <metra/metra.h>

#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
#include <cuda_bf16.h>
#endif

#include <array>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <string>
#include <sstream>
#include <filesystem>
#include <fstream>

namespace rlt = rl_tools;
using rlt::prologue;
using rlt::epilogue;

#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_STATE_ESTIMATION
static constexpr bool STATE_ESTIMATION_MODE = true;
#else
static constexpr bool STATE_ESTIMATION_MODE = false;
#endif
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_BLIND_TRAINING
static constexpr bool BLIND_TRAINING = true;
#else
static constexpr bool BLIND_TRAINING = false;
#endif

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
using T = float;
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
using DEVICE_COMPUTE_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
using DEVICE_COMPUTE = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_COMPUTE_SPEC>;
using TYPE_POLICY = rlt::numeric_types::Policy<float,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Parameter, __nv_bfloat16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Activation, __nv_bfloat16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Gradient, __nv_bfloat16>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, float>>;
static constexpr const char* COMPUTE_DEVICE_NAME = "cuda";
#else
using DEVICE_COMPUTE = DEVICE;
using TYPE_POLICY = rlt::numeric_types::Policy<float>;
static constexpr const char* COMPUTE_DEVICE_NAME = "cpu";
#endif
using T_ACTIVATION = TYPE_POLICY::GET<rlt::numeric_types::categories::Activation>;
using T_GRADIENT = TYPE_POLICY::GET<rlt::numeric_types::categories::Gradient>;
using TEACHER_TYPE_POLICY = rlt::numeric_types::Policy<float>;
using TI = typename DEVICE::index_t;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_COMPUTE = typename DEVICE_COMPUTE::SPEC::RANDOM::ENGINE<>;

template <typename ANY_DEVICE>
void synchronize_compute(ANY_DEVICE&){}
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
template <typename CUDA_DEV_SPEC>
void synchronize_compute(rlt::devices::CUDA<CUDA_DEV_SPEC>& device){
    cudaDeviceSynchronize();
    rlt::check_status(device);
}
#endif

// =========================================================================
// L2F dynamics configuration (identical to imitation_cuda.cu)
// =========================================================================
namespace l2f = rlt::rl::environments::l2f;
namespace obs = l2f::observation;

using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
static constexpr TI SIMULATION_FREQUENCY = 100;
static constexpr TI EPISODE_STEP_LIMIT = 500;
using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
struct DOMAIN_RANDOMIZATION_OPTIONS {
    static constexpr bool THRUST_TO_WEIGHT = false;
    static constexpr bool MASS = false;
    static constexpr bool TORQUE_TO_INERTIA = false;
    static constexpr bool MASS_SIZE_DEVIATION = false;
    static constexpr bool ROTOR_TORQUE_CONSTANT = false;
    static constexpr bool DISTURBANCE_FORCE = false;
    static constexpr bool ROTOR_TIME_CONSTANT = false;
};
using PARAMETERS_TYPE = l2f::ParametersDomainRandomization<l2f::ParametersDomainRandomizationSpecification<T, TI, DOMAIN_RANDOMIZATION_OPTIONS, l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>>>;

static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie_openmv;

static constexpr REWARD_FUNCTION reward_function = {
    false, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    {0.00, 0.00, 0.00, 0.00}, {0.00, 0.00, 0.00, 0.00}, 0.00
};
static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
    1.0, 0.0, 0.3, 1.0, 1.0, true, -1, +1,
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    true, 1.0, 0, 10, 35, 10000, 50000,
};
static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::Integration integration = {
    static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)
};
static constexpr typename PARAMETERS_TYPE::MDP mdp = { init, reward_function, {}, {}, termination };
static constexpr T DISTURBANCE_FORCE_STD = 0;
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = { {0, DISTURBANCE_FORCE_STD}, {0, 0} };
static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static constexpr PARAMETERS_TYPE nominal_parameters = { {{dynamics, integration, mdp}, disturbances}, domain_randomization };

static constexpr TI ACTION_HISTORY_LENGTH = 8;

struct STATIC_PARAMETERS {
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI CLOSED_FORM = false;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, STATE_BASE>>>>>>>>;
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

using ACTOR_STATE_OBS = obs::OrientationWorldZ<obs::OrientationWorldZSpecification<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI, obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>;
static constexpr TI STATE_OBS_DIM = ACTOR_STATE_OBS::DIM;

// =========================================================================
// Scene / rollout profile
// =========================================================================
// RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_SMALL: functional smoke profile for machines without
// the memory or compute for the full epoch dataset (CPU-compute builds); the full profile is
// the imitation_cuda.cu configuration
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_SMALL
static constexpr TI N_TOTAL_SCENES = 2;
static constexpr TI N_ENVIRONMENTS_PER_SCENE = 16;
static constexpr TI STEPS_PER_ENV = 100;
static constexpr TI N_TRAIN_PASSES = 1;
static constexpr TI N_EXAMPLES = 64;
static constexpr TI ENV_GRID_SIDE = 4;
#else
static constexpr TI N_TOTAL_SCENES = 25;
static constexpr TI N_ENVIRONMENTS_PER_SCENE = 64;
static constexpr TI STEPS_PER_ENV = 500;
static constexpr TI N_TRAIN_PASSES = 4;
static constexpr TI N_EXAMPLES = 512;
static constexpr TI ENV_GRID_SIDE = 8;
#endif
static constexpr TI N_ACTIVE_SCENES = 2;
static constexpr TI N_ENVIRONMENTS = N_ACTIVE_SCENES * N_ENVIRONMENTS_PER_SCENE;
static constexpr TI CAM_WIDTH = 80;
static constexpr TI CAM_HEIGHT = 50;
static constexpr TI NUM_PROBES = 64;
static constexpr T CAMERA_FOV = 63.8;
static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = static_cast<T>(5.0) / static_cast<T>(180) * rlt::math::PI<T>;
static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE =
    STATE_ESTIMATION_MODE ? static_cast<T>(0) : static_cast<T>(10.0) / static_cast<T>(180) * rlt::math::PI<T>;
static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.5;
static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0.25;

using RENDER_SHADING = rlt::rendering::raytracing::High;
static constexpr bool RENDER_ENABLE_MOTION_BLUR = false;
static constexpr TI RENDER_MOTION_BLUR_SAMPLES = 1;
static constexpr bool RENDER_ENABLE_ANTI_ALIASING = true;
static constexpr TI RENDER_ANTI_ALIASING_GRID_SIZE = 2;
static constexpr T RENDER_SHUTTER_FRACTION_MIN = static_cast<T>(0.25);
static constexpr T RENDER_SHUTTER_FRACTION_MAX = static_cast<T>(1);

static constexpr TI FRAME_STACK_N = 10;
static constexpr TI FRAME_STACK_STRIDE = 10;
static constexpr TI FRAME_STACK_HISTORY_LENGTH = FRAME_STACK_STRIDE * (FRAME_STACK_N - 1) + 1;

struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, STATIC_PARAMETERS>{
    static constexpr TI INSTANCES_PER_ENVIRONMENT = N_ENVIRONMENTS_PER_SCENE;
    static constexpr TI CAM_WIDTH = ::CAM_WIDTH;
    static constexpr TI CAM_HEIGHT = ::CAM_HEIGHT;
    static constexpr TI NUM_PROBES = ::NUM_PROBES;
    using SHADING = RENDER_SHADING;
    static constexpr bool ENABLE_MOTION_BLUR = RENDER_ENABLE_MOTION_BLUR;
    static constexpr TI MOTION_BLUR_SAMPLES = RENDER_MOTION_BLUR_SAMPLES;
    static constexpr bool ENABLE_ANTI_ALIASING = RENDER_ENABLE_ANTI_ALIASING;
    static constexpr TI ANTI_ALIASING_GRID_SIZE = RENDER_ANTI_ALIASING_GRID_SIZE;
    static constexpr TI HISTORY_LENGTH = FRAME_STACK_HISTORY_LENGTH;
    static constexpr T CAMERA_FOV = ::CAMERA_FOV;
    static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = ::CAMERA_FOV_RANDOMIZATION_RANGE;
    static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = 0;
    static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = 0;
    static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = ::BRIGHTNESS_RANDOMIZATION_RANGE;
    static constexpr T SHUTTER_FRACTION_MIN = RENDER_SHUTTER_FRACTION_MIN;
    static constexpr T SHUTTER_FRACTION_MAX = RENDER_SHUTTER_FRACTION_MAX;
};
using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
static_assert(BASE_WORLD::RENDERER_SPEC::SHADING::PBR_SHADING, "l2f visual imitation must use the High renderer profile");
struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD>{
    static constexpr TI IMAGE_STACK_N = FRAME_STACK_N;
    static constexpr TI IMAGE_STACK_STRIDE = FRAME_STACK_STRIDE;
    static constexpr TI PAD_CHANNELS_TO = 8;
    static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = ::TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE;
    static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = ::TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE;
};
using TASK_WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
static constexpr TI NUMBER_OF_ENVIRONMENTS = N_ACTIVE_SCENES;
using MULTI_ENVIRONMENT = rlt::rl::environments::hyperdrone::MultiEnvironment<TASK_WORLD, NUMBER_OF_ENVIRONMENTS>;
using ENVIRONMENT = TASK_WORLD;
static constexpr bool RENDER_MOTION_BLUR_ACTIVE = BASE_WORLD::RENDERER_SPEC::ENABLE_MOTION_BLUR;
static constexpr bool RENDER_ANTI_ALIASING_ACTIVE = BASE_WORLD::RENDERER_SPEC::ENABLE_ANTI_ALIASING;
static_assert(TASK_WORLD::INSTANCES == N_ENVIRONMENTS_PER_SCENE);
static_assert(MULTI_ENVIRONMENT::INSTANCES == N_ENVIRONMENTS);
static constexpr TI OBSERVATION_DIM = BASE_WORLD::OBSERVATION_DIM;
static constexpr TI IMG_H = BASE_WORLD::Observation::HEIGHT;
static constexpr TI IMG_W = BASE_WORLD::Observation::WIDTH;
static constexpr TI IMG_C = BASE_WORLD::Observation::CHANNELS;
static constexpr TI COMBINED_IMG_C = TASK_WORLD::OBSERVATION_CHANNELS;
static constexpr TI COMBINED_OBS_DIM = TASK_WORLD::OBSERVATION_DIM;
static_assert(COMBINED_IMG_C == ((IMG_C * (FRAME_STACK_N + 1) + 7) & ~((TI)7)), "composed channel layout must match the imitation_cuda.cu padding");
static_assert(COMBINED_OBS_DIM == IMG_H * IMG_W * COMBINED_IMG_C);

// =========================================================================
// RAPTOR teacher
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
// Student and training configuration (identical to imitation_cuda.cu)
// =========================================================================
static constexpr TI ACTOR_HIDDEN_DIM = 64;
static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
static constexpr TI STATE_ESTIMATION_TARGET_DIM = 3 + 3 + 9;
static constexpr TI STATE_ESTIMATION_NUM_METRICS = 4;
static constexpr TI TARGET_DIM = STATE_ESTIMATION_MODE ? STATE_ESTIMATION_TARGET_DIM : ACTION_DIM;
static constexpr TI BATCH_SIZE = 512;
static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * N_ENVIRONMENTS;
static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
static constexpr TI NUM_EPOCHS = 1000000;
static constexpr T TEACHER_FORCING_FRACTION = 0.0;
static constexpr T EFFECTIVE_TEACHER_FORCING_FRACTION = STATE_ESTIMATION_MODE ? static_cast<T>(1) : TEACHER_FORCING_FRACTION;
// the episode statistics split teacher-forced and student episodes per mode, not per instance
static_assert(EFFECTIVE_TEACHER_FORCING_FRACTION == static_cast<T>(0) || EFFECTIVE_TEACHER_FORCING_FRACTION == static_cast<T>(1), "fractional teacher forcing needs a per-instance tag joined against the episode log");
static constexpr TI CHECKPOINT_CADENCE = 1000;
static constexpr bool EXPORT_CHECKPOINT_TAR = false;
static constexpr bool EXPORT_CHECKPOINT_CODE = false;
static constexpr TI REDUCED_BATCH_SIZE = 2;
static_assert(REDUCED_BATCH_SIZE <= N_EXAMPLES);
static constexpr TI SCENE_GRID_COLS = N_ACTIVE_SCENES;
static constexpr TI SCENE_GRID_ROWS = (N_ACTIVE_SCENES + SCENE_GRID_COLS - 1) / SCENE_GRID_COLS;
static_assert(ENV_GRID_SIDE * ENV_GRID_SIDE == N_ENVIRONMENTS_PER_SCENE, "N_ENVIRONMENTS_PER_SCENE must be a perfect square for the per-scene video mosaic");
static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");
static constexpr TI TRAJECTORY_NUM_ENVS = 10;
static constexpr TI TRAJECTORY_MAX_EPISODES = 10;
static_assert(TRAJECTORY_NUM_ENVS <= N_ENVIRONMENTS);

struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
    static constexpr T ALPHA = 3e-4;
    static constexpr T EPSILON = 1e-5;
    static constexpr T EPSILON_SQRT = 1e-5;
};

template<typename CAPABILITY, typename T_TYPE_POLICY = TYPE_POLICY>
struct StudentActor{
    static constexpr TI STEPS = 1;
    static constexpr TI FORWARD_BATCH_SIZE = BATCH_SIZE;
    using IMAGE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
    using STATE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, STATE_OBS_DIM>;

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

    using STATE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, TI>;
    using STATE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STATE_STANDARDIZE_CONFIG>;
    using STATE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using STATE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<STATE_DENSE_EMBED_CONFIG>;
    using STATE_BRANCH = rlt::nn_models::sequential::Module<STATE_STANDARDIZE, STATE_DENSE_EMBED>;

    using HEAD_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using HEAD_DENSE1 = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE1_CONFIG>;
    using HEAD_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using HEAD_DENSE2 = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE2_CONFIG>;
    using HEAD_DENSE_OUT_CONFIG = rlt::nn::layers::dense::Configuration<T_TYPE_POLICY, TI, TARGET_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using HEAD_DENSE_OUT = rlt::nn::layers::dense::BindConfiguration<HEAD_DENSE_OUT_CONFIG>;
    using SEQUENTIAL_HEAD = rlt::nn_models::sequential::Module<HEAD_DENSE1, HEAD_DENSE2, HEAD_DENSE_OUT>;
    using BRANCH_IMAGE = rlt::nn_models::parallel::Branch<IMAGE_BRANCH, IMAGE_INPUT_SHAPE>;
    using BRANCH_STATE = rlt::nn_models::parallel::Branch<STATE_BRANCH, STATE_INPUT_SHAPE>;
    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, SEQUENTIAL_HEAD, BRANCH_IMAGE, BRANCH_STATE>;
};

using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, true>;
using STUDENT_TYPE = typename StudentActor<CAPABILITY_ADAM>::MODEL;
using STUDENT_BUFFERS = typename STUDENT_TYPE::Buffer<true>;
using CAPABILITY_FORWARD_CPU = rlt::nn::capability::Forward<true>;
using CPU_STUDENT_TYPE = typename StudentActor<CAPABILITY_FORWARD_CPU, TEACHER_TYPE_POLICY>::MODEL;
using CPU_STUDENT_INIT_TYPE = typename StudentActor<CAPABILITY_ADAM, TEACHER_TYPE_POLICY>::MODEL;
using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, ADAM_PARAMETERS, true>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
using ROLLOUT_STUDENT_TYPE = typename STUDENT_TYPE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>;
// the batched on-policy runner collects the composed observations (student input, activation
// precision) and the teacher's privileged observations per row; the teacher's recurrent state is
// the rollout policy state (reset from the runner's episode accounting)
using RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<TYPE_POLICY, MULTI_ENVIRONMENT, typename RAPTOR_MODEL::State<true>, typename TASK_WORLD::Observation, RAPTOR_OBSERVATION_TYPE, T_ACTIVATION, T, EPISODE_STEP_LIMIT>;
using RUNNER = rlt::rl::components::OnPolicyRunner<RUNNER_SPEC>;
using RUNNER_BUFFER = rlt::rl::components::on_policy_runner::Buffer<RUNNER_SPEC>;
using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<RUNNER_SPEC, STEPS_PER_ENV>;
using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;
using EPISODE_STATISTICS = rlt::rl::components::on_policy_runner::EpisodeStatistics<T, TI>;
static_assert(RUNNER_SPEC::STEP_LIMIT == EPISODE_STEP_LIMIT);

// =========================================================================
// Trajectory recording for the extrack UI
// =========================================================================
struct TrajectoryStep {
    typename ENVIRONMENT::State state;
    T actions[ACTION_DIM];
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

std::string parameters_to_json(DEVICE& device, TASK_WORLD& world, const typename ENVIRONMENT::Parameters& parameters){
    std::string parameters_json = rlt::json(device, world.dynamics, parameters.dynamics);
    parameters_json = parameters_json.substr(0, parameters_json.size() - 1);
    parameters_json += ", \"visual\": {\"scene_translation\": [" + std::to_string(parameters.scene_translation[0]) + ", " + std::to_string(parameters.scene_translation[1]) + ", " + std::to_string(parameters.scene_translation[2]) + "]"
        + ", \"scene_yaw_cos\": " + std::to_string(parameters.scene_yaw_cos)
        + ", \"scene_yaw_sin\": " + std::to_string(parameters.scene_yaw_sin)
        + ", \"cam_width\": " + std::to_string(WORLD_SPEC::CAM_WIDTH)
        + ", \"cam_height\": " + std::to_string(WORLD_SPEC::CAM_HEIGHT)
        + ", \"fov\": " + std::to_string(parameters.fov) + "}}";
    return parameters_json;
}

std::string trajectory_episodes_to_json(DEVICE& device, TASK_WORLD& world, const std::vector<CompletedEpisode>& episodes, T dt){
    if(episodes.empty()) return "[]";
    std::string json = "[";
    for(TI ep_i = 0; ep_i < episodes.size(); ep_i++){
        auto& episode = episodes[ep_i];
        auto& parameters = episode.parameters;
        json += "{\"parameters\": " + parameters_to_json(device, world, parameters) + ",\n";
        json += "\"trajectory\": [";
        for(TI step_i = 0; step_i < episode.steps.size(); step_i++){
            auto& s = episode.steps[step_i];
            json += "{\"state\":" + rlt::json(device, world.dynamics, parameters.dynamics, s.state) + ",";
            json += "\"action\":[";
            for(TI a = 0; a < ACTION_DIM; a++){
                json += std::to_string(s.actions[a]);
                if(a < ACTION_DIM - 1) json += ",";
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

// state-estimation label: body-frame relative target position (target = origin), body-frame
// linear velocity, and the transposed body-to-world rotation matrix
void write_state_estimation_target(const typename ENVIRONMENT::State& state, T* target_ptr){
    T conjugate_orientation[4] = {state.orientation[0], -state.orientation[1], -state.orientation[2], -state.orientation[3]};
    T relative_position_world[3] = {-state.position[0], -state.position[1], -state.position[2]};
    T relative_position_body[3];
    l2f::rotate_vector_by_quaternion<DEVICE, T>(conjugate_orientation, relative_position_world, relative_position_body);
    T linear_velocity_body[3];
    l2f::rotate_vector_by_quaternion<DEVICE, T>(conjugate_orientation, state.linear_velocity, linear_velocity_body);
    for(TI axis_i = 0; axis_i < 3; axis_i++){
        target_ptr[axis_i] = relative_position_body[axis_i];
        target_ptr[3 + axis_i] = linear_velocity_body[axis_i];
    }
    T orientation_body_to_world[3][3];
    l2f::quaternion_to_rotation_matrix<DEVICE, T>(state.orientation, orientation_body_to_world);
    for(TI row_i = 0; row_i < 3; row_i++){
        for(TI col_i = 0; col_i < 3; col_i++){
            target_ptr[6 + row_i * 3 + col_i] = orientation_body_to_world[col_i][row_i];
        }
    }
}

template <typename OUTPUT, typename TARGETS>
T batch_mse(DEVICE& device, const OUTPUT& output, const TARGETS& targets){
    T acc = 0;
    for(TI sample_i = 0; sample_i < BATCH_SIZE; sample_i++){
        for(TI dim_i = 0; dim_i < TARGET_DIM; dim_i++){
            T diff = rlt::get(device, output, 0, sample_i, dim_i) - rlt::get(device, targets, sample_i, dim_i);
            acc += diff * diff;
        }
    }
    return acc / static_cast<T>(BATCH_SIZE * TARGET_DIM);
}

template <typename OUTPUT, typename TARGETS>
void state_estimation_batch_metrics(DEVICE& device, const OUTPUT& output, const TARGETS& targets, T metrics[STATE_ESTIMATION_NUM_METRICS]){
    T position_mse = 0, linear_velocity_mse = 0, orientation_mse = 0, orientation_angle_error = 0;
    for(TI sample_i = 0; sample_i < BATCH_SIZE; sample_i++){
        for(TI axis_i = 0; axis_i < 3; axis_i++){
            T diff_position = rlt::get(device, output, 0, sample_i, axis_i) - rlt::get(device, targets, sample_i, axis_i);
            T diff_linear_velocity = rlt::get(device, output, 0, sample_i, 3 + axis_i) - rlt::get(device, targets, sample_i, 3 + axis_i);
            position_mse += diff_position * diff_position;
            linear_velocity_mse += diff_linear_velocity * diff_linear_velocity;
        }
        T rotation_trace = 0;
        for(TI rotation_i = 0; rotation_i < 9; rotation_i++){
            T pred = rlt::get(device, output, 0, sample_i, 6 + rotation_i);
            T target = rlt::get(device, targets, sample_i, 6 + rotation_i);
            T diff_orientation = pred - target;
            orientation_mse += diff_orientation * diff_orientation;
            rotation_trace += pred * target;
        }
        T cos_angle = std::clamp((rotation_trace - static_cast<T>(1)) / static_cast<T>(2), static_cast<T>(-1), static_cast<T>(1));
        orientation_angle_error += std::acos(cos_angle);
    }
    metrics[0] = position_mse / static_cast<T>(BATCH_SIZE * 3);
    metrics[1] = linear_velocity_mse / static_cast<T>(BATCH_SIZE * 3);
    metrics[2] = orientation_mse / static_cast<T>(BATCH_SIZE * 9);
    metrics[3] = orientation_angle_error / static_cast<T>(BATCH_SIZE);
}

// latest rendered student frame and cached target frame of every instance of one World, pulled
// from the compute device into host staging
void fetch_world_frames(DEVICE& device, DEVICE_COMPUTE& device_compute, TASK_WORLD& world, float* frames, float* target_frames){
    constexpr TI M = TASK_WORLD::INSTANCES;
    const TI history_slot = (world.history_step - 1) % FRAME_STACK_HISTORY_LENGTH;
    {
        auto history_row = rlt::view(device_compute, world.history, history_slot);
        rlt::Tensor<rlt::tensor::Specification<float, TI, rlt::tensor::Shape<TI, M * BASE_WORLD::N_VIEWS, BASE_WORLD::FRAME_DIM>>> frame_alias;
        frame_alias._data = frames;
        rlt::copy(device_compute, device, history_row, frame_alias);
    }
    {
        rlt::Tensor<typename TASK_WORLD::TARGET_FRAMES_SPEC> target_alias;
        target_alias._data = target_frames;
        rlt::copy(device_compute, device, world.target_frames, target_alias);
    }
}

// =========================================================================
// Main
// =========================================================================
int main(int argc, char** argv){
    TI seed = 0;
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <scene_directory or scene.glb> [seed]" << std::endl;
        return 1;
    }
    if(argc > 2){
        seed = std::atoi(argv[2]);
    }

    // scene corpus: the first N_TOTAL_SCENES of a directory ordered by the trailing integer of
    // the stem (the imitation_cuda.cu rule), or one .glb replicated across all slots
    rlt::rendering::datasets::procthor::GLB scene_dataset{{}, {}};
    const char* scene_arg = argv[1];
    if(std::filesystem::is_directory(scene_arg)){
        std::vector<std::string> all_glbs;
        for(auto& entry : std::filesystem::directory_iterator(scene_arg)){
            if(entry.path().extension() == ".glb"){
                all_glbs.push_back(entry.path().string());
            }
        }
        auto extract_number = [](const std::string& path) -> int {
            auto filename = std::filesystem::path(path).stem().string();
            auto pos = filename.rfind('-');
            if(pos != std::string::npos){
                try { return std::stoi(filename.substr(pos + 1)); } catch(...) {}
            }
            return 0;
        };
        std::sort(all_glbs.begin(), all_glbs.end(), [&](const std::string& a, const std::string& b){
            return extract_number(a) < extract_number(b);
        });
        if(static_cast<TI>(all_glbs.size()) < N_TOTAL_SCENES){
            std::cerr << "Need at least " << N_TOTAL_SCENES << " GLB scenes, found " << all_glbs.size() << std::endl;
            return 1;
        }
        scene_dataset.references.assign(all_glbs.begin(), all_glbs.begin() + N_TOTAL_SCENES);
        std::cout << "Selected " << scene_dataset.references.size() << " scenes from " << scene_arg << std::endl;
        for(TI i = 0; i < scene_dataset.references.size(); i++){
            std::cout << "  [" << i << "] " << std::filesystem::path(scene_dataset.references[i]).filename().string() << std::endl;
        }
    } else {
        scene_dataset.references.assign(N_TOTAL_SCENES, scene_arg);
        std::cout << "Replicating single scene across " << N_TOTAL_SCENES << " renderers: " << scene_arg << std::endl;
    }

    DEVICE device;
    rlt::init(device);
#ifdef RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_COMPUTE_CUDA
    DEVICE_COMPUTE device_compute;
    rlt::init(device_compute);
#else
    DEVICE_COMPUTE& device_compute = device;
#endif

    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = STATE_ESTIMATION_MODE ? "l2f_visual_state_estimation_hyperdrone" : "l2f_visual_imitation_hyperdrone";
    rlt::init(device, extrack_config, extrack_paths, seed);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    RNG_COMPUTE rng_compute;
    rlt::malloc(device_compute, rng_compute);
    rlt::init(device_compute, rng_compute, seed);

    // ---------------------------------------------------------------------
    // RAPTOR teacher: host copy (weights) and compute copy (rollout)
    // ---------------------------------------------------------------------
    RAPTOR_MODEL raptor;
    rlt::malloc(device, raptor);
    rlt::copy(device, device, rl_tools::checkpoint::actor::module, raptor);

    RAPTOR_MODEL raptor_compute;
    typename RAPTOR_MODEL::Buffer<true> raptor_buffer_compute;
    RUNNER runner;
    RUNNER_BUFFER runner_buffer;
    rlt::malloc(device_compute, runner);
    rlt::malloc(device_compute, runner_buffer);
    auto& raptor_state_compute = runner.policy_state;
    rlt::malloc(device_compute, raptor_compute);
    rlt::malloc(device_compute, raptor_buffer_compute);
    rlt::copy(device, device_compute, raptor, raptor_compute);
    rlt::reset(device_compute, raptor_compute, raptor_state_compute, rng_compute);

    // ---------------------------------------------------------------------
    // Student: fp32 host init, then the compute copies
    // ---------------------------------------------------------------------
    CPU_STUDENT_INIT_TYPE student_cpu;
    rlt::malloc(device, student_cpu);
    rlt::init_weights(device, student_cpu, rng);

    STUDENT_TYPE student;
    STUDENT_BUFFERS student_buffers;
    OPTIMIZER optimizer;
    rlt::malloc(device_compute, student);
    rlt::malloc(device_compute, student_buffers);
    rlt::malloc(device_compute, optimizer);
    rlt::copy(device, device_compute, student_cpu, student);
    rlt::init(device_compute, optimizer);
    rlt::reset_optimizer_state(device_compute, optimizer, student);
    ROLLOUT_STUDENT_TYPE rollout_student;
    typename ROLLOUT_STUDENT_TYPE::template Buffer<true> rollout_student_buffers;
    rlt::malloc(device_compute, rollout_student);
    rlt::malloc(device_compute, rollout_student_buffers);
    rlt::copy(device, device_compute, student_cpu, rollout_student);

    // ---------------------------------------------------------------------
    // Environment: MultiEnvironment of target-frame Worlds over the scene corpus
    // ---------------------------------------------------------------------
    auto* env_storage = new MULTI_ENVIRONMENT{};
    MULTI_ENVIRONMENT& env = *env_storage;
    rlt::malloc(device_compute, env);
    typename decltype(scene_dataset)::Corpus scene_corpus;
    rlt::rendering::datasets::procthor::enumerate(device, scene_dataset, scene_corpus);
    if(static_cast<TI>(scene_corpus.references.size()) != N_TOTAL_SCENES){
        std::cerr << "Expected exactly " << N_TOTAL_SCENES << " scenes, got " << scene_corpus.references.size() << std::endl;
        return 1;
    }
    for(TI member_i = 0; member_i < NUMBER_OF_ENVIRONMENTS; member_i++){
        const TI first = member_i * N_TOTAL_SCENES / NUMBER_OF_ENVIRONMENTS;
        const TI last = (member_i + 1) * N_TOTAL_SCENES / NUMBER_OF_ENVIRONMENTS;
        std::cout << "Initializing World " << member_i << " with scenes [" << first << ", " << last << ")" << std::endl;
        rlt::init(device_compute, env.environments[member_i], env.shared, scene_dataset, scene_corpus, first, last - first, member_i);
        for(TI slot_i = 0; slot_i < last - first; slot_i++){
            std::cout << "  [" << first + slot_i << "] " << std::filesystem::path(scene_corpus.references[first + slot_i]).filename().string() << " — " << env.environments[member_i].slots[slot_i].annotations.num_positions << " indoor positions" << std::endl;
        }
    }
    std::cout << "Loaded " << N_TOTAL_SCENES << " scenes" << std::endl;

    {
        std::string ui = rlt::get_ui(device, env.environments[0].dynamics);
        if(!ui.empty()){
            std::filesystem::create_directories(extrack_paths.seed);
            std::ofstream ui_file(extrack_paths.seed / "ui.esm.js");
            ui_file << ui;
        }
    }

    // ---------------------------------------------------------------------
    // Per-instance data on the compute device
    // ---------------------------------------------------------------------
    DATASET dataset;
    auto& parameters = runner.env_parameters;
    auto& states = runner.states;
    auto& next_states = runner_buffer.next_states;
    auto& actions_step = runner_buffer.actions;
    auto& all_combined_observations = dataset.all_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ACTION_DIM>>> teacher_actions;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, TARGET_DIM>>> student_output_step;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, STATE_OBS_DIM>>> all_state_observations;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, TARGET_DIM>>> all_targets;
    rlt::Matrix<rlt::matrix::Specification<T_GRADIENT, TI, BATCH_SIZE, TARGET_DIM>> d_action_train;
    rlt::Tensor<rlt::tensor::Specification<T_ACTIVATION, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, TARGET_DIM>>> student_output_train;
    rlt::malloc(device_compute, teacher_actions);
    rlt::malloc(device_compute, student_output_step);
    rlt::malloc(device_compute, dataset);
    rlt::malloc(device_compute, all_state_observations);
    rlt::malloc(device_compute, all_targets);
    rlt::malloc(device_compute, d_action_train);
    rlt::malloc(device_compute, student_output_train);

    // ---------------------------------------------------------------------
    // Host-side staging
    // ---------------------------------------------------------------------
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS>>> reset_mask_host, terminated_host;
    rlt::Tensor<rlt::tensor::Specification<typename ENVIRONMENT::State, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS>>> states_host;
    rlt::Tensor<rlt::tensor::Specification<typename ENVIRONMENT::Parameters, TI, rlt::tensor::Shape<TI, TRAJECTORY_NUM_ENVS>>> parameters_trajectory_host;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TRAJECTORY_NUM_ENVS, ACTION_DIM>>> actions_trajectory_host;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, TARGET_DIM>>> targets_host;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH_SIZE, TARGET_DIM>>> student_output_train_host;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH_SIZE, TARGET_DIM>>> targets_batch_host;
    rlt::malloc(device, reset_mask_host);
    rlt::malloc(device, terminated_host);
    rlt::malloc(device, states_host);
    rlt::malloc(device, parameters_trajectory_host);
    rlt::malloc(device, actions_trajectory_host);
    rlt::malloc(device, targets_host);
    rlt::malloc(device, student_output_train_host);
    rlt::malloc(device, targets_batch_host);
    // Only the compact episode fields are mirrored for the host-side per-epoch summary; the
    // teacher's recurrent state resets from the runner's reset mask.
    RUNNER runner_host;
    DATASET dataset_host;
    rlt::malloc(device, runner_host.episode_step);
    rlt::malloc(device, dataset_host.episode_end_reason);
    rlt::malloc(device, dataset_host.episode_length);
    rlt::malloc(device, dataset_host.episode_return);
    auto reset_mask = rlt::matrix_view(device_compute, runner.reset);
    rlt::Mode<rlt::mode::sequential::ResetMask<rlt::mode::Default<>, rlt::mode::sequential::ResetMaskSpecification<decltype(reset_mask)>>> mode_reset_mask;
    mode_reset_mask.mask = reset_mask;
    using NO_AUTO_RESET_MODE = rlt::Mode<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Default<>>>;
    NO_AUTO_RESET_MODE no_auto_reset_mode;

    EpisodeRecorder episode_recorders[TRAJECTORY_NUM_ENVS];
    std::vector<CompletedEpisode> completed_episodes;
    T simulation_dt = static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY);
    TI global_step = 0;

    static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    static constexpr TI MOSAIC_W = SCENE_GRID_COLS * ENV_GRID_SIDE * CAM_WIDTH * 2;
    static constexpr TI MOSAIC_H = SCENE_GRID_ROWS * ENV_GRID_SIDE * CAM_HEIGHT;
    std::vector<float> video_frames(N_ENVIRONMENTS * OBSERVATION_DIM);
    std::vector<float> video_target_frames(N_ENVIRONMENTS * OBSERVATION_DIM);
    std::vector<uint8_t> mosaic_frame(MOSAIC_W * MOSAIC_H * 3);

    const std::string metra_prefix = "l2f_visual_imitation";
    const std::string target_name = STATE_ESTIMATION_MODE ? "l2f_visual_state_estimation_hyperdrone" : "l2f_visual_imitation_hyperdrone";
    metra::log_raw(metra_prefix + "/target", "\"" + target_name + "\"");
    metra::log_raw(metra_prefix + "/compute_device", std::string("\"") + COMPUTE_DEVICE_NAME + "\"");
    metra::log(metra_prefix + "/n_environments", static_cast<double>(N_ENVIRONMENTS));
    metra::log(metra_prefix + "/steps_per_env", static_cast<double>(STEPS_PER_ENV));
    metra::log(metra_prefix + "/seed", static_cast<double>(seed));

    std::cout << "Starting imitation learning (visual L2F hover, hyperdrone, compute=" << COMPUTE_DEVICE_NAME << ")" << std::endl;
    if constexpr(BLIND_TRAINING){
        std::cout << "  [BLIND_TRAINING] enabled - visual observations zeroed before the policy" << std::endl;
    }
    std::cout << "  N_TOTAL_SCENES: " << N_TOTAL_SCENES << std::endl;
    std::cout << "  N_ENVIRONMENTS: " << N_ENVIRONMENTS << std::endl;
    std::cout << "  STEPS_PER_ENV: " << STEPS_PER_ENV << std::endl;
    std::cout << "  BATCH_SIZE: " << BATCH_SIZE << std::endl;
    std::cout << "  N_BATCHES: " << N_BATCHES << std::endl;
    std::cout << "  N_TRAIN_PASSES: " << N_TRAIN_PASSES << std::endl;
    std::cout << "  OBSERVATION_DIM (image): " << OBSERVATION_DIM << std::endl;
    std::cout << "  STATE_OBS_DIM: " << STATE_OBS_DIM << std::endl;
    std::cout << "  RAPTOR_OBS_DIM: " << RAPTOR_OBS_DIM << std::endl;
    std::cout << "  TARGET_DIM: " << TARGET_DIM << std::endl;
    if constexpr(STATE_ESTIMATION_MODE){
        std::cout << "  [STATE_ESTIMATION] enabled - targets are relative position, linear velocity, and relative orientation in body frame" << std::endl;
    }
    std::cout << "  TEACHER_FORCING_FRACTION: " << EFFECTIVE_TEACHER_FORCING_FRACTION << std::endl;
    std::cout << "  FRAME_STACK_N: " << FRAME_STACK_N << std::endl;
    std::cout << "  FRAME_STACK_STRIDE: " << FRAME_STACK_STRIDE << " (" << SIMULATION_FREQUENCY / FRAME_STACK_STRIDE << " Hz)" << std::endl;
    std::cout << "  COMBINED_IMG_C: " << COMBINED_IMG_C << std::endl;
    std::cout << "  RENDER_AA: " << (RENDER_ANTI_ALIASING_ACTIVE ? "on" : "off") << " grid=" << (RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : (TI)1) << std::endl;
    std::cout << "  RENDER_MOTION_BLUR: " << (RENDER_MOTION_BLUR_ACTIVE ? "on" : "off") << " samples=" << (RENDER_MOTION_BLUR_ACTIVE ? RENDER_MOTION_BLUR_SAMPLES : (TI)1) << " shutter=[" << RENDER_SHUTTER_FRACTION_MIN << ", " << RENDER_SHUTTER_FRACTION_MAX << "]" << std::endl;
    std::cout << "  TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE: " << TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE << " rad (" << TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE * static_cast<T>(180) / rlt::math::PI<T> << " deg)" << std::endl;
    std::cout << "  TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE: " << TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE << std::endl;

    auto training_start = std::chrono::high_resolution_clock::now();
    const bool full_teacher_forcing = STATE_ESTIMATION_MODE;

    for(TI epoch_i = 0; epoch_i < NUM_EPOCHS; epoch_i++){
        auto epoch_start = std::chrono::high_resolution_clock::now();
        // scene set for this epoch: deterministic round-robin over each World's partition (the
        // original draws a random pair); every instance restarts on the new scene
        if(epoch_i > 0){
            rlt::rotate_scene(device_compute, env);
        }
        // epoch boundary: every instance restarts and the episodes cut by the boundary are
        // discarded (they were counted in the previous epoch's in-progress statistics)
        rlt::init(device_compute, runner, env, rng_compute);
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
            std::filesystem::create_directories(step_folder);
            auto video_path = step_folder / "video.mp4";
            char ffmpeg_cmd[1024];
            std::snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
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
        // Data collection
        // =================================================================
        // row 0 (composed observation + teacher input) is observed by the runner's prologue and
        // rows t + 1 by its epilogue
        prologue(device_compute, dataset, runner, env, rng_compute);
        for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
            // 1. the reset applied for this row (epoch start or the previous epilogue): teacher state
            rlt::reset(device_compute, raptor_compute, raptor_state_compute, rng_compute, mode_reset_mask);
            if(record_trajectories){
                rlt::copy(device_compute, device, runner.reset, reset_mask_host);
                rlt::copy(device_compute, device, states, states_host);
                auto parameters_trajectory_view = rlt::view_range(device_compute, parameters, (TI)0, rlt::tensor::ViewSpec<0, TRAJECTORY_NUM_ENVS>{});
                rlt::copy(device_compute, device, parameters_trajectory_view, parameters_trajectory_host);
            }

            // 2. the student's state branch and this row's runner observations
            auto state_observations_step = rlt::view_range(device_compute, all_state_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            rlt::observe(device_compute, env, parameters, states, ACTOR_STATE_OBS{}, state_observations_step, rng_compute);
            auto combined_observations_step = rlt::view_range(device_compute, all_combined_observations, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            auto teacher_observations = rlt::view_range(device_compute, dataset.all_observations_privileged, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            if constexpr(BLIND_TRAINING){
                rlt::set_all(device_compute, combined_observations_step, (T_ACTIVATION)0);
            }

            // 3. teacher
            rlt::evaluate_step(device_compute, raptor_compute, teacher_observations, raptor_state_compute, teacher_actions, raptor_buffer_compute, rng_compute, no_auto_reset_mode);

            // 4. video mosaic: (target | actual) per instance
            if(record_video && ffmpeg_pipe){
                synchronize_compute(device_compute);
                for(TI member_i = 0; member_i < NUMBER_OF_ENVIRONMENTS; member_i++){
                    fetch_world_frames(device, device_compute, env.environments[member_i], video_frames.data() + member_i * TASK_WORLD::INSTANCES * OBSERVATION_DIM, video_target_frames.data() + member_i * TASK_WORLD::INSTANCES * OBSERVATION_DIM);
                }
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
                                const float* env_obs = video_frames.data() + env_i * OBSERVATION_DIM;
                                const float* env_target_obs = video_target_frames.data() + env_i * OBSERVATION_DIM;
                                TI cell_x = (scene_col * ENV_GRID_SIDE + local_col) * CAM_WIDTH * 2;
                                TI cell_y = (scene_row * ENV_GRID_SIDE + local_row) * CAM_HEIGHT;
                                for(TI py = 0; py < CAM_HEIGHT; py++){
                                    for(TI px = 0; px < CAM_WIDTH; px++){
                                        TI pixel_i = py * CAM_WIDTH + px;
                                        TI mosaic_y = cell_y + py;
                                        TI target_idx = (mosaic_y * MOSAIC_W + cell_x + px) * 3;
                                        TI actual_idx = (mosaic_y * MOSAIC_W + cell_x + CAM_WIDTH + px) * 3;
                                        for(TI channel_i = 0; channel_i < 3; channel_i++){
                                            mosaic_frame[target_idx + channel_i] = static_cast<uint8_t>(std::clamp(env_target_obs[pixel_i * 3 + channel_i] * 255.0f, 0.0f, 255.0f));
                                            mosaic_frame[actual_idx + channel_i] = static_cast<uint8_t>(std::clamp(env_obs[pixel_i * 3 + channel_i] * 255.0f, 0.0f, 255.0f));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                fwrite(mosaic_frame.data(), 1, mosaic_frame.size(), ffmpeg_pipe);
            }

            // 5. student
            {
                auto state_observations_reshaped = rlt::reshape_row_major(device_compute, state_observations_step, rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, STATE_OBS_DIM>{});
                using ROLLOUT_IMG_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, IMG_H, IMG_W, COMBINED_IMG_C>;
                auto combined_observations_reshaped = rlt::reshape_row_major(device_compute, combined_observations_step, ROLLOUT_IMG_SHAPE{});
                auto inputs = rlt::nn_models::parallel::pack_inputs(combined_observations_reshaped, state_observations_reshaped);
                rlt::evaluate(device_compute, rollout_student, inputs, student_output_step, rollout_student_buffers, rng_compute);
            }

            // 6. labels and the action applied to the simulation
            auto targets_step = rlt::view_range(device_compute, all_targets, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            if constexpr(STATE_ESTIMATION_MODE){
                if(!record_trajectories){
                    rlt::copy(device_compute, device, states, states_host);
                }
                for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                    T target_row[TARGET_DIM];
                    write_state_estimation_target(rlt::get(device, states_host, env_i), target_row);
                    for(TI dim_i = 0; dim_i < TARGET_DIM; dim_i++){
                        rlt::set(device, targets_host, target_row[dim_i], env_i, dim_i);
                    }
                }
                rlt::copy(device, device_compute, targets_host, targets_step);
                rlt::copy(device_compute, device_compute, teacher_actions, actions_step);
            } else {
                rlt::copy(device_compute, device_compute, teacher_actions, targets_step);
                rlt::copy(device_compute, device_compute, student_output_step, actions_step);
            }

            // 7. environment step and episode accounting (the runner: step, reward, autoreset, next row)
            epilogue(device_compute, dataset, runner, runner_buffer, env, rng_compute, step_i);
            if(record_trajectories){
                rlt::copy(device_compute, device, runner_buffer.terminated, terminated_host);
                auto actions_trajectory_view = rlt::view_range(device_compute, actions_step, (TI)0, rlt::tensor::ViewSpec<0, TRAJECTORY_NUM_ENVS>{});
                rlt::copy(device_compute, device, actions_trajectory_view, actions_trajectory_host);
                for(TI env_i = 0; env_i < TRAJECTORY_NUM_ENVS; env_i++){
                    auto& rec = episode_recorders[env_i];
                    bool needs_reset = rlt::get(device, reset_mask_host, env_i);
                    if(needs_reset && rec.episode_started){
                        if(completed_episodes.size() < TRAJECTORY_MAX_EPISODES){
                            completed_episodes.push_back({rec.parameters_snapshot, std::move(rec.current_episode)});
                        }
                        rec.current_episode.clear();
                        rec.episode_started = false;
                    }
                    if(completed_episodes.size() >= TRAJECTORY_MAX_EPISODES) continue;
                    if(!rec.episode_started){
                        rec.parameters_snapshot = rlt::get(device, parameters_trajectory_host, env_i);
                        rec.episode_started = true;
                    }
                    TrajectoryStep ts;
                    ts.state = rlt::get(device, states_host, env_i);
                    for(TI a = 0; a < ACTION_DIM; a++){
                        ts.actions[a] = rlt::get(device, actions_trajectory_host, env_i, a);
                    }
                    ts.reward = 0;
                    ts.terminated = rlt::get(device, terminated_host, env_i);
                    rec.current_episode.push_back(ts);
                }
            }
            global_step += N_ENVIRONMENTS;
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
        synchronize_compute(device_compute);
        auto collection_end = std::chrono::high_resolution_clock::now();

        // =================================================================
        // Training
        // =================================================================
        T epoch_loss_sum = 0;
        TI epoch_loss_count = 0;
        T epoch_state_estimation_metrics[STATE_ESTIMATION_NUM_METRICS] = {0, 0, 0, 0};
        for(TI pass = 0; pass < N_TRAIN_PASSES; pass++){
            TI batch_order[N_BATCHES];
            for(TI i = 0; i < N_BATCHES; i++) batch_order[i] = i;
            for(TI i = N_BATCHES - 1; i > 0; i--){
                TI j = rlt::random::uniform_int_distribution(device.random, (TI)0, i, rng);
                std::swap(batch_order[i], batch_order[j]);
            }
            for(TI batch_idx = 0; batch_idx < N_BATCHES; batch_idx++){
                TI batch_i = batch_order[batch_idx];
                TI batch_offset = batch_i * BATCH_SIZE;
                rlt::zero_gradient(device_compute, student);
                using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
                auto combined_batch = rlt::view_range(device_compute, all_combined_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto combined_batch_reshaped = rlt::reshape_row_major(device_compute, combined_batch, ACTOR_INPUT_SHAPE{});
                auto state_batch = rlt::view_range(device_compute, all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto state_batch_reshaped = rlt::reshape_row_major(device_compute, state_batch, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});
                {
                    auto inputs = rlt::nn_models::parallel::pack_inputs(combined_batch_reshaped, state_batch_reshaped);
                    rlt::forward(device_compute, student, inputs, student_output_train, student_buffers, rng_compute);
                }
                auto student_output_matrix = rlt::matrix_view(device_compute, student_output_train);
                auto target_batch_tensor = rlt::view_range(device_compute, all_targets, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto target_batch = rlt::matrix_view(device_compute, target_batch_tensor);
                rlt::nn::loss_functions::mse::gradient(device_compute, student_output_matrix, target_batch, d_action_train, (T)0.5);
                if(pass == 0){
                    rlt::copy(device_compute, device, student_output_train, student_output_train_host);
                    rlt::copy(device_compute, device, target_batch_tensor, targets_batch_host);
                    epoch_loss_sum += batch_mse(device, student_output_train_host, targets_batch_host);
                    if constexpr(STATE_ESTIMATION_MODE){
                        T metrics[STATE_ESTIMATION_NUM_METRICS];
                        state_estimation_batch_metrics(device, student_output_train_host, targets_batch_host, metrics);
                        for(TI metric_i = 0; metric_i < STATE_ESTIMATION_NUM_METRICS; metric_i++){
                            epoch_state_estimation_metrics[metric_i] += metrics[metric_i];
                        }
                    }
                    epoch_loss_count++;
                }
                auto d_action_tensor = rlt::to_tensor(device_compute, d_action_train);
                auto d_action_reshaped = rlt::reshape_row_major(device_compute, d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, TARGET_DIM>{});
                {
                    auto inputs = rlt::nn_models::parallel::pack_inputs(combined_batch_reshaped, state_batch_reshaped);
                    rlt::backward(device_compute, student, inputs, d_action_reshaped, student_buffers);
                }
                rlt::step(device_compute, optimizer, student);
            }
        }
        T epoch_loss = epoch_loss_count > 0 ? epoch_loss_sum / static_cast<T>(epoch_loss_count) : (T)0;
        if(epoch_loss_count > 0){
            for(TI metric_i = 0; metric_i < STATE_ESTIMATION_NUM_METRICS; metric_i++){
                epoch_state_estimation_metrics[metric_i] /= static_cast<T>(epoch_loss_count);
            }
        }
        rlt::copy(device_compute, device_compute, student, rollout_student);
        synchronize_compute(device_compute);

        // =================================================================
        // Statistics and logging
        // =================================================================
        // finished episodes from the dataset; the episodes still in progress at the epoch end count once
        // with their current length (as in imitation_cuda.cu)
        EPISODE_STATISTICS statistics;
        rlt::copy(device_compute, device, dataset.episode_end_reason, dataset_host.episode_end_reason);
        rlt::copy(device_compute, device, dataset.episode_length, dataset_host.episode_length);
        rlt::copy(device_compute, device, dataset.episode_return, dataset_host.episode_return);
        rlt::copy(device_compute, device, runner.episode_step, runner_host.episode_step);
        rlt::summarize(device, dataset_host, runner_host, statistics);
        TI complete_episode_count = statistics.finished;
        TI episode_count_terminated = statistics.terminated;
        TI episode_count = statistics.finished + statistics.in_progress;
        TI episode_count_tf = full_teacher_forcing ? episode_count : 0;
        TI episode_count_student = full_teacher_forcing ? 0 : episode_count;
        T episode_terminated_share = episode_count > 0 ? static_cast<T>(episode_count_terminated) / static_cast<T>(episode_count) : (T)0;
        T complete_terminated_share = statistics.terminated_share;
        T complete_episode_length = statistics.mean_length;
        T mean_episode_length = episode_count > 0 ? (statistics.length_sum + statistics.in_progress_length_sum) / static_cast<T>(episode_count) : (T)0;
        T mean_episode_length_tf = full_teacher_forcing ? mean_episode_length : (T)0;
        T mean_episode_length_student = full_teacher_forcing ? (T)0 : mean_episode_length;

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> training_elapsed = now - training_start;
        std::chrono::duration<T> epoch_elapsed = now - epoch_start;
        std::chrono::duration<T> collection_elapsed = collection_end - epoch_start;
        std::chrono::duration<T> train_elapsed = now - collection_end;
        T fps = epoch_elapsed.count() > 0 ? static_cast<T>(STEPS_TOTAL) / epoch_elapsed.count() : 0;
        T collection_fps = collection_elapsed.count() > 0 ? static_cast<T>(STEPS_TOTAL) / collection_elapsed.count() : 0;
        T collection_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * collection_elapsed.count() / epoch_elapsed.count() : 0;
        T train_pct = epoch_elapsed.count() > 0 ? static_cast<T>(100) * train_elapsed.count() / epoch_elapsed.count() : 0;
        T epoch_orientation_angle_error_deg = epoch_state_estimation_metrics[3] * static_cast<T>(180) / rlt::math::PI<T>;

        std::cout << (full_teacher_forcing ? "[TF] " : "[TF=" + std::to_string((int)(EFFECTIVE_TEACHER_FORCING_FRACTION * 100)) + "%] ")
                  << "Epoch: " << std::setw(5) << epoch_i
                  << " MSE: " << std::setw(10) << std::setprecision(6) << std::fixed << epoch_loss
                  << " mean_ep_len: " << std::setw(6) << std::setprecision(1) << mean_episode_length
                  << " ep_limit: " << std::setw(3) << runner.episode_step_limit
                  << " episodes: " << std::setw(5) << episode_count
                  << " term_share: " << std::setw(5) << std::setprecision(2) << std::fixed << episode_terminated_share
                  << " fps: " << std::setw(7) << std::setprecision(0) << fps
                  << " collect: " << std::setw(6) << std::setprecision(1) << collection_elapsed.count() << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << collection_pct << "%"
                  << " " << std::setw(7) << std::setprecision(0) << collection_fps << " fps)"
                  << " train: " << std::setw(6) << std::setprecision(1) << train_elapsed.count() << "s"
                  << " (" << std::setw(4) << std::setprecision(1) << train_pct << "%)"
                  << " epoch_time: " << std::setw(6) << std::setprecision(1) << epoch_elapsed.count() << "s"
                  << " total: " << std::setw(8) << std::setprecision(1) << training_elapsed.count() << "s";
        if constexpr(STATE_ESTIMATION_MODE){
            std::cout << " pos_mse: " << std::setw(10) << std::setprecision(6) << std::fixed << epoch_state_estimation_metrics[0]
                      << " vel_mse: " << std::setw(10) << std::setprecision(6) << std::fixed << epoch_state_estimation_metrics[1]
                      << " ori_mse: " << std::setw(10) << std::setprecision(6) << std::fixed << epoch_state_estimation_metrics[2]
                      << " ori_deg: " << std::setw(7) << std::setprecision(2) << std::fixed << epoch_orientation_angle_error_deg;
        }
        std::cout << std::endl;

        metra::log(metra_prefix + "/mse_loss", static_cast<double>(epoch_loss));
        metra::log(metra_prefix + "/episode_length", static_cast<double>(mean_episode_length));
        metra::log(metra_prefix + "/episodes", static_cast<double>(episode_count));
        metra::log(metra_prefix + "/terminated_share", static_cast<double>(episode_terminated_share));
        metra::log(metra_prefix + "/complete_episode_length", static_cast<double>(complete_episode_length));
        metra::log(metra_prefix + "/complete_terminated_share", static_cast<double>(complete_terminated_share));
        metra::log(metra_prefix + "/fps", static_cast<double>(fps));
        metra::log(metra_prefix + "/epoch_time_s", static_cast<double>(epoch_elapsed.count()));
        if constexpr(STATE_ESTIMATION_MODE){
            metra::log(metra_prefix + "/state_estimation/position_mse", static_cast<double>(epoch_state_estimation_metrics[0]));
            metra::log(metra_prefix + "/state_estimation/linear_velocity_mse", static_cast<double>(epoch_state_estimation_metrics[1]));
            metra::log(metra_prefix + "/state_estimation/orientation_mse", static_cast<double>(epoch_state_estimation_metrics[2]));
            metra::log(metra_prefix + "/state_estimation/orientation_angle_error_deg", static_cast<double>(epoch_orientation_angle_error_deg));
        }

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
        rlt::set_step(device, device.logger, epoch_i);
        rlt::add_scalar(device, device.logger, "training/mse_loss", epoch_loss);
        if constexpr(STATE_ESTIMATION_MODE){
            rlt::add_scalar(device, device.logger, "training/state_estimation/position_mse", epoch_state_estimation_metrics[0]);
            rlt::add_scalar(device, device.logger, "training/state_estimation/linear_velocity_mse", epoch_state_estimation_metrics[1]);
            rlt::add_scalar(device, device.logger, "training/state_estimation/orientation_mse", epoch_state_estimation_metrics[2]);
            rlt::add_scalar(device, device.logger, "training/state_estimation/orientation_angle_error_rad", epoch_state_estimation_metrics[3]);
            rlt::add_scalar(device, device.logger, "training/state_estimation/orientation_angle_error_deg", epoch_orientation_angle_error_deg);
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
        rlt::add_scalar(device, device.logger, "timing/collection_time_s", collection_elapsed.count());
        rlt::add_scalar(device, device.logger, "timing/collection_fps", collection_fps);
        rlt::add_scalar(device, device.logger, "timing/collection_pct", collection_pct);
        rlt::add_scalar(device, device.logger, "timing/train_time_s", train_elapsed.count());
        rlt::add_scalar(device, device.logger, "timing/train_pct", train_pct);
        rlt::add_scalar(device, device.logger, "timing/epoch_time_s", epoch_elapsed.count());
        rlt::add_scalar(device, device.logger, "timing/total_time_s", training_elapsed.count());
        rlt::add_scalar(device, device.logger, "training/teacher_forcing", full_teacher_forcing ? (T)1 : EFFECTIVE_TEACHER_FORCING_FRACTION);
        rlt::add_scalar(device, device.logger, "training/terminated_share", episode_terminated_share);
        rlt::add_scalar(device, device.logger, "training/terminated_episodes", static_cast<T>(episode_count_terminated));
        rlt::add_scalar(device, device.logger, "training/complete_terminated_share", complete_terminated_share);
        rlt::add_scalar(device, device.logger, "training/complete_episode_length", complete_episode_length);
        rlt::add_scalar(device, device.logger, "training/complete_episodes", static_cast<T>(complete_episode_count));
        rlt::add_scalar(device, device.logger, "curriculum/episode_step_limit", static_cast<T>(runner.episode_step_limit));
        rlt::add_scalar(device, device.logger, "rendering/anti_aliasing_grid_size", RENDER_ANTI_ALIASING_ACTIVE ? static_cast<T>(RENDER_ANTI_ALIASING_GRID_SIZE) : static_cast<T>(1));
        rlt::add_scalar(device, device.logger, "rendering/motion_blur_samples", RENDER_MOTION_BLUR_ACTIVE ? static_cast<T>(RENDER_MOTION_BLUR_SAMPLES) : static_cast<T>(1));
        rlt::add_scalar(device, device.logger, "rendering/target_frame_roll_pitch_randomization_range", TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE);
        rlt::add_scalar(device, device.logger, "rendering/target_frame_brightness_mismatch_range", TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE);
#endif

        // =================================================================
        // Checkpoint
        // =================================================================
        if(epoch_i % CHECKPOINT_CADENCE == 0){
            auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, epoch_end_step);
            std::filesystem::create_directories(step_folder);
            using EVAL_TYPE = typename CPU_STUDENT_TYPE::template CHANGE_BATCH_SIZE<TI, N_EXAMPLES>;
            EVAL_TYPE eval_student;
            rlt::malloc(device, eval_student);
            rlt::copy(device_compute, device, student, eval_student);
            char fov_buf[32];
            std::snprintf(fov_buf, sizeof(fov_buf), "%.6g", (double)WORLD_SPEC::CAMERA_FOV);
            std::string state_obs_string = rlt::string(device, env.environments[0].dynamics, ACTOR_STATE_OBS{});
            std::string image_obs_string = std::string("CameraRGBStackedWithTarget(") + fov_buf + ", "
                + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ", "
                + std::to_string(FRAME_STACK_STRIDE) + ", " + std::to_string(FRAME_STACK_N) + ")";
            std::string obs_string = image_obs_string + ", " + state_obs_string;
            std::string rendering_string = std::string("{\"anti_aliasing\": ") + (RENDER_ANTI_ALIASING_ACTIVE ? "true" : "false")
                + ", \"anti_aliasing_grid_size\": " + std::to_string(RENDER_ANTI_ALIASING_ACTIVE ? RENDER_ANTI_ALIASING_GRID_SIZE : (TI)1)
                + ", \"motion_blur\": " + (RENDER_MOTION_BLUR_ACTIVE ? "true" : "false")
                + ", \"motion_blur_samples\": " + std::to_string(RENDER_MOTION_BLUR_ACTIVE ? RENDER_MOTION_BLUR_SAMPLES : (TI)1)
                + ", \"shutter_fraction_min\": " + std::to_string(RENDER_SHUTTER_FRACTION_MIN)
                + ", \"shutter_fraction_max\": " + std::to_string(RENDER_SHUTTER_FRACTION_MAX)
                + ", \"target_frame_roll_pitch_randomization_range\": " + std::to_string(TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE)
                + ", \"target_frame_brightness_mismatch_range\": " + std::to_string(TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE) + "}";
            std::string output_string = STATE_ESTIMATION_MODE
                ? "StateEstimation(RelativeTargetPositionBody,LinearVelocityBody,RelativeTargetOrientationBodyRotationMatrix)"
                : "Action";
            std::string meta = "{\"environment\": {\"name\": \"l2f_visual\", \"observation\": \"" + obs_string + "\", \"output\": \"" + output_string + "\", \"rendering\": " + rendering_string + "}}";
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, N_EXAMPLES, IMG_H, IMG_W, COMBINED_IMG_C>, true>> example_input_0_image;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, N_EXAMPLES, STATE_OBS_DIM>, true>> example_input_1_state;
            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_EXAMPLES, TARGET_DIM>, true>> example_output;
            rlt::malloc(device, example_input_0_image);
            rlt::malloc(device, example_input_1_state);
            rlt::malloc(device, example_output);
            {
                {
                    // the tail of the rollout: every frame-stack slot is populated there
                    static constexpr TI EXAMPLE_ROW_OFFSET = STEPS_TOTAL - N_EXAMPLES;
                    static_assert(EXAMPLE_ROW_OFFSET >= (FRAME_STACK_N - 1) * FRAME_STACK_STRIDE * N_ENVIRONMENTS,
                        "N_EXAMPLES too large: sampled window would include steps with clamped frame-stack history");
                    auto src_combined = rlt::view_range(device_compute, all_combined_observations, EXAMPLE_ROW_OFFSET, rlt::tensor::ViewSpec<0, N_EXAMPLES>{});
                    auto src_state = rlt::view_range(device_compute, all_state_observations, EXAMPLE_ROW_OFFSET, rlt::tensor::ViewSpec<0, N_EXAMPLES>{});
                    auto dst_image_2d = rlt::reshape_row_major(device, example_input_0_image, rlt::tensor::Shape<TI, N_EXAMPLES, COMBINED_OBS_DIM>{});
                    auto dst_state_2d = rlt::reshape_row_major(device, example_input_1_state, rlt::tensor::Shape<TI, N_EXAMPLES, STATE_OBS_DIM>{});
                    rlt::copy(device_compute, device, src_combined, dst_image_2d);
                    rlt::copy(device_compute, device, src_state, dst_state_2d);
                }
                using BRANCH_0 = typename rlt::utils::tuple_element<0, typename EVAL_TYPE::SPEC::BRANCH_TUPLE>::type;
                using BRANCH_1 = typename rlt::utils::tuple_element<1, typename EVAL_TYPE::SPEC::BRANCH_TUPLE>::type;
                using BRANCH_0_OUTPUT_SHAPE = rlt::nn_models::parallel::detail::output_shape<typename EVAL_TYPE::SPEC::CAPABILITY, BRANCH_0>;
                using BRANCH_1_OUTPUT_SHAPE = rlt::nn_models::parallel::detail::output_shape<typename EVAL_TYPE::SPEC::CAPABILITY, BRANCH_1>;
                static constexpr TI BRANCH_0_DIM = rlt::get_last(BRANCH_0_OUTPUT_SHAPE{});
                static constexpr TI BRANCH_1_DIM = rlt::get_last(BRANCH_1_OUTPUT_SHAPE{});
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_EXAMPLES, BRANCH_0_DIM>, true>> branch_0_out;
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_EXAMPLES, BRANCH_1_DIM>, true>> branch_1_out;
                rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_EXAMPLES, BRANCH_0_DIM + BRANCH_1_DIM>, true>> concat_out;
                typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<0>(eval_student.pipelines))>::template Buffer<true> buffer_0;
                typename rlt::utils::typing::remove_reference_t<decltype(rlt::get<1>(eval_student.pipelines))>::template Buffer<true> buffer_1;
                rlt::malloc(device, branch_0_out);
                rlt::malloc(device, branch_1_out);
                rlt::malloc(device, concat_out);
                rlt::malloc(device, buffer_0);
                rlt::malloc(device, buffer_1);
                rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
                auto image_eval_view = rlt::view_memory<rlt::tensor::Shape<TI, N_EXAMPLES, IMG_H, IMG_W, COMBINED_IMG_C>>(device, example_input_0_image);
                auto state_eval_view = rlt::view_memory<rlt::tensor::Shape<TI, N_EXAMPLES, STATE_OBS_DIM>>(device, example_input_1_state);
                rlt::evaluate(device, rlt::get<0>(eval_student.pipelines), image_eval_view, branch_0_out, buffer_0, rng, eval_mode);
                rlt::evaluate(device, rlt::get<1>(eval_student.pipelines), state_eval_view, branch_1_out, buffer_1, rng, eval_mode);
                auto concat_0 = rlt::view_range(device, concat_out, (TI)0, rlt::tensor::ViewSpec<1, BRANCH_0_DIM>{});
                auto concat_1 = rlt::view_range(device, concat_out, (TI)BRANCH_0_DIM, rlt::tensor::ViewSpec<1, BRANCH_1_DIM>{});
                rlt::copy(device, device, branch_0_out, concat_0);
                rlt::copy(device, device, branch_1_out, concat_1);
                typename decltype(eval_student.head)::template Buffer<true> head_buffer;
                rlt::malloc(device, head_buffer);
                rlt::evaluate(device, eval_student.head, concat_out, example_output, head_buffer, rng, eval_mode);
                rlt::free(device, head_buffer);
                rlt::free(device, branch_0_out);
                rlt::free(device, branch_1_out);
                rlt::free(device, concat_out);
                rlt::free(device, buffer_0);
                rlt::free(device, buffer_1);
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
                rlt::save(device, example_input_1_state, inputs_group, "1");
                auto outputs_group = rlt::create_group(device, example_group, "outputs");
                auto example_output_canonical = rlt::reshape_row_major(device, example_output, rlt::tensor::Shape<TI, 1, N_EXAMPLES, TARGET_DIM>{});
                rlt::save(device, example_output_canonical, outputs_group, "0");
                rlt::persist::backends::tar::finalize(device, writer);
                std::ofstream f(checkpoint_path, std::ios::binary);
                f.write(writer.buffer.data(), writer.buffer.size());
            }
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
            auto save_hdf5 = [&](auto batch_size_tag){
                static constexpr TI SAVE_BATCH_SIZE = decltype(batch_size_tag)::value;
                using SIZED_EVAL_TYPE = typename CPU_STUDENT_TYPE::template CHANGE_BATCH_SIZE<TI, SAVE_BATCH_SIZE>;
                SIZED_EVAL_TYPE sized_eval_student;
                rlt::malloc(device, sized_eval_student);
                rlt::copy(device_compute, device, student, sized_eval_student);
                std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
                std::filesystem::path checkpoint_path = step_folder / (std::string("checkpoint_") + std::to_string(SAVE_BATCH_SIZE) + "examples.h5");
                rlt::persist::backends::hdf5::File root_file(checkpoint_path.string(), rlt::persist::backends::hdf5::Mode::WRITE);
                auto actor_group = rlt::create_group(device, root_file, "actor");
                rlt::set_attribute(device, actor_group, "checkpoint_name", step_folder.string().c_str());
                rlt::set_attribute(device, actor_group, "meta", meta.c_str());
                rlt::save(device, sized_eval_student, actor_group);
                auto example_group = rlt::create_group(device, root_file, "example");
                auto inputs_group = rlt::create_group(device, example_group, "inputs");
                auto example_input_0_image_view = rlt::view_range(device, example_input_0_image, (TI)0, rlt::tensor::ViewSpec<1, SAVE_BATCH_SIZE>{});
                auto example_input_1_state_view = rlt::view_range(device, example_input_1_state, (TI)0, rlt::tensor::ViewSpec<1, SAVE_BATCH_SIZE>{});
                rlt::save(device, example_input_0_image_view, inputs_group, "0");
                rlt::save(device, example_input_1_state_view, inputs_group, "1");
                auto outputs_group = rlt::create_group(device, example_group, "outputs");
                auto example_output_canonical = rlt::reshape_row_major(device, example_output, rlt::tensor::Shape<TI, 1, N_EXAMPLES, TARGET_DIM>{});
                auto example_output_view = rlt::view_range(device, example_output_canonical, (TI)0, rlt::tensor::ViewSpec<1, SAVE_BATCH_SIZE>{});
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
                output_ss << "\n" << rlt::save_code(device, example_input_1_state, std::string("_1"), true);
                output_ss << "\n" << "}";
                output_ss << "\n" << "namespace rl_tools::checkpoint::example::outputs{";
                {
                    auto example_output_canonical = rlt::reshape_row_major(device, example_output, rlt::tensor::Shape<TI, 1, N_EXAMPLES, TARGET_DIM>{});
                    output_ss << "\n" << rlt::save_code(device, example_output_canonical, std::string("_0"), true);
                }
                output_ss << "\n" << "}";
                output_ss << "\n" << "namespace rl_tools::checkpoint::meta{";
                output_ss << "\n" << "   " << "char name[] = \"" << step_folder.string() << "\";";
                output_ss << "\n" << "   " << "char commit_hash[] = \"" << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
                output_ss << "\n" << "   " << "char observation[] = \"" << obs_string << "\";";
                output_ss << "\n" << "}";
                std::string code_string = output_ss.str();
#ifdef RL_TOOLS_ENABLE_ZLIB
                {
                    std::filesystem::path checkpoint_code_path = step_folder / "checkpoint.h.gz";
                    std::vector<uint8_t> compressed;
                    rlt::compress_zlib(code_string, compressed);
                    std::ofstream f(checkpoint_code_path, std::ios::binary);
                    f.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
                }
#endif
                {
                    std::filesystem::path checkpoint_code_path = step_folder / "checkpoint.h";
                    std::ofstream f(checkpoint_code_path);
                    f << code_string;
                }
            }
            rlt::free(device, example_input_0_image);
            rlt::free(device, example_input_1_state);
            rlt::free(device, example_output);
            rlt::free(device, eval_student);
            {
                std::string trajectories_json = trajectory_episodes_to_json(device, env.environments[0], completed_episodes, simulation_dt);
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
    }

    std::cout << "Training finished." << std::endl;
#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
    rlt::free(device, device.logger);
#endif

    // =========================================================================
    // Cleanup
    // =========================================================================
    rlt::free(device_compute, env);
    delete env_storage;
    rlt::free(device, raptor);
    rlt::free(device, student_cpu);
    rlt::free(device, reset_mask_host);
    rlt::free(device, terminated_host);
    rlt::free(device, runner_host.episode_step);
    rlt::free(device, dataset_host.episode_end_reason);
    rlt::free(device, dataset_host.episode_length);
    rlt::free(device, dataset_host.episode_return);
    rlt::free(device_compute, runner);
    rlt::free(device_compute, runner_buffer);
    rlt::free(device_compute, dataset);
    rlt::free(device, states_host);
    rlt::free(device, parameters_trajectory_host);
    rlt::free(device, actions_trajectory_host);
    rlt::free(device, targets_host);
    rlt::free(device, student_output_train_host);
    rlt::free(device, targets_batch_host);
    rlt::free(device, rng);
    rlt::free(device_compute, rng_compute);
    rlt::free(device_compute, raptor_compute);
    rlt::free(device_compute, raptor_buffer_compute);
    rlt::free(device_compute, student);
    rlt::free(device_compute, student_buffers);
    rlt::free(device_compute, optimizer);
    rlt::free(device_compute, rollout_student);
    rlt::free(device_compute, rollout_student_buffers);
    rlt::free(device_compute, teacher_actions);
    rlt::free(device_compute, student_output_step);
    rlt::free(device_compute, all_state_observations);
    rlt::free(device_compute, all_targets);
    rlt::free(device_compute, d_action_train);
    rlt::free(device_compute, student_output_train);
    return 0;
}
