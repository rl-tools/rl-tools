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
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_cuda.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cuda.h>

#include <rl_tools/nn/loss_functions/mse/operations_generic.h>
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>

#include "../../../../src/nn_models/port_checkpoint/raptor/policy.h"

#include <array>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <numeric>

namespace rlt = rl_tools;

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
using TYPE_POLICY = rlt::numeric_types::Policy<float>;
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
static constexpr TI EPISODE_STEP_LIMIT = 500;
using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie;

static constexpr REWARD_FUNCTION reward_function = {
    false, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
};
static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
    0.2, 1.0, 0.3, 1.0, 1.0, true, -1, +1,
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    true, 1.5, 10, 35, 10000, 50000,
};
static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::Integration integration = {
    static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)
};
static constexpr typename PARAMETERS_TYPE::MDP mdp = { init, reward_function, {}, {}, termination };
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = { {0, 0}, {0, 0} };
static constexpr PARAMETERS_TYPE nominal_parameters = { {dynamics, integration, mdp}, disturbances };

// =========================================================================
// Environment static parameters
// =========================================================================
static constexpr TI ACTION_HISTORY_LENGTH = 1;

struct STATIC_PARAMETERS {
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI CLOSED_FORM = false;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
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
    static constexpr T STATE_LIMIT_POSITION = 100000;
    static constexpr T STATE_LIMIT_VELOCITY = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
};

// using ACTOR_STATE_OBS = obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI>>>>;
using ACTOR_STATE_OBS = STATIC_PARAMETERS::OBSERVATION_TYPE;
static constexpr TI STATE_OBS_DIM = ACTOR_STATE_OBS::DIM; // 12


// =========================================================================
// Visual environment specification
// =========================================================================
static constexpr TI N_ENVIRONMENTS = 64;
static constexpr TI CAM_WIDTH = 64;
static constexpr TI CAM_HEIGHT = 64;
static constexpr TI NUM_PROBES = 64;

using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, N_ENVIRONMENTS, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES>;
using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;

// =========================================================================
// RAPTOR teacher (CPU only)
// =========================================================================
static constexpr TI RAPTOR_HIDDEN_DIM = 16;
static constexpr TI RAPTOR_OBS_DIM = 22; // Position(3) + RotMat(9) + LinVel(3) + AngVel(3) + ActionHistory(4)

using RAPTOR_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
using RAPTOR_DENSE1 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE1_CONFIG>;
using RAPTOR_GRU_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM>;
using RAPTOR_GRU = rlt::nn::layers::gru::BindConfiguration<RAPTOR_GRU_CONFIG>;
using RAPTOR_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
using RAPTOR_DENSE2 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE2_CONFIG>;

using RAPTOR_MODULE = rlt::nn_models::sequential::Module<RAPTOR_DENSE1, RAPTOR_GRU, RAPTOR_DENSE2>;
using RAPTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, RAPTOR_OBS_DIM>;
using RAPTOR_CAPABILITY = rlt::nn::capability::Forward<true, false>;
using RAPTOR_MODEL = rlt::nn_models::sequential::Build<RAPTOR_CAPABILITY, RAPTOR_MODULE, RAPTOR_INPUT_SHAPE>;

// =========================================================================
// Student CNN (GPU)
// =========================================================================
static constexpr TI ACTOR_HIDDEN_DIM = 64;
static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
static constexpr TI BATCH_SIZE = 512;
static constexpr TI STEPS_PER_ENV = 500;
static constexpr TI STEPS_TOTAL = STEPS_PER_ENV * N_ENVIRONMENTS;
static constexpr TI N_BATCHES = STEPS_TOTAL / BATCH_SIZE;
static constexpr TI NUM_EPOCHS = 1000;
static constexpr TI TEACHER_FORCING_EPOCHS = 100;
static constexpr TI N_TRAIN_PASSES = 4;
static constexpr TI VIDEO_CADENCE = 10;
static constexpr TI GRID_SIDE = 8; // sqrt(N_ENVIRONMENTS)
static_assert(GRID_SIDE * GRID_SIDE == N_ENVIRONMENTS, "N_ENVIRONMENTS must be a perfect square for video mosaic");

static_assert(N_BATCHES > 0, "STEPS_TOTAL must be >= BATCH_SIZE");

struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
    static constexpr T ALPHA = 3e-4;
    static constexpr T EPSILON = 1e-5;
    static constexpr T EPSILON_SQRT = 1e-5;
};

template<typename CAPABILITY>
struct StudentActor{
    static constexpr TI STEPS = 1;
    static constexpr TI FORWARD_BATCH_SIZE = BATCH_SIZE;
    using OBS_SHAPE = typename ENVIRONMENT::Observation::SHAPE;
    using IMAGE_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
    using STATE_INPUT_SHAPE = rlt::tensor::Shape<TI, STEPS, FORWARD_BATCH_SIZE, STATE_OBS_DIM>;

    static constexpr TI IMG_H = ENVIRONMENT::Observation::HEIGHT;
    static constexpr TI IMG_W = ENVIRONMENT::Observation::WIDTH;
    static constexpr TI IMG_C = ENVIRONMENT::Observation::CHANNELS;

    // Image branch: Flatten→Standardize→Unflatten→Conv1→Conv2→Flatten→Dense(64)
    using INPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using INPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<INPUT_FLATTEN_CONFIG>;
    using IMAGE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
    using IMAGE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<IMAGE_STANDARDIZE_CONFIG>;
    using UNFLATTEN_CONFIG = rlt::nn::layers::unflatten::Configuration<TYPE_POLICY, TI, IMG_H, IMG_W, IMG_C>;
    using UNFLATTEN = rlt::nn::layers::unflatten::BindConfiguration<UNFLATTEN_CONFIG>;
    using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 16, 3, 3, 2, 2, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
    using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 32, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
    using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
    using IMAGE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using IMAGE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<IMAGE_DENSE_EMBED_CONFIG>;
    using IMAGE_BRANCH = rlt::nn_models::sequential::Module<INPUT_FLATTEN, IMAGE_STANDARDIZE, UNFLATTEN, CONV1, CONV2, OUTPUT_FLATTEN, IMAGE_DENSE_EMBED>;

    // State branch: Standardize→Dense(64)
    using STATE_STANDARDIZE_CONFIG = rlt::nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
    using STATE_STANDARDIZE = rlt::nn::layers::standardize::BindConfiguration<STATE_STANDARDIZE_CONFIG>;
    using STATE_DENSE_EMBED_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION>;
    using STATE_DENSE_EMBED = rlt::nn::layers::dense::BindConfiguration<STATE_DENSE_EMBED_CONFIG>;
    using STATE_BRANCH = rlt::nn_models::sequential::Module<STATE_STANDARDIZE, STATE_DENSE_EMBED>;

    // Head MLP: 128D → 64 → 64 → ACTION_DIM (plain MLP, no log_std)
    using MLP_HEAD_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, ACTION_DIM, 2, ACTOR_HIDDEN_DIM, ACTOR_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
    using MLP_HEAD = rlt::nn_models::mlp::BindConfiguration<MLP_HEAD_CONFIG>;

    using MODEL = rlt::nn_models::parallel::Build<CAPABILITY, IMAGE_BRANCH, STATE_BRANCH, IMAGE_INPUT_SHAPE, STATE_INPUT_SHAPE, MLP_HEAD>;
};

using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, true>;
using STUDENT_TYPE = typename StudentActor<CAPABILITY_ADAM>::MODEL;
using STUDENT_BUFFERS = typename STUDENT_TYPE::Buffer<true>;
using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, ADAM_PARAMETERS, true>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;

static constexpr TI IMG_H = ENVIRONMENT::Observation::HEIGHT;
static constexpr TI IMG_W = ENVIRONMENT::Observation::WIDTH;
static constexpr TI IMG_C = ENVIRONMENT::Observation::CHANNELS;

// =========================================================================
// Main
// =========================================================================
int main(int argc, char** argv){
    const char* scene_path = nullptr;
    TI seed = 0;
    if(argc > 1){
        scene_path = argv[1];
    }
    else{
        std::cout << "Usage: " << argv[0] << " <scene_path> [seed]" << std::endl;
        return 1;
    }
    if(argc > 2){
        seed = std::atoi(argv[2]);
    }

    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device);

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

    // CPU state observations buffer (per step)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, STATE_OBS_DIM>>> cpu_state_obs_step;
    rlt::malloc(device, cpu_state_obs_step);

    // All teacher actions for entire epoch (CPU)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, ACTION_DIM>>> cpu_all_teacher_actions;
    rlt::malloc(device, cpu_all_teacher_actions);

    // =========================================================================
    // Student (CPU copy for warmup)
    // =========================================================================
    STUDENT_TYPE student_cpu;
    rlt::malloc(device, student_cpu);
    rlt::init_weights(device, student_cpu, rng);

    // Warmup observations (CPU)
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, OBSERVATION_DIM>>> warmup_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, STATE_OBS_DIM>>> warmup_state_observations;
    rlt::malloc(device, warmup_observations);
    rlt::malloc(device, warmup_state_observations);

    // =========================================================================
    // Environment setup (shared renderer)
    // =========================================================================
    ENVIRONMENT envs[N_ENVIRONMENTS];
    typename ENVIRONMENT::Parameters env_parameters[N_ENVIRONMENTS];

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::malloc(device, envs[env_i]);
    }

    auto& env0 = envs[0];
    for(TI env_i = 1; env_i < N_ENVIRONMENTS; env_i++){
        if(envs[env_i].owns_renderer && envs[env_i].renderer != nullptr){
            rlt::free(device, *envs[env_i].renderer);
            delete envs[env_i].renderer;
        }
        if(envs[env_i].scene != nullptr){
            delete envs[env_i].scene;
        }
        envs[env_i].renderer = env0.renderer;
        envs[env_i].scene = env0.scene;
        envs[env_i].owns_renderer = false;
    }

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        envs[env_i].use_target_mode = true;
        envs[env_i].scene_path = scene_path;
        if(env_i > 0){
            envs[env_i].renderer_initialized = true;
        }
    }

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::init(device, envs[env_i]);
    }

    // Filter indoor positions by clearance (min probe distance >= 1m)
    static constexpr T MIN_CLEARANCE = 1.0;
    struct FilteredPosition { T translation[3]; };
    std::vector<FilteredPosition> filtered_positions;
    {
        TI num_positions = env0.scene->num_indoor_positions;
        std::cout << "Filtering " << num_positions << " indoor positions (min clearance: " << MIN_CLEARANCE << "m)..." << std::endl;
        T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);
        T look_ahead = 1.0;
        owl::vec3f up(0.f, 1.f, 0.f);
        std::array<rlt::CameraData, N_ENVIRONMENTS> filter_cameras{};

        for(TI batch_start = 0; batch_start < num_positions; batch_start += N_ENVIRONMENTS){
            TI batch_count = std::min(N_ENVIRONMENTS, num_positions - batch_start);
            for(TI i = 0; i < batch_count; i++){
                auto& pos = env0.scene->indoor_positions[batch_start + i];
                owl::vec3f position(pos.position[0], pos.position[1] + env0.eye_height, pos.position[2]);
                owl::vec3f look_at(
                    pos.position[0] + look_ahead * std::cos(pos.yaw),
                    pos.position[1] + env0.eye_height,
                    pos.position[2] + look_ahead * std::sin(pos.yaw));
                filter_cameras[i] = rlt::make_camera_data(position, look_at, up, env0.cos_fov, aspect);
            }
            rlt::set_cameras(device, *env0.renderer, filter_cameras.data(), batch_count);
            rlt::render(device, *env0.renderer);
            for(TI i = 0; i < batch_count; i++){
                T clearance = rlt::rendering::raytracing::scene::procthor::evaluate_clearance(device, *env0.renderer, i);
                if(clearance >= MIN_CLEARANCE){
                    auto& pos = env0.scene->indoor_positions[batch_start + i];
                    FilteredPosition fp;
                    fp.translation[0] = pos.position[0];
                    fp.translation[1] = pos.position[1] + env0.eye_height;
                    fp.translation[2] = pos.position[2];
                    filtered_positions.push_back(fp);
                }
            }
        }
        std::cout << "Kept " << filtered_positions.size() << " / " << num_positions << " positions with >= " << MIN_CLEARANCE << "m clearance." << std::endl;
        if(filtered_positions.empty()){
            std::cerr << "No indoor positions with sufficient clearance. Exiting." << std::endl;
            return 1;
        }
    }

    // =========================================================================
    // Observation normalization warmup (CPU)
    // =========================================================================
    {
        STUDENT_BUFFERS student_buffers_cpu;
        rlt::malloc(device, student_buffers_cpu);

        std::cout << "Running observation normalization warmup..." << std::endl;
        for(TI obs_row = 0; obs_row < STEPS_TOTAL; obs_row++){
            TI env_i = obs_row % N_ENVIRONMENTS;
            typename ENVIRONMENT::State warmup_state;
            rlt::sample_initial_parameters(device, envs[env_i], env_parameters[env_i], rng);
            rlt::sample_initial_state(device, envs[env_i], env_parameters[env_i], warmup_state, rng);

            auto obs_slice = rlt::view(device, warmup_observations, obs_row);
            auto obs_matrix = rlt::matrix_view(device, obs_slice);
            rlt::observe(device, envs[env_i], env_parameters[env_i], warmup_state, typename ENVIRONMENT::Observation{}, obs_matrix, rng);

            auto state_obs_slice = rlt::view(device, warmup_state_observations, obs_row);
            auto state_obs_matrix = rlt::matrix_view(device, state_obs_slice);
            rlt::observe(device, envs[env_i].dynamics, env_parameters[env_i].dynamics, warmup_state, ACTOR_STATE_OBS{}, state_obs_matrix, rng);
        }

        using IMAGE_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<typename ENVIRONMENT::Observation::SHAPE, BATCH_SIZE>, (TI)1>;
        using STATE_INPUT_SHAPE_WARMUP = rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>;
        rlt::Mode<rlt::nn::layers::standardize::AccumulateMode<>> accumulate_mode;
        for(TI batch_i = 0; batch_i < N_BATCHES; batch_i++){
            auto batch_observations = rlt::view_range(device, warmup_observations, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            auto batch_observations_reshaped = rlt::reshape_row_major(device, batch_observations, IMAGE_INPUT_SHAPE{});
            auto batch_state_observations = rlt::view_range(device, warmup_state_observations, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
            auto batch_state_observations_reshaped = rlt::reshape_row_major(device, batch_state_observations, STATE_INPUT_SHAPE_WARMUP{});
            rlt::forward(device, student_cpu, batch_observations_reshaped, batch_state_observations_reshaped, student_buffers_cpu, rng, accumulate_mode);
        }
        std::cout << "Observation normalization warmup complete." << std::endl;
        rlt::free(device, student_buffers_cpu);
    }
    rlt::free(device, warmup_observations);
    rlt::free(device, warmup_state_observations);

    // =========================================================================
    // GPU init (after warmup renders change CUDA context)
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

    // GPU tensors
    static constexpr TI GPU_OBS_ROWS = STEPS_TOTAL + BATCH_SIZE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, OBSERVATION_DIM>>> gpu_all_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, GPU_OBS_ROWS, STATE_OBS_DIM>>> gpu_all_state_observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, STEPS_TOTAL, ACTION_DIM>>> gpu_all_teacher_actions;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_d_action_train;
    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> gpu_actions_eval;
    rlt::Matrix<rlt::matrix::Specification<T, TI, N_ENVIRONMENTS, ACTION_DIM>> cpu_actions_eval;
    rlt::malloc(device_gpu, gpu_all_observations);
    rlt::malloc(device_gpu, gpu_all_state_observations);
    rlt::malloc(device_gpu, gpu_all_teacher_actions);
    rlt::malloc(device_gpu, gpu_d_action_train);
    rlt::malloc(device_gpu, gpu_actions_eval);
    rlt::malloc(device, cpu_actions_eval);

    // =========================================================================
    // Environment state tracking
    // =========================================================================
    typename ENVIRONMENT::State states[N_ENVIRONMENTS];
    typename ENVIRONMENT::State next_states[N_ENVIRONMENTS];
    bool terminated[N_ENVIRONMENTS];
    TI episode_step[N_ENVIRONMENTS];
    T episode_length_sum = 0;
    TI episode_count = 0;

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        terminated[env_i] = true;
        episode_step[env_i] = 0;
    }

    // =========================================================================
    // Training loop
    // =========================================================================
    std::cout << "Starting imitation learning (visual L2F hover, CUDA)" << std::endl;
    std::cout << "  N_ENVIRONMENTS: " << N_ENVIRONMENTS << std::endl;
    std::cout << "  STEPS_PER_ENV: " << STEPS_PER_ENV << std::endl;
    std::cout << "  BATCH_SIZE: " << BATCH_SIZE << std::endl;
    std::cout << "  N_BATCHES: " << N_BATCHES << std::endl;
    std::cout << "  N_TRAIN_PASSES: " << N_TRAIN_PASSES << std::endl;
    std::cout << "  OBSERVATION_DIM (image): " << OBSERVATION_DIM << std::endl;
    std::cout << "  STATE_OBS_DIM: " << STATE_OBS_DIM << std::endl;
    std::cout << "  RAPTOR_OBS_DIM: " << RAPTOR_OBS_DIM << std::endl;
    std::cout << "  TEACHER_FORCING_EPOCHS: " << TEACHER_FORCING_EPOCHS << std::endl;

    using NO_AUTO_RESET_MODE = rlt::Mode<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Default<>>>;
    NO_AUTO_RESET_MODE no_auto_reset_mode;

    auto training_start = std::chrono::high_resolution_clock::now();
    std::array<rlt::CameraData, N_ENVIRONMENTS> cameras;

    // Video recording buffers
    static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    static constexpr TI MOSAIC_W = GRID_SIDE * CAM_WIDTH;
    static constexpr TI MOSAIC_H = GRID_SIDE * CAM_HEIGHT;
    std::vector<uint32_t> video_pixel_buffer(N_ENVIRONMENTS * CAM_PIXELS);
    std::vector<uint8_t> mosaic_frame(MOSAIC_W * MOSAIC_H * 3);

    for(TI epoch_i = 0; epoch_i < NUM_EPOCHS; epoch_i++){
        auto epoch_start = std::chrono::high_resolution_clock::now();
        bool teacher_forcing = epoch_i < TEACHER_FORCING_EPOCHS;
        bool record_video = (epoch_i % VIDEO_CADENCE == 0);
        FILE* ffmpeg_pipe = nullptr;
        if(record_video){
            char video_filename[256];
            snprintf(video_filename, sizeof(video_filename), "epoch_%05lu.mp4", (unsigned long)epoch_i);
            char ffmpeg_cmd[512];
            snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
                "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %lux%lu -framerate 25 -i - "
                "-c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast -loglevel warning %s",
                (unsigned long)MOSAIC_W, (unsigned long)MOSAIC_H, video_filename);
            ffmpeg_pipe = popen(ffmpeg_cmd, "w");
            if(!ffmpeg_pipe){
                std::cerr << "Failed to open ffmpeg pipe for " << video_filename << std::endl;
                record_video = false;
            }
        }

        // =================================================================
        // Data collection (CPU + GPU rendering)
        // =================================================================
        for(TI step_i = 0; step_i < STEPS_PER_ENV; step_i++){
            for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                if(terminated[env_i] || episode_step[env_i] >= EPISODE_STEP_LIMIT){
                    if(episode_step[env_i] > 0){
                        episode_length_sum += episode_step[env_i];
                        episode_count++;
                    }
                    // Pick a random indoor position
                    TI pos_idx = rlt::random::uniform_int_distribution(device.random, (TI)0, (TI)(filtered_positions.size() - 1), rng);
                    auto& fp = filtered_positions[pos_idx];
                    for(TI j = 0; j < 3; j++){
                        envs[env_i].target_scene_translation[j] = fp.translation[j];
                    }
                    rlt::sample_initial_parameters(device, envs[env_i], env_parameters[env_i], rng);
                    rlt::sample_initial_state(device, envs[env_i], env_parameters[env_i], states[env_i], rng);
                    episode_step[env_i] = 0;
                    terminated[env_i] = false;

                    // Reset RAPTOR GRU state for this environment
                    auto& gru_layer = rlt::nn_models::sequential::layer<1>(raptor);
                    auto& gru_state = rlt::nn_models::sequential::content_state<1>(raptor_state.content_state);
                    auto state_row = rlt::view(device, gru_state.state, env_i);
                    rlt::copy(device, device, gru_layer.initial_hidden_state.parameters, state_row);
                    rlt::set(device, gru_state.step, (TI)0, env_i);
                }

                // Teacher observation (22D)
                auto teacher_obs_row = rlt::view(device, teacher_obs, env_i);
                auto teacher_obs_matrix = rlt::matrix_view(device, teacher_obs_row);
                rlt::observe(device, envs[env_i], env_parameters[env_i], states[env_i], typename STATIC_PARAMETERS::OBSERVATION_TYPE{}, teacher_obs_matrix, rng);

                // Student state observation (12D)
                auto state_obs_row = rlt::view(device, cpu_state_obs_step, env_i);
                auto state_obs_matrix = rlt::matrix_view(device, state_obs_row);
                rlt::observe(device, envs[env_i].dynamics, env_parameters[env_i].dynamics, states[env_i], ACTOR_STATE_OBS{}, state_obs_matrix, rng);

                // Camera
                cameras[env_i] = rlt::rl::environments::l2f_visual::make_camera_for_state(device, envs[env_i], env_parameters[env_i], states[env_i]);
            }

            // RAPTOR single-step inference (batched, CPU)
            rlt::evaluate_step(device, raptor, teacher_obs, raptor_state, teacher_actions, raptor_buffer, rng, no_auto_reset_mode);

            // GPU batch render
            T* obs_ptr = rlt::data(gpu_all_observations) + (TI)(step_i * N_ENVIRONMENTS) * OBSERVATION_DIM;
            rlt::observe_batch_render_gpu(device, env0, cameras.data(), N_ENVIRONMENTS, obs_ptr);

            // Video: read back frame buffer and write mosaic frame
            if(record_video && ffmpeg_pipe){
                rlt::read_frame_buffer(device, *env0.renderer, video_pixel_buffer.data(), video_pixel_buffer.size());
                for(TI grid_row = 0; grid_row < GRID_SIDE; grid_row++){
                    for(TI grid_col = 0; grid_col < GRID_SIDE; grid_col++){
                        TI env_i = grid_row * GRID_SIDE + grid_col;
                        for(TI py = 0; py < CAM_HEIGHT; py++){
                            for(TI px = 0; px < CAM_WIDTH; px++){
                                uint32_t rgba = video_pixel_buffer[env_i * CAM_PIXELS + py * CAM_WIDTH + px];
                                TI mosaic_x = grid_col * CAM_WIDTH + px;
                                TI mosaic_y = grid_row * CAM_HEIGHT + py;
                                TI out_idx = (mosaic_y * MOSAIC_W + mosaic_x) * 3;
                                mosaic_frame[out_idx + 0] = (rgba >>  0) & 0xFF;
                                mosaic_frame[out_idx + 1] = (rgba >>  8) & 0xFF;
                                mosaic_frame[out_idx + 2] = (rgba >> 16) & 0xFF;
                            }
                        }
                    }
                }
                fwrite(mosaic_frame.data(), 1, mosaic_frame.size(), ffmpeg_pipe);
            }

            // CPU→GPU: copy state observations for this step
            cudaMemcpy(
                rlt::data(gpu_all_state_observations) + (TI)(step_i * N_ENVIRONMENTS) * STATE_OBS_DIM,
                rlt::data(cpu_state_obs_step),
                N_ENVIRONMENTS * STATE_OBS_DIM * sizeof(T),
                cudaMemcpyHostToDevice);

            // Store teacher actions (CPU)
            auto actions_dest = rlt::view_range(device, cpu_all_teacher_actions, step_i * N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, N_ENVIRONMENTS>{});
            rlt::copy(device, device, teacher_actions, actions_dest);

            if(!teacher_forcing){
                // Student rollout: evaluate student on GPU to get stepping actions
                auto gpu_obs_slice = rlt::view_range(device_gpu, gpu_all_observations, (TI)(step_i * N_ENVIRONMENTS), rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                using EVAL_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<typename ENVIRONMENT::Observation::SHAPE, BATCH_SIZE>, (TI)1>;
                auto gpu_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_obs_slice, EVAL_INPUT_SHAPE{});
                auto gpu_state_obs_slice = rlt::view_range(device_gpu, gpu_all_state_observations, (TI)(step_i * N_ENVIRONMENTS), rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_state_obs_reshaped = rlt::reshape_row_major(device_gpu, gpu_state_obs_slice, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});
                auto gpu_actions_eval_tensor = rlt::to_tensor(device_gpu, gpu_actions_eval);
                auto gpu_actions_eval_reshaped = rlt::reshape_row_major(device_gpu, gpu_actions_eval_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                rlt::evaluate(device_gpu, student_gpu, gpu_obs_reshaped, gpu_state_obs_reshaped, gpu_actions_eval_reshaped, student_buffers, rng_gpu);
                cudaDeviceSynchronize();

                // GPU→CPU: copy first N_ENVIRONMENTS actions
                auto gpu_actions_first_n = rlt::view(device_gpu, gpu_actions_eval, rlt::matrix::ViewSpec<N_ENVIRONMENTS, ACTION_DIM>(), 0, 0);
                rlt::copy(device_gpu, device, gpu_actions_first_n, cpu_actions_eval);
            }

            // Step environment with chosen actions
            for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
                if(teacher_forcing){
                    auto action_row = rlt::view(device, teacher_actions, env_i);
                    auto action_matrix = rlt::matrix_view(device, action_row);
                    rlt::step(device, envs[env_i].dynamics, env_parameters[env_i].dynamics, states[env_i], action_matrix, next_states[env_i], rng);
                } else {
                    auto action_row = rlt::view(device, cpu_actions_eval, rlt::matrix::ViewSpec<1, ACTION_DIM>(), env_i, 0);
                    rlt::step(device, envs[env_i].dynamics, env_parameters[env_i].dynamics, states[env_i], action_row, next_states[env_i], rng);
                }
                terminated[env_i] = rlt::terminated(device, envs[env_i].dynamics, env_parameters[env_i].dynamics, next_states[env_i], rng);
                states[env_i] = next_states[env_i];
                episode_step[env_i]++;
            }
        }

        // Close video pipe
        if(ffmpeg_pipe){
            pclose(ffmpeg_pipe);
            ffmpeg_pipe = nullptr;
        }

        // Reset CUDA state after OptiX renders
        {
            cudaError_t err;
            do { err = cudaGetLastError(); } while(err != cudaSuccess);
            cudaDeviceSynchronize();
            cublasDestroy(device_gpu.handle);
            cublasCreate(&device_gpu.handle);
            if(device_gpu.stream != 0) cublasSetStream(device_gpu.handle, device_gpu.stream);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
            cudnnDestroy(device_gpu.cudnn_handle);
            cudnnCreate(&device_gpu.cudnn_handle);
            if(device_gpu.stream != 0) cudnnSetStream(device_gpu.cudnn_handle, device_gpu.stream);
#endif
        }

        // Copy all teacher actions to GPU
        cudaMemcpy(
            rlt::data(gpu_all_teacher_actions),
            rlt::data(cpu_all_teacher_actions),
            STEPS_TOTAL * ACTION_DIM * sizeof(T),
            cudaMemcpyHostToDevice);

        // =================================================================
        // Training (GPU)
        // =================================================================
        T epoch_loss = 0;
        TI loss_count = 0;

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
                auto gpu_obs_batch = rlt::view_range(device_gpu, gpu_all_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                using ACTOR_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<typename ENVIRONMENT::Observation::SHAPE, BATCH_SIZE>, (TI)1>;
                auto gpu_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_obs_batch, ACTOR_INPUT_SHAPE{});
                auto gpu_state_obs_batch = rlt::view_range(device_gpu, gpu_all_state_observations, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto gpu_state_obs_batch_reshaped = rlt::reshape_row_major(device_gpu, gpu_state_obs_batch, rlt::tensor::Shape<TI, 1, BATCH_SIZE, STATE_OBS_DIM>{});

                rlt::forward(device_gpu, student_gpu, gpu_obs_batch_reshaped, gpu_state_obs_batch_reshaped, student_buffers, rng_gpu);
                cudaDeviceSynchronize();

                // MSE loss gradient
                auto student_output_tensor = rlt::output(device_gpu, student_gpu);
                auto student_output_matrix = rlt::matrix_view(device_gpu, student_output_tensor);
                auto target_batch_tensor = rlt::view_range(device_gpu, gpu_all_teacher_actions, batch_offset, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
                auto target_batch = rlt::matrix_view(device_gpu, target_batch_tensor);
                rlt::nn::loss_functions::mse::gradient(device_gpu, student_output_matrix, target_batch, gpu_d_action_train, (T)0.5);
                cudaDeviceSynchronize();

                // Compute loss for logging (sample every N_BATCHES batches)
                if(batch_idx == 0 && pass == 0){
                    rlt::Matrix<rlt::matrix::Specification<T, TI, BATCH_SIZE, ACTION_DIM>> cpu_student_output, cpu_target;
                    rlt::malloc(device, cpu_student_output);
                    rlt::malloc(device, cpu_target);
                    rlt::copy(device_gpu, device, student_output_matrix, cpu_student_output);
                    rlt::copy(device_gpu, device, target_batch, cpu_target);
                    T batch_loss = 0;
                    for(TI i = 0; i < BATCH_SIZE; i++){
                        for(TI j = 0; j < ACTION_DIM; j++){
                            T diff = rlt::get(cpu_student_output, i, j) - rlt::get(cpu_target, i, j);
                            batch_loss += diff * diff;
                        }
                    }
                    epoch_loss = batch_loss / (BATCH_SIZE * ACTION_DIM);
                    rlt::free(device, cpu_student_output);
                    rlt::free(device, cpu_target);
                }

                // Student backward + Adam step
                auto gpu_d_action_tensor = rlt::to_tensor(device_gpu, gpu_d_action_train);
                auto gpu_d_action_reshaped = rlt::reshape_row_major(device_gpu, gpu_d_action_tensor, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ACTION_DIM>{});
                rlt::backward(device_gpu, student_gpu, gpu_obs_batch_reshaped, gpu_state_obs_batch_reshaped, gpu_d_action_reshaped, student_buffers);
                cudaDeviceSynchronize();
                rlt::step(device_gpu, optimizer_gpu, student_gpu);
                cudaDeviceSynchronize();

                loss_count++;
            }
        }

        // Logging
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> training_elapsed = now - training_start;
        std::chrono::duration<T> epoch_elapsed = now - epoch_start;
        T mean_episode_length = episode_count > 0 ? episode_length_sum / episode_count : 0;

        std::cout << (teacher_forcing ? "[TF] " : "[SR] ")
                  << "Epoch: " << std::setw(5) << epoch_i
                  << " MSE: " << std::setw(10) << std::setprecision(6) << std::fixed << epoch_loss
                  << " mean_ep_len: " << std::setw(6) << std::setprecision(1) << mean_episode_length
                  << " episodes: " << std::setw(5) << episode_count
                  << " epoch_time: " << std::setw(6) << std::setprecision(1) << epoch_elapsed.count() << "s"
                  << " total: " << std::setw(8) << std::setprecision(1) << training_elapsed.count() << "s"
                  << std::endl;

        episode_length_sum = 0;
        episode_count = 0;
    }

    std::cout << "Training finished." << std::endl;

    // =========================================================================
    // Cleanup
    // =========================================================================
    for(TI env_i = 1; env_i < N_ENVIRONMENTS; env_i++){
        envs[env_i].renderer = nullptr;
        envs[env_i].scene = nullptr;
    }
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++){
        rlt::free(device, envs[env_i]);
    }

    rlt::free(device, raptor);
    rlt::free(device, raptor_buffer);
    rlt::free(device, raptor_state);
    rlt::free(device, teacher_obs);
    rlt::free(device, teacher_actions);
    rlt::free(device, cpu_state_obs_step);
    rlt::free(device, cpu_all_teacher_actions);
    rlt::free(device, student_cpu);

    rlt::free(device_gpu, student_gpu);
    rlt::free(device_gpu, student_buffers);
    rlt::free(device_gpu, optimizer_gpu);
    rlt::free(device_gpu, gpu_all_observations);
    rlt::free(device_gpu, gpu_all_state_observations);
    rlt::free(device_gpu, gpu_all_teacher_actions);
    rlt::free(device_gpu, gpu_d_action_train);
    rlt::free(device_gpu, gpu_actions_eval);
    rlt::free(device, cpu_actions_eval);

    return 0;
}
