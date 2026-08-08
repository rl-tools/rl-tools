#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cuda.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/gru/helper_operations_cuda.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cuda.h>

#include "../../../../src/nn_models/port_checkpoint/raptor/policy.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace rlt = rl_tools;

using LOGGER = rlt::devices::logging::CPU;
using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE::index_t;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;

namespace l2f = rlt::rl::environments::l2f;
namespace obs = l2f::observation;

using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
static constexpr T SIMULATION_DT = static_cast<T>(0.01);
static constexpr TI VIDEO_FPS = static_cast<TI>(static_cast<T>(1) / SIMULATION_DT + static_cast<T>(0.5));
static constexpr TI EPISODE_STEP_LIMIT = 10000;
static constexpr TI MOSAIC_GRID = 64;
static constexpr TI N_ENVIRONMENTS = MOSAIC_GRID * MOSAIC_GRID;
static constexpr TI MAX_SCENES = 128;
static constexpr TI MIN_SCENES = 64;
static constexpr TI N_ENVIRONMENTS_PER_SCENE = 64;
static constexpr TI WAYPOINTS_PER_ENV = 128;
static constexpr T WAYPOINT_REACHED_RADIUS = static_cast<T>(0.35);
static constexpr T WAYPOINT_MIN_DISTANCE = static_cast<T>(1.0);
static constexpr T TARGET_POSITION_ERROR_CLIP = static_cast<T>(1.0);
static constexpr TI CAM_WIDTH = 64;
static constexpr TI CAM_HEIGHT = 64;
static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
static constexpr TI NUM_PROBES = 64;
static constexpr T CAMERA_FOV = static_cast<T>(79.6) / static_cast<T>(180) * rlt::math::PI<T>;
static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = static_cast<T>(0);
static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_X = static_cast<T>(0);
static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Y = static_cast<T>(0);
static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE_Z = static_cast<T>(0);
static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_X = static_cast<T>(0);
static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Y = static_cast<T>(0);
static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE_Z = static_cast<T>(0);

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
    0.0, 0.0, 0.0, 0.0, 1.0, true, -1, +1,
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    false, 1.0, 0, 10, 35, 10000, 50000,
};
static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::Integration integration = {SIMULATION_DT};
static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, reward_function, {}, {}, termination};
static constexpr typename PARAMETERS_TYPE::IMU imu = {{static_cast<T>(0.02), static_cast<T>(0), static_cast<T>(0)}};
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
static constexpr PARAMETERS_TYPE nominal_parameters = {{{{dynamics, integration, mdp}, imu}, disturbances}, domain_randomization};

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

using RENDER_SHADING = rlt::rendering::raytracing::High;
static constexpr bool RENDER_ENABLE_MOTION_BLUR = false;
static constexpr TI RENDER_MOTION_BLUR_SAMPLES = 1;
static constexpr bool RENDER_ENABLE_ANTI_ALIASING = true;
static constexpr TI RENDER_ANTI_ALIASING_GRID_SIZE = 2;
using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, N_ENVIRONMENTS_PER_SCENE, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES, RENDER_SHADING, RENDER_ENABLE_MOTION_BLUR, RENDER_MOTION_BLUR_SAMPLES, RENDER_ENABLE_ANTI_ALIASING, RENDER_ANTI_ALIASING_GRID_SIZE>;
using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;
using CAMERA_DATA = rlt::rendering::raytracing::Camera<T>;

static constexpr TI RAPTOR_HIDDEN_DIM = 16;
using RAPTOR_OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI,
        obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
        obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
        obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI,
        obs::ActionHistory<obs::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
static constexpr TI RAPTOR_OBS_DIM = RAPTOR_OBSERVATION_TYPE::DIM;
using RAPTOR_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
using RAPTOR_DENSE1 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE1_CONFIG>;
using RAPTOR_GRU_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, RAPTOR_HIDDEN_DIM>;
using RAPTOR_GRU = rlt::nn::layers::gru::BindConfiguration<RAPTOR_GRU_CONFIG>;
using RAPTOR_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
using RAPTOR_DENSE2 = rlt::nn::layers::dense::BindConfiguration<RAPTOR_DENSE2_CONFIG>;
using RAPTOR_MODULE = rlt::nn_models::sequential::Module<RAPTOR_DENSE1, RAPTOR_GRU, RAPTOR_DENSE2>;
using RAPTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, RAPTOR_OBS_DIM>;
using RAPTOR_CAPABILITY = rlt::nn::capability::Forward<true, false>;
using RAPTOR_MODEL = rlt::nn_models::sequential::Build<RAPTOR_CAPABILITY, RAPTOR_MODULE, RAPTOR_INPUT_SHAPE>;

struct Options {
    std::string scene_path;
    std::string output_path = "l2f_visual_teacher_stitch.mp4";
    std::string ffmpeg = "ffmpeg";
    TI seed = 0;
    TI frames = 1000;
    TI scenes = MAX_SCENES;
    bool allow_repeat_scenes = false;
    bool help = false;
};

static int validate_scene_has_initial_states(const std::string& scene_path) {
    DEVICE device;
    rlt::init(device);

    ENVIRONMENT env;
    rlt::malloc(device, env);
    env.scene_path = scene_path.c_str();
    env.renderer_initialized = false;
    rlt::init(device, env);
    const bool valid = env.scene != nullptr && env.scene->num_indoor_positions > 0;
    rlt::free(device, env);
    return valid ? 0 : 1;
}

static void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --scene-dir <dir> [options]\n"
        << "       " << argv0 << " <scene-dir-or-glb> [options]\n"
        << "Options:\n"
        << "  --scene-dir <dir>             Directory containing .glb scenes\n"
        << "  --scene <path.glb>            Single scene; requires --allow-repeat-scenes for multi-scene mosaic\n"
        << "  --output <path.mp4>           Output MP4 path (default: l2f_visual_teacher_stitch.mp4)\n"
        << "  --ffmpeg <path>               ffmpeg binary (default: ffmpeg)\n"
        << "  --seed <n>                    Random seed (default: 0)\n"
        << "  --frames <n>                  Number of video frames (default: 1000)\n"
        << "  --scenes <n>                  Active scenes, 64..128 (default: 128)\n"
        << "  --allow-repeat-scenes         Repeat scene files if fewer than --scenes are available\n";
}

static bool parse_ti(const std::string& value, TI& out) {
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if(end == value.c_str() || *end != '\0') {
        return false;
    }
    out = static_cast<TI>(parsed);
    return true;
}

static bool option_value(int& i, int argc, char** argv, const std::string& arg, const char* name, std::string& out) {
    std::string prefix = std::string(name) + "=";
    if(arg.compare(0, prefix.size(), prefix) == 0) {
        out = arg.substr(prefix.size());
        return true;
    }
    if(arg == name) {
        if(i + 1 >= argc) {
            std::cerr << "Missing value for " << name << std::endl;
            return false;
        }
        out = argv[++i];
        return true;
    }
    return false;
}

static bool parse_options(int argc, char** argv, Options& options) {
    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        std::string value;
        if(arg == "-h" || arg == "--help") {
            options.help = true;
            return true;
        }
        else if(option_value(i, argc, argv, arg, "--scene-dir", value) || option_value(i, argc, argv, arg, "--scene", value)) {
            options.scene_path = value;
        }
        else if(option_value(i, argc, argv, arg, "--output", value)) {
            options.output_path = value;
        }
        else if(option_value(i, argc, argv, arg, "--ffmpeg", value)) {
            options.ffmpeg = value;
        }
        else if(option_value(i, argc, argv, arg, "--seed", value)) {
            if(!parse_ti(value, options.seed)) {
                std::cerr << "Invalid --seed: " << value << std::endl;
                return false;
            }
        }
        else if(option_value(i, argc, argv, arg, "--frames", value)) {
            if(!parse_ti(value, options.frames) || options.frames == 0) {
                std::cerr << "Invalid --frames: " << value << std::endl;
                return false;
            }
        }
        else if(option_value(i, argc, argv, arg, "--scenes", value)) {
            if(!parse_ti(value, options.scenes)) {
                std::cerr << "Invalid --scenes: " << value << std::endl;
                return false;
            }
        }
        else if(arg == "--allow-repeat-scenes") {
            options.allow_repeat_scenes = true;
        }
        else if(arg.size() > 0 && arg[0] != '-' && options.scene_path.empty()) {
            options.scene_path = arg;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    if(options.scene_path.empty() && !options.help) {
        std::cerr << "--scene-dir or --scene is required" << std::endl;
        return false;
    }
    if(options.scenes < MIN_SCENES || options.scenes > MAX_SCENES) {
        std::cerr << "--scenes must be in [" << MIN_SCENES << ", " << MAX_SCENES << "]" << std::endl;
        return false;
    }
    return true;
}

static std::string shell_quote(const std::string& value) {
    std::string out = "'";
    for(char c : value) {
        if(c == '\'') {
            out += "'\\''";
        }
        else {
            out += c;
        }
    }
    out += "'";
    return out;
}

static int scene_sort_key(const std::filesystem::path& path) {
    std::string stem = path.stem().string();
    size_t pos = stem.rfind('-');
    if(pos != std::string::npos) {
        try {
            return std::stoi(stem.substr(pos + 1));
        }
        catch(...) {}
    }
    return 0;
}

static bool scene_has_initial_states(const char* executable_path, const std::string& scene_path) {
    std::string command = shell_quote(executable_path) + " --validate-scene " + shell_quote(scene_path) + " >/dev/null 2>&1";
    return std::system(command.c_str()) == 0;
}

static bool collect_scene_paths(const Options& options, const char* executable_path, std::vector<std::string>& scene_paths) {
    std::vector<std::string> available;
    std::filesystem::path scene_path(options.scene_path);
    if(std::filesystem::is_directory(scene_path)) {
        for(const auto& entry : std::filesystem::directory_iterator(scene_path)) {
            if(entry.path().extension() == ".glb") {
                available.push_back(entry.path().string());
            }
        }
        std::sort(available.begin(), available.end(), [](const std::string& a, const std::string& b) {
            int ka = scene_sort_key(a);
            int kb = scene_sort_key(b);
            if(ka != kb) {
                return ka < kb;
            }
            return a < b;
        });
    }
    else if(std::filesystem::is_regular_file(scene_path)) {
        available.push_back(scene_path.string());
    }
    else {
        std::cerr << "Scene path does not exist: " << options.scene_path << std::endl;
        return false;
    }
    if(available.empty()) {
        std::cerr << "No .glb scenes found at: " << options.scene_path << std::endl;
        return false;
    }

    scene_paths.clear();
    for(const std::string& candidate_path : available) {
        std::cout << "Checking scene candidate: " << std::filesystem::path(candidate_path).filename().string() << std::flush;
        if(scene_has_initial_states(executable_path, candidate_path)) {
            scene_paths.push_back(candidate_path);
            std::cout << " ok" << std::endl;
            if(static_cast<TI>(scene_paths.size()) >= options.scenes) {
                break;
            }
        }
        else {
            std::cout << " skipped: no initial states" << std::endl;
        }
    }
    if(scene_paths.empty()) {
        std::cerr << "No scenes with valid initial states found at: " << options.scene_path << std::endl;
        return false;
    }
    if(static_cast<TI>(scene_paths.size()) < options.scenes && !options.allow_repeat_scenes) {
        std::cerr << "Need " << options.scenes << " scenes with valid initial states, found "
                  << scene_paths.size() << ". Use --allow-repeat-scenes to repeat usable files." << std::endl;
        return false;
    }
    return true;
}

static T squared_distance(const T* a, const T* b) {
    T dx = a[0] - b[0];
    T dy = a[1] - b[1];
    T dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

template <typename SCENE>
static void build_route_for_env(const SCENE& scene, TI env_i, std::mt19937_64& route_rng, std::vector<T>& route_positions) {
    const TI count = scene.num_indoor_positions;
    const TI base = env_i * WAYPOINTS_PER_ENV * 3;
    if(count == 0) {
        for(TI waypoint_i = 0; waypoint_i < WAYPOINTS_PER_ENV; waypoint_i++) {
            route_positions[base + waypoint_i * 3 + 0] = 0;
            route_positions[base + waypoint_i * 3 + 1] = 0;
            route_positions[base + waypoint_i * 3 + 2] = 0;
        }
        return;
    }
    std::vector<unsigned char> used(count, 0);
    std::uniform_int_distribution<unsigned long long> initial_position_distribution(0, count - 1);
    TI current = static_cast<TI>(initial_position_distribution(route_rng));
    for(TI waypoint_i = 0; waypoint_i < WAYPOINTS_PER_ENV; waypoint_i++) {
        const auto& p = scene.indoor_positions[current].position;
        route_positions[base + waypoint_i * 3 + 0] = p[0];
        route_positions[base + waypoint_i * 3 + 1] = p[1];
        route_positions[base + waypoint_i * 3 + 2] = p[2];
        used[current] = 1;

        TI next = current;
        T best = std::numeric_limits<T>::max();
        T current_pos[3] = {p[0], p[1], p[2]};
        T min_dist_sq = WAYPOINT_MIN_DISTANCE * WAYPOINT_MIN_DISTANCE;
        for(TI candidate_i = 0; candidate_i < count; candidate_i++) {
            if(used[candidate_i]) {
                continue;
            }
            const auto& c = scene.indoor_positions[candidate_i].position;
            T candidate_pos[3] = {c[0], c[1], c[2]};
            T dist_sq = squared_distance(current_pos, candidate_pos);
            if(dist_sq >= min_dist_sq && dist_sq < best) {
                best = dist_sq;
                next = candidate_i;
            }
        }
        if(next == current) {
            for(TI candidate_i = 0; candidate_i < count; candidate_i++) {
                if(candidate_i == current || used[candidate_i]) {
                    continue;
                }
                const auto& c = scene.indoor_positions[candidate_i].position;
                T candidate_pos[3] = {c[0], c[1], c[2]};
                T dist_sq = squared_distance(current_pos, candidate_pos);
                if(dist_sq < best) {
                    best = dist_sq;
                    next = candidate_i;
                }
            }
        }
        if(next == current) {
            std::fill(used.begin(), used.end(), 0);
            next = static_cast<TI>(initial_position_distribution(route_rng));
        }
        current = next;
    }
}

namespace stitch_kernels {
    __device__ T clamp_target_error(T value) {
        return value < -TARGET_POSITION_ERROR_CLIP ? -TARGET_POSITION_ERROR_CLIP :
               value >  TARGET_POSITION_ERROR_CLIP ?  TARGET_POSITION_ERROR_CLIP : value;
    }

    template<typename DEVICE, typename RNG>
    __global__ void reset_observe_kernel(
        DEVICE device,
        ENVIRONMENT* envs,
        typename ENVIRONMENT::Parameters* env_params,
        typename ENVIRONMENT::State* states,
        bool* terminated,
        TI* episode_steps,
        TI* waypoint_indices,
        const T* route_positions,
        T* teacher_obs,
        T* raptor_gru_state,
        const T* raptor_gru_initial_hidden,
        TI* raptor_gru_step,
        RNG rng
    ) {
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) {
            return;
        }
        auto& rng_state = rl_tools::get(rng.states, 0, env_i);
        auto& env = envs[env_i];
        auto& params = env_params[env_i];
        auto& state = states[env_i];
        if(terminated[env_i] || episode_steps[env_i] >= EPISODE_STEP_LIMIT) {
            rl_tools::sample_initial_parameters(device, env, params, rng_state);
            rl_tools::initial_state(device, env.dynamics, params.dynamics, state);
            TI route_base = env_i * WAYPOINTS_PER_ENV * 3;
            for(TI axis_i = 0; axis_i < 3; axis_i++) {
                state.position[axis_i] = route_positions[route_base + axis_i];
            }
            waypoint_indices[env_i] = WAYPOINTS_PER_ENV > 1 ? 1 : 0;
            episode_steps[env_i] = 0;
            terminated[env_i] = false;
            for(TI hidden_i = 0; hidden_i < RAPTOR_HIDDEN_DIM; hidden_i++) {
                raptor_gru_state[env_i * RAPTOR_HIDDEN_DIM + hidden_i] = raptor_gru_initial_hidden[hidden_i];
            }
            raptor_gru_step[env_i] = 0;
        }

        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, RAPTOR_OBS_DIM, true, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> obs_mat;
        obs_mat._data = teacher_obs + env_i * RAPTOR_OBS_DIM;
        rl_tools::observe(device, env.dynamics, params.dynamics, state, RAPTOR_OBSERVATION_TYPE{}, obs_mat, rng_state);

        TI target_i = waypoint_indices[env_i] % WAYPOINTS_PER_ENV;
        const T* target = route_positions + (env_i * WAYPOINTS_PER_ENV + target_i) * 3;
        for(TI axis_i = 0; axis_i < 3; axis_i++) {
            obs_mat._data[axis_i] = clamp_target_error(state.position[axis_i] - target[axis_i]);
        }
    }

    template<typename DEVICE>
    __global__ void make_cameras_kernel(
        DEVICE device,
        const typename ENVIRONMENT::Parameters* env_params,
        const typename ENVIRONMENT::State* states,
        CAMERA_DATA* cameras,
        const T* scene_translation,
        const T* scene_yaw_cos,
        const T* scene_yaw_sin,
        T aspect
    ) {
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) {
            return;
        }
        cameras[env_i] = rlt::rl::environments::l2f_visual::cuda::make_camera_for_state<DEVICE, VISUAL_SPEC>(
            device,
            env_params[env_i],
            states[env_i],
            aspect,
            scene_translation + env_i * 3,
            scene_yaw_cos[env_i],
            scene_yaw_sin[env_i]
        );
    }

    __global__ void gather_scene_cameras_kernel(
        const CAMERA_DATA* global_cameras,
        CAMERA_DATA* scene_cameras,
        const TI* scene_env_indices,
        TI scene_i
    ) {
        TI local_camera_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(local_camera_i >= N_ENVIRONMENTS_PER_SCENE) {
            return;
        }
        TI index_offset = scene_i * N_ENVIRONMENTS_PER_SCENE + local_camera_i;
        TI env_i = scene_env_indices[index_offset];
        if(env_i >= N_ENVIRONMENTS) {
            env_i = scene_env_indices[scene_i * N_ENVIRONMENTS_PER_SCENE];
            if(env_i >= N_ENVIRONMENTS) {
                env_i = 0;
            }
        }
        scene_cameras[index_offset] = global_cameras[env_i];
    }

    __global__ void scatter_scene_framebuffer_kernel(
        const uint32_t* framebuffer,
        uint8_t* mosaic,
        const TI* scene_env_indices,
        TI scene_i
    ) {
        TI tid = threadIdx.x + blockIdx.x * blockDim.x;
        TI total = N_ENVIRONMENTS_PER_SCENE * CAM_PIXELS;
        if(tid >= total) {
            return;
        }
        TI local_camera_i = tid / CAM_PIXELS;
        TI pixel_i = tid % CAM_PIXELS;
        TI env_i = scene_env_indices[scene_i * N_ENVIRONMENTS_PER_SCENE + local_camera_i];
        if(env_i >= N_ENVIRONMENTS) {
            return;
        }
        TI env_row = env_i / MOSAIC_GRID;
        TI env_col = env_i % MOSAIC_GRID;
        TI py = pixel_i / CAM_WIDTH;
        TI px = pixel_i % CAM_WIDTH;
        TI mosaic_width = MOSAIC_GRID * CAM_WIDTH;
        TI dst = ((env_row * CAM_HEIGHT + py) * mosaic_width + env_col * CAM_WIDTH + px) * 3;
        uint32_t rgba = framebuffer[local_camera_i * CAM_PIXELS + pixel_i];
        mosaic[dst + 0] = static_cast<uint8_t>((rgba >> 0) & 0xff);
        mosaic[dst + 1] = static_cast<uint8_t>((rgba >> 8) & 0xff);
        mosaic[dst + 2] = static_cast<uint8_t>((rgba >> 16) & 0xff);
    }

    template<typename DEVICE, typename RNG>
    __global__ void step_teacher_kernel(
        DEVICE device,
        ENVIRONMENT* envs,
        typename ENVIRONMENT::Parameters* env_params,
        typename ENVIRONMENT::State* states,
        bool* terminated,
        TI* episode_steps,
        TI* waypoint_indices,
        const T* route_positions,
        const T* teacher_actions,
        RNG rng
    ) {
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i >= N_ENVIRONMENTS) {
            return;
        }
        auto& rng_state = rl_tools::get(rng.states, 0, env_i);
        T action_arr[ENVIRONMENT::ACTION_DIM];
        for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++) {
            action_arr[action_i] = teacher_actions[env_i * ENVIRONMENT::ACTION_DIM + action_i];
        }
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM, true, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> action_matrix;
        action_matrix._data = action_arr;
        typename ENVIRONMENT::State next_state;
        rl_tools::step(device, envs[env_i], env_params[env_i], states[env_i], action_matrix, next_state, rng_state);
        terminated[env_i] = rl_tools::terminated(device, envs[env_i], env_params[env_i], next_state, rng_state);
        states[env_i] = next_state;
        episode_steps[env_i]++;

        TI target_i = waypoint_indices[env_i] % WAYPOINTS_PER_ENV;
        const T* target = route_positions + (env_i * WAYPOINTS_PER_ENV + target_i) * 3;
        T dx = next_state.position[0] - target[0];
        T dy = next_state.position[1] - target[1];
        T dz = next_state.position[2] - target[2];
        T dist_sq = dx * dx + dy * dy + dz * dz;
        if(dist_sq < WAYPOINT_REACHED_RADIUS * WAYPOINT_REACHED_RADIUS) {
            waypoint_indices[env_i] = (target_i + 1) % WAYPOINTS_PER_ENV;
        }
    }
}

int main(int argc, char** argv) {
#define CUDA_CHECK(MSG) do { \
    cudaError_t e = cudaGetLastError(); \
    if(e != cudaSuccess) { std::cerr << "CUDA error [" << MSG << "]: " << cudaGetErrorString(e) << std::endl; return 1; } \
} while(false)

    if(argc == 3 && std::string(argv[1]) == "--validate-scene") {
        return validate_scene_has_initial_states(argv[2]);
    }

    Options options;
    if(!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 1;
    }
    if(options.help) {
        print_usage(argv[0]);
        return 0;
    }

    std::vector<std::string> scene_paths;
    if(!collect_scene_paths(options, argv[0], scene_paths)) {
        return 1;
    }

    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device);
    rlt::init(device_gpu);

    RNG rng;
    RNG_GPU rng_gpu;
    rlt::malloc(device, rng);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device, rng, options.seed);
    rlt::init(device_gpu, rng_gpu, options.seed);

    using RENDERER_TYPE = rlt::rendering::raytracing::Renderer<typename ENVIRONMENT::SPEC::RENDERER_SPEC>;
    using SCENE_TYPE = rlt::rendering::raytracing::scene::procthor::Scene<typename ENVIRONMENT::SPEC::SCENE_SPEC>;
    std::array<RENDERER_TYPE*, MAX_SCENES> renderers{};
    std::array<SCENE_TYPE*, MAX_SCENES> scenes{};

    {
        TI scene_i = 0;
        for(; scene_i < options.scenes; scene_i++) {
            const std::string& scene_path = scene_paths[scene_i % scene_paths.size()];
            std::cout << "Loading scene " << scene_i << "/" << options.scenes << ": "
                      << std::filesystem::path(scene_path).filename().string() << std::flush;
            ENVIRONMENT loader_env;
            rlt::malloc(device, loader_env);
            loader_env.scene_path = scene_path.c_str();
            loader_env.renderer_initialized = false;
            rlt::init(device, loader_env);
            if(loader_env.scene->num_indoor_positions == 0) {
                std::cerr << "\nScene passed preflight but has no valid free-space positions: " << scene_path << std::endl;
                return 1;
            }
            std::cout << " (" << loader_env.scene->num_indoor_positions << " free-space points)" << std::endl;
            renderers[scene_i] = loader_env.renderer;
            scenes[scene_i] = loader_env.scene;
            loader_env.renderer = nullptr;
            loader_env.scene = nullptr;
            loader_env.owns_renderer = false;
        }
    }

    std::vector<TI> env_scene(N_ENVIRONMENTS);
    std::vector<TI> scene_env_indices(MAX_SCENES * N_ENVIRONMENTS_PER_SCENE, N_ENVIRONMENTS);
    std::vector<TI> scene_counts(options.scenes, 0);
    for(TI row = 0; row < MOSAIC_GRID; row++) {
        for(TI col = 0; col < MOSAIC_GRID; col++) {
            TI env_i = row * MOSAIC_GRID + col;
            TI scene_i = (row * 17 + col * 31) % options.scenes;
            TI local_i = scene_counts[scene_i]++;
            if(local_i >= N_ENVIRONMENTS_PER_SCENE) {
                std::cerr << "Scene " << scene_i << " received more than " << N_ENVIRONMENTS_PER_SCENE
                          << " mosaic cells. Increase --scenes." << std::endl;
                return 1;
            }
            env_scene[env_i] = scene_i;
            scene_env_indices[scene_i * N_ENVIRONMENTS_PER_SCENE + local_i] = env_i;
        }
    }

    std::vector<T> route_positions(N_ENVIRONMENTS * WAYPOINTS_PER_ENV * 3);
    std::mt19937_64 route_rng(options.seed);
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++) {
        build_route_for_env(*scenes[env_scene[env_i]], env_i, route_rng, route_positions);
    }

    std::vector<ENVIRONMENT> envs(N_ENVIRONMENTS);
    std::vector<typename ENVIRONMENT::Parameters> env_parameters(N_ENVIRONMENTS);
    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++) {
        TI scene_i = env_scene[env_i];
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
    }

    RAPTOR_MODEL raptor;
    typename RAPTOR_MODEL::Buffer<true> raptor_buffer;
    typename RAPTOR_MODEL::State<true> raptor_state;
    rlt::malloc(device, raptor);
    rlt::malloc(device, raptor_buffer);
    rlt::malloc(device, raptor_state);
    rlt::copy(device, device, rl_tools::checkpoint::actor::module, raptor);
    rlt::reset(device, raptor, raptor_state, rng);

    RAPTOR_MODEL raptor_gpu;
    typename RAPTOR_MODEL::Buffer<true> raptor_buffer_gpu;
    typename RAPTOR_MODEL::State<true> raptor_state_gpu;
    rlt::malloc(device_gpu, raptor_gpu);
    rlt::malloc(device_gpu, raptor_buffer_gpu);
    rlt::malloc(device_gpu, raptor_state_gpu);
    rlt::copy(device, device_gpu, raptor, raptor_gpu);
    rlt::reset(device_gpu, raptor_gpu, raptor_state_gpu, rng_gpu);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, RAPTOR_OBS_DIM>>> gpu_teacher_obs;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS, ENVIRONMENT::ACTION_DIM>>> gpu_teacher_actions;
    rlt::malloc(device_gpu, gpu_teacher_obs);
    rlt::malloc(device_gpu, gpu_teacher_actions);

    ENVIRONMENT* gpu_envs = nullptr;
    typename ENVIRONMENT::Parameters* gpu_params = nullptr;
    typename ENVIRONMENT::State* gpu_states = nullptr;
    bool* gpu_terminated = nullptr;
    TI* gpu_episode_steps = nullptr;
    TI* gpu_waypoint_indices = nullptr;
    T* gpu_route_positions = nullptr;
    T* gpu_scene_translation = nullptr;
    T* gpu_scene_yaw_cos = nullptr;
    T* gpu_scene_yaw_sin = nullptr;
    CAMERA_DATA* gpu_cameras = nullptr;
    CAMERA_DATA* gpu_scene_cameras = nullptr;
    TI* gpu_scene_env_indices = nullptr;
    uint8_t* gpu_mosaic = nullptr;

    cudaMalloc(&gpu_envs, N_ENVIRONMENTS * sizeof(ENVIRONMENT));
    cudaMalloc(&gpu_params, N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::Parameters));
    cudaMalloc(&gpu_states, N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::State));
    cudaMalloc(&gpu_terminated, N_ENVIRONMENTS * sizeof(bool));
    cudaMalloc(&gpu_episode_steps, N_ENVIRONMENTS * sizeof(TI));
    cudaMalloc(&gpu_waypoint_indices, N_ENVIRONMENTS * sizeof(TI));
    cudaMalloc(&gpu_route_positions, route_positions.size() * sizeof(T));
    cudaMalloc(&gpu_scene_translation, N_ENVIRONMENTS * 3 * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_cos, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_scene_yaw_sin, N_ENVIRONMENTS * sizeof(T));
    cudaMalloc(&gpu_cameras, N_ENVIRONMENTS * sizeof(CAMERA_DATA));
    cudaMalloc(&gpu_scene_cameras, MAX_SCENES * N_ENVIRONMENTS_PER_SCENE * sizeof(CAMERA_DATA));
    cudaMalloc(&gpu_scene_env_indices, scene_env_indices.size() * sizeof(TI));
    static constexpr TI MOSAIC_W = MOSAIC_GRID * CAM_WIDTH;
    static constexpr TI MOSAIC_H = MOSAIC_GRID * CAM_HEIGHT;
    static constexpr size_t MOSAIC_BYTES = static_cast<size_t>(MOSAIC_W) * static_cast<size_t>(MOSAIC_H) * 3;
    cudaMalloc(&gpu_mosaic, MOSAIC_BYTES);
    CUDA_CHECK("allocation");

    cudaMemcpy(gpu_envs, envs.data(), N_ENVIRONMENTS * sizeof(ENVIRONMENT), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_params, env_parameters.data(), N_ENVIRONMENTS * sizeof(typename ENVIRONMENT::Parameters), cudaMemcpyHostToDevice);
    std::vector<unsigned char> terminated_init(N_ENVIRONMENTS, 1);
    std::vector<TI> episode_steps_init(N_ENVIRONMENTS, 0);
    std::vector<TI> waypoint_indices_init(N_ENVIRONMENTS, 1);
    std::vector<T> scene_translation_init(N_ENVIRONMENTS * 3, 0);
    std::vector<T> scene_yaw_cos_init(N_ENVIRONMENTS, 1);
    std::vector<T> scene_yaw_sin_init(N_ENVIRONMENTS, 0);
    cudaMemcpy(gpu_terminated, terminated_init.data(), N_ENVIRONMENTS * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_episode_steps, episode_steps_init.data(), N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_waypoint_indices, waypoint_indices_init.data(), N_ENVIRONMENTS * sizeof(TI), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_route_positions, route_positions.data(), route_positions.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_scene_translation, scene_translation_init.data(), scene_translation_init.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_scene_yaw_cos, scene_yaw_cos_init.data(), scene_yaw_cos_init.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_scene_yaw_sin, scene_yaw_sin_init.data(), scene_yaw_sin_init.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_scene_env_indices, scene_env_indices.data(), scene_env_indices.size() * sizeof(TI), cudaMemcpyHostToDevice);
    CUDA_CHECK("initial copy");

    cudaEvent_t cameras_ready_event;
    cudaEventCreateWithFlags(&cameras_ready_event, cudaEventDisableTiming);
    std::vector<cudaEvent_t> scene_done_events(options.scenes);
    std::vector<cudaStream_t> scene_streams(options.scenes);
    std::vector<void*> scene_camera_buffers(options.scenes);
    std::vector<const uint32_t*> scene_framebuffers(options.scenes);
    for(TI scene_i = 0; scene_i < options.scenes; scene_i++) {
        cudaEventCreateWithFlags(&scene_done_events[scene_i], cudaEventDisableTiming);
        OWLParams rgb_lp = (OWLParams)renderers[scene_i]->backend.launch_params;
        scene_streams[scene_i] = (cudaStream_t)owlParamsGetCudaStream(rgb_lp, 0);
        scene_camera_buffers[scene_i] = (void*)owlBufferGetPointer((OWLBuffer)renderers[scene_i]->backend.cameras_buffer, 0);
        scene_framebuffers[scene_i] = rlt::get_framebuffer_device_ptr(device, *renderers[scene_i]);
    }

    std::string ffmpeg_cmd;
    {
        std::ostringstream cmd;
        cmd << shell_quote(options.ffmpeg)
            << " -y -f rawvideo -pixel_format rgb24"
            << " -video_size " << MOSAIC_W << "x" << MOSAIC_H
            << " -framerate " << VIDEO_FPS
            << " -i - -c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast -loglevel warning "
            << shell_quote(options.output_path);
        ffmpeg_cmd = cmd.str();
    }
    FILE* ffmpeg_pipe = popen(ffmpeg_cmd.c_str(), "w");
    if(ffmpeg_pipe == nullptr) {
        std::cerr << "Failed to start ffmpeg: " << ffmpeg_cmd << std::endl;
        return 1;
    }

    std::vector<uint8_t> host_mosaic(MOSAIC_BYTES);
    constexpr TI BLOCK = 256;
    constexpr TI ENV_BLOCKS = (N_ENVIRONMENTS + BLOCK - 1) / BLOCK;
    constexpr TI CAMERA_BLOCKS = (N_ENVIRONMENTS_PER_SCENE + BLOCK - 1) / BLOCK;
    constexpr TI SCATTER_BLOCKS = (N_ENVIRONMENTS_PER_SCENE * CAM_PIXELS + BLOCK - 1) / BLOCK;
    rlt::devices::cuda::TAG<DEVICE_GPU, true> tag_device{};
    using NO_AUTO_RESET_MODE = rlt::Mode<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Default<>>>;
    NO_AUTO_RESET_MODE no_auto_reset_mode;
    auto& raptor_gru_layer = rlt::nn_models::sequential::layer<1>(raptor_gpu);
    auto& raptor_gru_state_content = rlt::nn_models::sequential::content_state<1>(raptor_state_gpu.content_state);
    T aspect = static_cast<T>(CAM_WIDTH) / static_cast<T>(CAM_HEIGHT);

    std::cout << "Writing " << options.frames << " frames at " << VIDEO_FPS << " fps"
              << " (" << MOSAIC_W << "x" << MOSAIC_H << ") to " << options.output_path << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for(TI frame_i = 0; frame_i < options.frames; frame_i++) {
        stitch_kernels::reset_observe_kernel<<<ENV_BLOCKS, BLOCK, 0, device_gpu.stream>>>(
            tag_device,
            gpu_envs,
            gpu_params,
            gpu_states,
            gpu_terminated,
            gpu_episode_steps,
            gpu_waypoint_indices,
            gpu_route_positions,
            rlt::data(gpu_teacher_obs),
            rlt::data(raptor_gru_state_content.state),
            rlt::data(raptor_gru_layer.initial_hidden_state.parameters),
            rlt::data(raptor_gru_state_content.step),
            rng_gpu
        );
        stitch_kernels::make_cameras_kernel<<<ENV_BLOCKS, BLOCK, 0, device_gpu.stream>>>(
            tag_device,
            gpu_params,
            gpu_states,
            gpu_cameras,
            gpu_scene_translation,
            gpu_scene_yaw_cos,
            gpu_scene_yaw_sin,
            aspect
        );
        for(TI scene_i = 0; scene_i < options.scenes; scene_i++) {
            stitch_kernels::gather_scene_cameras_kernel<<<CAMERA_BLOCKS, BLOCK, 0, device_gpu.stream>>>(
                gpu_cameras,
                gpu_scene_cameras,
                gpu_scene_env_indices,
                scene_i
            );
        }
        rlt::evaluate_step(device_gpu, raptor_gpu, gpu_teacher_obs, raptor_state_gpu, gpu_teacher_actions, raptor_buffer_gpu, rng_gpu, no_auto_reset_mode);
        stitch_kernels::step_teacher_kernel<<<ENV_BLOCKS, BLOCK, 0, device_gpu.stream>>>(
            tag_device,
            gpu_envs,
            gpu_params,
            gpu_states,
            gpu_terminated,
            gpu_episode_steps,
            gpu_waypoint_indices,
            gpu_route_positions,
            rlt::data(gpu_teacher_actions),
            rng_gpu
        );
        cudaEventRecord(cameras_ready_event, device_gpu.stream);
        for(TI scene_i = 0; scene_i < options.scenes; scene_i++) {
            cudaStream_t stream = scene_streams[scene_i];
            cudaStreamWaitEvent(stream, cameras_ready_event, 0);
            cudaMemcpyAsync(
                scene_camera_buffers[scene_i],
                gpu_scene_cameras + scene_i * N_ENVIRONMENTS_PER_SCENE,
                N_ENVIRONMENTS_PER_SCENE * sizeof(CAMERA_DATA),
                cudaMemcpyDeviceToDevice,
                stream
            );
            rlt::render_launch(device, *renderers[scene_i]);
            stitch_kernels::scatter_scene_framebuffer_kernel<<<SCATTER_BLOCKS, BLOCK, 0, stream>>>(
                scene_framebuffers[scene_i],
                gpu_mosaic,
                gpu_scene_env_indices,
                scene_i
            );
            cudaEventRecord(scene_done_events[scene_i], stream);
        }
        for(TI scene_i = 0; scene_i < options.scenes; scene_i++) {
            cudaStreamWaitEvent(device_gpu.stream, scene_done_events[scene_i], 0);
        }
        cudaMemcpyAsync(host_mosaic.data(), gpu_mosaic, MOSAIC_BYTES, cudaMemcpyDeviceToHost, device_gpu.stream);
        cudaStreamSynchronize(device_gpu.stream);
        CUDA_CHECK("frame");

        size_t written = std::fwrite(host_mosaic.data(), 1, host_mosaic.size(), ffmpeg_pipe);
        if(written != host_mosaic.size()) {
            std::cerr << "Failed writing frame " << frame_i << " to ffmpeg" << std::endl;
            pclose(ffmpeg_pipe);
            return 1;
        }
        if((frame_i + 1) % 10 == 0 || frame_i + 1 == options.frames) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> elapsed = now - start_time;
            T fps = elapsed.count() > 0 ? static_cast<T>(frame_i + 1) / elapsed.count() : static_cast<T>(0);
            std::cout << "Frame " << (frame_i + 1) << "/" << options.frames
                      << " render_rate=" << fps << " fps" << std::endl;
        }
    }

    int ffmpeg_status = pclose(ffmpeg_pipe);
    ffmpeg_pipe = nullptr;
    if(ffmpeg_status != 0) {
        std::cerr << "ffmpeg exited with status " << ffmpeg_status << std::endl;
        return 1;
    }

    cudaEventDestroy(cameras_ready_event);
    for(TI scene_i = 0; scene_i < options.scenes; scene_i++) {
        cudaEventDestroy(scene_done_events[scene_i]);
    }

    rlt::free(device_gpu, gpu_teacher_obs);
    rlt::free(device_gpu, gpu_teacher_actions);
    rlt::free(device_gpu, raptor_gpu);
    rlt::free(device_gpu, raptor_buffer_gpu);
    rlt::free(device_gpu, raptor_state_gpu);
    rlt::free(device_gpu, rng_gpu);
    cudaFree(gpu_envs);
    cudaFree(gpu_params);
    cudaFree(gpu_states);
    cudaFree(gpu_terminated);
    cudaFree(gpu_episode_steps);
    cudaFree(gpu_waypoint_indices);
    cudaFree(gpu_route_positions);
    cudaFree(gpu_scene_translation);
    cudaFree(gpu_scene_yaw_cos);
    cudaFree(gpu_scene_yaw_sin);
    cudaFree(gpu_cameras);
    cudaFree(gpu_scene_cameras);
    cudaFree(gpu_scene_env_indices);
    cudaFree(gpu_mosaic);

    rlt::free(device, raptor);
    rlt::free(device, raptor_buffer);
    rlt::free(device, raptor_state);
    rlt::free(device, rng);

    for(TI env_i = 0; env_i < N_ENVIRONMENTS; env_i++) {
        envs[env_i].renderer = nullptr;
        envs[env_i].scene = nullptr;
        rlt::free(device, envs[env_i].dynamics);
    }
    for(TI scene_i = 0; scene_i < options.scenes; scene_i++) {
        if(renderers[scene_i] != nullptr) {
            rlt::free(device, *renderers[scene_i]);
            delete renderers[scene_i];
        }
        if(scenes[scene_i] != nullptr) {
            delete scenes[scene_i];
        }
    }

    std::cout << "Video written: " << options.output_path << std::endl;
    return 0;
}
