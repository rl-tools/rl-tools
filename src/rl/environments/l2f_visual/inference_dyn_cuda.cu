#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cuda.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>

#include <rl_tools/persist/backends/tar/operations_cpu.h>
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
#include <rl_tools/persist/backends/hdf5/hdf5.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#endif

#include <rl_tools/dyn/model.h>
#include <rl_tools/dyn/operations_generic.h>
#include <rl_tools/dyn/persist.h>

#include <rl_tools/utils/extrack/operations_cpu.h>

#include <conta/conta.h>

#include <array>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cstring>
#include <string>
#include <filesystem>
#include <fstream>

namespace rlt = rl_tools;

// =========================================================================
// Device types
// =========================================================================
using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, rlt::devices::logging::CPU>;
using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;

using T = float;
using TI = typename DEVICE::index_t;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;

// =========================================================================
// L2F dynamics configuration (identical to training)
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
    false, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    {0.00, 0.00, 0.00, 0.00}, {0.00, 0.00, 0.00, 0.00}, 0.00
};
static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
    0.2, 0.0, 0.3, 0.0, 1.0, true, -1, +1,
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    true, 1.0, 0, 10, 35, 10000, 50000,
};
static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::Integration integration = {
    static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)
};
static constexpr typename PARAMETERS_TYPE::MDP mdp = { init, reward_function, {}, {}, termination };
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = { {0, 0}, {0, 0} };
static constexpr PARAMETERS_TYPE nominal_parameters = { {dynamics, integration, mdp}, disturbances };

// =========================================================================
// Environment static parameters (identical to training)
// =========================================================================
static constexpr TI ACTION_HISTORY_LENGTH = 4;
static constexpr TI ACTION_DIM = 4;

struct STATIC_PARAMETERS {
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI CLOSED_FORM = false;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>>>;
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

using ACTOR_STATE_OBS = obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI, obs::LinearAccelerationBodyFrame<obs::LinearAccelerationBodyFrameSpecification<T, TI>>>>;
static constexpr TI STATE_OBS_DIM = ACTOR_STATE_OBS::DIM;

// =========================================================================
// Visual environment specification
// =========================================================================
static constexpr TI N_ENVIRONMENTS = 1;
static constexpr TI CAM_WIDTH = 80;
static constexpr TI CAM_HEIGHT = 50;
static constexpr TI NUM_PROBES = 64;
using RENDER_SHADING = rlt::rendering::raytracing::High;

using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, N_ENVIRONMENTS, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES, RENDER_SHADING>;
static_assert(VISUAL_SPEC::RENDERER_SPEC::SHADING::PBR_SHADING, "l2f visual inference must use the High renderer profile");
using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;

// =========================================================================
// Frame stacking configuration (identical to training)
// =========================================================================
static constexpr TI IMG_H = CAM_HEIGHT;
static constexpr TI IMG_W = CAM_WIDTH;
static constexpr TI IMG_C = 3;
static constexpr TI FRAME_STACK_N = 5;
static constexpr TI FRAME_STACK_STRIDE = 20;
static constexpr TI STACKED_IMG_C = IMG_C * FRAME_STACK_N;
static constexpr TI COMBINED_IMG_C = STACKED_IMG_C + IMG_C;
static constexpr TI IMG_PIXELS = IMG_H * IMG_W;
static constexpr TI OBS_DIM_SINGLE = IMG_PIXELS * IMG_C;
static constexpr TI STACKED_OBS_DIM = IMG_PIXELS * STACKED_IMG_C;
static constexpr TI COMBINED_OBS_DIM = IMG_PIXELS * COMBINED_IMG_C;
static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
static constexpr TI TOTAL_INPUT_DIM = COMBINED_OBS_DIM + STATE_OBS_DIM;
static constexpr TI FRAME_HISTORY_SIZE = (FRAME_STACK_N - 1) * FRAME_STACK_STRIDE + 1;

static_assert(OBSERVATION_DIM == OBS_DIM_SINGLE, "Single-frame observation dim mismatch");

// =========================================================================
// Rendering perturbation config (hooks for future robustness testing)
// =========================================================================
struct InferenceConfig {
    T brightness_scale = 1.0;
    T brightness_offset = 0.0;
    T gaussian_noise_std = 0.0;
    TI num_episodes = 10;
    TI seed = 0;
    bool record_video = true;
};

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

void render_target_frame(
    DEVICE& device,
    ENVIRONMENT& env,
    const typename ENVIRONMENT::Parameters& parameters,
    float* target_frame_out
){
    typename ENVIRONMENT::State target_state = {};
    target_state.orientation[0] = (T)1;
    target_state.orientation[1] = (T)0;
    target_state.orientation[2] = (T)0;
    target_state.orientation[3] = (T)0;
    auto camera = rlt::rl::environments::l2f_visual::make_camera_for_state(device, env, parameters, target_state);
    cudaMemcpy(rlt::data(rlt::cameras(device, *env.renderer)), &camera, sizeof(camera), cudaMemcpyHostToDevice);
    rlt::render(device, *env.renderer);
    std::vector<float> obs_staging((size_t)IMG_PIXELS * 3);
    cudaMemcpy(obs_staging.data(), rlt::data(rlt::observation(device, *env.renderer)), obs_staging.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for(TI p = 0; p < IMG_PIXELS; p++){
        target_frame_out[p * IMG_C + 0] = static_cast<T>(obs_staging[p * 3 + 0]);
        target_frame_out[p * IMG_C + 1] = static_cast<T>(obs_staging[p * 3 + 1]);
        target_frame_out[p * IMG_C + 2] = static_cast<T>(obs_staging[p * 3 + 2]);
    }
}

int main(int argc, char** argv){
    if(argc < 3){
        std::cerr << "Usage: " << argv[0] << " <conta:HASH or scene.glb> <checkpoint.tar|.h5> [seed] [num_episodes]" << std::endl;
        return 1;
    }

    InferenceConfig config;
    if(argc > 3) config.seed = std::atoi(argv[3]);
    if(argc > 4) config.num_episodes = std::atoi(argv[4]);

    // =====================================================================
    // Resolve scene path
    // =====================================================================
    std::string resolved_scene_path;
    rlt::rl::environments::l2f_visual::SceneHash scene_hash;
    const char* scene_arg = argv[1];
    if(std::strncmp(scene_arg, "conta:", 6) == 0){
        const char* hash_str = scene_arg + 6;
        if(std::strlen(hash_str) != 40){
            std::cerr << "Invalid conta hash: expected 40 hex characters, got " << std::strlen(hash_str) << std::endl;
            return 1;
        }
        if(!parse_hex_hash(hash_str, scene_hash.hash, rlt::rl::environments::l2f_visual::SceneHash::HASH_SIZE)){
            std::cerr << "Invalid conta hash: contains non-hex characters" << std::endl;
            return 1;
        }
        std::string conta_error;
        if(!conta::resolve(hash_str, resolved_scene_path, conta_error)){
            std::cerr << conta_error << std::endl;
            return 1;
        }
    } else {
        resolved_scene_path = scene_arg;
        std::memset(scene_hash.hash, 0, rlt::rl::environments::l2f_visual::SceneHash::HASH_SIZE);
    }
    const char* scene_path = resolved_scene_path.c_str();
    const char* checkpoint_path = argv[2];

    // =====================================================================
    // Device and extrack
    // =====================================================================
    DEVICE device;
    rlt::init(device);

    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = "l2f_visual_inference_dyn";
    rlt::init(device, extrack_config, extrack_paths, config.seed);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, config.seed);

    // =====================================================================
    // Load checkpoint via dyn
    // =====================================================================
    std::cout << "Loading checkpoint: " << checkpoint_path << std::endl;
    rlt::dyn::Layer<TI> model;
    std::string checkpoint_obs;

    // Tuple input: image branch [1, H, W, C] + state branch [1, STATE_OBS_DIM]
    rlt::dyn::TensorTuple<TI> dyn_inputs;
    dyn_inputs.num_tensors = 2;
    {
        TI img_shape[] = {(TI)1, IMG_H, IMG_W, COMBINED_IMG_C};
        rlt::dyn::set_shape(dyn_inputs.tensors[0], (TI)4, img_shape);
        dyn_inputs.tensors[0].type = rlt::dyn::Type::FLOAT32;
        rlt::malloc(device, dyn_inputs.tensors[0]);
        TI state_shape[] = {(TI)1, STATE_OBS_DIM};
        rlt::dyn::set_shape(dyn_inputs.tensors[1], (TI)2, state_shape);
        dyn_inputs.tensors[1].type = rlt::dyn::Type::FLOAT32;
        rlt::malloc(device, dyn_inputs.tensors[1]);
    }

    {
        std::string cp_path(checkpoint_path);
        bool is_hdf5 = (cp_path.size() >= 3 && cp_path.substr(cp_path.size() - 3) == ".h5");

        auto parse_meta_obs = [](const char* meta_buf) -> std::string {
            std::string meta_str(meta_buf);
            auto obs_pos = meta_str.find("\"observation\": \"");
            if(obs_pos != std::string::npos){
                obs_pos += 16;
                auto obs_end = meta_str.find("\"", obs_pos);
                if(obs_end != std::string::npos) return meta_str.substr(obs_pos, obs_end - obs_pos);
            }
            return "";
        };

        auto verify_example = [&](auto& file) {
            auto example_group = rlt::get_group(device, file, "example");
            auto inputs_group = rlt::get_group(device, example_group, "inputs");
            auto outputs_group = rlt::get_group(device, example_group, "outputs");
            rlt::dyn::TensorTuple<TI> example_tuple;
            example_tuple.num_tensors = 0;
            for(TI i = 0; i < rlt::dyn::TensorTuple<TI>::MAX_TENSORS; i++){
                char name[4] = {(char)('0' + i), '\0', '\0', '\0'};
                if(!rlt::load(device, example_tuple.tensors[i], inputs_group, name)) break;
                example_tuple.num_tensors = i + 1;
            }
            rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> example_output;
            bool have_example = example_tuple.num_tensors > 0 && rlt::load(device, example_output, outputs_group, "0");
            if(have_example){
                rlt::dyn::propagate_shapes(model, example_tuple);
                rlt::dyn::Buffer<TI> verify_buffer;
                verify_buffer.layer = &model;
                rlt::malloc(device, verify_buffer);
                rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> verify_output;
                TI verify_output_shape[] = {model.output_size};
                rlt::dyn::set_shape(verify_output, (TI)1, verify_output_shape);
                verify_output.type = rlt::dyn::Type::FLOAT32;
                rlt::malloc(device, verify_output);
                if(rlt::evaluate(device, model, example_tuple, verify_output, verify_buffer)){
                    float max_diff = 0;
                    for(TI i = 0; i < example_output.size(); i++){
                        float diff = std::fabs(rlt::get(device, verify_output, i) - rlt::get(device, example_output, i));
                        if(diff > max_diff) max_diff = diff;
                    }
                    std::cout << "Checkpoint verification: max_diff=" << max_diff << (max_diff < 1e-5f ? " PASS" : " WARNING: large difference") << std::endl;
                } else {
                    std::cerr << "WARNING: checkpoint verification evaluate failed" << std::endl;
                }
                rlt::free(device, verify_output);
                rlt::free(device, verify_buffer);
                for(TI i = 0; i < example_tuple.num_tensors; i++) rlt::free(device, example_tuple.tensors[i]);
                rlt::free(device, example_output);
            }
        };

        bool load_ok = false;
        if(is_hdf5){
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
            rlt::persist::backends::hdf5::File hdf5_file(checkpoint_path, rlt::persist::backends::hdf5::Mode::READ);
            auto actor_group = rlt::get_group(device, hdf5_file, "actor");
            load_ok = rlt::load(device, model, actor_group);
            if(load_ok){
                constexpr TI META_BUF_SIZE = 512;
                char meta_buf[META_BUF_SIZE]; meta_buf[0] = '\0';
                if(rlt::attribute_exists(device, actor_group, "meta")) rlt::get_attribute<char*>(device, actor_group, "meta", meta_buf, META_BUF_SIZE);
                checkpoint_obs = parse_meta_obs(meta_buf);
            }
            verify_example(hdf5_file);
#else
            std::cerr << "HDF5 support not compiled (RL_TOOLS_ENABLE_HDF5 not defined)" << std::endl;
            return 1;
#endif
        } else {
            rlt::persist::backends::tar::File<TI> tar_file(checkpoint_path, rlt::persist::backends::tar::Mode::READ);
            auto actor_group = rlt::get_group(device, tar_file, "actor");
            load_ok = rlt::load(device, model, actor_group);
            if(load_ok){
                constexpr TI META_BUF_SIZE = 512;
                char meta_buf[META_BUF_SIZE]; meta_buf[0] = '\0';
                if(rlt::attribute_exists(device, actor_group, "meta")) rlt::get_attribute<char*>(device, actor_group, "meta", meta_buf, META_BUF_SIZE);
                checkpoint_obs = parse_meta_obs(meta_buf);
            }
            verify_example(tar_file);
        }
        if(!load_ok){
            std::cerr << "Failed to load model from checkpoint" << std::endl;
            return 1;
        }
    }

    // Propagate shapes using tuple input
    rlt::dyn::propagate_shapes(model, dyn_inputs);

    // Allocate dyn buffers
    rlt::dyn::Buffer<TI> buffer;
    buffer.layer = &model;
    rlt::malloc(device, buffer);

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> dyn_output;
    TI output_shape[] = {model.output_size};
    rlt::dyn::set_shape(dyn_output, (TI)1, output_shape);
    dyn_output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_output);

    // =====================================================================
    // Environment setup
    // =====================================================================
    ENVIRONMENT env;
    typename ENVIRONMENT::Parameters env_parameters;
    rlt::malloc(device, env);
    using RENDERER_SPEC = typename ENVIRONMENT::SPEC::RENDERER_SPEC;
    using LIBRARY_TYPE = rlt::rendering::raytracing::AssetLibrary<RENDERER_SPEC>;
    using ANNOTATIONS_TYPE = rlt::rendering::datasets::procthor::Annotations<typename ENVIRONMENT::SPEC::ANNOTATIONS_SPEC>;
    auto* library = new LIBRARY_TYPE{};
    auto* renderer = new rlt::rendering::raytracing::Renderer<RENDERER_SPEC>{};
    auto* annotations = new ANNOTATIONS_TYPE{};
    rlt::malloc(device, *library);
    rlt::malloc(device, *renderer, *library);
    rlt::rendering::Bundle<T> bundle;
    rlt::load<typename RENDERER_SPEC::SHADING, RENDERER_SPEC::HAS_RGB>(device, bundle, scene_path);
    auto scene_id = rlt::insert(device, *library, bundle);
    rlt::init(device, *renderer, *library, scene_id);
    {
        const T scene_fov = typename ENVIRONMENT::Parameters{}.fov;
        rlt::generate_probe_directions(device, *renderer);
        rlt::rendering::datasets::procthor::annotate(device, *annotations, bundle.metadata, *renderer, scene_fov, (T)CAM_WIDTH / (T)CAM_HEIGHT);
    }
    env.renderer = renderer;
    env.annotations = annotations;
    env.use_target_mode = true;
    rlt::init(device, env);

    env_parameters.scene_hash = scene_hash;

    // Validate checkpoint observation config against our setup
    {
        char fov_buf[32];
        std::snprintf(fov_buf, sizeof(fov_buf), "%.6g", (double)env_parameters.fov);
        std::string expected_image_obs = std::string("CameraRGBStackedWithTarget(") + fov_buf + ", "
            + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ", "
            + std::to_string(FRAME_STACK_STRIDE) + ", " + std::to_string(FRAME_STACK_N) + ")";
        std::string expected_state_obs = rlt::string(device, env.dynamics, ACTOR_STATE_OBS{});
        std::string expected_obs_string = expected_image_obs + ", " + expected_state_obs;
        std::cout << "Checkpoint observation: \"" << checkpoint_obs << "\"" << std::endl;
        std::cout << "Expected observation:   \"" << expected_obs_string << "\"" << std::endl;
        if(!checkpoint_obs.empty() && checkpoint_obs != expected_obs_string){
            std::cerr << "FATAL: observation config mismatch between checkpoint and inference setup" << std::endl;
            return 1;
        }
    }

    {
        std::string ui = rlt::get_ui(device, env.dynamics);
        if(!ui.empty()){
            std::filesystem::create_directories(extrack_paths.seed);
            std::ofstream ui_file(extrack_paths.seed / "ui.esm.js");
            ui_file << ui;
        }
    }

    // =====================================================================
    // Inference buffers
    // =====================================================================
    std::vector<float> frame_history(FRAME_HISTORY_SIZE * OBS_DIM_SINGLE, 0.0f);
    std::vector<float> combined_obs(COMBINED_OBS_DIM, 0.0f);
    std::vector<float> target_frame_buf(OBS_DIM_SINGLE, 0.0f);
    std::vector<uint8_t> video_frame(IMG_PIXELS * 3);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, STATE_OBS_DIM, true, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> state_obs_mat;
    T state_obs_data[STATE_OBS_DIM];
    state_obs_mat._data = state_obs_data;

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_DIM, true, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> action_mat;
    T action_data[ACTION_DIM];
    action_mat._data = action_data;

    // =====================================================================
    // Noise sweep
    // =====================================================================
    std::cout << "Starting noise sweep" << std::endl;
    std::cout << "  Episodes per noise level: " << config.num_episodes << std::endl;
    std::cout << "  Seed: " << config.seed << std::endl;

    T noise_levels[] = {0.1, 0.25, 0.5, 0.75, 1.0};
    constexpr TI NUM_NOISE_LEVELS = sizeof(noise_levels) / sizeof(noise_levels[0]);

    std::cout << std::endl;
    std::cout << std::setw(12) << "brightness" << std::setw(14) << "mean_length" << std::setw(14) << "survival" << std::endl;
    std::cout << std::string(40, '-') << std::endl;

    TI global_step = 0;

    for(TI noise_i = 0; noise_i < NUM_NOISE_LEVELS; noise_i++){
    config.brightness_scale = noise_levels[noise_i];

    RNG sweep_rng;
    rlt::malloc(device, sweep_rng);
    rlt::init(device, sweep_rng, config.seed);

    std::vector<TI> episode_lengths;
    std::vector<bool> episode_terminated;

    for(TI episode_i = 0; episode_i < config.num_episodes; episode_i++){
        auto episode_start = std::chrono::high_resolution_clock::now();

        rlt::sample_initial_parameters(device, env, env_parameters, sweep_rng);
        {
            auto indoor_pos = rlt::rendering::datasets::procthor::sample_free_position(device, *env.annotations, sweep_rng);
            env_parameters.scene_translation[0] = indoor_pos.position[0];
            env_parameters.scene_translation[1] = indoor_pos.position[1];
            env_parameters.scene_translation[2] = indoor_pos.position[2];
            env_parameters.scene_yaw = rlt::random::uniform_real_distribution(device.random, (T)0, (T)(2.0 * 3.14159265358979323846), sweep_rng);
        }
        typename ENVIRONMENT::State state;
        rlt::sample_initial_state(device, env, env_parameters, state, sweep_rng);

        render_target_frame(device, env, env_parameters, target_frame_buf.data());
        if(config.brightness_scale != 1.0f || config.brightness_offset != 0.0f){
            for(TI i = 0; i < OBS_DIM_SINGLE; i++){
                target_frame_buf[i] = std::clamp(target_frame_buf[i] * config.brightness_scale + config.brightness_offset, 0.0f, 1.0f);
            }
        }

        std::fill(frame_history.begin(), frame_history.end(), 0.0f);

        TI episode_end_step = global_step + EPISODE_STEP_LIMIT;
        FILE* ffmpeg_pipe = nullptr;
        if(config.record_video){
            auto step_folder = rlt::get_step_folder(device, extrack_config, extrack_paths, episode_end_step);
            auto video_path = step_folder / "video.mp4";
            char ffmpeg_cmd[1024];
            snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
                "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %lux%lu -framerate %lu -i - "
                "-c:v libx264 -pix_fmt yuv420p -crf 23 -r %lu -preset fast -loglevel warning %s",
                (unsigned long)CAM_WIDTH, (unsigned long)CAM_HEIGHT, (unsigned long)SIMULATION_FREQUENCY,
                (unsigned long)SIMULATION_FREQUENCY, video_path.c_str());
            ffmpeg_pipe = popen(ffmpeg_cmd, "w");
            if(!ffmpeg_pipe){
                std::cerr << "Failed to open ffmpeg pipe for " << video_path << std::endl;
            }
        }

        TI step_i = 0;
        bool terminated = false;
        for(; step_i < EPISODE_STEP_LIMIT && !terminated; step_i++){
            // State observation
            rlt::observe(device, env.dynamics, env_parameters.dynamics, state, ACTOR_STATE_OBS{}, state_obs_mat, sweep_rng);

            // Render
            auto camera = rlt::rl::environments::l2f_visual::make_camera_for_state(device, env, env_parameters, state);
            cudaMemcpy(rlt::data(rlt::cameras(device, *env.renderer)), &camera, sizeof(camera), cudaMemcpyHostToDevice);
            rlt::render(device, *env.renderer);

            // renderer observation output (float) into the frame ring buffer
            TI ring_idx = step_i % FRAME_HISTORY_SIZE;
            float* frame_dst = frame_history.data() + ring_idx * OBS_DIM_SINGLE;
            cudaMemcpy(frame_dst, rlt::data(rlt::observation(device, *env.renderer)), (size_t)IMG_PIXELS * 3 * sizeof(float), cudaMemcpyDeviceToHost);

            // Apply rendering perturbations
            if(config.brightness_scale != 1.0f || config.brightness_offset != 0.0f){
                for(TI i = 0; i < OBS_DIM_SINGLE; i++){
                    frame_dst[i] = std::clamp(frame_dst[i] * config.brightness_scale + config.brightness_offset, 0.0f, 1.0f);
                }
            }
            if(config.gaussian_noise_std > 0.0f){
                for(TI i = 0; i < OBS_DIM_SINGLE; i++){
                    T noise = rlt::random::normal_distribution::sample(device.random, (T)0, config.gaussian_noise_std, sweep_rng);
                    frame_dst[i] = std::clamp(frame_dst[i] + noise, 0.0f, 1.0f);
                }
            }

            // Assemble combined frame observation: [stacked_5_frames | target_frame] per pixel
            for(TI p = 0; p < IMG_PIXELS; p++){
                for(TI f = 0; f < FRAME_STACK_N; f++){
                    TI back = f * FRAME_STACK_STRIDE;
                    TI src_step = step_i >= back ? step_i - back : 0;
                    TI src_ring = src_step % FRAME_HISTORY_SIZE;
                    const float* src_frame = frame_history.data() + src_ring * OBS_DIM_SINGLE;
                    for(TI c = 0; c < IMG_C; c++){
                        combined_obs[p * COMBINED_IMG_C + f * IMG_C + c] = src_frame[p * IMG_C + c];
                    }
                }
                for(TI c = 0; c < IMG_C; c++){
                    combined_obs[p * COMBINED_IMG_C + FRAME_STACK_N * IMG_C + c] = target_frame_buf[p * IMG_C + c];
                }
            }

            // Fill tuple inputs
            std::memcpy(dyn_inputs.tensors[0].data, combined_obs.data(), COMBINED_OBS_DIM * sizeof(float));
            std::memcpy(dyn_inputs.tensors[1].data, state_obs_data, STATE_OBS_DIM * sizeof(float));

            if(!rlt::evaluate(device, model, dyn_inputs, dyn_output, buffer)){
                std::cerr << "dyn::evaluate failed at episode " << episode_i << " step " << step_i << std::endl;
                return 1;
            }

            // Extract action
            for(TI a = 0; a < ACTION_DIM; a++){
                action_data[a] = rlt::get(device, dyn_output, a);
            }

            // Step environment
            typename ENVIRONMENT::State next_state;
            rlt::step(device, env.dynamics, env_parameters.dynamics, state, action_mat, next_state, sweep_rng);
            terminated = rlt::terminated(device, env.dynamics, env_parameters.dynamics, next_state, sweep_rng);

            // Record video frame (packed uint8 frame buffer, rendered alongside the observation)
            if(ffmpeg_pipe){
                std::vector<uint32_t> fb_staging((size_t)IMG_PIXELS);
                cudaMemcpy(fb_staging.data(), rlt::data(rlt::frame_buffer(device, *env.renderer)), fb_staging.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                for(TI p = 0; p < IMG_PIXELS; p++){
                    uint32_t rgba = fb_staging[p];
                    video_frame[p * 3 + 0] = (rgba >>  0) & 0xFF;
                    video_frame[p * 3 + 1] = (rgba >>  8) & 0xFF;
                    video_frame[p * 3 + 2] = (rgba >> 16) & 0xFF;
                }
                fwrite(video_frame.data(), 1, video_frame.size(), ffmpeg_pipe);
            }

            state = next_state;
        }

        if(ffmpeg_pipe){
            pclose(ffmpeg_pipe);
        }

        auto episode_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> episode_elapsed = episode_end - episode_start;

        episode_lengths.push_back(step_i);
        episode_terminated.push_back(terminated);
        global_step += EPISODE_STEP_LIMIT;
    }

    T mean_length = 0;
    TI terminated_count = 0;
    for(TI i = 0; i < episode_lengths.size(); i++){
        mean_length += episode_lengths[i];
        if(episode_terminated[i]) terminated_count++;
    }
    mean_length /= episode_lengths.size();
    T survival = (T)(config.num_episodes - terminated_count) / (T)config.num_episodes * 100;
    std::cout << std::setw(12) << std::setprecision(4) << std::fixed << config.brightness_scale
              << std::setw(14) << std::setprecision(1) << mean_length
              << std::setw(13) << std::setprecision(1) << survival << "%" << std::endl;

    rlt::free(device, sweep_rng);
    } // noise sweep

    // =====================================================================
    // Cleanup
    // =====================================================================
    for(TI i = 0; i < dyn_inputs.num_tensors; i++) rlt::free(device, dyn_inputs.tensors[i]);
    rlt::free(device, dyn_output);
    rlt::free(device, buffer);
    rlt::free(device, model);
    rlt::free(device, env);
    rlt::free(device, rng);

    return 0;
}
