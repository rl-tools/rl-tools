#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/operations_generic.h>

#include <rl_tools/nn/operations_cuda.h>

#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>

#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>

#include "model.h"
#include "scene.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace rlt = rl_tools;
namespace fs = std::filesystem;

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)

using T = float;
using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_CUDA = rlt::devices::DEVICE_FACTORY_CUDA<>;
using TI = DEVICE_CPU::index_t;
using TI_CUDA = DEVICE_CUDA::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<float>;

namespace yp = rlt::rendering::raytracing::yaw_prediction;

static constexpr TI_CUDA BATCH_SIZE = yp::SCENE_NUM_CAMERAS / 2;
static constexpr TI CAM_WIDTH = yp::SCENE_CAM_WIDTH;
static constexpr TI CAM_HEIGHT = yp::SCENE_CAM_HEIGHT;
static constexpr TI OUTPUT_DIM = 3;
static constexpr TI TOTAL_OUTPUT_ELEMENTS = BATCH_SIZE * OUTPUT_DIM;

struct EvalConfig {
    static constexpr float MAX_ANGLE = 3.14159265358979323846f / 6.0f;
    static constexpr float COS_FOV_MIN = 0.3f;
    static constexpr float COS_FOV_MAX = 1.2f;
};

using GPU_CAPABILITY = rlt::nn::capability::Forward<>;
using GPU_MODEL = yp::MODEL<GPU_CAPABILITY, TYPE_POLICY, TI_CUDA, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH>;
using CPU_CAPABILITY = rlt::nn::capability::Forward<>;
using CPU_MODEL = yp::MODEL<CPU_CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH>;

using GPU_INPUT_SHAPE = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH, 3>;
using GPU_INPUT_SPEC = rlt::tensor::Specification<T, TI_CUDA, GPU_INPUT_SHAPE>;
using GPU_OUTPUT_SHAPE = typename GPU_MODEL::OUTPUT_SHAPE;
using GPU_OUTPUT_SPEC = rlt::tensor::Specification<T, TI_CUDA, GPU_OUTPUT_SHAPE>;

static constexpr int ENCODER_DIM = GPU_MODEL::SPEC::LAST_DIM_A;
static constexpr int CONCAT_DIM = GPU_MODEL::SPEC::LAST_DIM;
static constexpr int ENCODER_TOTAL = rlt::product(typename GPU_MODEL::SPEC::OUTPUT_SHAPE_A{});
static constexpr int CONCAT_ROWS = ENCODER_TOTAL / ENCODER_DIM;

__global__ void rgba_to_float_kernel(
    const uint32_t* __restrict__ framebuffer,
    float* __restrict__ output,
    int cam_width, int cam_height, int camera_offset, int dst_offset
) {
    const int sample_idx = blockIdx.x;
    const int camera_idx = camera_offset + sample_idx;
    const int cam_pixels = cam_width * cam_height;

    const uint32_t* src = framebuffer + camera_idx * cam_pixels;
    float* dst = output + (dst_offset + sample_idx) * cam_height * cam_width * 3;

    for (int pixel = threadIdx.x; pixel < cam_pixels; pixel += blockDim.x) {
        const uint32_t rgba = src[pixel];
        const float r = static_cast<float>((rgba >> 0) & 0xFF) / 255.0f;
        const float g = static_cast<float>((rgba >> 8) & 0xFF) / 255.0f;
        const float b = static_cast<float>((rgba >> 16) & 0xFF) / 255.0f;

        const int out_base = pixel * 3;
        dst[out_base + 0] = r;
        dst[out_base + 1] = g;
        dst[out_base + 2] = b;
    }
}

__global__ void concatenate_kernel(
    const float* __restrict__ a, int d_a,
    const float* __restrict__ b, int d_b,
    float* __restrict__ output,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int d_out = d_a + d_b;
    const int total = n * d_out;
    if (idx >= total) return;
    const int row = idx / d_out;
    const int col = idx % d_out;
    if (col < d_a) {
        output[idx] = a[row * d_a + col];
    } else {
        output[idx] = b[row * d_b + (col - d_a)];
    }
}

struct Metrics {
    double mse = 0.0;
    double mae_px = 0.0;
    double mae_py = 0.0;
    double mae_roll = 0.0;
    double max_abs_px = 0.0;
    double max_abs_py = 0.0;
    double max_abs_roll = 0.0;
    TI samples = 0;
};

struct WorstPair {
    std::string scene_name;
    TI sample_index = 0;
    double mae = 0.0;
    double err_px = 0.0;
    double err_py = 0.0;
    double err_roll = 0.0;
    std::vector<unsigned char> rgb_pair;
};

static std::string format_metric(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << value;
    std::string s = oss.str();
    for (char& c : s) {
        if (c == '-') {
            c = 'm';
        } else if (c == '.') {
            c = 'p';
        }
    }
    return s;
}

static std::vector<unsigned char> make_pair_rgb(const uint32_t* framebuffer, TI sample_index) {
    std::vector<unsigned char> rgb_pair(CAM_HEIGHT * (CAM_WIDTH * 2) * 3);
    const TI pixels_per_cam = CAM_WIDTH * CAM_HEIGHT;
    const uint32_t* image_a = framebuffer + sample_index * pixels_per_cam;
    const uint32_t* image_b = framebuffer + (BATCH_SIZE + sample_index) * pixels_per_cam;

    for (TI y = 0; y < CAM_HEIGHT; y++) {
        for (TI x = 0; x < CAM_WIDTH; x++) {
            const TI src_idx = y * CAM_WIDTH + x;
            const uint32_t rgba_a = image_a[src_idx];
            const uint32_t rgba_b = image_b[src_idx];
            const TI dst_a = (y * (CAM_WIDTH * 2) + x) * 3;
            const TI dst_b = (y * (CAM_WIDTH * 2) + (CAM_WIDTH + x)) * 3;

            rgb_pair[dst_a + 0] = static_cast<unsigned char>((rgba_a >> 0) & 0xFF);
            rgb_pair[dst_a + 1] = static_cast<unsigned char>((rgba_a >> 8) & 0xFF);
            rgb_pair[dst_a + 2] = static_cast<unsigned char>((rgba_a >> 16) & 0xFF);

            rgb_pair[dst_b + 0] = static_cast<unsigned char>((rgba_b >> 0) & 0xFF);
            rgb_pair[dst_b + 1] = static_cast<unsigned char>((rgba_b >> 8) & 0xFF);
            rgb_pair[dst_b + 2] = static_cast<unsigned char>((rgba_b >> 16) & 0xFF);
        }
    }
    return rgb_pair;
}

static void update_worst_pairs(std::vector<WorstPair>& worst_pairs, WorstPair&& candidate, TI limit) {
    worst_pairs.push_back(std::move(candidate));
    std::sort(worst_pairs.begin(), worst_pairs.end(), [](const WorstPair& a, const WorstPair& b) {
        return a.mae > b.mae;
    });
    if (worst_pairs.size() > limit) {
        worst_pairs.resize(limit);
    }
}

static Metrics evaluate_scene(
    DEVICE_CUDA& device_cuda,
    DEVICE_CPU& device_cpu,
    GPU_MODEL& model,
    typename GPU_MODEL::template Buffer<true>& model_buffer,
    rlt::Tensor<GPU_INPUT_SPEC>& gpu_input_a,
    rlt::Tensor<GPU_INPUT_SPEC>& gpu_input_b,
    rlt::Tensor<GPU_OUTPUT_SPEC>& gpu_output,
    DEVICE_CUDA::SPEC::RANDOM::ENGINE<>& rng_cuda,
    yp::SceneHandle* scene,
    const std::string& scene_name,
    std::vector<WorstPair>& worst_pairs,
    TI worst_pair_limit
) {
    std::vector<rlt::CameraData> cameras(yp::SCENE_NUM_CAMERAS);
    std::vector<float> targets(TOTAL_OUTPUT_ELEMENTS);
    std::vector<float> predictions(TOTAL_OUTPUT_ELEMENTS);
    std::vector<uint32_t> framebuffer_host(yp::SCENE_NUM_CAMERAS * CAM_WIDTH * CAM_HEIGHT);

    yp::sample_camera_batch(
        scene,
        cameras.data(),
        targets.data(),
        BATCH_SIZE,
        EvalConfig::MAX_ANGLE,
        EvalConfig::COS_FOV_MIN,
        EvalConfig::COS_FOV_MAX
    );
    yp::render_batch(scene, cameras.data());

    uint32_t* device_fb = yp::get_framebuffer_device_ptr(scene);
    rgba_to_float_kernel<<<BATCH_SIZE, 256, 0, device_cuda.stream>>>(
        device_fb, gpu_input_a._data, CAM_WIDTH, CAM_HEIGHT, 0, 0
    );
    rgba_to_float_kernel<<<BATCH_SIZE, 256, 0, device_cuda.stream>>>(
        device_fb, gpu_input_b._data, CAM_WIDTH, CAM_HEIGHT, BATCH_SIZE, 0
    );

    auto eval_mode = rlt::Mode<rlt::mode::Evaluation<>>{};
    rlt::evaluate(device_cuda, model.pipeline_a, gpu_input_a, model_buffer.intermediate_a, model_buffer.buffer_a, rng_cuda, eval_mode);
    rlt::evaluate(device_cuda, model.pipeline_b, gpu_input_b, model_buffer.intermediate_b, model_buffer.buffer_b, rng_cuda, eval_mode);

    {
        constexpr int total = CONCAT_ROWS * CONCAT_DIM;
        constexpr int threads = 256;
        constexpr int blocks = (total + threads - 1) / threads;
        concatenate_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
            model_buffer.intermediate_a._data, ENCODER_DIM,
            model_buffer.intermediate_b._data, ENCODER_DIM,
            model_buffer.concatenated._data, CONCAT_ROWS
        );
    }

    rlt::evaluate(device_cuda, model.head, model_buffer.concatenated, gpu_output, model_buffer.head_buffer, rng_cuda, eval_mode);

    cudaStreamSynchronize(device_cuda.stream);
    cudaMemcpy(
        framebuffer_host.data(),
        device_fb,
        framebuffer_host.size() * sizeof(uint32_t),
        cudaMemcpyDeviceToHost
    );
    cudaMemcpy(
        predictions.data(),
        gpu_output._data,
        TOTAL_OUTPUT_ELEMENTS * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    Metrics metrics;
    metrics.samples = BATCH_SIZE;
    for (TI i = 0; i < BATCH_SIZE; i++) {
        const double diff_px = std::abs(predictions[i * OUTPUT_DIM + 0] - targets[i * OUTPUT_DIM + 0]);
        const double diff_py = std::abs(predictions[i * OUTPUT_DIM + 1] - targets[i * OUTPUT_DIM + 1]);
        const double diff_roll = std::abs(predictions[i * OUTPUT_DIM + 2] - targets[i * OUTPUT_DIM + 2]);

        metrics.mae_px += diff_px;
        metrics.mae_py += diff_py;
        metrics.mae_roll += diff_roll;
        metrics.max_abs_px = std::max(metrics.max_abs_px, diff_px);
        metrics.max_abs_py = std::max(metrics.max_abs_py, diff_py);
        metrics.max_abs_roll = std::max(metrics.max_abs_roll, diff_roll);
        metrics.mse +=
            diff_px * diff_px +
            diff_py * diff_py +
            diff_roll * diff_roll;

        WorstPair candidate;
        candidate.scene_name = scene_name;
        candidate.sample_index = i;
        candidate.err_px = diff_px;
        candidate.err_py = diff_py;
        candidate.err_roll = diff_roll;
        candidate.mae = (diff_px + diff_py + diff_roll) / 3.0;
        candidate.rgb_pair = make_pair_rgb(framebuffer_host.data(), i);
        update_worst_pairs(worst_pairs, std::move(candidate), worst_pair_limit);
    }

    const double denom = static_cast<double>(BATCH_SIZE);
    metrics.mae_px /= denom;
    metrics.mae_py /= denom;
    metrics.mae_roll /= denom;
    metrics.mse /= static_cast<double>(TOTAL_OUTPUT_ELEMENTS);
    return metrics;
}

int main(int argc, char** argv) {
    std::string scene_dir;
    std::string checkpoint_path;
    TI num_scenes = 80;
    TI num_val_scenes = 10;
    const fs::path output_dir = fs::path("src/rendering/raytracing/yaw_prediction/pairs");
    static constexpr TI WORST_PAIR_LIMIT = 100;

#ifdef RL_TOOLS_TEST_DATA_PATH
    checkpoint_path = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/yaw-predictor4.h5";
#endif

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--scene-dir" && i + 1 < argc) {
            scene_dir = argv[++i];
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (arg == "--num-scenes" && i + 1 < argc) {
            num_scenes = std::atoi(argv[++i]);
        } else if (arg == "--num-val-scenes" && i + 1 < argc) {
            num_val_scenes = std::atoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " --scene-dir <dir>"
                      << " [--checkpoint <path.h5>]"
                      << " [--num-scenes <train_offset>]"
                      << " [--num-val-scenes <count>]" << std::endl;
            return 1;
        }
    }

    if (scene_dir.empty()) {
        std::cerr << "Error: --scene-dir is required" << std::endl;
        return 1;
    }
    if (checkpoint_path.empty()) {
        std::cerr << "Error: no checkpoint path configured; pass --checkpoint or configure RL_TOOLS_TEST_DATA_PATH" << std::endl;
        return 1;
    }

    std::vector<std::string> found_paths;
    for (auto& entry : fs::directory_iterator(scene_dir)) {
        if (entry.path().extension() == ".glb") {
            found_paths.push_back(entry.path().string());
        }
    }
    std::sort(found_paths.begin(), found_paths.end());

    if (found_paths.empty()) {
        std::cerr << "No .glb files found in " << scene_dir << std::endl;
        return 1;
    }

    const TI total_needed = num_scenes + num_val_scenes;
    if (total_needed > found_paths.size()) {
        std::cerr << "Need " << total_needed
                  << " scenes (train offset=" << num_scenes
                  << " + val=" << num_val_scenes
                  << ") but only found " << found_paths.size() << std::endl;
        return 1;
    }

    std::vector<std::string> val_scene_paths(
        found_paths.begin() + num_scenes,
        found_paths.begin() + total_needed
    );

    std::cout << "Yaw prediction evaluation" << std::endl;
    std::cout << "  Checkpoint: " << checkpoint_path << std::endl;
    std::cout << "  Scene dir: " << scene_dir << std::endl;
    std::cout << "  Train offset (--num-scenes): " << num_scenes << std::endl;
    std::cout << "  Num val scenes: " << num_val_scenes << std::endl;
    std::cout << "  Batch size per scene: " << BATCH_SIZE << std::endl;

    struct LoadedScene {
        yp::SceneHandle* handle = nullptr;
        std::string path;
    };
    std::vector<LoadedScene> val_scenes(val_scene_paths.size());
    for (TI s = 0; s < val_scene_paths.size(); s++) {
        val_scenes[s].path = val_scene_paths[s];
        std::cout << "Loading val scene " << s << "/" << val_scene_paths.size() << ": "
                  << fs::path(val_scenes[s].path).filename().string() << std::endl;
        val_scenes[s].handle = yp::create_scene(val_scenes[s].path.c_str());
        std::cout << "  Indoor states: " << yp::get_num_indoor_states(val_scenes[s].handle) << std::endl;
    }

    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cpu);
    rlt::init(device_cuda);

    CPU_MODEL model_cpu;
    rlt::malloc(device_cpu, model_cpu);
    {
        auto file = HighFive::File(checkpoint_path, HighFive::File::ReadOnly);
        auto mg = rlt::get_group(device_cpu, file, "model");
        const bool success = rlt::load(device_cpu, model_cpu, mg);
        if (!success) {
            std::cerr << "Failed to load checkpoint: " << checkpoint_path << std::endl;
            return 1;
        }
    }

    GPU_MODEL model;
    typename GPU_MODEL::template Buffer<true> model_buffer;
    rlt::Tensor<GPU_INPUT_SPEC> gpu_input_a;
    rlt::Tensor<GPU_INPUT_SPEC> gpu_input_b;
    rlt::Tensor<GPU_OUTPUT_SPEC> gpu_output;
    DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;

    rlt::malloc(device_cuda, model);
    rlt::malloc(device_cuda, model_buffer);
    rlt::malloc(device_cuda, gpu_input_a);
    rlt::malloc(device_cuda, gpu_input_b);
    rlt::malloc(device_cuda, gpu_output);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 42);
    rlt::copy(device_cpu, device_cuda, model_cpu, model);

    Metrics aggregate;
    std::vector<WorstPair> worst_pairs;
    fs::create_directories(output_dir);
    for (const auto& scene : val_scenes) {
        const Metrics scene_metrics = evaluate_scene(
            device_cuda,
            device_cpu,
            model,
            model_buffer,
            gpu_input_a,
            gpu_input_b,
            gpu_output,
            rng_cuda,
            scene.handle,
            fs::path(scene.path).stem().string(),
            worst_pairs,
            WORST_PAIR_LIMIT
        );

        aggregate.mse += scene_metrics.mse * static_cast<double>(TOTAL_OUTPUT_ELEMENTS);
        aggregate.mae_px += scene_metrics.mae_px * static_cast<double>(scene_metrics.samples);
        aggregate.mae_py += scene_metrics.mae_py * static_cast<double>(scene_metrics.samples);
        aggregate.mae_roll += scene_metrics.mae_roll * static_cast<double>(scene_metrics.samples);
        aggregate.max_abs_px = std::max(aggregate.max_abs_px, scene_metrics.max_abs_px);
        aggregate.max_abs_py = std::max(aggregate.max_abs_py, scene_metrics.max_abs_py);
        aggregate.max_abs_roll = std::max(aggregate.max_abs_roll, scene_metrics.max_abs_roll);
        aggregate.samples += scene_metrics.samples;

        std::cout << fs::path(scene.path).filename().string()
                  << "  mse=" << scene_metrics.mse
                  << "  mae_px=" << scene_metrics.mae_px
                  << "  mae_py=" << scene_metrics.mae_py
                  << "  mae_roll=" << scene_metrics.mae_roll
                  << std::endl;
    }

    if (aggregate.samples > 0) {
        aggregate.mse /= static_cast<double>(aggregate.samples * OUTPUT_DIM);
        aggregate.mae_px /= static_cast<double>(aggregate.samples);
        aggregate.mae_py /= static_cast<double>(aggregate.samples);
        aggregate.mae_roll /= static_cast<double>(aggregate.samples);
    }

    std::cout << "Aggregate"
              << "  mse=" << aggregate.mse
              << "  mae_px=" << aggregate.mae_px
              << "  mae_py=" << aggregate.mae_py
              << "  mae_roll=" << aggregate.mae_roll
              << "  max_px=" << aggregate.max_abs_px
              << "  max_py=" << aggregate.max_abs_py
              << "  max_roll=" << aggregate.max_abs_roll
              << std::endl;

    std::sort(worst_pairs.begin(), worst_pairs.end(), [](const WorstPair& a, const WorstPair& b) {
        return a.mae > b.mae;
    });
    for (TI i = 0; i < worst_pairs.size(); i++) {
        const auto& pair = worst_pairs[i];
        std::ostringstream filename;
        filename << std::setfill('0') << std::setw(3) << i
                 << "_" << pair.scene_name
                 << "_sample" << pair.sample_index
                 << "_mae" << format_metric(pair.mae)
                 << "_px" << format_metric(pair.err_px)
                 << "_py" << format_metric(pair.err_py)
                 << "_roll" << format_metric(pair.err_roll)
                 << ".png";
        const fs::path output_path = output_dir / filename.str();
        stbi_write_png(
            output_path.c_str(),
            CAM_WIDTH * 2,
            CAM_HEIGHT,
            3,
            pair.rgb_pair.data(),
            (CAM_WIDTH * 2) * 3
        );
    }
    std::cout << "Saved " << worst_pairs.size() << " worst pairs to " << output_dir << std::endl;

    for (auto& scene : val_scenes) {
        yp::destroy_scene(scene.handle);
    }
    rlt::free(device_cuda, rng_cuda);
    rlt::free(device_cuda, gpu_output);
    rlt::free(device_cuda, gpu_input_b);
    rlt::free(device_cuda, gpu_input_a);
    rlt::free(device_cuda, model_buffer);
    rlt::free(device_cuda, model);
    rlt::free(device_cpu, model_cpu);

    return 0;
}
