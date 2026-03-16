#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

// Adam instance operations must come before layer operations (see CLAUDE.md)
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>

// Layer operations
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>

// Model operations
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>

// CUDA-specific operations
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn_models/operations_generic.h>

// Adam optimizer operations (after model operations so ADL finds model's _reset_optimizer_state)
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

// Tensor operations
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>

// Model definition
#include "model.h"

// Scene management (compiled separately as .cpp to avoid NVCC issues with environment code)

#include "scene.h"

#include "../example/environment/environment.h"
#include "../example/environment/operations_cpu.h"

// Experiment tracking (logging, tensorboard)
#include <rl_tools/utils/extrack/extrack.h>
#include <rl_tools/utils/extrack/operations_cpu.h>

// Checkpointing (optional, requires HDF5)
#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <random>
#include <algorithm>
#include <filesystem>
#include <csignal>
#include <thread>
#include <mutex>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace fs = std::filesystem;

namespace rlt = rl_tools;

static volatile std::sig_atomic_t signal_received = 0;
static void signal_handler(int signal) {
    signal_received = signal;
}

// ---- Configuration ----
using T = __nv_bfloat16;
using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_CUDA = rlt::devices::DEVICE_FACTORY_CUDA<>;
using TI = DEVICE_CPU::index_t;
using TI_CUDA = DEVICE_CUDA::index_t;

namespace yp = rlt::rendering::raytracing::yaw_prediction;
static constexpr TI_CUDA BATCH_SIZE = yp::SCENE_NUM_CAMERAS / 2;
static constexpr TI_CUDA NUM_CAMERAS = yp::SCENE_NUM_CAMERAS;
static constexpr TI CAM_WIDTH = yp::SCENE_CAM_WIDTH;
static constexpr TI CAM_HEIGHT = yp::SCENE_CAM_HEIGHT;
static constexpr TI NUM_ITERATIONS = 10000;

struct TrainingConfig {
    static constexpr float LEARNING_RATE = 1e-3f;
    static constexpr float MAX_ANGLE = 3.14159265358979323846f / 6.0f; // 30 degrees
    static constexpr float COS_FOV_MIN = 0.3f;  // wide FOV
    static constexpr float COS_FOV_MAX = 1.2f;   // narrow FOV
    static constexpr float HUBER_DELTA = 0.1f;
    static constexpr TI LOG_INTERVAL = 100;
    static constexpr TI VAL_INTERVAL = 1000;
    static constexpr TI CHECKPOINT_INTERVAL = 100000;
};

using TYPE_POLICY = rlt::numeric_types::Policy<float,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Parameter, T>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Activation, T>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Gradient, T>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, float>>;
using T_ACTIVATION = TYPE_POLICY::GET<rlt::numeric_types::categories::Activation>;
using T_GRADIENT = TYPE_POLICY::GET<rlt::numeric_types::categories::Gradient>;

struct AdamParams : rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY> {
    static constexpr float ALPHA = TrainingConfig::LEARNING_RATE;
};

using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI_CUDA, AdamParams>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
using GPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using GPU_MODEL = rlt::rendering::raytracing::yaw_prediction::MODEL<GPU_CAPABILITY, TYPE_POLICY, TI_CUDA, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH>;

using CPU_TYPE_POLICY = rlt::numeric_types::Policy<float>;
using CPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using CPU_MODEL = rlt::rendering::raytracing::yaw_prediction::MODEL<CPU_CAPABILITY, CPU_TYPE_POLICY, TI, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH>;
using CPU_MODEL_INFERENCE = typename CPU_MODEL::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;

using GPU_INPUT_SHAPE = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH, 3>;
using GPU_INPUT_SPEC = rlt::tensor::Specification<T_ACTIVATION, TI_CUDA, GPU_INPUT_SHAPE>;
using GPU_OUTPUT_SHAPE = typename GPU_MODEL::OUTPUT_SHAPE;
using GPU_D_OUTPUT_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, GPU_OUTPUT_SHAPE>;

// Encoder output: last dim is channel count, leading dims are spatial
static constexpr int ENCODER_DIM = GPU_MODEL::SPEC::LAST_DIM_A;
static constexpr int CONCAT_DIM = GPU_MODEL::SPEC::LAST_DIM;
// Number of rows for concatenation (product of all dims / last dim)
static constexpr int ENCODER_TOTAL = rlt::product(typename GPU_MODEL::SPEC::OUTPUT_SHAPE_A{});
static constexpr int CONCAT_ROWS = ENCODER_TOTAL / ENCODER_DIM;

// ---- CUDA kernels ----

// Concatenate two 2D tensors [N, D_A] and [N, D_B] into [N, D_A + D_B] along last dim
__global__ void concatenate_kernel(
    const T_ACTIVATION* __restrict__ a, int d_a,
    const T_ACTIVATION* __restrict__ b, int d_b,
    T_ACTIVATION* __restrict__ output,
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

// Split [N, D_A + D_B] into [N, D_A] and [N, D_B] along last dim
__global__ void split_kernel(
    const T_ACTIVATION* __restrict__ input, int d_a, int d_b,
    T_ACTIVATION* __restrict__ a,
    T_ACTIVATION* __restrict__ b,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int d_in = d_a + d_b;
    const int total = n * d_in;
    if (idx >= total) return;
    const int row = idx / d_in;
    const int col = idx % d_in;
    if (col < d_a) {
        a[row * d_a + col] = input[idx];
    } else {
        b[row * d_b + (col - d_a)] = input[idx];
    }
}

// Convert RGBA uint32 framebuffer pixels to bf16 RGB tensor [BATCH, H, W, 3] normalized to [0,1]
// dst_offset: write starting at output[(dst_offset + sample_idx) * H * W * 3]
__global__ void rgba_to_activation_kernel(
    const uint32_t* __restrict__ framebuffer,
    T_ACTIVATION* __restrict__ output,
    int cam_width, int cam_height, int camera_offset, int dst_offset
) {
    const int sample_idx = blockIdx.x;
    const int camera_idx = camera_offset + sample_idx;
    const int cam_pixels = cam_width * cam_height;

    const uint32_t* src = framebuffer + camera_idx * cam_pixels;
    T_ACTIVATION* dst = output + (dst_offset + sample_idx) * cam_height * cam_width * 3;

    for (int pixel = threadIdx.x; pixel < cam_pixels; pixel += blockDim.x) {
        const uint32_t rgba = src[pixel];
        const float r = static_cast<float>((rgba >>  0) & 0xFF) / 255.0f;
        const float g = static_cast<float>((rgba >>  8) & 0xFF) / 255.0f;
        const float b = static_cast<float>((rgba >> 16) & 0xFF) / 255.0f;

        const int out_base = pixel * 3;
        dst[out_base + 0] = (T_ACTIVATION)r;
        dst[out_base + 1] = (T_ACTIVATION)g;
        dst[out_base + 2] = (T_ACTIVATION)b;
    }
}

// Element-wise Huber loss and gradient (operates on all BATCH_SIZE * OUTPUT_DIM elements)
// Reads bf16 predictions, float targets; writes bf16 gradient, float losses
__global__ void huber_loss_gradient_kernel(
    const T_ACTIVATION* __restrict__ predictions,
    const float* __restrict__ targets,
    T_GRADIENT* __restrict__ d_output,
    float* __restrict__ losses,
    int total_elements,
    float delta
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const float pred = (float)predictions[idx];
    const float tgt = targets[idx];
    const float diff = pred - tgt;
    const float abs_diff = abs(diff);

    const float scale = 1.0f / static_cast<float>(total_elements);
    if (abs_diff <= delta) {
        losses[idx] = 0.5f * diff * diff;
        d_output[idx] = (T_GRADIENT)(scale * diff);
    } else {
        losses[idx] = delta * (abs_diff - 0.5f * delta);
        d_output[idx] = (T_GRADIENT)(scale * delta * ((diff > 0.0f) - (diff < 0.0f)));
    }
}

__global__ void reduce_loss_kernel(
    const float* __restrict__ losses,
    float* __restrict__ total_loss,
    int total_elements
) {
    float sum = 0.0f;
    for (int i = 0; i < total_elements; i++) {
        sum += losses[i];
    }
    *total_loss = sum / static_cast<float>(total_elements);
}

int main(int argc, char** argv) {
    std::string scene_dir;
    std::string single_scene;
    TI num_scenes = 20;
    TI num_scenes_per_batch = 8;
    TI num_iterations = 100000;

    TI num_val_scenes = 10;
    TI num_load_threads = 8;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--scene-dir" && i + 1 < argc) {
            scene_dir = argv[++i];
        } else if (arg == "--scene" && i + 1 < argc) {
            single_scene = argv[++i];
        } else if (arg == "--num-scenes" && i + 1 < argc) {
            num_scenes = std::atoi(argv[++i]);
        } else if (arg == "--num-scenes-per-batch" && i + 1 < argc) {
            num_scenes_per_batch = std::atoi(argv[++i]);
        } else if (arg == "--num-iterations" && i + 1 < argc) {
            num_iterations = std::atoi(argv[++i]);
        } else if (arg == "--num-val-scenes" && i + 1 < argc) {
            num_val_scenes = std::atoi(argv[++i]);
        } else if (arg == "--num-load-threads" && i + 1 < argc) {
            num_load_threads = std::atoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [--scene-dir <dir> | --scene <path.glb>]"
                      << " [--num-scenes N] [--num-scenes-per-batch M]"
                      << " [--num-iterations N] [--num-val-scenes N]"
                      << " [--num-load-threads N]" << std::endl;
            return 1;
        }
    }

    // Collect available scene paths (sorted, then split into train + val)
    std::vector<std::string> all_scene_paths;
    std::vector<std::string> val_scene_paths;
    if (!scene_dir.empty()) {
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
        // First num_scenes for training, next num_val_scenes for validation
        TI total_needed = num_scenes + num_val_scenes;
        if (total_needed > found_paths.size()) {
            std::cerr << "Need " << total_needed << " scenes (train=" << num_scenes << " + val=" << num_val_scenes << ") but only found " << found_paths.size() << std::endl;
            return 1;
        }
        all_scene_paths.assign(found_paths.begin(), found_paths.begin() + num_scenes);
        val_scene_paths.assign(found_paths.begin() + num_scenes, found_paths.begin() + total_needed);
    } else if (!single_scene.empty()) {
        all_scene_paths.push_back(single_scene);
        num_scenes_per_batch = 1;
        num_val_scenes = 0;
    } else {
        std::cerr << "Must specify --scene-dir or --scene" << std::endl;
        return 1;
    }

    if (num_scenes_per_batch > all_scene_paths.size()) {
        num_scenes_per_batch = all_scene_paths.size();
    }

    const TI samples_per_scene = BATCH_SIZE / num_scenes_per_batch;
    if (samples_per_scene * num_scenes_per_batch != BATCH_SIZE) {
        std::cerr << "BATCH_SIZE (" << BATCH_SIZE << ") must be divisible by num_scenes_per_batch (" << num_scenes_per_batch << ")" << std::endl;
        return 1;
    }

    constexpr float PI = 3.14159265358979323846f;

    std::cout << "Rotation prediction training (multi-scene)" << std::endl;
    std::cout << "  Training scenes: " << all_scene_paths.size() << std::endl;
    std::cout << "  Validation scenes: " << val_scene_paths.size() << std::endl;
    std::cout << "  Scenes per batch: " << num_scenes_per_batch << std::endl;
    std::cout << "  Samples per scene: " << samples_per_scene << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Image resolution: " << CAM_WIDTH << "x" << CAM_HEIGHT << std::endl;
    std::cout << "  Max angle: " << TrainingConfig::MAX_ANGLE * 180.0f / PI << " degrees" << std::endl;
    std::cout << "  Learning rate: " << TrainingConfig::LEARNING_RATE << std::endl;
    std::cout << "  Num iterations: " << num_iterations << std::endl;

    // ---- Load all scenes in parallel ----
    struct LoadedScene {
        yp::SceneHandle* handle = nullptr;
        std::string path;
    };
    std::vector<LoadedScene> loaded_scenes(all_scene_paths.size());
    std::vector<LoadedScene> val_scenes(val_scene_paths.size());

    {
        std::mutex cout_mutex;
        TI total_scenes = all_scene_paths.size() + val_scene_paths.size();
        TI actual_load_threads = std::min(num_load_threads, total_scenes);

        // Build a flat list of load tasks (train first, then val)
        struct LoadTask {
            LoadedScene* target;
            const std::string* path;
            const char* label;
            TI index;
            TI count;
        };
        std::vector<LoadTask> tasks;
        tasks.reserve(total_scenes);
        for (TI s = 0; s < all_scene_paths.size(); s++) {
            loaded_scenes[s].path = all_scene_paths[s];
            tasks.push_back({&loaded_scenes[s], &loaded_scenes[s].path, "train", s, (TI)all_scene_paths.size()});
        }
        for (TI s = 0; s < val_scene_paths.size(); s++) {
            val_scenes[s].path = val_scene_paths[s];
            tasks.push_back({&val_scenes[s], &val_scenes[s].path, "val", s, (TI)val_scene_paths.size()});
        }

        std::atomic<TI> next_task{0};
        std::vector<std::thread> load_threads;
        load_threads.reserve(actual_load_threads);
        for (TI t = 0; t < actual_load_threads; t++) {
            load_threads.emplace_back([&]() {
                while (true) {
                    TI task_idx = next_task.fetch_add(1);
                    if (task_idx >= tasks.size()) break;
                    auto& task = tasks[task_idx];
                    auto* handle = yp::create_scene(task.path->c_str());
                    task.target->handle = handle;
                    std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "Loaded " << task.label << " scene " << task.index << "/" << task.count << ": "
                              << fs::path(*task.path).filename().string()
                              << " (" << yp::get_num_indoor_states(handle) << " indoor states)" << std::endl;
                }
            });
        }
        for (auto& t : load_threads) t.join();
        std::cout << "All " << total_scenes << " scenes loaded (" << actual_load_threads << " threads)." << std::endl;
    }

    // RNG for selecting scenes per batch
    std::mt19937 scene_rng(42);

    // ---- Device setup ----
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cpu);
    rlt::init(device_cuda);

    DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 42);

    // ---- Model setup ----
    OPTIMIZER optimizer;
    GPU_MODEL model;
    typename GPU_MODEL::template Buffer<true> model_buffer;
    rlt::malloc(device_cuda, optimizer);
    rlt::malloc(device_cuda, model);
    rlt::malloc(device_cuda, model_buffer);
    rlt::init(device_cuda, optimizer);

    // Init weights on CPU, then copy to GPU
    CPU_MODEL model_cpu;
    CPU_MODEL_INFERENCE model_cpu_inference;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, model_cpu_inference);
    {
        DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
        rlt::malloc(device_cpu, rng_cpu);
        rlt::init(device_cpu, rng_cpu, 42);
        rlt::init_weights(device_cpu, model_cpu, rng_cpu);
        rlt::free(device_cpu, rng_cpu);
    }
    rlt::copy(device_cpu, device_cuda, model_cpu, model);
    rlt::reset_optimizer_state(device_cuda, optimizer, model);
    std::cout << "Model initialized" << std::endl;

    // ---- GPU tensors ----
    rlt::Tensor<GPU_INPUT_SPEC> gpu_input_a, gpu_input_b;
    rlt::malloc(device_cuda, gpu_input_a);
    rlt::malloc(device_cuda, gpu_input_b);
    using GPU_D_INPUT_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, GPU_INPUT_SHAPE>;
    rlt::Tensor<GPU_D_INPUT_SPEC> gpu_d_input_a, gpu_d_input_b;
    rlt::malloc(device_cuda, gpu_d_input_a);
    rlt::malloc(device_cuda, gpu_d_input_b);

    rlt::Tensor<GPU_D_OUTPUT_SPEC> gpu_d_output;
    rlt::malloc(device_cuda, gpu_d_output);
    rlt::disable_dynamic_memory_allocation(device_cuda);

    static constexpr TI OUTPUT_DIM = 3;
    static constexpr TI TOTAL_OUTPUT_ELEMENTS = BATCH_SIZE * OUTPUT_DIM;

    float* gpu_targets;
    cudaMalloc(&gpu_targets, TOTAL_OUTPUT_ELEMENTS * sizeof(float));

    float* gpu_losses;
    cudaMalloc(&gpu_losses, TOTAL_OUTPUT_ELEMENTS * sizeof(float));

    float* gpu_total_loss;
    cudaMalloc(&gpu_total_loss, sizeof(float));

    // CPU-side buffers
    std::vector<float> cpu_targets(TOTAL_OUTPUT_ELEMENTS);
    std::vector<rlt::CameraData> cameras(NUM_CAMERAS);

    // ---- Extrack setup (experiment tracking, tensorboard logging) ----
    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = "yaw-prediction";
    rlt::init(device_cpu, extrack_config, extrack_paths, 0);

    // ---- Signal handler for clean shutdown ----
    std::signal(SIGINT, signal_handler);

    // ---- Training loop ----
    auto train_mode = rlt::Mode<rlt::mode::Default<>>{};
    using CLOCK = std::chrono::high_resolution_clock;
    CLOCK::time_point total_start;
    bool throughput_timer_started = false;

    // Indices for shuffled scene selection
    std::vector<TI> scene_indices(loaded_scenes.size());
    std::iota(scene_indices.begin(), scene_indices.end(), 0);

    for (TI iteration = 0; iteration < num_iterations && !signal_received; iteration++) {
        if (!throughput_timer_started && iteration == 1) {
            total_start = CLOCK::now();
            throughput_timer_started = true;
        }
        rlt::set_step(device_cpu, device_cpu.logger, iteration);

        // ---- Select random scenes for this batch ----
        std::shuffle(scene_indices.begin(), scene_indices.end(), scene_rng);

        // ---- Sample and launch async renders for all scenes ----
        for (TI s = 0; s < num_scenes_per_batch; s++) {
            TI scene_idx = scene_indices[s];
            const TI batch_offset = s * samples_per_scene;

            yp::sample_camera_batch(
                loaded_scenes[scene_idx].handle,
                cameras.data(),
                cpu_targets.data() + batch_offset * OUTPUT_DIM,
                samples_per_scene,
                TrainingConfig::MAX_ANGLE,
                TrainingConfig::COS_FOV_MIN,
                TrainingConfig::COS_FOV_MAX
            );

            yp::render_batch<true>(loaded_scenes[scene_idx].handle, cameras.data());
        }

        // ---- Sync all renders ----
        for (TI s = 0; s < num_scenes_per_batch; s++) {
            TI scene_idx = scene_indices[s];
            auto& handle = loaded_scenes[scene_idx].handle;
            rlt::render_rgb_only_sync(handle->device, *handle->env.renderer);
        }

        // ---- Convert rendered framebuffers to float tensors ----
        for (TI s = 0; s < num_scenes_per_batch; s++) {
            TI scene_idx = scene_indices[s];
            const TI batch_offset = s * samples_per_scene;
            uint32_t* device_fb = yp::get_framebuffer_device_ptr(loaded_scenes[scene_idx].handle);

            rgba_to_activation_kernel<<<samples_per_scene, 256, 0, device_cuda.stream>>>(
                device_fb, gpu_input_a._data, CAM_WIDTH, CAM_HEIGHT, 0, batch_offset
            );
            rgba_to_activation_kernel<<<samples_per_scene, 256, 0, device_cuda.stream>>>(
                device_fb, gpu_input_b._data, CAM_WIDTH, CAM_HEIGHT, samples_per_scene, batch_offset
            );
        }
        cudaMemcpyAsync(gpu_targets, cpu_targets.data(), TOTAL_OUTPUT_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice, device_cuda.stream);

        // ---- Forward ----
        rlt::zero_gradient(device_cuda, model);



        rlt::forward(device_cuda, model.pipeline_a, gpu_input_a, model_buffer.buffer_a, rng_cuda, train_mode);
        rlt::forward(device_cuda, model.pipeline_b, gpu_input_b, model_buffer.buffer_b, rng_cuda, train_mode);

        auto output_a = rlt::output(device_cuda, model.pipeline_a);
        auto output_b = rlt::output(device_cuda, model.pipeline_b);
        rlt::copy(device_cuda, device_cuda, output_a, model_buffer.intermediate_a);
        rlt::copy(device_cuda, device_cuda, output_b, model_buffer.intermediate_b);

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

        rlt::forward(device_cuda, model.head, model_buffer.concatenated, model_buffer.head_buffer, rng_cuda, train_mode);
        {
            auto head_output = rlt::output(device_cuda, model.head);
            rlt::copy(device_cuda, device_cuda, head_output, model.output);
        }
        auto output_view = rlt::output(device_cuda, model);

        // ---- Loss + gradient ----
        {
            constexpr int threads = 256;
            constexpr int blocks = (TOTAL_OUTPUT_ELEMENTS + threads - 1) / threads;
            huber_loss_gradient_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                output_view._data, gpu_targets, gpu_d_output._data, gpu_losses, TOTAL_OUTPUT_ELEMENTS, TrainingConfig::HUBER_DELTA
            );
        }

        // ---- Backward ----
        rlt::backward_full(device_cuda, model.head, model_buffer.concatenated, gpu_d_output, model_buffer.d_concatenated, model_buffer.head_buffer);

        {
            constexpr int total = CONCAT_ROWS * CONCAT_DIM;
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            split_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                model_buffer.d_concatenated._data, ENCODER_DIM, ENCODER_DIM,
                model_buffer.d_output_a._data,
                model_buffer.d_output_b._data,
                CONCAT_ROWS
            );
        }

        rlt::backward_full(device_cuda, model.pipeline_a, gpu_input_a, model_buffer.d_output_a, gpu_d_input_a, model_buffer.buffer_a);
        rlt::backward_full(device_cuda, model.pipeline_b, gpu_input_b, model_buffer.d_output_b, gpu_d_input_b, model_buffer.buffer_b);

        // ---- Optimizer step ----
        rlt::step(device_cuda, optimizer, model);

        // ---- Checkpointing ----
#ifdef RL_TOOLS_ENABLE_HDF5
        if (iteration % TrainingConfig::CHECKPOINT_INTERVAL == 0 || iteration == num_iterations - 1) {
            cudaStreamSynchronize(device_cuda.stream);
            rlt::copy(device_cuda, device_cpu, model, model_cpu);
            rlt::copy(device_cpu, device_cpu, model_cpu, model_cpu_inference);
            auto step_folder = rlt::get_step_folder(device_cpu, extrack_config, extrack_paths, iteration);
            auto file = HighFive::File((step_folder / "checkpoint.h5").string(), HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
            auto mg = rlt::create_group(device_cpu, file, "model");
            rlt::save(device_cpu, model_cpu_inference, mg);
            std::cout << "  Checkpoint: " << (step_folder / "checkpoint.h5").string() << std::endl;
        }
#endif

        // ---- Logging ----
        if (iteration % TrainingConfig::LOG_INTERVAL == 0 || iteration == num_iterations - 1) {
            reduce_loss_kernel<<<1, 1, 0, device_cuda.stream>>>(
                gpu_losses, gpu_total_loss, TOTAL_OUTPUT_ELEMENTS
            );
            cudaStreamSynchronize(device_cuda.stream);

            float loss_val;
            cudaMemcpy(&loss_val, gpu_total_loss, sizeof(float), cudaMemcpyDeviceToHost);

            std::vector<T_ACTIVATION> pred_buf_raw(TOTAL_OUTPUT_ELEMENTS);
            cudaMemcpy(pred_buf_raw.data(), output_view._data, TOTAL_OUTPUT_ELEMENTS * sizeof(T_ACTIVATION), cudaMemcpyDeviceToHost);

            float err_px = 0.0f, err_py = 0.0f, err_roll = 0.0f;
            for (TI_CUDA i = 0; i < BATCH_SIZE; i++) {
                err_px   += std::abs((float)pred_buf_raw[i * OUTPUT_DIM + 0] - cpu_targets[i * OUTPUT_DIM + 0]);
                err_py   += std::abs((float)pred_buf_raw[i * OUTPUT_DIM + 1] - cpu_targets[i * OUTPUT_DIM + 1]);
                err_roll += std::abs((float)pred_buf_raw[i * OUTPUT_DIM + 2] - cpu_targets[i * OUTPUT_DIM + 2]);
            }
            err_px /= BATCH_SIZE;
            err_py /= BATCH_SIZE;
            err_roll /= BATCH_SIZE;

            auto now = CLOCK::now();
            double elapsed_s = throughput_timer_started
                ? std::chrono::duration<double>(now - total_start).count()
                : 0.0;
            double samples_per_s = elapsed_s > 0.0 && iteration > 0
                ? static_cast<double>(iteration * BATCH_SIZE) / elapsed_s
                : 0.0;

            rlt::add_scalar(device_cpu, device_cpu.logger, "train/loss", loss_val);
            rlt::add_scalar(device_cpu, device_cpu.logger, "train/err_px", err_px);
            rlt::add_scalar(device_cpu, device_cpu.logger, "train/err_py", err_py);
            rlt::add_scalar(device_cpu, device_cpu.logger, "train/err_roll", err_roll);
            rlt::add_scalar(device_cpu, device_cpu.logger, "train/samples_per_s", samples_per_s);

            std::cout << "[iter " << iteration << "/" << num_iterations << "]"
                      << "  loss=" << loss_val
                      << "  err_px=" << err_px
                      << "  err_py=" << err_py
                      << "  err_roll=" << err_roll
                      << "  samples_per_s=" << samples_per_s
                      << "  time=" << elapsed_s << "s"
                      << std::endl;
        }

        // ---- Validation ----
        if (val_scenes.size() > 0 && (iteration % TrainingConfig::VAL_INTERVAL == 0 || iteration == num_iterations - 1)) {
            auto eval_mode = rlt::Mode<rlt::mode::Evaluation<>>{};
            float val_loss_sum = 0.0f;
            float val_err_px_sum = 0.0f, val_err_py_sum = 0.0f, val_err_roll_sum = 0.0f;
            TI val_total_samples = 0;

            for (TI vs = 0; vs < val_scenes.size(); vs++) {
                // Sample a full batch from this validation scene
                yp::sample_camera_batch(
                    val_scenes[vs].handle,
                    cameras.data(),
                    cpu_targets.data(),
                    BATCH_SIZE,
                    TrainingConfig::MAX_ANGLE,
                    TrainingConfig::COS_FOV_MIN,
                    TrainingConfig::COS_FOV_MAX
                );
                yp::render_batch<false>(val_scenes[vs].handle, cameras.data());
                uint32_t* device_fb = yp::get_framebuffer_device_ptr(val_scenes[vs].handle);

                rgba_to_activation_kernel<<<BATCH_SIZE, 256, 0, device_cuda.stream>>>(
                    device_fb, gpu_input_a._data, CAM_WIDTH, CAM_HEIGHT, 0, 0
                );
                rgba_to_activation_kernel<<<BATCH_SIZE, 256, 0, device_cuda.stream>>>(
                    device_fb, gpu_input_b._data, CAM_WIDTH, CAM_HEIGHT, BATCH_SIZE, 0
                );

                cudaMemcpyAsync(gpu_targets, cpu_targets.data(), TOTAL_OUTPUT_ELEMENTS * sizeof(float),
                                cudaMemcpyHostToDevice, device_cuda.stream);

                // Forward pass (evaluation mode, no gradient)
                rlt::forward(device_cuda, model.pipeline_a, gpu_input_a, model_buffer.buffer_a, rng_cuda, eval_mode);
                rlt::forward(device_cuda, model.pipeline_b, gpu_input_b, model_buffer.buffer_b, rng_cuda, eval_mode);

                auto val_output_a = rlt::output(device_cuda, model.pipeline_a);
                auto val_output_b = rlt::output(device_cuda, model.pipeline_b);
                rlt::copy(device_cuda, device_cuda, val_output_a, model_buffer.intermediate_a);
                rlt::copy(device_cuda, device_cuda, val_output_b, model_buffer.intermediate_b);

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

                rlt::forward(device_cuda, model.head, model_buffer.concatenated, model_buffer.head_buffer, rng_cuda, eval_mode);
                {
                    auto head_output = rlt::output(device_cuda, model.head);
                    rlt::copy(device_cuda, device_cuda, head_output, model.output);
                }
                auto val_output_view = rlt::output(device_cuda, model);

                // Compute loss
                {
                    constexpr int threads = 256;
                    constexpr int blocks = (TOTAL_OUTPUT_ELEMENTS + threads - 1) / threads;
                    huber_loss_gradient_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                        val_output_view._data, gpu_targets, gpu_d_output._data, gpu_losses, TOTAL_OUTPUT_ELEMENTS, TrainingConfig::HUBER_DELTA
                    );
                }
                reduce_loss_kernel<<<1, 1, 0, device_cuda.stream>>>(
                    gpu_losses, gpu_total_loss, TOTAL_OUTPUT_ELEMENTS
                );
                cudaStreamSynchronize(device_cuda.stream);

                float scene_loss;
                cudaMemcpy(&scene_loss, gpu_total_loss, sizeof(float), cudaMemcpyDeviceToHost);

                std::vector<T_ACTIVATION> val_pred_buf_raw(TOTAL_OUTPUT_ELEMENTS);
                cudaMemcpy(val_pred_buf_raw.data(), val_output_view._data, TOTAL_OUTPUT_ELEMENTS * sizeof(T_ACTIVATION), cudaMemcpyDeviceToHost);

                float scene_err_px = 0.0f, scene_err_py = 0.0f, scene_err_roll = 0.0f;
                for (TI_CUDA i = 0; i < BATCH_SIZE; i++) {
                    scene_err_px   += std::abs((float)val_pred_buf_raw[i * OUTPUT_DIM + 0] - cpu_targets[i * OUTPUT_DIM + 0]);
                    scene_err_py   += std::abs((float)val_pred_buf_raw[i * OUTPUT_DIM + 1] - cpu_targets[i * OUTPUT_DIM + 1]);
                    scene_err_roll += std::abs((float)val_pred_buf_raw[i * OUTPUT_DIM + 2] - cpu_targets[i * OUTPUT_DIM + 2]);
                }

                val_loss_sum += scene_loss * TOTAL_OUTPUT_ELEMENTS;
                val_err_px_sum += scene_err_px;
                val_err_py_sum += scene_err_py;
                val_err_roll_sum += scene_err_roll;
                val_total_samples += BATCH_SIZE;
            }

            float val_loss = val_loss_sum / (val_total_samples * OUTPUT_DIM);
            float val_err_px = val_err_px_sum / val_total_samples;
            float val_err_py = val_err_py_sum / val_total_samples;
            float val_err_roll = val_err_roll_sum / val_total_samples;

            rlt::add_scalar(device_cpu, device_cpu.logger, "val/loss", val_loss);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/err_px", val_err_px);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/err_py", val_err_py);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/err_roll", val_err_roll);

            std::cout << "[iter " << iteration << "] VAL loss=" << val_loss
                      << "  err_px=" << val_err_px
                      << "  err_py=" << val_err_py
                      << "  err_roll=" << val_err_roll << std::endl;
        }
    }

    if (signal_received) {
        std::cout << "Training interrupted by signal " << signal_received << "." << std::endl;
    } else {
        std::cout << "Training complete." << std::endl;
    }

    // ---- Cleanup ----
    cudaFree(gpu_targets);
    cudaFree(gpu_losses);
    cudaFree(gpu_total_loss);
    rlt::free(device_cuda, gpu_input_a);
    rlt::free(device_cuda, gpu_input_b);
    rlt::free(device_cuda, gpu_d_input_a);
    rlt::free(device_cuda, gpu_d_input_b);
    rlt::free(device_cuda, gpu_d_output);
    rlt::free(device_cuda, model_buffer);
    rlt::free(device_cuda, model);
    rlt::free(device_cuda, optimizer);
    rlt::free(device_cuda, rng_cuda);
    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, model_cpu_inference);
    for (auto& s : loaded_scenes) {
        yp::destroy_scene(s.handle);
    }
    for (auto& s : val_scenes) {
        yp::destroy_scene(s.handle);
    }
    rlt::free(device_cpu, device_cpu.logger);

    return 0;
}
