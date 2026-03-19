#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/dynamic_conv2d/operations_generic.h>
#include <rl_tools/nn/layers/dynamic_conv2d/operations_cuda.h>

#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>

#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn_models/operations_generic.h>

#include "model_operations.h"
#include "model_forward_cuda.h"

#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>

#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>

#include "model.h"
#include "model_student.h"
#include "scene.h"

#include "../example/environment/environment.h"
#include "../example/environment/operations_cpu.h"

#include <rl_tools/utils/extrack/extrack.h>
#include <rl_tools/utils/extrack/operations_cpu.h>

#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn/layers/dynamic_conv2d/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>
#include "model_persist.h"
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

struct DistillConfig {
    static constexpr float LEARNING_RATE = 1e-3f;
    static constexpr float MAX_ANGLE = 3.14159265358979323846f / 6.0f;
    static constexpr float COS_FOV_MIN = 0.3f;
    static constexpr float COS_FOV_MAX = 1.2f;
    static constexpr float HUBER_DELTA = 0.1f;
    static constexpr float TEACHER_ERROR_THRESHOLD = 0.3f;
    static constexpr float LOSS_WEIGHT_COSINE = 1.0f;
    static constexpr float LOSS_WEIGHT_TASK = 1.0f;
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

// ---- Teacher model ----
using CPU_TYPE_POLICY = rlt::numeric_types::Policy<float>;

// Load teacher on CPU with Gradient capability (needed to copy head to backward-capable GPU copy)
using TEACHER_CPU_GRAD_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>;
using TEACHER_CPU_MODEL_GRAD = yp::MODEL<TEACHER_CPU_GRAD_CAPABILITY, CPU_TYPE_POLICY, TI, BATCH_SIZE>;
// Forward-only CPU copy (intermediate for CPU Gradient → GPU Forward transfer)
using TEACHER_CPU_FWD_CAPABILITY = rlt::nn::capability::Forward<>;
using TEACHER_CPU_MODEL_FWD = yp::MODEL<TEACHER_CPU_FWD_CAPABILITY, CPU_TYPE_POLICY, TI, BATCH_SIZE>;

// GPU teacher model (Forward only, frozen, for encoder eval + head eval)
using TEACHER_GPU_CAPABILITY = rlt::nn::capability::Forward<>;
using TEACHER_GPU_MODEL = yp::MODEL<TEACHER_GPU_CAPABILITY, TYPE_POLICY, TI_CUDA, BATCH_SIZE>;

// ---- Student model (trainable) ----
struct AdamParams : rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY> {
    static constexpr float ALPHA = DistillConfig::LEARNING_RATE;
};
using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI_CUDA, AdamParams>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
using STUDENT_GPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using STUDENT_GPU_MODEL = yp::STUDENT_MODEL<STUDENT_GPU_CAPABILITY, TYPE_POLICY, TI_CUDA, BATCH_SIZE>;

using STUDENT_CPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using STUDENT_CPU_MODEL = yp::STUDENT_MODEL<STUDENT_CPU_CAPABILITY, CPU_TYPE_POLICY, TI, BATCH_SIZE>;
using STUDENT_CPU_MODEL_INFERENCE = typename STUDENT_CPU_MODEL::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;

// Student head (trainable, same architecture as teacher head)
using STUDENT_CONCAT_SHAPE = typename STUDENT_GPU_MODEL::SPEC::CONCAT_OUTPUT_SHAPE;
using STUDENT_HEAD_GPU = typename yp::HEAD_MODULE<TYPE_POLICY, TI_CUDA, yp::ModelConfig<TI_CUDA>>::template Layer<STUDENT_GPU_CAPABILITY, STUDENT_CONCAT_SHAPE>;
using STUDENT_HEAD_CPU_CONCAT_SHAPE = typename STUDENT_CPU_MODEL::SPEC::CONCAT_OUTPUT_SHAPE;
using STUDENT_HEAD_CPU = typename yp::HEAD_MODULE<CPU_TYPE_POLICY, TI, yp::ModelConfig<TI>>::template Layer<STUDENT_CPU_CAPABILITY, STUDENT_HEAD_CPU_CONCAT_SHAPE>;
using STUDENT_HEAD_CPU_INFERENCE = typename STUDENT_HEAD_CPU::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;

// ---- Tensor shapes ----
using GPU_INPUT_SHAPE = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH, 3>;
using GPU_INPUT_SPEC = rlt::tensor::Specification<T_ACTIVATION, TI_CUDA, GPU_INPUT_SHAPE>;

// Teacher encoder output dimensions
static constexpr int TEACHER_ENCODER_DIM_VAL = TEACHER_GPU_MODEL::SPEC::LAST_DIM_A;
static constexpr int TEACHER_CONCAT_DIM = TEACHER_GPU_MODEL::SPEC::LAST_DIM;
static constexpr int TEACHER_ENCODER_TOTAL = rlt::product(typename TEACHER_GPU_MODEL::SPEC::OUTPUT_SHAPE_A{});
static constexpr int ENCODER_SPATIAL = TEACHER_ENCODER_TOTAL / TEACHER_ENCODER_DIM_VAL;
static constexpr int SPATIAL_PER_SAMPLE = ENCODER_SPATIAL / BATCH_SIZE;

// Student encoder output dimensions (should match teacher after projection)
static constexpr int STUDENT_ENCODER_DIM_VAL = STUDENT_GPU_MODEL::SPEC::LAST_DIM_A;
static constexpr int STUDENT_CONCAT_DIM = STUDENT_GPU_MODEL::SPEC::LAST_DIM;
// Note: student per-branch dim (TEACHER_ENCODER_DIM=512) != teacher per-branch dim (LATE_CH=256)
// The student projects to the teacher's *concatenated* dim per branch, not per-branch dim.
// Cosine similarity operates on the min of the two dims or requires separate alignment.

// Head output shape (3 values: px, py, roll)
using HEAD_OUTPUT_SHAPE = typename STUDENT_HEAD_GPU::OUTPUT_SHAPE;
static constexpr TI OUTPUT_DIM = 3;
static constexpr TI TOTAL_OUTPUT_ELEMENTS = BATCH_SIZE * OUTPUT_DIM;

// ---- CUDA kernels ----
// concatenate_kernel, split_kernel, rgba_to_activation_kernel are provided by
// yp::cuda_kernels:: in model_forward_cuda.h

// Compute per-sample teacher MAE and generate binary filter mask
__global__ void compute_teacher_error_mask_kernel(
    const T_ACTIVATION* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ mask,
    int batch_size,
    float threshold
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float err = 0.0f;
    for (int d = 0; d < 3; d++) {
        err += fabsf((float)predictions[idx * 3 + d] - targets[idx * 3 + d]);
    }
    err /= 3.0f;

    mask[idx] = (err < threshold) ? 1.0f : 0.0f;
}

// Huber loss with per-sample mask
__global__ void huber_loss_gradient_masked_kernel(
    const T_ACTIVATION* __restrict__ predictions,
    const float* __restrict__ targets,
    T_GRADIENT* __restrict__ d_output,
    float* __restrict__ losses,
    const float* __restrict__ mask,
    int batch_size,
    int output_dim,
    float delta,
    float loss_weight
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * output_dim;
    if (idx >= total) return;

    const int sample_idx = idx / output_dim;
    const float m = mask[sample_idx];

    const float pred = (float)predictions[idx];
    const float tgt = targets[idx];
    const float diff = pred - tgt;
    const float abs_diff = fabsf(diff);

    const float scale = m * loss_weight / static_cast<float>(total);
    if (abs_diff <= delta) {
        losses[idx] = m * 0.5f * diff * diff;
        d_output[idx] = (T_GRADIENT)(scale * diff);
    } else {
        losses[idx] = m * delta * (abs_diff - 0.5f * delta);
        d_output[idx] = (T_GRADIENT)(scale * delta * ((diff > 0.0f) - (diff < 0.0f)));
    }
}

// Cosine similarity loss and gradient per spatial position
// For each (batch, h, w) position, computes cosine similarity across channel dimension
// Loss = 1 - cos_sim, gradient w.r.t. student features
__global__ void cosine_similarity_loss_gradient_kernel(
    const T_ACTIVATION* __restrict__ student,
    const T_ACTIVATION* __restrict__ teacher,
    T_GRADIENT* __restrict__ d_student,
    float* __restrict__ losses,
    const float* __restrict__ mask,
    int batch_size,
    int spatial_size,
    int channels,
    float loss_weight
) {
    const int pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_positions = batch_size * spatial_size;
    if (pos_idx >= total_positions) return;

    const int sample_idx = pos_idx / spatial_size;
    const float m = mask[sample_idx];
    const int offset = pos_idx * channels;

    float dot = 0.0f;
    float norm_s_sq = 0.0f;
    float norm_t_sq = 0.0f;
    for (int c = 0; c < channels; c++) {
        const float s = (float)student[offset + c];
        const float t = (float)teacher[offset + c];
        dot += s * t;
        norm_s_sq += s * s;
        norm_t_sq += t * t;
    }

    const float eps = 1e-8f;
    const float norm_s = sqrtf(norm_s_sq + eps);
    const float norm_t = sqrtf(norm_t_sq + eps);
    const float cos_sim = dot / (norm_s * norm_t);
    losses[pos_idx] = m * (1.0f - cos_sim);

    // d(1 - cos_sim)/ds_i = -(t_i / (ns * nt) - dot * s_i / (ns^3 * nt))
    const float scale = m * loss_weight / static_cast<float>(total_positions);
    const float inv_ns_nt = 1.0f / (norm_s * norm_t);
    const float dot_over_ns3_nt = dot / (norm_s * norm_s_sq * norm_t + eps);

    for (int c = 0; c < channels; c++) {
        const float s = (float)student[offset + c];
        const float t = (float)teacher[offset + c];
        const float grad = -scale * (t * inv_ns_nt - s * dot_over_ns3_nt);
        d_student[offset + c] = (T_GRADIENT)grad;
    }
}

// Add gradient tensors element-wise: a += b
__global__ void add_gradients_kernel(
    T_GRADIENT* __restrict__ a,
    const T_GRADIENT* __restrict__ b,
    int total
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    a[idx] = (T_GRADIENT)((float)a[idx] + (float)b[idx]);
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
    std::string checkpoint_path;
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
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            checkpoint_path = argv[++i];
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
                      << " --checkpoint <teacher.h5>"
                      << " [--scene-dir <dir> | --scene <path.glb>]"
                      << " [--num-scenes N] [--num-scenes-per-batch M]"
                      << " [--num-iterations N] [--num-val-scenes N]"
                      << " [--num-load-threads N]" << std::endl;
            return 1;
        }
    }

    if (checkpoint_path.empty()) {
        std::cerr << "Error: --checkpoint <teacher.h5> is required" << std::endl;
        return 1;
    }

    // ---- Collect scene paths ----
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
        TI total_needed = num_scenes + num_val_scenes;
        if (total_needed > found_paths.size()) {
            std::cerr << "Need " << total_needed << " scenes but only found " << found_paths.size() << std::endl;
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

    std::cout << "Distillation training (teacher → student)" << std::endl;
    std::cout << "  Teacher checkpoint: " << checkpoint_path << std::endl;
    std::cout << "  Training scenes: " << all_scene_paths.size() << std::endl;
    std::cout << "  Validation scenes: " << val_scene_paths.size() << std::endl;
    std::cout << "  Scenes per batch: " << num_scenes_per_batch << std::endl;
    std::cout << "  Samples per scene: " << samples_per_scene << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Image resolution: " << CAM_WIDTH << "x" << CAM_HEIGHT << std::endl;
    std::cout << "  Teacher error threshold: " << DistillConfig::TEACHER_ERROR_THRESHOLD << std::endl;
    std::cout << "  Cosine loss weight: " << DistillConfig::LOSS_WEIGHT_COSINE << std::endl;
    std::cout << "  Task loss weight: " << DistillConfig::LOSS_WEIGHT_TASK << std::endl;
    std::cout << "  Learning rate: " << DistillConfig::LEARNING_RATE << std::endl;
    std::cout << "  Num iterations: " << num_iterations << std::endl;

    // ---- Load scenes ----
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
        std::cout << "All " << total_scenes << " scenes loaded." << std::endl;
    }

    std::mt19937 scene_rng(42);

    // ---- Device setup ----
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cpu);
    rlt::init(device_cuda);

    DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 42);

    // ---- Load teacher model ----
#ifdef RL_TOOLS_ENABLE_HDF5
    // Load on CPU with Gradient capability (needed to copy head to backward-capable GPU model)
    TEACHER_CPU_MODEL_GRAD teacher_cpu_grad;
    rlt::malloc(device_cpu, teacher_cpu_grad);
    {
        auto file = HighFive::File(checkpoint_path, HighFive::File::ReadOnly);
        auto mg = rlt::get_group(device_cpu, file, "model");
        const bool success = rlt::load(device_cpu, teacher_cpu_grad, mg);
        if (!success) {
            std::cerr << "Failed to load teacher checkpoint: " << checkpoint_path << std::endl;
            return 1;
        }
    }
    std::cout << "Teacher model loaded from " << checkpoint_path << std::endl;

    // Copy CPU Gradient → CPU Forward (needed as intermediate for GPU Forward transfer)
    TEACHER_CPU_MODEL_FWD teacher_cpu_fwd;
    rlt::malloc(device_cpu, teacher_cpu_fwd);
    rlt::copy(device_cpu, device_cpu, teacher_cpu_grad, teacher_cpu_fwd);

    // Copy CPU Forward → GPU Forward (teacher for encoder eval + head eval)
    TEACHER_GPU_MODEL teacher;
    typename TEACHER_GPU_MODEL::template Buffer<true> teacher_buffer;
    rlt::malloc(device_cuda, teacher);
    rlt::malloc(device_cuda, teacher_buffer);
    rlt::copy(device_cpu, device_cuda, teacher_cpu_fwd, teacher);
    rlt::free(device_cpu, teacher_cpu_fwd);

    rlt::free(device_cpu, teacher_cpu_grad);
#else
    std::cerr << "Error: HDF5 support required for loading teacher checkpoint" << std::endl;
    return 1;
#endif

    // ---- Student model setup ----
    OPTIMIZER optimizer;
    STUDENT_GPU_MODEL student;
    typename STUDENT_GPU_MODEL::template Buffer<true> student_buffer;
    rlt::malloc(device_cuda, optimizer);
    rlt::malloc(device_cuda, student);
    rlt::malloc(device_cuda, student_buffer);
    rlt::init(device_cuda, optimizer);

    STUDENT_HEAD_GPU student_head;
    typename STUDENT_HEAD_GPU::template Buffer<true> student_head_buffer;
    rlt::malloc(device_cuda, student_head);
    rlt::malloc(device_cuda, student_head_buffer);

    STUDENT_CPU_MODEL student_cpu;
    STUDENT_CPU_MODEL_INFERENCE student_cpu_inference;
    STUDENT_HEAD_CPU student_head_cpu;
    STUDENT_HEAD_CPU_INFERENCE student_head_cpu_inference;
    rlt::malloc(device_cpu, student_cpu);
    rlt::malloc(device_cpu, student_cpu_inference);
    rlt::malloc(device_cpu, student_head_cpu);
    rlt::malloc(device_cpu, student_head_cpu_inference);
    {
        DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
        rlt::malloc(device_cpu, rng_cpu);
        rlt::init(device_cpu, rng_cpu, 42);
        rlt::init_weights(device_cpu, student_cpu, rng_cpu);
        rlt::init_weights(device_cpu, student_head_cpu, rng_cpu);
        rlt::free(device_cpu, rng_cpu);
    }
    rlt::copy(device_cpu, device_cuda, student_cpu, student);
    rlt::copy(device_cpu, device_cuda, student_head_cpu, student_head);
    rlt::reset_optimizer_state(device_cuda, optimizer, student);
    rlt::reset_optimizer_state(device_cuda, optimizer, student_head);
    std::cout << "Student model initialized" << std::endl;

    // ---- GPU tensors ----
    rlt::Tensor<GPU_INPUT_SPEC> gpu_input_a, gpu_input_b;
    rlt::malloc(device_cuda, gpu_input_a);
    rlt::malloc(device_cuda, gpu_input_b);

    using GPU_D_INPUT_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, GPU_INPUT_SHAPE>;
    rlt::Tensor<GPU_D_INPUT_SPEC> gpu_d_input_a, gpu_d_input_b;
    rlt::malloc(device_cuda, gpu_d_input_a);
    rlt::malloc(device_cuda, gpu_d_input_b);

    // Head output gradient
    using HEAD_D_OUTPUT_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, HEAD_OUTPUT_SHAPE>;
    rlt::Tensor<HEAD_D_OUTPUT_SPEC> gpu_d_head_output;
    rlt::malloc(device_cuda, gpu_d_head_output);

    // d_concatenated from teacher head backward_input
    using D_CONCAT_SHAPE = typename TEACHER_GPU_MODEL::SPEC::CONCAT_OUTPUT_SHAPE;
    using D_CONCAT_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, D_CONCAT_SHAPE, true, rlt::tensor::RowMajorStride<D_CONCAT_SHAPE>>;
    rlt::Tensor<D_CONCAT_SPEC> gpu_d_student_concat_task;
    rlt::malloc(device_cuda, gpu_d_student_concat_task);

    // Cosine similarity gradient buffers for each encoder branch
    using ENCODER_OUTPUT_SHAPE = typename STUDENT_GPU_MODEL::SPEC::OUTPUT_SHAPE_A;
    using D_ENCODER_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, ENCODER_OUTPUT_SHAPE, true, rlt::tensor::RowMajorStride<ENCODER_OUTPUT_SHAPE>>;
    rlt::Tensor<D_ENCODER_SPEC> gpu_d_student_a_cos, gpu_d_student_b_cos;
    rlt::malloc(device_cuda, gpu_d_student_a_cos);
    rlt::malloc(device_cuda, gpu_d_student_b_cos);

    // Teacher head evaluation output (for computing predictions and mask)
    using TEACHER_PRED_SPEC = rlt::tensor::Specification<T_ACTIVATION, TI_CUDA, rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, OUTPUT_DIM>>;
    rlt::Tensor<TEACHER_PRED_SPEC> gpu_teacher_predictions;
    rlt::malloc(device_cuda, gpu_teacher_predictions);

    rlt::disable_dynamic_memory_allocation(device_cuda);

    // Raw GPU buffers
    float* gpu_targets;
    cudaMalloc(&gpu_targets, TOTAL_OUTPUT_ELEMENTS * sizeof(float));
    float* gpu_mask;
    cudaMalloc(&gpu_mask, BATCH_SIZE * sizeof(float));
    float* gpu_task_losses;
    cudaMalloc(&gpu_task_losses, TOTAL_OUTPUT_ELEMENTS * sizeof(float));
    float* gpu_cos_losses;
    cudaMalloc(&gpu_cos_losses, ENCODER_SPATIAL * sizeof(float));
    float* gpu_total_loss;
    cudaMalloc(&gpu_total_loss, 2 * sizeof(float));

    std::vector<float> cpu_targets(TOTAL_OUTPUT_ELEMENTS);
    std::vector<rlt::CameraData> cameras(NUM_CAMERAS);

    // ---- Extrack setup ----
    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = "yaw-prediction-distill";
    extrack_config.population_variates = "cross-conv_channel-multiplier_resolution";
    extrack_config.population_values = std::to_string(ABLATION_USE_CROSS_CONV) + "_" + std::to_string(ABLATION_CHANNEL_MULTIPLIER) + "_" + std::to_string(ABLATION_RESOLUTION);
    rlt::init(device_cpu, extrack_config, extrack_paths, 0);

    std::signal(SIGINT, signal_handler);

    // ---- Training loop ----
    auto train_mode = rlt::Mode<rlt::mode::Default<>>{};
    auto eval_mode = rlt::Mode<rlt::mode::Evaluation<>>{};
    using CLOCK = std::chrono::high_resolution_clock;
    CLOCK::time_point total_start;
    bool throughput_timer_started = false;

    std::vector<TI> scene_indices(loaded_scenes.size());
    std::iota(scene_indices.begin(), scene_indices.end(), 0);

    for (TI iteration = 0; iteration < num_iterations && !signal_received; iteration++) {
        if (!throughput_timer_started && iteration == 1) {
            total_start = CLOCK::now();
            throughput_timer_started = true;
        }
        rlt::set_step(device_cpu, device_cpu.logger, iteration);

        // ---- Select random scenes ----
        std::shuffle(scene_indices.begin(), scene_indices.end(), scene_rng);

        // ---- Sample and render ----
        for (TI s = 0; s < num_scenes_per_batch; s++) {
            TI scene_idx = scene_indices[s];
            const TI batch_offset = s * samples_per_scene;
            yp::sample_camera_batch(
                loaded_scenes[scene_idx].handle,
                cameras.data(),
                cpu_targets.data() + batch_offset * OUTPUT_DIM,
                samples_per_scene,
                DistillConfig::MAX_ANGLE,
                DistillConfig::COS_FOV_MIN,
                DistillConfig::COS_FOV_MAX
            );
            yp::render_batch<true>(loaded_scenes[scene_idx].handle, cameras.data());
        }

        for (TI s = 0; s < num_scenes_per_batch; s++) {
            TI scene_idx = scene_indices[s];
            auto& handle = loaded_scenes[scene_idx].handle;
            rlt::render_rgb_only_sync(handle->device, *handle->env.renderer);
        }

        for (TI s = 0; s < num_scenes_per_batch; s++) {
            TI scene_idx = scene_indices[s];
            const TI batch_offset = s * samples_per_scene;
            uint32_t* device_fb = yp::get_framebuffer_device_ptr(loaded_scenes[scene_idx].handle);
            yp::cuda_kernels::rgba_to_activation_kernel<<<samples_per_scene, 256, 0, device_cuda.stream>>>(
                device_fb, gpu_input_a._data, CAM_WIDTH, CAM_HEIGHT, 0, batch_offset
            );
            yp::cuda_kernels::rgba_to_activation_kernel<<<samples_per_scene, 256, 0, device_cuda.stream>>>(
                device_fb, gpu_input_b._data, CAM_WIDTH, CAM_HEIGHT, samples_per_scene, batch_offset
            );
        }
        cudaMemcpyAsync(gpu_targets, cpu_targets.data(), TOTAL_OUTPUT_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice, device_cuda.stream);

        // ---- Teacher forward (eval mode, frozen) ----
        // evaluate_cuda runs all submodules and writes predictions directly
        rlt::evaluate_cuda(device_cuda, teacher, gpu_input_a, gpu_input_b, gpu_teacher_predictions, teacher_buffer, rng_cuda, eval_mode);

        // Compute per-sample filter mask based on teacher error
        {
            constexpr int threads = 256;
            const int blocks = (BATCH_SIZE + threads - 1) / threads;
            compute_teacher_error_mask_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                gpu_teacher_predictions._data, gpu_targets, gpu_mask, BATCH_SIZE, DistillConfig::TEACHER_ERROR_THRESHOLD
            );
        }

        // ---- Student forward (train mode) ----
        rlt::zero_gradient(device_cuda, student);
        rlt::zero_gradient(device_cuda, student_head);

        rlt::forward(device_cuda, student.pipeline_a, gpu_input_a, student_buffer.buffer_a, rng_cuda, train_mode);
        rlt::forward(device_cuda, student.pipeline_b, gpu_input_b, student_buffer.buffer_b, rng_cuda, train_mode);

        auto student_output_a = rlt::output(device_cuda, student.pipeline_a);
        auto student_output_b = rlt::output(device_cuda, student.pipeline_b);
        rlt::copy(device_cuda, device_cuda, student_output_a, student_buffer.intermediate_a);
        rlt::copy(device_cuda, device_cuda, student_output_b, student_buffer.intermediate_b);

        // ---- Dense supervision: cosine similarity loss on encoder features ----
        {
            constexpr int total_positions = ENCODER_SPATIAL;
            constexpr int threads = 256;
            constexpr int blocks = (total_positions + threads - 1) / threads;
            cosine_similarity_loss_gradient_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                student_buffer.intermediate_a._data,
                teacher_buffer.intermediate_a._data,
                gpu_d_student_a_cos._data,
                gpu_cos_losses,
                gpu_mask,
                BATCH_SIZE,
                SPATIAL_PER_SAMPLE,
                TEACHER_ENCODER_DIM_VAL,
                DistillConfig::LOSS_WEIGHT_COSINE
            );
            cosine_similarity_loss_gradient_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                student_buffer.intermediate_b._data,
                teacher_buffer.intermediate_b._data,
                gpu_d_student_b_cos._data,
                gpu_cos_losses,
                gpu_mask,
                BATCH_SIZE,
                SPATIAL_PER_SAMPLE,
                TEACHER_ENCODER_DIM_VAL,
                DistillConfig::LOSS_WEIGHT_COSINE
            );
        }

        // ---- Task loss: student features through teacher head ----
        {
            constexpr int total = ENCODER_SPATIAL * STUDENT_CONCAT_DIM;
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            yp::cuda_kernels::concatenate_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                student_buffer.intermediate_a._data, STUDENT_ENCODER_DIM_VAL,
                student_buffer.intermediate_b._data, STUDENT_ENCODER_DIM_VAL,
                student_buffer.concatenated._data, ENCODER_SPATIAL
            );
        }

        // Forward through student head
        rlt::forward(device_cuda, student_head, student_buffer.concatenated, student_head_buffer, rng_cuda, train_mode);
        {
            auto head_output = rlt::output(device_cuda, student_head);
            // Compute Huber loss against ground truth targets
            {
                constexpr int threads = 256;
                constexpr int blocks = (TOTAL_OUTPUT_ELEMENTS + threads - 1) / threads;
                huber_loss_gradient_masked_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                    head_output._data, gpu_targets, gpu_d_head_output._data, gpu_task_losses, gpu_mask,
                    BATCH_SIZE, OUTPUT_DIM, DistillConfig::HUBER_DELTA, DistillConfig::LOSS_WEIGHT_TASK
                );
            }
        }

        // Backward through student head to get d_student_concatenated
        rlt::backward_full(device_cuda, student_head, student_buffer.concatenated, gpu_d_head_output, gpu_d_student_concat_task, student_head_buffer);

        // Split task gradient into per-branch gradients and add cosine similarity gradients
        {
            constexpr int total = ENCODER_SPATIAL * STUDENT_CONCAT_DIM;
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            yp::cuda_kernels::split_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                gpu_d_student_concat_task._data, STUDENT_ENCODER_DIM_VAL, STUDENT_ENCODER_DIM_VAL,
                student_buffer.d_output_a._data,
                student_buffer.d_output_b._data,
                ENCODER_SPATIAL
            );
        }

        // Add cosine similarity gradients to task gradients
        {
            constexpr int total = ENCODER_SPATIAL * STUDENT_ENCODER_DIM_VAL;
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            add_gradients_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                student_buffer.d_output_a._data, gpu_d_student_a_cos._data, total
            );
            add_gradients_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                student_buffer.d_output_b._data, gpu_d_student_b_cos._data, total
            );
        }

        // ---- Student backward ----
        rlt::backward_full(device_cuda, student.pipeline_a, gpu_input_a, student_buffer.d_output_a, gpu_d_input_a, student_buffer.buffer_a);
        rlt::backward_full(device_cuda, student.pipeline_b, gpu_input_b, student_buffer.d_output_b, gpu_d_input_b, student_buffer.buffer_b);

        // ---- Optimizer step ----
        rlt::step(device_cuda, optimizer, student);
        rlt::step(device_cuda, optimizer, student_head);

        // ---- Checkpointing ----
#ifdef RL_TOOLS_ENABLE_HDF5
        if (iteration % DistillConfig::CHECKPOINT_INTERVAL == 0 || iteration == num_iterations - 1) {
            cudaStreamSynchronize(device_cuda.stream);
            rlt::copy(device_cuda, device_cpu, student, student_cpu);
            rlt::copy(device_cpu, device_cpu, student_cpu, student_cpu_inference);
            rlt::copy(device_cuda, device_cpu, student_head, student_head_cpu);
            rlt::copy(device_cpu, device_cpu, student_head_cpu, student_head_cpu_inference);
            auto step_folder = rlt::get_step_folder(device_cpu, extrack_config, extrack_paths, iteration);
            auto file = HighFive::File((step_folder / "checkpoint.h5").string(), HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
            auto mg = rlt::create_group(device_cpu, file, "student");
            rlt::save(device_cpu, student_cpu_inference, mg);
            auto mg_head = rlt::create_group(device_cpu, file, "student_head");
            rlt::save(device_cpu, student_head_cpu_inference, mg_head);
            std::cout << "  Checkpoint: " << (step_folder / "checkpoint.h5").string() << std::endl;
        }
#endif

        // ---- Logging ----
        if (iteration % DistillConfig::LOG_INTERVAL == 0 || iteration == num_iterations - 1) {
            reduce_loss_kernel<<<1, 1, 0, device_cuda.stream>>>(
                gpu_task_losses, gpu_total_loss, TOTAL_OUTPUT_ELEMENTS
            );
            reduce_loss_kernel<<<1, 1, 0, device_cuda.stream>>>(
                gpu_cos_losses, gpu_total_loss + 1, ENCODER_SPATIAL
            );
            cudaStreamSynchronize(device_cuda.stream);

            float loss_vals[2];
            cudaMemcpy(loss_vals, gpu_total_loss, 2 * sizeof(float), cudaMemcpyDeviceToHost);

            auto head_output = rlt::output(device_cuda, student_head);
            std::vector<T_ACTIVATION> pred_buf_raw(TOTAL_OUTPUT_ELEMENTS);
            cudaMemcpy(pred_buf_raw.data(), head_output._data, TOTAL_OUTPUT_ELEMENTS * sizeof(T_ACTIVATION), cudaMemcpyDeviceToHost);

            float err_px = 0.0f, err_py = 0.0f, err_roll = 0.0f;
            for (TI_CUDA i = 0; i < BATCH_SIZE; i++) {
                err_px   += std::abs((float)pred_buf_raw[i * OUTPUT_DIM + 0] - cpu_targets[i * OUTPUT_DIM + 0]);
                err_py   += std::abs((float)pred_buf_raw[i * OUTPUT_DIM + 1] - cpu_targets[i * OUTPUT_DIM + 1]);
                err_roll += std::abs((float)pred_buf_raw[i * OUTPUT_DIM + 2] - cpu_targets[i * OUTPUT_DIM + 2]);
            }
            err_px /= BATCH_SIZE;
            err_py /= BATCH_SIZE;
            err_roll /= BATCH_SIZE;

            // Count masked samples
            std::vector<float> mask_buf(BATCH_SIZE);
            cudaMemcpy(mask_buf.data(), gpu_mask, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
            TI num_valid = 0;
            for (TI i = 0; i < BATCH_SIZE; i++) {
                if (mask_buf[i] > 0.5f) num_valid++;
            }

            auto now = CLOCK::now();
            double elapsed_s = throughput_timer_started
                ? std::chrono::duration<double>(now - total_start).count()
                : 0.0;
            double samples_per_s = elapsed_s > 0.0 && iteration > 0
                ? static_cast<double>(iteration * BATCH_SIZE) / elapsed_s
                : 0.0;

            rlt::add_scalar(device_cpu, device_cpu.logger, "train/task_loss", loss_vals[0]);
            rlt::add_scalar(device_cpu, device_cpu.logger, "train/cosine_loss", loss_vals[1]);
            rlt::add_scalar(device_cpu, device_cpu.logger, "train/err_px", err_px);
            rlt::add_scalar(device_cpu, device_cpu.logger, "train/err_py", err_py);
            rlt::add_scalar(device_cpu, device_cpu.logger, "train/err_roll", err_roll);
            rlt::add_scalar(device_cpu, device_cpu.logger, "train/valid_samples", (float)num_valid);
            rlt::add_scalar(device_cpu, device_cpu.logger, "train/samples_per_s", samples_per_s);

            std::cout << "[iter " << iteration << "/" << num_iterations << "]"
                      << "  task=" << loss_vals[0]
                      << "  cos=" << loss_vals[1]
                      << "  err_px=" << err_px
                      << "  err_py=" << err_py
                      << "  err_roll=" << err_roll
                      << "  valid=" << num_valid << "/" << BATCH_SIZE
                      << "  samples/s=" << samples_per_s
                      << std::endl;
        }

        // ---- Validation ----
        if (val_scenes.size() > 0 && (iteration % DistillConfig::VAL_INTERVAL == 0 || iteration == num_iterations - 1)) {
            float val_err_px_sum = 0.0f, val_err_py_sum = 0.0f, val_err_roll_sum = 0.0f;
            float teacher_err_px_sum = 0.0f, teacher_err_py_sum = 0.0f, teacher_err_roll_sum = 0.0f;
            TI val_total_samples = 0;

            for (TI vs = 0; vs < val_scenes.size(); vs++) {
                yp::sample_camera_batch(
                    val_scenes[vs].handle,
                    cameras.data(),
                    cpu_targets.data(),
                    BATCH_SIZE,
                    DistillConfig::MAX_ANGLE,
                    DistillConfig::COS_FOV_MIN,
                    DistillConfig::COS_FOV_MAX
                );
                yp::render_batch<false>(val_scenes[vs].handle, cameras.data());
                uint32_t* device_fb = yp::get_framebuffer_device_ptr(val_scenes[vs].handle);

                yp::cuda_kernels::rgba_to_activation_kernel<<<BATCH_SIZE, 256, 0, device_cuda.stream>>>(
                    device_fb, gpu_input_a._data, CAM_WIDTH, CAM_HEIGHT, 0, 0
                );
                yp::cuda_kernels::rgba_to_activation_kernel<<<BATCH_SIZE, 256, 0, device_cuda.stream>>>(
                    device_fb, gpu_input_b._data, CAM_WIDTH, CAM_HEIGHT, BATCH_SIZE, 0
                );
                cudaMemcpyAsync(gpu_targets, cpu_targets.data(), TOTAL_OUTPUT_ELEMENTS * sizeof(float),
                                cudaMemcpyHostToDevice, device_cuda.stream);

                // Student forward (eval mode)
                rlt::forward(device_cuda, student.pipeline_a, gpu_input_a, student_buffer.buffer_a, rng_cuda, eval_mode);
                rlt::forward(device_cuda, student.pipeline_b, gpu_input_b, student_buffer.buffer_b, rng_cuda, eval_mode);

                auto val_output_a = rlt::output(device_cuda, student.pipeline_a);
                auto val_output_b = rlt::output(device_cuda, student.pipeline_b);
                rlt::copy(device_cuda, device_cuda, val_output_a, student_buffer.intermediate_a);
                rlt::copy(device_cuda, device_cuda, val_output_b, student_buffer.intermediate_b);

                {
                    constexpr int total = ENCODER_SPATIAL * STUDENT_CONCAT_DIM;
                    constexpr int threads = 256;
                    constexpr int blocks = (total + threads - 1) / threads;
                    yp::cuda_kernels::concatenate_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                        student_buffer.intermediate_a._data, STUDENT_ENCODER_DIM_VAL,
                        student_buffer.intermediate_b._data, STUDENT_ENCODER_DIM_VAL,
                        student_buffer.concatenated._data, ENCODER_SPATIAL
                    );
                }

                // Through student head (eval mode)
                rlt::evaluate(device_cuda, student_head, student_buffer.concatenated, gpu_teacher_predictions, student_head_buffer, rng_cuda, eval_mode);

                // Teacher forward (eval mode) for expert baseline
                rlt::evaluate_cuda(device_cuda, teacher, gpu_input_a, gpu_input_b, gpu_teacher_predictions, teacher_buffer, rng_cuda, eval_mode);

                cudaStreamSynchronize(device_cuda.stream);

                std::vector<T_ACTIVATION> teacher_pred_buf_raw(TOTAL_OUTPUT_ELEMENTS);
                cudaMemcpy(teacher_pred_buf_raw.data(), gpu_teacher_predictions._data, TOTAL_OUTPUT_ELEMENTS * sizeof(T_ACTIVATION), cudaMemcpyDeviceToHost);

                for (TI_CUDA i = 0; i < BATCH_SIZE; i++) {
                    teacher_err_px_sum   += std::abs((float)teacher_pred_buf_raw[i * OUTPUT_DIM + 0] - cpu_targets[i * OUTPUT_DIM + 0]);
                    teacher_err_py_sum   += std::abs((float)teacher_pred_buf_raw[i * OUTPUT_DIM + 1] - cpu_targets[i * OUTPUT_DIM + 1]);
                    teacher_err_roll_sum += std::abs((float)teacher_pred_buf_raw[i * OUTPUT_DIM + 2] - cpu_targets[i * OUTPUT_DIM + 2]);
                }

                // Rerun student through head for student error (reuse gpu_teacher_predictions buffer)
                rlt::evaluate(device_cuda, student_head, student_buffer.concatenated, gpu_teacher_predictions, student_head_buffer, rng_cuda, eval_mode);

                cudaStreamSynchronize(device_cuda.stream);

                std::vector<T_ACTIVATION> val_pred_buf_raw(TOTAL_OUTPUT_ELEMENTS);
                cudaMemcpy(val_pred_buf_raw.data(), gpu_teacher_predictions._data, TOTAL_OUTPUT_ELEMENTS * sizeof(T_ACTIVATION), cudaMemcpyDeviceToHost);

                for (TI_CUDA i = 0; i < BATCH_SIZE; i++) {
                    val_err_px_sum   += std::abs((float)val_pred_buf_raw[i * OUTPUT_DIM + 0] - cpu_targets[i * OUTPUT_DIM + 0]);
                    val_err_py_sum   += std::abs((float)val_pred_buf_raw[i * OUTPUT_DIM + 1] - cpu_targets[i * OUTPUT_DIM + 1]);
                    val_err_roll_sum += std::abs((float)val_pred_buf_raw[i * OUTPUT_DIM + 2] - cpu_targets[i * OUTPUT_DIM + 2]);
                }
                val_total_samples += BATCH_SIZE;
            }

            float val_err_px = val_err_px_sum / val_total_samples;
            float val_err_py = val_err_py_sum / val_total_samples;
            float val_err_roll = val_err_roll_sum / val_total_samples;
            float teacher_err_px = teacher_err_px_sum / val_total_samples;
            float teacher_err_py = teacher_err_py_sum / val_total_samples;
            float teacher_err_roll = teacher_err_roll_sum / val_total_samples;

            rlt::add_scalar(device_cpu, device_cpu.logger, "val/err_px", val_err_px);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/err_py", val_err_py);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/err_roll", val_err_roll);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/teacher_err_px", teacher_err_px);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/teacher_err_py", teacher_err_py);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/teacher_err_roll", teacher_err_roll);

            std::cout << "[iter " << iteration << "] VAL"
                      << "  err_px=" << val_err_px
                      << "  err_py=" << val_err_py
                      << "  err_roll=" << val_err_roll
                      << "  teacher_px=" << teacher_err_px
                      << "  teacher_py=" << teacher_err_py
                      << "  teacher_roll=" << teacher_err_roll << std::endl;
        }
    }

    if (signal_received) {
        std::cout << "Training interrupted by signal " << signal_received << "." << std::endl;
    } else {
        std::cout << "Distillation complete." << std::endl;
    }

    // ---- Cleanup ----
    cudaFree(gpu_targets);
    cudaFree(gpu_mask);
    cudaFree(gpu_task_losses);
    cudaFree(gpu_cos_losses);
    cudaFree(gpu_total_loss);
    rlt::free(device_cuda, gpu_input_a);
    rlt::free(device_cuda, gpu_input_b);
    rlt::free(device_cuda, gpu_d_input_a);
    rlt::free(device_cuda, gpu_d_input_b);
    rlt::free(device_cuda, gpu_d_head_output);
    rlt::free(device_cuda, gpu_d_student_concat_task);
    rlt::free(device_cuda, gpu_d_student_a_cos);
    rlt::free(device_cuda, gpu_d_student_b_cos);
    rlt::free(device_cuda, gpu_teacher_predictions);
    rlt::free(device_cuda, student_buffer);
    rlt::free(device_cuda, student);
    rlt::free(device_cuda, optimizer);
    rlt::free(device_cuda, student_head_buffer);
    rlt::free(device_cuda, student_head);
    rlt::free(device_cuda, teacher_buffer);
    rlt::free(device_cuda, teacher);
    rlt::free(device_cuda, rng_cuda);
    rlt::free(device_cpu, student_cpu);
    rlt::free(device_cpu, student_cpu_inference);
    rlt::free(device_cpu, student_head_cpu);
    rlt::free(device_cpu, student_head_cpu_inference);
    for (auto& s : loaded_scenes) {
        yp::destroy_scene(s.handle);
    }
    for (auto& s : val_scenes) {
        yp::destroy_scene(s.handle);
    }
    rlt::free(device_cpu, device_cpu.logger);

    return 0;
}
