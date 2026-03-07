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

// Checkpointing (optional, requires HDF5)
#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/utils/extrack/extrack.h>
#include <rl_tools/utils/extrack/operations_cpu.h>
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
#include <cuda_runtime.h>

namespace rlt = rl_tools;

// ---- Configuration ----
using T = float;
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
    static constexpr TI LOG_INTERVAL = 100;
    static constexpr TI CHECKPOINT_INTERVAL = 10000;
};

using TYPE_POLICY = rlt::numeric_types::Policy<float>;

struct AdamParams : rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY> {
    static constexpr float ALPHA = TrainingConfig::LEARNING_RATE;
};

using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI_CUDA, AdamParams>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
using GPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using GPU_MODEL = rlt::rendering::raytracing::yaw_prediction::MODEL<GPU_CAPABILITY, TYPE_POLICY, TI_CUDA, BATCH_SIZE>;

using CPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using CPU_MODEL = rlt::rendering::raytracing::yaw_prediction::MODEL<CPU_CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE>;

using GPU_INPUT_SHAPE = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH, 3>;
using GPU_INPUT_SPEC = rlt::tensor::Specification<T, TI_CUDA, GPU_INPUT_SHAPE>;
using GPU_OUTPUT_SHAPE = typename GPU_MODEL::OUTPUT_SHAPE;
using GPU_D_OUTPUT_SPEC = rlt::tensor::Specification<T, TI_CUDA, GPU_OUTPUT_SHAPE>;

// Encoder output dimension (256 from global avgpool of 4x4x256)
static constexpr int ENCODER_DIM = GPU_MODEL::SPEC::LAST_DIM_A;
static constexpr int CONCAT_DIM = GPU_MODEL::SPEC::LAST_DIM;

// ---- CUDA kernels ----

// Concatenate two 2D tensors [N, D_A] and [N, D_B] into [N, D_A + D_B] along last dim
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

// Split [N, D_A + D_B] into [N, D_A] and [N, D_B] along last dim
__global__ void split_kernel(
    const float* __restrict__ input, int d_a, int d_b,
    float* __restrict__ a,
    float* __restrict__ b,
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

// Convert RGBA uint32 framebuffer pixels to float RGB tensor [BATCH, H, W, 3] normalized to [0,1]
__global__ void rgba_to_float_kernel(
    const uint32_t* __restrict__ framebuffer,
    float* __restrict__ output,
    int cam_width, int cam_height, int camera_offset
) {
    const int sample_idx = blockIdx.x;
    const int camera_idx = camera_offset + sample_idx;
    const int cam_pixels = cam_width * cam_height;

    const uint32_t* src = framebuffer + camera_idx * cam_pixels;
    float* dst = output + sample_idx * cam_height * cam_width * 3;

    for (int pixel = threadIdx.x; pixel < cam_pixels; pixel += blockDim.x) {
        const uint32_t rgba = src[pixel];
        const float r = static_cast<float>((rgba >>  0) & 0xFF) / 255.0f;
        const float g = static_cast<float>((rgba >>  8) & 0xFF) / 255.0f;
        const float b = static_cast<float>((rgba >> 16) & 0xFF) / 255.0f;

        const int out_base = pixel * 3;
        dst[out_base + 0] = r;
        dst[out_base + 1] = g;
        dst[out_base + 2] = b;
    }
}

// MSE loss on sin/cos outputs and write gradient to d_output
__global__ void mse_loss_gradient_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ d_output,
    float* __restrict__ losses,
    int batch_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float pred_sin = predictions[idx * 2 + 0];
    const float pred_cos = predictions[idx * 2 + 1];
    const float tgt_sin = targets[idx * 2 + 0];
    const float tgt_cos = targets[idx * 2 + 1];

    const float diff_sin = pred_sin - tgt_sin;
    const float diff_cos = pred_cos - tgt_cos;

    losses[idx] = diff_sin * diff_sin + diff_cos * diff_cos;

    const float scale = 2.0f / static_cast<float>(batch_size);
    d_output[idx * 2 + 0] = scale * diff_sin;
    d_output[idx * 2 + 1] = scale * diff_cos;
}

__global__ void reduce_loss_kernel(
    const float* __restrict__ losses,
    float* __restrict__ total_loss,
    int batch_size
) {
    float sum = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        sum += losses[i];
    }
    *total_loss = sum / static_cast<float>(batch_size);
}

int main(int argc, char** argv) {
    const char* scene_path = "ProcTHOR-Test-0-new.glb";
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) {
            scene_path = argv[++i];
        } else {
            std::cerr << "Usage: " << argv[0] << " [--scene <path.glb>]" << std::endl;
            return 1;
        }
    }

    constexpr float PI = 3.14159265358979323846f;

    std::cout << "Yaw prediction training" << std::endl;
    std::cout << "  Scene: " << scene_path << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Image resolution: " << CAM_WIDTH << "x" << CAM_HEIGHT << std::endl;
    std::cout << "  Max angle: " << TrainingConfig::MAX_ANGLE * 180.0f / PI << " degrees" << std::endl;
    std::cout << "  Learning rate: " << TrainingConfig::LEARNING_RATE << std::endl;
    std::cout << "  Encoder dim: " << ENCODER_DIM << ", concat dim: " << CONCAT_DIM << std::endl;

    // ---- Scene setup ----
    auto* scene = yp::create_scene(scene_path);
    std::cout << "Indoor initial states: " << yp::get_num_indoor_states(scene) << std::endl;

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
    rlt::malloc(device_cpu, model_cpu);
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

    rlt::Tensor<GPU_D_OUTPUT_SPEC> gpu_d_output;
    rlt::malloc(device_cuda, gpu_d_output);

    float* gpu_targets;
    cudaMalloc(&gpu_targets, BATCH_SIZE * 2 * sizeof(float));

    float* gpu_losses;
    cudaMalloc(&gpu_losses, BATCH_SIZE * sizeof(float));

    float* gpu_total_loss;
    cudaMalloc(&gpu_total_loss, sizeof(float));

    // CPU-side buffers
    std::vector<float> cpu_targets(BATCH_SIZE * 2);
    std::vector<float> cpu_delta_yaws(BATCH_SIZE);
    std::vector<rlt::CameraData> cameras(NUM_CAMERAS);

    // ---- Extrack setup (checkpointing) ----
#ifdef RL_TOOLS_ENABLE_HDF5
    rlt::utils::extrack::Config<TI> extrack_config;
    rlt::utils::extrack::Paths extrack_paths;
    extrack_config.name = "yaw-prediction";
    rlt::init(device_cpu, extrack_config, extrack_paths, 0);
#endif

    // ---- Training loop ----
    auto train_mode = rlt::Mode<rlt::mode::Default<>>{};
    auto total_start = std::chrono::high_resolution_clock::now();

    for (TI iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        // ---- Sample batch ----
        yp::sample_camera_batch(
            scene, cameras.data(), cpu_delta_yaws.data(), cpu_targets.data(),
            BATCH_SIZE, TrainingConfig::MAX_ANGLE
        );

        // ---- Render ----
        yp::render_batch(scene, cameras.data());

        // ---- Preprocess pixels on GPU ----
        uint32_t* device_fb = yp::get_framebuffer_device_ptr(scene);

        rgba_to_float_kernel<<<BATCH_SIZE, 256, 0, device_cuda.stream>>>(
            device_fb, gpu_input_a._data, CAM_WIDTH, CAM_HEIGHT, 0
        );
        rgba_to_float_kernel<<<BATCH_SIZE, 256, 0, device_cuda.stream>>>(
            device_fb, gpu_input_b._data, CAM_WIDTH, CAM_HEIGHT, BATCH_SIZE
        );

        cudaMemcpyAsync(gpu_targets, cpu_targets.data(), BATCH_SIZE * 2 * sizeof(float),
                        cudaMemcpyHostToDevice, device_cuda.stream);

        // ---- Forward ----
        // Note: parallel model's _concatenate/_split use host-side loops that can't access
        // GPU memory, so we manually decompose the parallel forward/backward here with
        // CUDA kernels for the concatenation and splitting operations.
        rlt::zero_gradient(device_cuda, model);

        // Forward through both encoder pipelines
        rlt::forward(device_cuda, model.pipeline_a, gpu_input_a, model_buffer.buffer_a, rng_cuda, train_mode);
        rlt::forward(device_cuda, model.pipeline_b, gpu_input_b, model_buffer.buffer_b, rng_cuda, train_mode);

        // Copy encoder outputs to contiguous intermediates
        auto output_a = rlt::output(device_cuda, model.pipeline_a);
        auto output_b = rlt::output(device_cuda, model.pipeline_b);
        rlt::copy(device_cuda, device_cuda, output_a, model_buffer.intermediate_a);
        rlt::copy(device_cuda, device_cuda, output_b, model_buffer.intermediate_b);

        // GPU concatenation: [BATCH, ENCODER_DIM] + [BATCH, ENCODER_DIM] -> [BATCH, CONCAT_DIM]
        {
            constexpr int total = BATCH_SIZE * CONCAT_DIM;
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            concatenate_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                model_buffer.intermediate_a._data, ENCODER_DIM,
                model_buffer.intermediate_b._data, ENCODER_DIM,
                model_buffer.concatenated._data, BATCH_SIZE
            );
        }

        // Forward through head MLP
        rlt::forward(device_cuda, model.head, model_buffer.concatenated, model_buffer.head_buffer, rng_cuda, train_mode);
        {
            auto head_output = rlt::output(device_cuda, model.head);
            rlt::copy(device_cuda, device_cuda, head_output, model.output);
        }
        auto output_view = rlt::output(device_cuda, model);

        // ---- Loss + gradient ----
        {
            constexpr int threads = 256;
            constexpr int blocks = (BATCH_SIZE + threads - 1) / threads;
            mse_loss_gradient_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                output_view._data, gpu_targets, gpu_d_output._data, gpu_losses, BATCH_SIZE
            );
        }

        // ---- Backward ----
        // Head backward: computes gradients for head and d_concatenated
        rlt::backward_full(device_cuda, model.head, model_buffer.concatenated, gpu_d_output, model_buffer.d_concatenated, model_buffer.head_buffer);

        // GPU split: [BATCH, CONCAT_DIM] -> [BATCH, ENCODER_DIM] + [BATCH, ENCODER_DIM]
        {
            constexpr int total = BATCH_SIZE * CONCAT_DIM;
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            split_kernel<<<blocks, threads, 0, device_cuda.stream>>>(
                model_buffer.d_concatenated._data, ENCODER_DIM, ENCODER_DIM,
                model_buffer.d_output_a._data,
                model_buffer.d_output_b._data,
                BATCH_SIZE
            );
        }

        // Encoder backward: accumulate weight gradients
        rlt::backward(device_cuda, model.pipeline_a, gpu_input_a, model_buffer.d_output_a, model_buffer.buffer_a);
        rlt::backward(device_cuda, model.pipeline_b, gpu_input_b, model_buffer.d_output_b, model_buffer.buffer_b);

        // ---- Optimizer step ----
        rlt::step(device_cuda, optimizer, model);

        // ---- Checkpointing ----
#ifdef RL_TOOLS_ENABLE_HDF5
        if (iteration % TrainingConfig::CHECKPOINT_INTERVAL == 0 || iteration == NUM_ITERATIONS - 1) {
            cudaStreamSynchronize(device_cuda.stream);
            rlt::copy(device_cuda, device_cpu, model, model_cpu);
            auto step_folder = rlt::get_step_folder(device_cpu, extrack_config, extrack_paths, iteration);
            auto file = HighFive::File((step_folder / "checkpoint.h5").string(), HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
            auto mg = rlt::create_group(device_cpu, file, "model");
            rlt::save(device_cpu, model_cpu, mg);
            std::cout << "  Checkpoint: " << (step_folder / "checkpoint.h5").string() << std::endl;
        }
#endif

        // ---- Logging ----
        if (iteration % TrainingConfig::LOG_INTERVAL == 0 || iteration == NUM_ITERATIONS - 1) {
            reduce_loss_kernel<<<1, 1, 0, device_cuda.stream>>>(
                gpu_losses, gpu_total_loss, BATCH_SIZE
            );
            cudaStreamSynchronize(device_cuda.stream);

            float loss_val;
            cudaMemcpy(&loss_val, gpu_total_loss, sizeof(float), cudaMemcpyDeviceToHost);

            std::vector<float> pred_buf(BATCH_SIZE * 2);
            cudaMemcpy(pred_buf.data(), output_view._data, BATCH_SIZE * 2 * sizeof(float), cudaMemcpyDeviceToHost);

            float total_angle_error = 0.0f;
            for (TI_CUDA i = 0; i < BATCH_SIZE; i++) {
                const float pred_angle = std::atan2(pred_buf[i * 2 + 0], pred_buf[i * 2 + 1]);
                const float angle_error = std::abs(pred_angle - cpu_delta_yaws[i]);
                total_angle_error += angle_error;
            }
            const float mean_angle_error_deg = (total_angle_error / BATCH_SIZE) * 180.0f / PI;

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - total_start).count();

            std::cout << "[iter " << iteration << "/" << NUM_ITERATIONS << "]"
                      << "  loss=" << loss_val
                      << "  angle_err=" << mean_angle_error_deg << " deg"
                      << "  time=" << elapsed_s << "s"
                      << std::endl;
        }
    }

    std::cout << "Training complete." << std::endl;

    // ---- Cleanup ----
    cudaFree(gpu_targets);
    cudaFree(gpu_losses);
    cudaFree(gpu_total_loss);
    rlt::free(device_cuda, gpu_input_a);
    rlt::free(device_cuda, gpu_input_b);
    rlt::free(device_cuda, gpu_d_output);
    rlt::free(device_cuda, model_buffer);
    rlt::free(device_cuda, model);
    rlt::free(device_cuda, optimizer);
    rlt::free(device_cuda, rng_cuda);
    rlt::free(device_cpu, model_cpu);
    yp::destroy_scene(scene);

    return 0;
}
