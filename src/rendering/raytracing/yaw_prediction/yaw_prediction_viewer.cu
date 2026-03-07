#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

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

// Tensor operations
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>

// Persist (HDF5)
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>

// Model definition
#include "model.h"

// Scene management
#include "scene.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>

namespace rlt = rl_tools;

// ---- Configuration ----
using T = float;
using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_CUDA = rlt::devices::DEVICE_FACTORY_CUDA<>;
using TI = DEVICE_CPU::index_t;
using TI_CUDA = DEVICE_CUDA::index_t;

namespace yp = rlt::rendering::raytracing::yaw_prediction;
static constexpr TI_CUDA BATCH_SIZE = yp::SCENE_NUM_CAMERAS / 2; // 128
static constexpr TI_CUDA NUM_CAMERAS = yp::SCENE_NUM_CAMERAS;    // 256
static constexpr TI CAM_WIDTH = yp::SCENE_CAM_WIDTH;
static constexpr TI CAM_HEIGHT = yp::SCENE_CAM_HEIGHT;

static constexpr float PI = 3.14159265358979323846f;
static constexpr float MAX_ANGLE = PI / 6.0f; // 30 degrees
static constexpr TI NUM_GRID_CAMERAS = 64;
static constexpr TI GRID_DIM = 8; // 8x8 grid

using TYPE_POLICY = rlt::numeric_types::Policy<float>;

// Inference-only model
using GPU_CAPABILITY = rlt::nn::capability::Forward<>;
using GPU_MODEL = yp::MODEL<GPU_CAPABILITY, TYPE_POLICY, TI_CUDA, BATCH_SIZE>;
using CPU_CAPABILITY = rlt::nn::capability::Forward<>;
using CPU_MODEL = yp::MODEL<CPU_CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE>;

using GPU_INPUT_SHAPE = rlt::tensor::Shape<TI_CUDA, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH, 3>;
using GPU_INPUT_SPEC = rlt::tensor::Specification<T, TI_CUDA, GPU_INPUT_SHAPE>;

// Head output tensor: [BATCH_SIZE, 2]
using HEAD_OUTPUT_SHAPE = typename GPU_MODEL::SPEC::HEAD_TYPE::OUTPUT_SHAPE;
using HEAD_OUTPUT_SPEC = rlt::tensor::Specification<T, TI_CUDA, HEAD_OUTPUT_SHAPE>;

static constexpr int ENCODER_DIM = GPU_MODEL::SPEC::LAST_DIM_A;
static constexpr int CONCAT_DIM = GPU_MODEL::SPEC::LAST_DIM;

// ---- CUDA kernels ----

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

// ---- Progress bead drawing ----
void draw_bead(std::vector<uint8_t>& image, int image_width, int tile_x, int tile_y,
               float angle, float max_angle, int bead_y, uint8_t r, uint8_t g, uint8_t b) {
    float normalized = (angle + max_angle) / (2.0f * max_angle);
    normalized = std::max(0.0f, std::min(1.0f, normalized));
    int bead_x = 2 + static_cast<int>(normalized * 59.0f);
    for (int dy = 0; dy < 3; dy++) {
        for (int dx = 0; dx < 5; dx++) {
            int px = tile_x + bead_x + dx;
            int py = tile_y + bead_y + dy;
            int idx = (py * image_width + px) * 3;
            image[idx + 0] = r;
            image[idx + 1] = g;
            image[idx + 2] = b;
        }
    }
}

int main(int argc, char** argv) {
    const char* scene_path = nullptr;
    const char* checkpoint_path = nullptr;
    const char* output_dir = "frames";
    int num_frames = 120;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) {
            scene_path = argv[++i];
        } else if (std::strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (std::strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--num-frames") == 0 && i + 1 < argc) {
            num_frames = std::atoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " --scene <path.glb> --checkpoint <path.h5>"
                      << " [--output-dir <dir>] [--num-frames <N>]" << std::endl;
            return 1;
        }
    }

    if (!scene_path || !checkpoint_path) {
        std::cerr << "Error: --scene and --checkpoint are required" << std::endl;
        return 1;
    }

    std::cout << "Yaw prediction viewer" << std::endl;
    std::cout << "  Scene: " << scene_path << std::endl;
    std::cout << "  Checkpoint: " << checkpoint_path << std::endl;
    std::cout << "  Output dir: " << output_dir << std::endl;
    std::cout << "  Num frames: " << num_frames << std::endl;

    // ---- Scene setup ----
    auto* scene = yp::create_scene(scene_path);
    std::cout << "Indoor initial states: " << yp::get_num_indoor_states(scene) << std::endl;

    // ---- Device setup ----
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cpu);
    rlt::init(device_cuda);

    // ---- Load model ----
    CPU_MODEL model_cpu;
    rlt::malloc(device_cpu, model_cpu);
    {
        auto file = HighFive::File(checkpoint_path, HighFive::File::ReadOnly);
        auto mg = rlt::get_group(device_cpu, file, "model");
        bool success = rlt::load(device_cpu, model_cpu, mg);
        if (!success) {
            std::cerr << "Failed to load checkpoint" << std::endl;
            return 1;
        }
    }
    std::cout << "Checkpoint loaded" << std::endl;

    GPU_MODEL model;
    typename GPU_MODEL::template Buffer<true> model_buffer;
    rlt::malloc(device_cuda, model);
    rlt::malloc(device_cuda, model_buffer);
    rlt::copy(device_cpu, device_cuda, model_cpu, model);

    // ---- GPU tensors ----
    rlt::Tensor<GPU_INPUT_SPEC> gpu_input_a, gpu_input_b;
    rlt::malloc(device_cuda, gpu_input_a);
    rlt::malloc(device_cuda, gpu_input_b);

    rlt::Tensor<HEAD_OUTPUT_SPEC> gpu_head_output;
    rlt::malloc(device_cuda, gpu_head_output);

    // ---- Sample 64 base camera positions ----
    // Use sample_camera_batch with max_angle=0 to get fixed indoor positions
    std::vector<rlt::CameraData> setup_cameras(NUM_CAMERAS);
    std::vector<float> setup_deltas(BATCH_SIZE);
    std::vector<float> setup_targets(BATCH_SIZE * 2);
    yp::sample_camera_batch(scene, setup_cameras.data(), setup_deltas.data(), setup_targets.data(), BATCH_SIZE, 0.0f);

    // Store first 64 cameras as base references
    std::vector<rlt::CameraData> base_cameras(NUM_GRID_CAMERAS);
    for (TI i = 0; i < NUM_GRID_CAMERAS; i++) {
        base_cameras[i] = setup_cameras[i];
    }

    // Rotate a camera's view direction by delta_yaw around the Y axis
    auto rotate_camera_yaw = [](const rlt::CameraData& cam, float delta_yaw) -> rlt::CameraData {
        rlt::CameraData rotated = cam;
        float cy = std::cos(delta_yaw);
        float sy = std::sin(delta_yaw);
        auto rotate_y = [cy, sy](owl::vec3f v) -> owl::vec3f {
            return owl::vec3f{
                cy * v.x + sy * v.z,
                v.y,
                -sy * v.x + cy * v.z
            };
        };
        rotated.dir_00 = rotate_y(cam.dir_00);
        rotated.dir_du = rotate_y(cam.dir_du);
        rotated.dir_dv = rotate_y(cam.dir_dv);
        return rotated;
    };

    // Create output directory
    std::filesystem::create_directories(output_dir);

    // ---- Frame generation loop ----
    auto eval_mode = rlt::Mode<rlt::mode::Default<>>{};
    DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 42);

    static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    std::vector<uint32_t> host_fb(NUM_CAMERAS * CAM_PIXELS);

    static constexpr TI IMAGE_WIDTH = GRID_DIM * CAM_WIDTH;
    static constexpr TI IMAGE_HEIGHT = GRID_DIM * CAM_HEIGHT;
    std::vector<uint8_t> output_image(IMAGE_WIDTH * IMAGE_HEIGHT * 3);

    std::vector<float> pred_buf(BATCH_SIZE * 2);
    std::vector<rlt::CameraData> cameras(NUM_CAMERAS);

    for (int frame = 0; frame < num_frames; frame++) {
        float t = (num_frames > 1) ? static_cast<float>(frame) / static_cast<float>(num_frames - 1) : 0.5f;
        float delta_yaw = -MAX_ANGLE + t * 2.0f * MAX_ANGLE;

        // Camera layout (matching training convention):
        //   [0..127]   = "image A" (reference views) -> gpu_input_a -> pipeline_a
        //   [128..255]  = "image B" (swept views)     -> gpu_input_b -> pipeline_b
        // First 64 of each batch are our grid cameras, rest are padding copies.
        for (TI i = 0; i < NUM_GRID_CAMERAS; i++) {
            cameras[i] = base_cameras[i];                                       // reference
            cameras[BATCH_SIZE + i] = rotate_camera_yaw(base_cameras[i], delta_yaw); // swept
        }
        // Pad remaining slots with copies
        for (TI i = NUM_GRID_CAMERAS; i < BATCH_SIZE; i++) {
            cameras[i] = cameras[i % NUM_GRID_CAMERAS];
            cameras[BATCH_SIZE + i] = cameras[BATCH_SIZE + (i % NUM_GRID_CAMERAS)];
        }

        // Render all 256 cameras
        yp::render_batch(scene, cameras.data());

        // RGBA -> float on GPU
        uint32_t* device_fb = yp::get_framebuffer_device_ptr(scene);
        rgba_to_float_kernel<<<BATCH_SIZE, 256, 0, device_cuda.stream>>>(
            device_fb, gpu_input_a._data, CAM_WIDTH, CAM_HEIGHT, 0
        );
        rgba_to_float_kernel<<<BATCH_SIZE, 256, 0, device_cuda.stream>>>(
            device_fb, gpu_input_b._data, CAM_WIDTH, CAM_HEIGHT, BATCH_SIZE
        );

        // Forward pass through both encoder pipelines
        rlt::evaluate(device_cuda, model.pipeline_a, gpu_input_a, model_buffer.intermediate_a, model_buffer.buffer_a, rng_cuda, eval_mode);
        rlt::evaluate(device_cuda, model.pipeline_b, gpu_input_b, model_buffer.intermediate_b, model_buffer.buffer_b, rng_cuda, eval_mode);

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
        rlt::evaluate(device_cuda, model.head, model_buffer.concatenated, gpu_head_output, model_buffer.head_buffer, rng_cuda, eval_mode);

        cudaStreamSynchronize(device_cuda.stream);

        // Read framebuffer for swept cameras [128..191] (first 64 of the B batch)
        cudaMemcpy(host_fb.data(), device_fb, NUM_CAMERAS * CAM_PIXELS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Read predictions
        cudaMemcpy(pred_buf.data(), gpu_head_output._data, BATCH_SIZE * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        // Composite 8x8 grid from swept cameras [128..191]
        std::memset(output_image.data(), 0, output_image.size());
        for (TI i = 0; i < NUM_GRID_CAMERAS; i++) {
            TI cam_idx = BATCH_SIZE + i; // cameras [128..191] = swept views
            TI grid_col = i % GRID_DIM;
            TI grid_row = i / GRID_DIM;
            TI tile_x = grid_col * CAM_WIDTH;
            TI tile_y = grid_row * CAM_HEIGHT;

            const uint32_t* cam_pixels_ptr = host_fb.data() + cam_idx * CAM_PIXELS;
            for (TI y = 0; y < CAM_HEIGHT; y++) {
                for (TI x = 0; x < CAM_WIDTH; x++) {
                    uint32_t rgba = cam_pixels_ptr[y * CAM_WIDTH + x];
                    TI px = tile_x + x;
                    TI py = tile_y + y;
                    TI idx = (py * IMAGE_WIDTH + px) * 3;
                    output_image[idx + 0] = (rgba >>  0) & 0xFF;
                    output_image[idx + 1] = (rgba >>  8) & 0xFF;
                    output_image[idx + 2] = (rgba >> 16) & 0xFF;
                }
            }

            // Draw progress beads: prediction[i] = delta between camera[i] and camera[128+i]
            float pred_sin = pred_buf[i * 2 + 0];
            float pred_cos = pred_buf[i * 2 + 1];
            float pred_angle = std::atan2(pred_sin, pred_cos);

            draw_bead(output_image, IMAGE_WIDTH, tile_x, tile_y, delta_yaw, MAX_ANGLE, 2, 0, 200, 0);     // GT green
            draw_bead(output_image, IMAGE_WIDTH, tile_x, tile_y, pred_angle, MAX_ANGLE, 5, 200, 0, 0);     // Pred red
        }

        // Save frame
        std::ostringstream filename;
        filename << output_dir << "/frame_" << std::setw(4) << std::setfill('0') << frame << ".png";
        stbi_write_png(filename.str().c_str(), IMAGE_WIDTH, IMAGE_HEIGHT, 3, output_image.data(), IMAGE_WIDTH * 3);

        if (frame % 10 == 0 || frame == num_frames - 1) {
            std::cout << "Frame " << frame << "/" << num_frames
                      << "  delta_yaw=" << delta_yaw * 180.0f / PI << " deg" << std::endl;
        }
    }

    std::cout << "Done. Frames saved to " << output_dir << "/" << std::endl;

    // ---- Cleanup ----
    rlt::free(device_cuda, gpu_head_output);
    rlt::free(device_cuda, gpu_input_a);
    rlt::free(device_cuda, gpu_input_b);
    rlt::free(device_cuda, model_buffer);
    rlt::free(device_cuda, model);
    rlt::free(device_cuda, rng_cuda);
    rlt::free(device_cpu, model_cpu);
    yp::destroy_scene(scene);

    return 0;
}
