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

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <cuda_runtime.h>
#include <cstdio>

namespace rlt = rl_tools;
namespace fs = std::filesystem;

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
    int cam_width, int cam_height, int camera_offset, int dst_offset
) {
    const int sample_idx = blockIdx.x;
    const int camera_idx = camera_offset + sample_idx;
    const int cam_pixels = cam_width * cam_height;

    const uint32_t* src = framebuffer + camera_idx * cam_pixels;
    float* dst = output + (dst_offset + sample_idx) * cam_height * cam_width * 3;

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
    std::string scene_dir;
    std::string single_scene;
    const char* checkpoint_path = nullptr;
    const char* output_path = "yaw_prediction_viewer.mp4";
    int num_frames = 120;
    TI num_scenes = 64;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--scene-dir" && i + 1 < argc) {
            scene_dir = argv[++i];
        } else if (arg == "--scene" && i + 1 < argc) {
            single_scene = argv[++i];
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--num-frames" && i + 1 < argc) {
            num_frames = std::atoi(argv[++i]);
        } else if (arg == "--num-scenes" && i + 1 < argc) {
            num_scenes = std::atoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " [--scene-dir <dir> | --scene <path.glb>]"
                      << " --checkpoint <path.h5>"
                      << " [--output <path.mp4>] [--num-frames <N>]"
                      << " [--num-scenes <N>]" << std::endl;
            return 1;
        }
    }

    if (!checkpoint_path) {
        std::cerr << "Error: --checkpoint is required" << std::endl;
        return 1;
    }

    // Collect scene paths
    std::vector<std::string> all_scene_paths;
    if (!scene_dir.empty()) {
        for (auto& entry : fs::directory_iterator(scene_dir)) {
            if (entry.path().extension() == ".glb") {
                all_scene_paths.push_back(entry.path().string());
            }
        }
        std::sort(all_scene_paths.begin(), all_scene_paths.end());
        if (all_scene_paths.empty()) {
            std::cerr << "No .glb files found in " << scene_dir << std::endl;
            return 1;
        }
        if (num_scenes < all_scene_paths.size()) {
            all_scene_paths.resize(num_scenes);
        }
    } else if (!single_scene.empty()) {
        all_scene_paths.push_back(single_scene);
    } else {
        std::cerr << "Must specify --scene-dir or --scene" << std::endl;
        return 1;
    }

    // Cap to 64 scenes (one per grid tile max)
    if (all_scene_paths.size() > NUM_GRID_CAMERAS) {
        all_scene_paths.resize(NUM_GRID_CAMERAS);
    }

    std::cout << "Yaw prediction viewer (multi-scene)" << std::endl;
    std::cout << "  Scenes: " << all_scene_paths.size() << std::endl;
    std::cout << "  Checkpoint: " << checkpoint_path << std::endl;
    std::cout << "  Output: " << output_path << std::endl;
    std::cout << "  Num frames: " << num_frames << std::endl;

    // ---- Load scenes ----
    struct LoadedScene {
        yp::SceneHandle* handle = nullptr;
        std::string path;
    };
    std::vector<LoadedScene> loaded_scenes(all_scene_paths.size());
    for (TI s = 0; s < all_scene_paths.size(); s++) {
        loaded_scenes[s].path = all_scene_paths[s];
        std::cout << "Loading scene " << s << "/" << all_scene_paths.size() << ": "
                  << fs::path(loaded_scenes[s].path).filename().string() << std::endl;
        loaded_scenes[s].handle = yp::create_scene(loaded_scenes[s].path.c_str());
        std::cout << "  Indoor states: " << yp::get_num_indoor_states(loaded_scenes[s].handle) << std::endl;
    }

    // Assign each grid tile to a scene (round-robin)
    std::vector<TI> tile_scene_idx(NUM_GRID_CAMERAS);
    for (TI i = 0; i < NUM_GRID_CAMERAS; i++) {
        tile_scene_idx[i] = i % loaded_scenes.size();
    }

    // Group tiles by scene
    std::vector<std::vector<TI>> scene_tiles(loaded_scenes.size());
    for (TI i = 0; i < NUM_GRID_CAMERAS; i++) {
        scene_tiles[tile_scene_idx[i]].push_back(i);
    }

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

    DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 42);

    // ---- Sample base cameras per scene ----
    // For each scene, sample base cameras for its assigned tiles
    std::vector<rlt::CameraData> base_cameras(NUM_GRID_CAMERAS);
    {
        std::vector<rlt::CameraData> setup_cameras(NUM_CAMERAS);
        std::vector<float> setup_deltas(BATCH_SIZE);
        std::vector<float> setup_targets(BATCH_SIZE * 2);

        for (TI s = 0; s < loaded_scenes.size(); s++) {
            auto& tiles = scene_tiles[s];
            if (tiles.empty()) continue;

            // Sample enough cameras from this scene
            yp::sample_camera_batch(loaded_scenes[s].handle, setup_cameras.data(),
                                    setup_deltas.data(), setup_targets.data(),
                                    BATCH_SIZE, 0.0f);

            for (TI t = 0; t < tiles.size(); t++) {
                base_cameras[tiles[t]] = setup_cameras[t % BATCH_SIZE];
            }
        }
    }

    // Rotate a camera's view direction by delta_yaw around the Y axis
    auto rotate_camera_yaw = [](const rlt::CameraData& cam, float delta_yaw) -> rlt::CameraData {
        rlt::CameraData rotated = cam;
        float cy = std::cos(delta_yaw);
        float sy = std::sin(delta_yaw);
        auto rotate_y = [cy, sy](owl::vec3f v) -> owl::vec3f {
            return owl::vec3f{
                cy * v.x - sy * v.z,
                v.y,
                sy * v.x + cy * v.z
            };
        };
        rotated.dir_00 = rotate_y(cam.dir_00);
        rotated.dir_du = rotate_y(cam.dir_du);
        rotated.dir_dv = rotate_y(cam.dir_dv);
        return rotated;
    };

    // ---- Open ffmpeg pipe ----
    static constexpr TI IMAGE_WIDTH = GRID_DIM * CAM_WIDTH;
    static constexpr TI IMAGE_HEIGHT = GRID_DIM * CAM_HEIGHT;
    std::ostringstream ffmpeg_cmd;
    ffmpeg_cmd
        << "ffmpeg -y -f rawvideo -pixel_format rgb24 "
        << "-video_size " << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << " "
        << "-framerate 30 -i - "
        << "-an -c:v libx264 -pix_fmt yuv420p "
        << output_path;
    FILE* mp4_pipe = popen(ffmpeg_cmd.str().c_str(), "w");
    if (!mp4_pipe) {
        std::cerr << "Failed to start ffmpeg" << std::endl;
        return 1;
    }

    // ---- Frame generation loop ----
    static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
    std::vector<uint32_t> host_fb(NUM_CAMERAS * CAM_PIXELS);
    std::vector<uint8_t> output_image(IMAGE_WIDTH * IMAGE_HEIGHT * 3);
    std::vector<float> pred_buf(BATCH_SIZE * 2);
    std::vector<rlt::CameraData> cameras(NUM_CAMERAS);

    // Per-tile host framebuffer for swept views (read back per scene)
    std::vector<std::vector<uint32_t>> tile_pixels(NUM_GRID_CAMERAS, std::vector<uint32_t>(CAM_PIXELS));

    auto eval_mode = rlt::Mode<rlt::mode::Default<>>{};

    for (int frame = 0; frame < num_frames; frame++) {
        float t = (num_frames > 1) ? static_cast<float>(frame) / static_cast<float>(num_frames - 1) : 0.5f;
        float delta_yaw = -MAX_ANGLE + t * 2.0f * MAX_ANGLE;

        // ---- Render each scene and fill GPU input tensors ----
        // We accumulate into gpu_input_a/b at the right batch offsets.
        // Batch layout: tiles [0..63] are our grid, [64..127] are padding.
        for (TI s = 0; s < loaded_scenes.size(); s++) {
            auto& tiles = scene_tiles[s];
            if (tiles.empty()) continue;

            TI num_pairs = tiles.size();

            // Set up cameras for this scene's tiles:
            // [0..num_pairs-1] = reference (A), [num_pairs..2*num_pairs-1] = swept (B)
            for (TI t = 0; t < num_pairs; t++) {
                cameras[t] = base_cameras[tiles[t]];
                cameras[num_pairs + t] = rotate_camera_yaw(base_cameras[tiles[t]], delta_yaw);
            }
            // Fill remaining camera slots with copies (renderer expects SCENE_NUM_CAMERAS)
            for (TI i = num_pairs * 2; i < NUM_CAMERAS; i++) {
                cameras[i] = cameras[i % (num_pairs * 2)];
            }

            yp::render_batch(loaded_scenes[s].handle, cameras.data());
            uint32_t* device_fb = yp::get_framebuffer_device_ptr(loaded_scenes[s].handle);

            // Copy reference cameras to gpu_input_a at tile positions
            for (TI t = 0; t < num_pairs; t++) {
                TI batch_pos = tiles[t]; // tile index = batch position for first 64
                rgba_to_float_kernel<<<1, 256, 0, device_cuda.stream>>>(
                    device_fb, gpu_input_a._data, CAM_WIDTH, CAM_HEIGHT, t, batch_pos
                );
            }
            // Copy swept cameras to gpu_input_b at tile positions
            for (TI t = 0; t < num_pairs; t++) {
                TI batch_pos = tiles[t];
                rgba_to_float_kernel<<<1, 256, 0, device_cuda.stream>>>(
                    device_fb, gpu_input_b._data, CAM_WIDTH, CAM_HEIGHT, num_pairs + t, batch_pos
                );
            }

            // Read back swept camera pixels for compositing
            cudaStreamSynchronize(device_cuda.stream);
            for (TI t = 0; t < num_pairs; t++) {
                TI cam_idx = num_pairs + t;
                cudaMemcpy(tile_pixels[tiles[t]].data(),
                           device_fb + cam_idx * CAM_PIXELS,
                           CAM_PIXELS * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost);
            }
        }

        // Pad batch positions [64..127] with copies of [0..63] for gpu_input_a/b
        for (TI i = NUM_GRID_CAMERAS; i < BATCH_SIZE; i++) {
            TI src_pos = i % NUM_GRID_CAMERAS;
            // Copy on GPU: just duplicate the already-written data
            cudaMemcpyAsync(
                gpu_input_a._data + i * CAM_HEIGHT * CAM_WIDTH * 3,
                gpu_input_a._data + src_pos * CAM_HEIGHT * CAM_WIDTH * 3,
                CAM_HEIGHT * CAM_WIDTH * 3 * sizeof(float),
                cudaMemcpyDeviceToDevice, device_cuda.stream
            );
            cudaMemcpyAsync(
                gpu_input_b._data + i * CAM_HEIGHT * CAM_WIDTH * 3,
                gpu_input_b._data + src_pos * CAM_HEIGHT * CAM_WIDTH * 3,
                CAM_HEIGHT * CAM_WIDTH * 3 * sizeof(float),
                cudaMemcpyDeviceToDevice, device_cuda.stream
            );
        }

        // ---- Model forward pass ----
        rlt::evaluate(device_cuda, model.pipeline_a, gpu_input_a, model_buffer.intermediate_a, model_buffer.buffer_a, rng_cuda, eval_mode);
        rlt::evaluate(device_cuda, model.pipeline_b, gpu_input_b, model_buffer.intermediate_b, model_buffer.buffer_b, rng_cuda, eval_mode);

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

        rlt::evaluate(device_cuda, model.head, model_buffer.concatenated, gpu_head_output, model_buffer.head_buffer, rng_cuda, eval_mode);
        cudaStreamSynchronize(device_cuda.stream);

        // Read predictions
        cudaMemcpy(pred_buf.data(), gpu_head_output._data, BATCH_SIZE * 2 * sizeof(float), cudaMemcpyDeviceToHost);

        // ---- Composite 8x8 grid ----
        std::memset(output_image.data(), 0, output_image.size());
        for (TI i = 0; i < NUM_GRID_CAMERAS; i++) {
            TI grid_col = i % GRID_DIM;
            TI grid_row = i / GRID_DIM;
            TI tile_x = grid_col * CAM_WIDTH;
            TI tile_y = grid_row * CAM_HEIGHT;

            const uint32_t* cam_pixels_ptr = tile_pixels[i].data();
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

            float pred_sin = pred_buf[i * 2 + 0];
            float pred_cos = pred_buf[i * 2 + 1];
            float pred_angle = std::atan2(pred_sin, pred_cos);

            draw_bead(output_image, IMAGE_WIDTH, tile_x, tile_y, delta_yaw, MAX_ANGLE, 2, 0, 200, 0);
            draw_bead(output_image, IMAGE_WIDTH, tile_x, tile_y, pred_angle, MAX_ANGLE, 5, 200, 0, 0);
        }

        // Write frame to ffmpeg pipe
        size_t frame_bytes = output_image.size();
        size_t written = fwrite(output_image.data(), 1, frame_bytes, mp4_pipe);
        if (written != frame_bytes) {
            std::cerr << "Failed writing frame " << frame << " to ffmpeg" << std::endl;
            pclose(mp4_pipe);
            mp4_pipe = nullptr;
            break;
        }

        if (frame % 10 == 0 || frame == num_frames - 1) {
            std::cout << "Frame " << frame << "/" << num_frames
                      << "  delta_yaw=" << delta_yaw * 180.0f / PI << " deg" << std::endl;
        }
    }

    if (mp4_pipe) {
        int ffmpeg_status = pclose(mp4_pipe);
        if (ffmpeg_status != 0) {
            std::cerr << "ffmpeg exited with status " << ffmpeg_status << std::endl;
        } else {
            std::cout << "Video written: " << output_path << std::endl;
        }
    }

    // ---- Cleanup ----
    rlt::free(device_cuda, gpu_head_output);
    rlt::free(device_cuda, gpu_input_a);
    rlt::free(device_cuda, gpu_input_b);
    rlt::free(device_cuda, model_buffer);
    rlt::free(device_cuda, model);
    rlt::free(device_cuda, rng_cuda);
    rlt::free(device_cpu, model_cpu);
    for (auto& s : loaded_scenes) {
        yp::destroy_scene(s.handle);
    }

    return 0;
}
