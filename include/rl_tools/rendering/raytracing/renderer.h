#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_RENDERER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_RENDERER_H

#include "types.h"
#include "../../containers/tensor/tensor.h"

#include <vector>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{

        template <typename T_T, typename T_TI, T_TI T_CAM_WIDTH, T_TI T_CAM_HEIGHT, T_TI T_NUM_CAMERAS, T_TI T_NUM_PROBES, bool T_HIGH_FIDELITY_SHADING = false>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            static constexpr TI CAM_WIDTH = T_CAM_WIDTH;
            static constexpr TI CAM_HEIGHT = T_CAM_HEIGHT;
            static constexpr TI NUM_CAMERAS = T_NUM_CAMERAS;
            static constexpr TI NUM_PROBES = T_NUM_PROBES;
            static constexpr bool HIGH_FIDELITY_SHADING = T_HIGH_FIDELITY_SHADING;
            static constexpr TI GRID_COLS = [](){
                TI cols = 1;
                while(cols * cols < NUM_CAMERAS) cols++;
                return cols;
            }();
            static constexpr TI GRID_ROWS = (NUM_CAMERAS + GRID_COLS - 1) / GRID_COLS;
            static constexpr TI FB_WIDTH = GRID_COLS * CAM_WIDTH;
            static constexpr TI FB_HEIGHT = GRID_ROWS * CAM_HEIGHT;
            static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
            static constexpr T BENCHMARK_SECONDS = 10.0;
            static constexpr T COS_FOVY = 0.66;
        };

        template <typename T_SPEC>
        struct MeshData{
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            std::vector<float> vertices; // vec3f flat
            std::vector<int> indices;    // vec3i flat
            std::vector<float> tex_coords; // vec2f flat
            float color[3];
            std::vector<uint8_t> tex_pixels; // RGBA8
            int tex_width = 0, tex_height = 0;
            bool has_texture = false;
            float metallic = 0.0f;
            float roughness = 1.0f;
            std::vector<float> normals;
            std::vector<uint8_t> normal_tex_pixels;
            int normal_tex_width = 0, normal_tex_height = 0;
            bool has_normal_map = false;
            std::vector<uint8_t> metallic_roughness_tex_pixels;
            int mr_tex_width = 0, mr_tex_height = 0;
            bool has_metallic_roughness_map = false;
            float emissive[3] = {0, 0, 0};
            std::vector<uint8_t> emissive_tex_pixels;
            int emissive_tex_width = 0, emissive_tex_height = 0;
            bool has_emissive_map = false;
            std::vector<uint8_t> occlusion_tex_pixels;
            int occlusion_tex_width = 0, occlusion_tex_height = 0;
            bool has_occlusion_map = false;
            float opacity = 1.0f;
        };

        template <typename T_SPEC>
        struct BackendContext{
            using SPEC = T_SPEC;
            void* context = nullptr;
            void* module = nullptr;
            void* ray_gen = nullptr;
            void* owl_frame_buffer = nullptr;
            void* owl_cameras_buffer = nullptr;
            void* world = nullptr;
            void* rgb_launch_params = nullptr;
            void* collision_ray_gen = nullptr;
            void* owl_collision_results_buffer = nullptr;
            void* probe_dirs_buffer = nullptr;
            void* coll_launch_params = nullptr;
        };

        template <typename T_SPEC>
        struct Renderer{
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;

            using CAMERA_TENSOR_SPEC = tensor::Specification<CameraData<T>, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS>, true>;
            Tensor<CAMERA_TENSOR_SPEC> cameras;

            using FB_TENSOR_SPEC = tensor::Specification<uint32_t, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH>, true>;
            Tensor<FB_TENSOR_SPEC> frame_buffer;

            using COLLISION_TENSOR_SPEC = tensor::Specification<CollisionResult, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::NUM_PROBES>, true>;
            Tensor<COLLISION_TENSOR_SPEC> collision_results;

            T scene_center[3] = {0, 0, 0};
            T camera_radius = 0;

            std::vector<MeshData<SPEC>> meshes;
            std::vector<rendering::raytracing::SceneLight> scene_lights;

            BackendContext<SPEC> backend;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif