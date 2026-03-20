#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_RENDERER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_RENDERER_H

#include <vector>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{
        struct CollisionResult {
            float distance;
            int hit;
        };

        template <typename T_T, typename T_TI, T_TI T_CAM_WIDTH, T_TI T_CAM_HEIGHT, T_TI T_NUM_CAMERAS, T_TI T_NUM_PROBES>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            static constexpr TI CAM_WIDTH = T_CAM_WIDTH;
            static constexpr TI CAM_HEIGHT = T_CAM_HEIGHT;
            static constexpr TI NUM_CAMERAS = T_NUM_CAMERAS;
            static constexpr TI NUM_PROBES = T_NUM_PROBES;
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
        };

        template <typename T_SPEC>
        struct Renderer{
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;

            // RGB rendering context (OWL handles stored as void* to avoid header leakage)
            void* context = nullptr;           // OWLContext
            void* module = nullptr;            // OWLModule
            void* ray_gen = nullptr;           // OWLRayGen
            void* frame_buffer = nullptr;      // OWLBuffer
            void* cameras_buffer = nullptr;    // OWLBuffer
            void* world = nullptr;             // OWLGroup (instance group)
            void* rgb_launch_params = nullptr; // OWLParams

            // Collision (shares context/module/world with RGB)
            void* collision_ray_gen = nullptr;
            void* collision_results_buffer = nullptr; // OWLBuffer (host-pinned)
            void* probe_dirs_buffer = nullptr;
            void* coll_launch_params = nullptr;

            // Scene parameters
            T scene_center[3] = {0, 0, 0};
            T camera_radius = 0;

            // Mesh data (kept for potential re-upload)
            std::vector<MeshData<SPEC>> meshes;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif