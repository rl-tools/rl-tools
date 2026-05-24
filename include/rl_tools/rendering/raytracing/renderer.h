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

        enum class OutputMode {
            RGB,
            RGBD,
            DEPTH
        };

        template <
            bool T_LOAD_TEXTURES,
            bool T_NORMAL_SHADING,
            bool T_METALLIC_REFLECTIONS,
            bool T_SRGB_OUTPUT,
            bool T_CHECKER_BACKGROUND,
            bool T_PBR_SHADING
        >
        struct ShadingOptions{
            static constexpr bool LOAD_TEXTURES = T_LOAD_TEXTURES;
            static constexpr bool NORMAL_SHADING = T_NORMAL_SHADING;
            static constexpr bool METALLIC_REFLECTIONS = T_METALLIC_REFLECTIONS;
            static constexpr bool SRGB_OUTPUT = T_SRGB_OUTPUT;
            static constexpr bool CHECKER_BACKGROUND = T_CHECKER_BACKGROUND;
            static constexpr bool PBR_SHADING = T_PBR_SHADING;
        };

        using BasicShading = ShadingOptions<true, true, true, true, true, false>;
        using HighFidelityShading = ShadingOptions<true, true, true, true, true, true>;
        using FastFlatShading = ShadingOptions<false, false, false, false, false, false>;

        template <bool T_HIGH_FIDELITY_SHADING>
        struct ShadingProfileFromHighFidelity{
            using type = BasicShading;
        };

        template <>
        struct ShadingProfileFromHighFidelity<true>{
            using type = HighFidelityShading;
        };

        template <typename T_T, typename T_TI, T_TI T_CAM_WIDTH, T_TI T_CAM_HEIGHT, T_TI T_NUM_CAMERAS, T_TI T_NUM_PROBES, bool T_HIGH_FIDELITY_SHADING = false, bool T_ENABLE_MOTION_BLUR = false, T_TI T_MOTION_BLUR_SAMPLES = 1, bool T_ENABLE_ANTI_ALIASING = false, T_TI T_ANTI_ALIASING_GRID_SIZE = 1, OutputMode T_OUTPUT_MODE = OutputMode::RGB, typename T_SHADING = typename ShadingProfileFromHighFidelity<T_HIGH_FIDELITY_SHADING>::type>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using SHADING = T_SHADING;
            static constexpr OutputMode OUTPUT_MODE = T_OUTPUT_MODE;
            static constexpr bool HAS_RGB = OUTPUT_MODE == OutputMode::RGB || OUTPUT_MODE == OutputMode::RGBD;
            static constexpr bool HAS_DEPTH = OUTPUT_MODE == OutputMode::RGBD || OUTPUT_MODE == OutputMode::DEPTH;
            static constexpr bool ENABLE_DEPTH = HAS_DEPTH;
            static constexpr bool ENABLE_RGB = HAS_RGB;
            static constexpr TI CAM_WIDTH = T_CAM_WIDTH;
            static constexpr TI CAM_HEIGHT = T_CAM_HEIGHT;
            static constexpr TI NUM_CAMERAS = T_NUM_CAMERAS;
            static constexpr TI NUM_PROBES = T_NUM_PROBES;
            static constexpr bool HIGH_FIDELITY_SHADING = SHADING::PBR_SHADING;
            static constexpr TI MOTION_BLUR_SAMPLES = T_MOTION_BLUR_SAMPLES;
            static constexpr bool ENABLE_MOTION_BLUR = T_ENABLE_MOTION_BLUR && MOTION_BLUR_SAMPLES > 1;
            static_assert(MOTION_BLUR_SAMPLES >= 1, "MOTION_BLUR_SAMPLES must be at least 1");
            static_assert(!ENABLE_MOTION_BLUR || MOTION_BLUR_SAMPLES == 2 || MOTION_BLUR_SAMPLES == 4 || MOTION_BLUR_SAMPLES == 8 || MOTION_BLUR_SAMPLES == 16 || MOTION_BLUR_SAMPLES == 32, "MOTION_BLUR_SAMPLES must be one of 2, 4, 8, 16, or 32");
            static constexpr TI ANTI_ALIASING_GRID_SIZE = T_ANTI_ALIASING_GRID_SIZE;
            static constexpr bool ENABLE_ANTI_ALIASING = T_ENABLE_ANTI_ALIASING && ANTI_ALIASING_GRID_SIZE > 1;
            static constexpr TI ANTI_ALIASING_SAMPLES = ENABLE_ANTI_ALIASING ? ANTI_ALIASING_GRID_SIZE * ANTI_ALIASING_GRID_SIZE : 1;
            static constexpr TI RGB_SAMPLES = (ENABLE_MOTION_BLUR ? MOTION_BLUR_SAMPLES : 1) * ANTI_ALIASING_SAMPLES;
            static constexpr TI DEPTH_SAMPLES = RGB_SAMPLES;
            static_assert(ANTI_ALIASING_GRID_SIZE >= 1, "ANTI_ALIASING_GRID_SIZE must be at least 1");
            static_assert(!ENABLE_ANTI_ALIASING || ANTI_ALIASING_GRID_SIZE == 2 || ANTI_ALIASING_GRID_SIZE == 3 || ANTI_ALIASING_GRID_SIZE == 4, "ANTI_ALIASING_GRID_SIZE must be one of 2, 3, or 4");
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
            static constexpr T COS_FOVY = 1.3962634015954636;
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
            int alpha_mode = 0;
            float alpha_cutoff = 0.5f;
        };

        template <typename T_SPEC, bool T_ENABLE_MOTION_BLUR>
        struct MotionBlurBackendContext {};

        template <typename T_SPEC>
        struct MotionBlurBackendContext<T_SPEC, true> {
            void* owl_cameras_open_buffer = nullptr;
        };

        template <typename T_SPEC>
        struct BackendContext: MotionBlurBackendContext<T_SPEC, T_SPEC::ENABLE_MOTION_BLUR>{
            using SPEC = T_SPEC;
            void* context = nullptr;
            void* module = nullptr;
            void* owl_cameras_buffer = nullptr;
            void* world = nullptr;
            void* launch_params = nullptr;
            void* collision_ray_gen = nullptr;
            void* owl_collision_results_buffer = nullptr;
            void* probe_dirs_buffer = nullptr;
            void* coll_launch_params = nullptr;
        };

        template <typename T_SPEC, bool T_HAS_RGB>
        struct RGBBackendContext {};

        template <typename T_SPEC>
        struct RGBBackendContext<T_SPEC, true> {
            void* ray_gen = nullptr;
            void* owl_frame_buffer = nullptr;
        };

        template <typename T_SPEC, bool T_HAS_DEPTH>
        struct DepthBackendContext {};

        template <typename T_SPEC>
        struct DepthBackendContext<T_SPEC, true> {
            void* depth_ray_gen = nullptr;
            void* owl_depth_buffer = nullptr;
        };

        template <typename T_SPEC>
        struct RendererBackend: BackendContext<T_SPEC>, RGBBackendContext<T_SPEC, T_SPEC::HAS_RGB>, DepthBackendContext<T_SPEC, T_SPEC::HAS_DEPTH> {};

        template <typename T_SPEC, bool T_ENABLE_MOTION_BLUR>
        struct MotionBlurRendererStorage {};

        template <typename T_SPEC>
        struct MotionBlurRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using CAMERA_TENSOR_SPEC = tensor::Specification<CameraData<T>, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS>, true>;
            Tensor<CAMERA_TENSOR_SPEC> cameras_open;
        };

        template <typename T_SPEC, bool T_HAS_RGB>
        struct RGBRendererStorage {};

        template <typename T_SPEC>
        struct RGBRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            using FB_TENSOR_SPEC = tensor::Specification<uint32_t, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH>, true>;
            Tensor<FB_TENSOR_SPEC> frame_buffer;
        };

        template <typename T_SPEC, bool T_HAS_DEPTH>
        struct DepthRendererStorage {};

        template <typename T_SPEC>
        struct DepthRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using DEPTH_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH>, true>;
            Tensor<DEPTH_TENSOR_SPEC> depth_buffer;
        };

        template <typename T_SPEC>
        struct Renderer: MotionBlurRendererStorage<T_SPEC, T_SPEC::ENABLE_MOTION_BLUR>, RGBRendererStorage<T_SPEC, T_SPEC::HAS_RGB>, DepthRendererStorage<T_SPEC, T_SPEC::HAS_DEPTH>{
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;

            using CAMERA_TENSOR_SPEC = tensor::Specification<CameraData<T>, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS>, true>;
            Tensor<CAMERA_TENSOR_SPEC> cameras;

            using COLLISION_TENSOR_SPEC = tensor::Specification<CollisionResult, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::NUM_PROBES>, true>;
            Tensor<COLLISION_TENSOR_SPEC> collision_results;

            T scene_center[3] = {0, 0, 0};
            T scene_half_extent[3] = {0, 0, 0};
            T camera_radius = 0;

            std::vector<MeshData<SPEC>> meshes;
            std::vector<rendering::raytracing::SceneLight> scene_lights;

            RendererBackend<SPEC> backend;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
