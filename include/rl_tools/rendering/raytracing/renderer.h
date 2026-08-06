#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_RENDERER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_RENDERER_H

#include "types.h"
#include "scene.h"
#include "../../containers/tensor/tensor.h"

#include <vector>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{

        enum class OutputMode {
            RGB,
            RGBD,
            DEPTH,
            // segmentation writes one uint32 instance index per pixel (miss = 0xFFFFFFFF), always
            // single-sample from the shutter-close camera: labels cannot be averaged, so
            // anti-aliasing and motion-blur settings do not apply to it
            SEGMENTATION,
            RGBD_SEGMENTATION
        };

        template <
            bool T_LOAD_TEXTURES,
            bool T_NORMAL_SHADING,
            bool T_METALLIC_REFLECTIONS,
            bool T_SRGB_OUTPUT,
            bool T_CHECKER_BACKGROUND,
            bool T_PBR_SHADING,
            bool T_PUNCTUAL_LIGHT_SHADOWS = false
        >
        struct ShadingOptions{
            static constexpr bool LOAD_TEXTURES = T_LOAD_TEXTURES;
            static constexpr bool NORMAL_SHADING = T_NORMAL_SHADING;
            static constexpr bool METALLIC_REFLECTIONS = T_METALLIC_REFLECTIONS;
            static constexpr bool SRGB_OUTPUT = T_SRGB_OUTPUT;
            static constexpr bool CHECKER_BACKGROUND = T_CHECKER_BACKGROUND;
            static constexpr bool PBR_SHADING = T_PBR_SHADING;
            static constexpr bool PUNCTUAL_LIGHT_SHADOWS = T_PUNCTUAL_LIGHT_SHADOWS;
        };

        using Medium = ShadingOptions<true, true, true, true, true, false>;
        using High = ShadingOptions<true, true, true, true, false, true>;
        using VeryHigh = ShadingOptions<true, true, true, true, false, true, true>;
        using Low = ShadingOptions<false, false, false, false, false, false>;

        using BasicShading = Medium;
        using HighFidelityShading = High;
        using VeryHighFidelityShading = VeryHigh;
        using FastFlatShading = Low;

        template <typename T_T, typename T_TI, T_TI T_CAM_WIDTH, T_TI T_CAM_HEIGHT, T_TI T_NUM_CAMERAS, T_TI T_NUM_PROBES, typename T_SHADING = Medium, bool T_ENABLE_MOTION_BLUR = false, T_TI T_MOTION_BLUR_SAMPLES = 1, bool T_ENABLE_ANTI_ALIASING = false, T_TI T_ANTI_ALIASING_GRID_SIZE = 1, OutputMode T_OUTPUT_MODE = OutputMode::RGB, T_TI T_NUM_OVERLAYS = 0, T_TI T_MAX_OVERLAY_INSTANCES = 0, T_TI T_MAX_OVERLAYS_PER_CAMERA = 0>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            using SHADING = T_SHADING;
            static constexpr OutputMode OUTPUT_MODE = T_OUTPUT_MODE;
            static constexpr bool HAS_RGB = OUTPUT_MODE == OutputMode::RGB || OUTPUT_MODE == OutputMode::RGBD || OUTPUT_MODE == OutputMode::RGBD_SEGMENTATION;
            static constexpr bool HAS_DEPTH = OUTPUT_MODE == OutputMode::RGBD || OUTPUT_MODE == OutputMode::DEPTH || OUTPUT_MODE == OutputMode::RGBD_SEGMENTATION;
            static constexpr bool HAS_SEGMENTATION = OUTPUT_MODE == OutputMode::SEGMENTATION || OUTPUT_MODE == OutputMode::RGBD_SEGMENTATION;
            static constexpr bool ENABLE_DEPTH = HAS_DEPTH;
            static constexpr bool ENABLE_RGB = HAS_RGB;
            static constexpr TI CAM_WIDTH = T_CAM_WIDTH;
            static constexpr TI CAM_HEIGHT = T_CAM_HEIGHT;
            static constexpr TI NUM_CAMERAS = T_NUM_CAMERAS;
            static constexpr TI NUM_PROBES = T_NUM_PROBES;
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
            // dynamic overlays: per-camera dynamic content composed onto the static shared world
            static constexpr TI NUM_OVERLAYS = T_NUM_OVERLAYS;
            static constexpr TI MAX_OVERLAY_INSTANCES = T_MAX_OVERLAY_INSTANCES;
            static constexpr TI MAX_OVERLAYS_PER_CAMERA = T_MAX_OVERLAYS_PER_CAMERA;
            static constexpr bool ENABLE_OVERLAYS = NUM_OVERLAYS > 0 && MAX_OVERLAY_INSTANCES > 0 && MAX_OVERLAYS_PER_CAMERA > 0;
            static_assert(ENABLE_OVERLAYS || (NUM_OVERLAYS == 0 && MAX_OVERLAY_INSTANCES == 0 && MAX_OVERLAYS_PER_CAMERA == 0), "overlay constants must be all zero (disabled) or all nonzero");
        };

        template <typename T_SPEC, bool T_ENABLE_MOTION_BLUR>
        struct MotionBlurBackendContext {};

        template <typename T_SPEC>
        struct MotionBlurBackendContext<T_SPEC, true> {
            void* cameras_open_buffer = nullptr;
        };

        template <typename T_SPEC>
        struct BackendContext: MotionBlurBackendContext<T_SPEC, T_SPEC::ENABLE_MOTION_BLUR>{
            using SPEC = T_SPEC;
            void* context = nullptr;
            void* module = nullptr;
            void* cameras_buffer = nullptr;
            void* world = nullptr;
            void* launch_params = nullptr;
            void* collision_ray_gen = nullptr;
            void* collision_results_buffer = nullptr;
            void* probe_dirs_buffer = nullptr;
            void* coll_launch_params = nullptr;
            void* overlay_state = nullptr;
        };

        template <typename T_SPEC, bool T_HAS_RGB>
        struct RGBBackendContext {};

        template <typename T_SPEC>
        struct RGBBackendContext<T_SPEC, true> {
            void* ray_gen = nullptr;
            void* frame_buffer_handle = nullptr;
        };

        template <typename T_SPEC, bool T_HAS_DEPTH>
        struct DepthBackendContext {};

        template <typename T_SPEC>
        struct DepthBackendContext<T_SPEC, true> {
            void* depth_ray_gen = nullptr;
            void* depth_buffer_handle = nullptr;
        };

        template <typename T_SPEC, bool T_HAS_SEGMENTATION>
        struct SegmentationBackendContext {};

        template <typename T_SPEC>
        struct SegmentationBackendContext<T_SPEC, true> {
            void* segmentation_ray_gen = nullptr;
            void* segmentation_buffer_handle = nullptr;
        };

        template <typename T_SPEC>
        struct RendererBackend: BackendContext<T_SPEC>, RGBBackendContext<T_SPEC, T_SPEC::HAS_RGB>, DepthBackendContext<T_SPEC, T_SPEC::HAS_DEPTH>, SegmentationBackendContext<T_SPEC, T_SPEC::HAS_SEGMENTATION> {};

        template <typename T_SPEC, bool T_ENABLE_MOTION_BLUR>
        struct MotionBlurRendererStorage {};

        template <typename T_SPEC>
        struct MotionBlurRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            using CAMERA_TENSOR_SPEC = tensor::Specification<Camera<T>, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS>, true>;
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

        template <typename T_SPEC, bool T_HAS_SEGMENTATION>
        struct SegmentationRendererStorage {};

        template <typename T_SPEC>
        struct SegmentationRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            using SEGMENTATION_TENSOR_SPEC = tensor::Specification<uint32_t, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH>, true>;
            Tensor<SEGMENTATION_TENSOR_SPEC> segmentation_buffer;
        };

        template <typename T_SPEC, bool T_ENABLE_OVERLAYS>
        struct OverlayRendererStorage {};

        template <typename T_SPEC>
        struct OverlayRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            static constexpr TI INVALID_OVERLAY = ~(TI)0;

            struct OverlaySlot {
                TI object = 0;          // global object index (scene objects + pool objects)
                float transform[12];    // object -> world
                bool active = false;
            };
            struct OverlayState {
                OverlaySlot slots[SPEC::MAX_OVERLAY_INSTANCES];
                bool dirty = false;
            };
            // pool assemblies flattened at init so spawn needs no pool access in the hot path
            struct AssetRecord {
                TI first_part;          // into asset_part_objects/asset_part_transforms
                TI num_parts;
            };
            OverlayState overlays[SPEC::NUM_OVERLAYS];
            TI attachments[SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA];
            bool attachments_dirty = false;
            std::vector<AssetRecord> assets;
            std::vector<TI> asset_part_objects;      // global object index per part
            std::vector<float> asset_part_transforms; // 12 per part, assembly-local

            OverlayRendererStorage(){
                for(TI attachment_i = 0; attachment_i < SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA; attachment_i++){
                    attachments[attachment_i] = INVALID_OVERLAY;
                }
            }
        };

        template <typename T_SPEC>
        struct Renderer: MotionBlurRendererStorage<T_SPEC, T_SPEC::ENABLE_MOTION_BLUR>, RGBRendererStorage<T_SPEC, T_SPEC::HAS_RGB>, DepthRendererStorage<T_SPEC, T_SPEC::HAS_DEPTH>, SegmentationRendererStorage<T_SPEC, T_SPEC::HAS_SEGMENTATION>, OverlayRendererStorage<T_SPEC, T_SPEC::ENABLE_OVERLAYS>{
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;

            using CAMERA_TENSOR_SPEC = tensor::Specification<Camera<T>, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS>, true>;
            Tensor<CAMERA_TENSOR_SPEC> cameras;

            using COLLISION_TENSOR_SPEC = tensor::Specification<CollisionResult, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::NUM_PROBES>, true>;
            Tensor<COLLISION_TENSOR_SPEC> collision_results;

            T scene_center[3] = {0, 0, 0};
            T scene_half_extent[3] = {0, 0, 0};
            T camera_radius = 0;

            RendererBackend<SPEC> backend;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
