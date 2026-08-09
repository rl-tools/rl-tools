#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_RENDERER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_RENDERER_H

#include "types.h"
#include "scene.h"
#include "../../containers/tensor/tensor.h"

#include <vector>
#include <deque>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{

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

        namespace config{
            // Fringe defaults only: the required geometry — CAM_WIDTH, CAM_HEIGHT, NUM_CAMERAS,
            // NUM_PROBES — is deliberately absent from this base, so a missing (or misspelled)
            // required member fails to compile instead of silently falling back to a default.
            template <typename T_T, typename T_TI>
            struct Default{
                using T = T_T;
                using TI = T_TI;
                using SHADING = Medium;
                static constexpr bool OUTPUT_RGB = true;
                static constexpr bool OUTPUT_DEPTH = false;
                // segmentation writes one uint32 instance index per pixel (miss = 0xFFFFFFFF),
                // always single-sample from the shutter-close camera: labels cannot be averaged,
                // so anti-aliasing and motion-blur settings do not apply to it
                static constexpr bool OUTPUT_SEGMENTATION = false;
                static constexpr bool SEMANTIC_SEGMENTATION = false;
                static constexpr bool ENABLE_MOTION_BLUR = false;
                static constexpr T_TI MOTION_BLUR_SAMPLES = 1;
                static constexpr bool ENABLE_ANTI_ALIASING = false;
                static constexpr T_TI ANTI_ALIASING_GRID_SIZE = 1;
                static constexpr T_TI NUM_OVERLAYS = 0;
                static constexpr T_TI MAX_OVERLAY_INSTANCES = 0;
                static constexpr T_TI MAX_OVERLAYS_PER_CAMERA = 0;
                // spec-driven observation output: the RGB ray gen writes OBSERVATION_T pixels
                // ([0,1] linear or sRGB per SHADING) directly — no format-conversion pass. The
                // packed frame buffer remains the video/golden output.
                static constexpr bool OUTPUT_OBSERVATION = false;
                using OBSERVATION_T = float;
            };
        }

        template <typename T_CONFIG>
        struct Specification{
            using CONFIG = T_CONFIG;
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            using SHADING = typename CONFIG::SHADING;
            static constexpr bool HAS_RGB = CONFIG::OUTPUT_RGB;
            static constexpr bool HAS_DEPTH = CONFIG::OUTPUT_DEPTH;
            static constexpr bool HAS_SEGMENTATION = CONFIG::OUTPUT_SEGMENTATION;
            static constexpr bool ENABLE_DEPTH = HAS_DEPTH;
            static constexpr bool ENABLE_RGB = HAS_RGB;
            static constexpr TI CAM_WIDTH = CONFIG::CAM_WIDTH;
            static constexpr TI CAM_HEIGHT = CONFIG::CAM_HEIGHT;
            static constexpr TI NUM_CAMERAS = CONFIG::NUM_CAMERAS;
            static constexpr TI NUM_PROBES = CONFIG::NUM_PROBES;
            static_assert(CAM_WIDTH > 0 && CAM_HEIGHT > 0 && NUM_CAMERAS > 0, "camera geometry must be nonzero");
            static_assert(HAS_RGB || HAS_DEPTH || HAS_SEGMENTATION || NUM_PROBES > 0, "the renderer must produce at least one output (an image target or collision probes)");
            static constexpr TI MOTION_BLUR_SAMPLES = CONFIG::MOTION_BLUR_SAMPLES;
            static constexpr bool ENABLE_MOTION_BLUR = CONFIG::ENABLE_MOTION_BLUR && MOTION_BLUR_SAMPLES > 1;
            static_assert(MOTION_BLUR_SAMPLES >= 1, "MOTION_BLUR_SAMPLES must be at least 1");
            static_assert(!ENABLE_MOTION_BLUR || MOTION_BLUR_SAMPLES == 2 || MOTION_BLUR_SAMPLES == 4 || MOTION_BLUR_SAMPLES == 8 || MOTION_BLUR_SAMPLES == 16 || MOTION_BLUR_SAMPLES == 32, "MOTION_BLUR_SAMPLES must be one of 2, 4, 8, 16, or 32");
            static constexpr TI ANTI_ALIASING_GRID_SIZE = CONFIG::ANTI_ALIASING_GRID_SIZE;
            static constexpr bool ENABLE_ANTI_ALIASING = CONFIG::ENABLE_ANTI_ALIASING && ANTI_ALIASING_GRID_SIZE > 1;
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
            static constexpr TI NUM_OVERLAYS = CONFIG::NUM_OVERLAYS;
            static constexpr TI MAX_OVERLAY_INSTANCES = CONFIG::MAX_OVERLAY_INSTANCES;
            static constexpr TI MAX_OVERLAYS_PER_CAMERA = CONFIG::MAX_OVERLAYS_PER_CAMERA;
            static constexpr bool ENABLE_OVERLAYS = NUM_OVERLAYS > 0 && MAX_OVERLAY_INSTANCES > 0 && MAX_OVERLAYS_PER_CAMERA > 0;
            static constexpr bool SEMANTIC_SEGMENTATION = CONFIG::SEMANTIC_SEGMENTATION; // segmentation output carries Object::segmentation_class instead of the instance id
            static_assert(!SEMANTIC_SEGMENTATION || HAS_SEGMENTATION, "SEMANTIC_SEGMENTATION requires OUTPUT_SEGMENTATION");
            static_assert(ENABLE_OVERLAYS || (NUM_OVERLAYS == 0 && MAX_OVERLAY_INSTANCES == 0 && MAX_OVERLAYS_PER_CAMERA == 0), "overlay constants must be all zero (disabled) or all nonzero");
            static constexpr bool HAS_OBSERVATION = CONFIG::OUTPUT_OBSERVATION;
            using OBSERVATION_T = typename CONFIG::OBSERVATION_T;
            static constexpr TI OBSERVATION_CHANNELS = 3;
            static_assert(!HAS_OBSERVATION || HAS_RGB, "OUTPUT_OBSERVATION requires OUTPUT_RGB (the RGB ray gen writes it)");
        };

        namespace backends {
            struct None {};
            struct Generic {};
            struct Optix {};
            struct Metal {};
            struct Vulkan {};

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)
            using Default = Metal;
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            using Default = Optix;
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_VULKAN)
            using Default = Vulkan;
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_GENERIC)
            using Default = Generic;
#else
            using Default = None;
#endif

            template <typename T_BACKEND, typename T_SPEC>
            struct RendererState;

            template <typename T_BACKEND, typename T_SPEC>
            struct LibraryState;

            template <typename T_BACKEND, typename T_SPEC>
            struct SceneState;
        }

        // shared scene store: renderers malloc'd against a library share one backend context and
        // one geometry/texture/BLAS build per unique scene — init(device, renderer, library,
        // path) deduplicates by file content hash and returns the unique-scene index, so callers
        // key their own per-scene data off it. The library owns the host scenes its builds
        // reference (deque: stable addresses). On backends without cross-renderer sharing
        // (generic/Metal/Vulkan) each renderer still builds its own device copy — the caller
        // code is uniform, the sharing is a backend property.
        template <typename T_SPEC, typename T_BACKEND = backends::Default>
        struct AssetLibrary {
            using SPEC = T_SPEC;
            using BACKEND = T_BACKEND;
            using TI = typename SPEC::TI;
            using BACKEND_STATE = backends::LibraryState<BACKEND, SPEC>;
            using SCENE_STATE = backends::SceneState<BACKEND, SPEC>;
            BACKEND_STATE* backend = nullptr;
            std::deque<Scene> scenes;
            std::deque<SCENE_STATE*> assets;
            std::vector<uint64_t> hashes;
            AssetPool pool;
        };

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

        template <typename T_SPEC, bool T_HAS_OBSERVATION>
        struct ObservationRendererStorage {};

        template <typename T_SPEC>
        struct ObservationRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            using OBSERVATION_TENSOR_SPEC = tensor::Specification<typename SPEC::OBSERVATION_T, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, SPEC::OBSERVATION_CHANNELS>, true>;
            Tensor<OBSERVATION_TENSOR_SPEC> observation;
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
                TI pose_slot = 0;       // root slot of the owning placement; the root's transforms entry moves the whole placement
                float part_local[12];   // static attachment (part -> assembly), captured at spawn
                float transform_entry[12]; // host mirror of the transforms-tensor entry (root: pose, non-root: articulation in the part frame)
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
            // dynamic transform input consumed by update(): entry [o][s] is the placement pose at
            // its root slot and a part-frame articulation elsewhere (world = pose ∘ part_local ∘
            // articulation). Backend-native residency (device memory on OptiX) — access via
            // transforms(device, renderer); host verbs stage through the slot mirrors instead.
            using TRANSFORMS_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, 12>, true>;
            Tensor<TRANSFORMS_TENSOR_SPEC> transforms;
            TI attachments[SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA];
            bool attachments_dirty = false;
            // the flattened pool tables live on the renderer (not in backend state) because the
            // backend-independent host verbs (spawn, set_transform) consume them
            std::vector<AssetRecord> assets;
            std::vector<TI> asset_part_objects;      // global object index per part
            std::vector<float> asset_part_transforms; // 12 per part, assembly-local
        };

        template <typename T_SPEC, typename T_BACKEND = backends::Default>
        struct Renderer: MotionBlurRendererStorage<T_SPEC, T_SPEC::ENABLE_MOTION_BLUR>, RGBRendererStorage<T_SPEC, T_SPEC::HAS_RGB>, DepthRendererStorage<T_SPEC, T_SPEC::HAS_DEPTH>, SegmentationRendererStorage<T_SPEC, T_SPEC::HAS_SEGMENTATION>, ObservationRendererStorage<T_SPEC, T_SPEC::HAS_OBSERVATION>, OverlayRendererStorage<T_SPEC, T_SPEC::ENABLE_OVERLAYS>{
            using SPEC = T_SPEC;
            using BACKEND = T_BACKEND;
            using BACKEND_STATE = backends::RendererState<BACKEND, SPEC>;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;

            // camera input consumed directly by render/probe: backend-native residency (device
            // memory on OptiX, host on generic, shared/mapped on Metal/Vulkan) — access via
            // cameras(device, renderer); under motion blur this is the shutter-close camera and
            // cameras_open holds the shutter-open pose (see cameras_open/cameras_close accessors)
            using CAMERA_TENSOR_SPEC = tensor::Specification<Camera<T>, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS>, true>;
            Tensor<CAMERA_TENSOR_SPEC> cameras;

            using COLLISION_TENSOR_SPEC = tensor::Specification<CollisionResult, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::NUM_PROBES>, true>;
            Tensor<COLLISION_TENSOR_SPEC> collision_results;

            T scene_center[3] = {0, 0, 0};
            T scene_half_extent[3] = {0, 0, 0};
            T camera_radius = 0;

            BACKEND_STATE* backend = nullptr;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
