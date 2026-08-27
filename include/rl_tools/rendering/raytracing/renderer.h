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
                // normals writes the world-frame (FLU) geometric unit normal of the hit surface,
                // oriented against the ray, as float3 per pixel (miss = 0,0,0); like segmentation
                // it is always single-sample from the shutter-close camera: unit normals cannot
                // be averaged, so anti-aliasing and motion-blur settings do not apply to it
                static constexpr bool OUTPUT_NORMALS = false;
                // flow writes the backward optical flow of the shutter-close frame as float2 per
                // pixel in pixel units (miss = 0,0): p_close - p_open, where p_open is the
                // shutter-open camera's projection of the surface point after applying its
                // instance's shutter motion. The two instants are the motion-blur pair inputs
                // (cameras_open/cameras_close, set_transform_pair) — a producer drives
                // frame-to-frame flow by writing last frame's state into the open slots. Always
                // single-sample from the shutter-close camera, like segmentation.
                static constexpr bool OUTPUT_FLOW = false;
                static constexpr bool ENABLE_MOTION_BLUR = false;
                static constexpr T_TI MOTION_BLUR_SAMPLES = 1;
                // dynamic motion blur renders MOTION_BLUR_SAMPLES sequential passes per frame,
                // rebuilding the overlay acceleration structures from per-sample transforms
                // (transforms_motion) between passes and accumulating linear radiance on the
                // device — overlays blur, not just the camera. Requires motion blur + overlays.
                static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = false;
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
            static constexpr bool HAS_NORMALS = CONFIG::OUTPUT_NORMALS;
            static constexpr bool HAS_FLOW = CONFIG::OUTPUT_FLOW;
            static constexpr bool ENABLE_DEPTH = HAS_DEPTH;
            static constexpr bool ENABLE_RGB = HAS_RGB;
            static constexpr TI CAM_WIDTH = CONFIG::CAM_WIDTH;
            static constexpr TI CAM_HEIGHT = CONFIG::CAM_HEIGHT;
            static constexpr TI NUM_CAMERAS = CONFIG::NUM_CAMERAS;
            static constexpr TI NUM_PROBES = CONFIG::NUM_PROBES;
            static_assert(CAM_WIDTH > 0 && CAM_HEIGHT > 0 && NUM_CAMERAS > 0, "camera geometry must be nonzero");
            static_assert(HAS_RGB || HAS_DEPTH || HAS_SEGMENTATION || HAS_NORMALS || HAS_FLOW || NUM_PROBES > 0, "the renderer must produce at least one output (an image target or collision probes)");
            static constexpr TI MOTION_BLUR_SAMPLES = CONFIG::MOTION_BLUR_SAMPLES;
            static constexpr bool ENABLE_MOTION_BLUR = CONFIG::ENABLE_MOTION_BLUR && MOTION_BLUR_SAMPLES > 1;
            // flow consumes the shutter-open camera (and overlay shutter poses) without motion
            // blur, so the pair storage is gated on this rather than on ENABLE_MOTION_BLUR
            static constexpr bool HAS_CAMERA_PAIR = ENABLE_MOTION_BLUR || HAS_FLOW;
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
            // dynamic overlays: per-camera dynamic content composed onto the static shared world
            static constexpr TI NUM_OVERLAYS = CONFIG::NUM_OVERLAYS;
            static constexpr TI MAX_OVERLAY_INSTANCES = CONFIG::MAX_OVERLAY_INSTANCES;
            static constexpr TI MAX_OVERLAYS_PER_CAMERA = CONFIG::MAX_OVERLAYS_PER_CAMERA;
            static constexpr bool ENABLE_OVERLAYS = NUM_OVERLAYS > 0 && MAX_OVERLAY_INSTANCES > 0 && MAX_OVERLAYS_PER_CAMERA > 0;
            static constexpr bool SEMANTIC_SEGMENTATION = CONFIG::SEMANTIC_SEGMENTATION; // segmentation output carries Object::segmentation_class instead of the instance id
            static_assert(!SEMANTIC_SEGMENTATION || HAS_SEGMENTATION, "SEMANTIC_SEGMENTATION requires OUTPUT_SEGMENTATION");
            static_assert(ENABLE_OVERLAYS || (NUM_OVERLAYS == 0 && MAX_OVERLAY_INSTANCES == 0 && MAX_OVERLAYS_PER_CAMERA == 0), "overlay constants must be all zero (disabled) or all nonzero");
            static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = CONFIG::ENABLE_DYNAMIC_MOTION_BLUR && ENABLE_MOTION_BLUR && ENABLE_OVERLAYS;
            static_assert(!CONFIG::ENABLE_DYNAMIC_MOTION_BLUR || (ENABLE_MOTION_BLUR && ENABLE_OVERLAYS), "ENABLE_DYNAMIC_MOTION_BLUR requires ENABLE_MOTION_BLUR (MOTION_BLUR_SAMPLES > 1) and overlays");
            // the transforms_pair producer input serves dynamic motion blur (slerped into
            // transforms_motion) and flow (composed into flow_deltas), so its storage is gated
            // on either consumer
            static constexpr bool HAS_TRANSFORM_PAIR = ENABLE_DYNAMIC_MOTION_BLUR || (HAS_FLOW && ENABLE_OVERLAYS);
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
            struct Webgpu {};

            template <typename T_BACKEND>
            struct Name;
            template <> struct Name<None>    { static constexpr const char* VALUE = "none"; };
            template <> struct Name<Generic> { static constexpr const char* VALUE = "generic"; };
            template <> struct Name<Optix>   { static constexpr const char* VALUE = "optix"; };
            template <> struct Name<Metal>   { static constexpr const char* VALUE = "metal"; };
            template <> struct Name<Vulkan>  { static constexpr const char* VALUE = "vulkan"; };
            template <> struct Name<Webgpu>  { static constexpr const char* VALUE = "webgpu"; };

            template <typename T_BACKEND>
            constexpr const char* name(){
                return Name<T_BACKEND>::VALUE;
            }

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)
            using Default = Metal;
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            using Default = Optix;
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_VULKAN)
            using Default = Vulkan;
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_WEBGPU)
            using Default = Webgpu;
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

            namespace optix { struct State; }
            namespace metal { struct Context; }
            namespace vulkan { struct Context; }
            namespace webgpu { struct Context; }
            // renderer.device: memory-domain handle for tensor copies at readback/upload
            // boundaries — copy(renderer.device, device, ...) dispatches on the backend residency
            // and orders the copy after the renderer's in-flight work; backend selection for the
            // verbs stays the Renderer's template argument
            template <typename T_BACKEND>
            struct Device {
                using index_t = size_t;
            };
            template <>
            struct Device<Optix>{
                using index_t = size_t;
                optix::State* state = nullptr;
            };
            template <>
            struct Device<Metal>{
                using index_t = size_t;
                metal::Context* context = nullptr;
            };
            template <>
            struct Device<Vulkan>{
                using index_t = size_t;
                vulkan::Context* context = nullptr;
            };
            template <>
            struct Device<Webgpu>{
                using index_t = size_t;
                webgpu::Context* context = nullptr;
            };
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

        template <typename T_SPEC, bool T_HAS_CAMERA_PAIR>
        struct CameraPairRendererStorage {};

        template <typename T_SPEC>
        struct CameraPairRendererStorage<T_SPEC, true> {
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

        template <typename T_SPEC, bool T_HAS_NORMALS>
        struct NormalsRendererStorage {};

        template <typename T_SPEC>
        struct NormalsRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            using NORMALS_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, 3>, true>;
            Tensor<NORMALS_TENSOR_SPEC> normals_buffer;
        };

        template <typename T_SPEC, bool T_HAS_FLOW>
        struct FlowRendererStorage {};

        template <typename T_SPEC>
        struct FlowRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            using FLOW_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, 2>, true>;
            Tensor<FLOW_TENSOR_SPEC> flow_buffer;
        };

        template <typename T_SPEC, bool T_HAS_FLOW_OVERLAYS>
        struct FlowOverlayRendererStorage {};

        template <typename T_SPEC>
        struct FlowOverlayRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            // per-slot shutter motion consumed by the flow pass: world_open ∘ world_close⁻¹,
            // composed by update() from the shutter-open mirrors (host verbs) or by the pair
            // expansion (transforms_pair producers); identity for slots held over the shutter.
            // Backend-native residency like transforms.
            using FLOW_DELTAS_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, 12>, true>;
            Tensor<FLOW_DELTAS_TENSOR_SPEC> flow_deltas;
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
                float transform_entry_open[12]; // shutter-open counterpart, kept in lockstep by the verbs (pair verbs write it, single-pose verbs replicate); consumed by the flow shutter-delta composition
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

        template <typename T_SPEC, bool T_HAS_TRANSFORM_PAIR>
        struct TransformPairRendererStorage {};

        template <typename T_SPEC>
        struct TransformPairRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            // device-producer input: shutter-open (index 0) and shutter-close (index 1) entries
            // per slot, expanded by expand_motion_transforms into transforms_motion (dynamic
            // motion blur), flow_deltas (flow), and the close state into transforms — on OptiX
            // both the producer write and the expansion stay on the device, so a sim kernel can
            // drive dynamic blur or flow with no host data path
            using TRANSFORMS_PAIR_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, 2, SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, 12>, true>;
            Tensor<TRANSFORMS_PAIR_TENSOR_SPEC> transforms_pair;
        };

        template <typename T_SPEC, bool T_ENABLE_DYNAMIC_MOTION_BLUR>
        struct DynamicMotionBlurRendererStorage {};

        template <typename T_SPEC>
        struct DynamicMotionBlurRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            // per-sample overlay transform input consumed by the dynamic-motion-blur render loop:
            // sample-major so slab s (base + s * NUM_OVERLAYS * MAX_OVERLAY_INSTANCES * 12) has
            // the exact layout of the transforms tensor and feeds the same build path. Row
            // semantics match transforms (root: pose, non-root: part-frame articulation).
            using TRANSFORMS_MOTION_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::MOTION_BLUR_SAMPLES, SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, 12>, true>;
            Tensor<TRANSFORMS_MOTION_TENSOR_SPEC> transforms_motion;
            // host mirror of transforms_motion for the verb path (the tensor is device-resident
            // on OptiX); staged per dirty overlay, same ownership contract as transform_entry
            std::vector<float> transforms_motion_staging;
            bool transforms_motion_dirty[SPEC::NUM_OVERLAYS] = {};
            // linear-radiance accumulators, resolved into the packed frame buffer / depth buffer
            // after the last sample pass; allocated only for the enabled outputs
            using RGB_ACCUMULATOR_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, 3>, true>;
            Tensor<RGB_ACCUMULATOR_TENSOR_SPEC> rgb_accumulator;
            using DEPTH_ACCUMULATOR_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH>, true>;
            Tensor<DEPTH_ACCUMULATOR_TENSOR_SPEC> depth_accumulator;
        };

        template <typename T_SPEC, typename T_BACKEND = backends::Default>
        struct Renderer: CameraPairRendererStorage<T_SPEC, T_SPEC::HAS_CAMERA_PAIR>, RGBRendererStorage<T_SPEC, T_SPEC::HAS_RGB>, DepthRendererStorage<T_SPEC, T_SPEC::HAS_DEPTH>, SegmentationRendererStorage<T_SPEC, T_SPEC::HAS_SEGMENTATION>, NormalsRendererStorage<T_SPEC, T_SPEC::HAS_NORMALS>, FlowRendererStorage<T_SPEC, T_SPEC::HAS_FLOW>, FlowOverlayRendererStorage<T_SPEC, T_SPEC::HAS_FLOW && T_SPEC::ENABLE_OVERLAYS>, ObservationRendererStorage<T_SPEC, T_SPEC::HAS_OBSERVATION>, OverlayRendererStorage<T_SPEC, T_SPEC::ENABLE_OVERLAYS>, TransformPairRendererStorage<T_SPEC, T_SPEC::HAS_TRANSFORM_PAIR>, DynamicMotionBlurRendererStorage<T_SPEC, T_SPEC::ENABLE_DYNAMIC_MOTION_BLUR>{
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

            backends::Device<BACKEND> device;
            BACKEND_STATE* backend = nullptr;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
