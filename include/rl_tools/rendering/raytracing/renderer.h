#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_RENDERER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_RENDERER_H

#include "types.h"
#include "outputs/outputs.h"
#include "scene.h"
#include "../../containers/tensor/tensor.h"

#include <vector>
#include <deque>
#include <cstdint>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing{

        using BasicShading = Medium;
        using HighFidelityShading = High;
        using VeryHighFidelityShading = VeryHigh;
        using FastFlatShading = Low;

        namespace config{
            template <typename T_T, typename T_TI>
            struct Default{
                using T = T_T;
                using TI = T_TI;
                using SHADING = Medium;
                using OUTPUTS = outputs::List<outputs::RGB<>>;
                static constexpr bool ENABLE_MOTION_BLUR = false;
                static constexpr T_TI MOTION_BLUR_SAMPLES = 1;
                // Dynamic motion blur uses transforms_motion for each shutter sample and averages linear radiance; requires motion blur and overlays.
                static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = false;
                static constexpr bool ENABLE_ANTI_ALIASING = false;
                static constexpr T_TI ANTI_ALIASING_GRID_SIZE = 1;
                static constexpr T_TI NUM_OVERLAYS = 0;
                static constexpr T_TI MAX_OVERLAY_INSTANCES = 0;
                static constexpr T_TI MAX_OVERLAYS_PER_CAMERA = 0;
            };
        }

        template <typename T_CONFIG>
        struct Specification{
            using CONFIG = T_CONFIG;
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            using SHADING = typename CONFIG::SHADING;
            using OUTPUTS = typename CONFIG::OUTPUTS;
            using PLAN = outputs::Requirements<OUTPUTS>;
            static constexpr bool HAS_RGB = PLAN::HAS_RGB;
            static constexpr bool HAS_RGBA8 = PLAN::HAS_RGBA8;
            static constexpr bool HAS_RGB32F = PLAN::HAS_RGB32F;
            static constexpr bool HAS_DEPTH = PLAN::HAS_DEPTH;
            static constexpr bool HAS_SEGMENTATION = PLAN::HAS_SEGMENTATION;
            static constexpr bool HAS_INSTANCE_SEGMENTATION = PLAN::HAS_INSTANCE_SEGMENTATION;
            static constexpr bool HAS_SEMANTIC_SEGMENTATION = PLAN::HAS_SEMANTIC_SEGMENTATION;
            static constexpr bool HAS_NORMALS = PLAN::HAS_NORMALS;
            static constexpr bool HAS_FLOW = PLAN::HAS_FLOW;
            static constexpr bool HAS_FLOW_BACKWARD = PLAN::HAS_FLOW_BACKWARD;
            static constexpr bool HAS_FLOW_FORWARD = PLAN::HAS_FLOW_FORWARD;
            static constexpr bool HAS_FLOW_BACKWARD_STATUS = PLAN::HAS_FLOW_BACKWARD_STATUS;
            static constexpr bool HAS_FLOW_FORWARD_STATUS = PLAN::HAS_FLOW_FORWARD_STATUS;
            static constexpr bool HAS_PROBES = PLAN::HAS_PROBES;
            static constexpr bool HAS_AUXILIARY = PLAN::HAS_AUXILIARY;
            static constexpr TI CAM_WIDTH = CONFIG::CAM_WIDTH;
            static constexpr TI CAM_HEIGHT = CONFIG::CAM_HEIGHT;
            static constexpr TI NUM_CAMERAS = CONFIG::NUM_CAMERAS;
            static constexpr TI NUM_PROBES = []() {
                if constexpr (HAS_PROBES) {
                    return CONFIG::NUM_PROBES;
                }
                else {
                    return TI(0);
                }
            }();
            static_assert(!HAS_PROBES || NUM_PROBES > 0, "configured probes require NUM_PROBES > 0");
            static_assert(CAM_WIDTH > 0 && CAM_HEIGHT > 0 && NUM_CAMERAS > 0, "camera geometry must be nonzero");
            static_assert(PLAN::HAS_IMAGE || HAS_PROBES, "the renderer must produce at least one output (an image target or collision probes)");
            static constexpr TI MOTION_BLUR_SAMPLES = CONFIG::MOTION_BLUR_SAMPLES;
            static constexpr bool ENABLE_MOTION_BLUR = HAS_RGB && CONFIG::ENABLE_MOTION_BLUR && MOTION_BLUR_SAMPLES > 1;
            static constexpr bool HAS_CAMERA_PAIR = ENABLE_MOTION_BLUR;
            static_assert(MOTION_BLUR_SAMPLES >= 1, "MOTION_BLUR_SAMPLES must be at least 1");
            static_assert(!ENABLE_MOTION_BLUR || MOTION_BLUR_SAMPLES == 2 || MOTION_BLUR_SAMPLES == 4 || MOTION_BLUR_SAMPLES == 8 || MOTION_BLUR_SAMPLES == 16 || MOTION_BLUR_SAMPLES == 32, "MOTION_BLUR_SAMPLES must be one of 2, 4, 8, 16, or 32");
            static constexpr TI ANTI_ALIASING_GRID_SIZE = CONFIG::ANTI_ALIASING_GRID_SIZE;
            static constexpr bool ENABLE_ANTI_ALIASING = HAS_RGB && CONFIG::ENABLE_ANTI_ALIASING && ANTI_ALIASING_GRID_SIZE > 1;
            static constexpr TI ANTI_ALIASING_SAMPLES = ENABLE_ANTI_ALIASING ? ANTI_ALIASING_GRID_SIZE * ANTI_ALIASING_GRID_SIZE : 1;
            static constexpr TI RGB_SAMPLES = (ENABLE_MOTION_BLUR ? MOTION_BLUR_SAMPLES : 1) * ANTI_ALIASING_SAMPLES;
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
            static_assert(ENABLE_OVERLAYS || (NUM_OVERLAYS == 0 && MAX_OVERLAY_INSTANCES == 0 && MAX_OVERLAYS_PER_CAMERA == 0), "overlay constants must be all zero (disabled) or all nonzero");
            static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = CONFIG::ENABLE_DYNAMIC_MOTION_BLUR && ENABLE_MOTION_BLUR && ENABLE_OVERLAYS;
            static_assert(!HAS_RGB || !CONFIG::ENABLE_DYNAMIC_MOTION_BLUR || (ENABLE_MOTION_BLUR && ENABLE_OVERLAYS), "ENABLE_DYNAMIC_MOTION_BLUR requires ENABLE_MOTION_BLUR (MOTION_BLUR_SAMPLES > 1) and overlays");
            static constexpr bool HAS_FLOW_OVERLAYS = HAS_FLOW && ENABLE_OVERLAYS;
            static constexpr bool FUSE_AUXILIARY = HAS_RGB && HAS_AUXILIARY && !ENABLE_ANTI_ALIASING && !ENABLE_MOTION_BLUR;
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

            template <typename T_BACKEND>
            struct RuntimeState {};

            template <>
            struct RuntimeState<Vulkan>;

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

        // A shared runtime must outlive every renderer allocated against it, regardless of SPEC.
        template <typename T_BACKEND = backends::Default>
        struct Runtime {
            using BACKEND = T_BACKEND;
            backends::RuntimeState<BACKEND> state{};
        };

        namespace device {
            template <typename T_RENDERING_DEVICE>
            struct Extension {
                using RENDERING_DEVICE = T_RENDERING_DEVICE;
                RENDERING_DEVICE rendering;
            };

            template <typename T_BASE_SPEC, typename T_RENDERING_DEVICE>
            struct Specification: T_BASE_SPEC {
                using DEVICE_EXTENSION = Extension<T_RENDERING_DEVICE>;
            };

            template <typename DEVICE, typename = void>
            struct HasRendering: utils::typing::false_type {};
            template <typename DEVICE>
            struct HasRendering<DEVICE, utils::typing::void_t<decltype(utils::typing::declared_lvalue<DEVICE>().rendering)>>: utils::typing::true_type {};
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
            std::vector<SceneMetadata<typename SPEC::T>> metadata;
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

        template <typename T_SPEC, bool T_HAS_FLOW>
        struct FlowCameraRendererStorage {};
        template <typename T_SPEC>
        struct FlowCameraRendererStorage<T_SPEC, true> {
            using CAMERA_TENSOR_SPEC = tensor::Specification<Camera<typename T_SPEC::T>, typename T_SPEC::TI, tensor::Shape<typename T_SPEC::TI, T_SPEC::NUM_CAMERAS>, true>;
            Tensor<CAMERA_TENSOR_SPEC> cameras_previous;
        };

        template <typename T_SPEC, bool T_HAS_FLOW_OVERLAYS>
        struct FlowOverlayRendererStorage {};

        template <typename T_SPEC>
        struct FlowOverlayRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            // the previous instant's overlay poses: same layout, residency and ownership contract
            // as transforms. advance_flow snapshots transforms into it (resident rows in the
            // backend's memory domain, rows the host has not yet published through the staging
            // mirror); update() composes the per-slot flow deltas from the two instants and
            // builds the previous instant's overlay acceleration structures the visibility rays
            // traverse.
            using TRANSFORMS_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, 12>, true>;
            Tensor<TRANSFORMS_TENSOR_SPEC> transforms_previous;
            std::vector<float> transforms_previous_staging;
            bool transforms_previous_pending[SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES] = {};
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
                float rest[12];         // part -> assembly frame captured at spawn; the whole-assembly verbs compose pose ∘ rest
                bool active = false;
            };
            struct OverlayState {
                OverlaySlot slots[SPEC::MAX_OVERLAY_INSTANCES];
                bool dirty = false;     // active/object changed since the last update
            };
            // pool assemblies flattened at init so spawn needs no pool access in the hot path
            struct AssetRecord {
                TI first_part;          // into asset_part_objects/asset_part_transforms
                TI num_parts;
            };
            OverlayState overlays[SPEC::NUM_OVERLAYS];
            // geometry-to-world transform per slot [o][s], consumed by update(): backend-native
            // residency (device memory on OptiX), accessed via transforms(device, renderer).
            // Producers write it in place; the host verbs write the staging mirror and mark their
            // rows pending, and update() publishes exactly those rows — a row belongs to whoever
            // wrote it last.
            using TRANSFORMS_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, 12>, true>;
            Tensor<TRANSFORMS_TENSOR_SPEC> transforms;
            std::vector<float> transforms_staging;
            bool transforms_pending[SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES] = {};
            TI attachments[SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA];
            bool attachments_dirty = false;
            // the flattened pool tables live on the renderer (not in backend state) because the
            // backend-independent host verbs (spawn, set_transform) consume them
            std::vector<AssetRecord> assets;
            std::vector<TI> asset_part_objects;      // global object index per part
            std::vector<float> asset_part_transforms; // 12 per part, assembly-local
        };

        template <typename T_SPEC, bool T_ENABLE_DYNAMIC_MOTION_BLUR>
        struct DynamicMotionBlurRendererStorage {};

        template <typename T_SPEC>
        struct DynamicMotionBlurRendererStorage<T_SPEC, true> {
            using SPEC = T_SPEC;
            using TI = typename SPEC::TI;
            // per-sample overlay transform input consumed by the dynamic-motion-blur render loop:
            // sample-major so slab s (base + s * NUM_OVERLAYS * MAX_OVERLAY_INSTANCES * 12) has
            // the exact layout of the transforms tensor and feeds the same build path, under the
            // same staging and ownership contract
            using TRANSFORMS_MOTION_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::MOTION_BLUR_SAMPLES, SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, 12>, true>;
            Tensor<TRANSFORMS_MOTION_TENSOR_SPEC> transforms_motion;
            std::vector<float> transforms_motion_staging;
            bool transforms_motion_pending[SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES] = {};
            // linear-radiance accumulators, resolved into the packed frame buffer
            // after the last sample pass; allocated only for the enabled outputs
            using RGB_ACCUMULATOR_TENSOR_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, 3>, true>;
            Tensor<RGB_ACCUMULATOR_TENSOR_SPEC> rgb_accumulator;
        };

        template <typename T_SPEC, typename T_BACKEND = backends::Default>
        struct Renderer: CameraPairRendererStorage<T_SPEC, T_SPEC::HAS_CAMERA_PAIR>, FlowCameraRendererStorage<T_SPEC, T_SPEC::HAS_FLOW>, FlowOverlayRendererStorage<T_SPEC, T_SPEC::HAS_FLOW_OVERLAYS>, OverlayRendererStorage<T_SPEC, T_SPEC::ENABLE_OVERLAYS>, DynamicMotionBlurRendererStorage<T_SPEC, T_SPEC::ENABLE_DYNAMIC_MOTION_BLUR>{
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

            rendering::raytracing::outputs::Buffers<SPEC, typename SPEC::OUTPUTS> outputs;

            // Loader-authoritative (SceneMetadata) probe ray budget.
            T max_ray_length = 0;
            // resolved Scene::Environment (background pair + ambient), staged into the backend
            // launch params; set by init
            T ambient_color[3] = {0, 0, 0};
            T miss_color_0[3] = {0, 0, 0};
            T miss_color_1[3] = {0, 0, 0};

            backends::Device<BACKEND> device;
            BACKEND_STATE* backend = nullptr;
        };
    }
    template <typename DEVICE>
    auto& get_rendering_device(DEVICE& device){
        if constexpr(rendering::raytracing::device::HasRendering<DEVICE>::value){
            return device.rendering;
        }
        else{
            return device;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
