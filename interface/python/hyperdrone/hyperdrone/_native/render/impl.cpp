#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>

#include "iface.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifndef HYPERDRONE_RENDER_WIDTH
#error "HYPERDRONE_RENDER_WIDTH must be defined"
#endif
#ifndef HYPERDRONE_RENDER_HEIGHT
#error "HYPERDRONE_RENDER_HEIGHT must be defined"
#endif
#ifndef HYPERDRONE_RENDER_NUM_CAMERAS
#error "HYPERDRONE_RENDER_NUM_CAMERAS must be defined"
#endif
#ifndef HYPERDRONE_RENDER_NUM_PROBES
#define HYPERDRONE_RENDER_NUM_PROBES 1
#endif
#ifndef HYPERDRONE_RENDER_SHADING
#define HYPERDRONE_RENDER_SHADING 2
#endif
#ifndef HYPERDRONE_RENDER_OUTPUT_MODE
#define HYPERDRONE_RENDER_OUTPUT_MODE 0
#endif
#ifndef HYPERDRONE_RENDER_MB_SAMPLES
#define HYPERDRONE_RENDER_MB_SAMPLES 1
#endif
#ifndef HYPERDRONE_RENDER_AA_GRID
#define HYPERDRONE_RENDER_AA_GRID 1
#endif
#ifndef HYPERDRONE_RENDER_NUM_OVERLAYS
#define HYPERDRONE_RENDER_NUM_OVERLAYS 0
#endif
#ifndef HYPERDRONE_RENDER_MAX_OVERLAY_INSTANCES
#define HYPERDRONE_RENDER_MAX_OVERLAY_INSTANCES 0
#endif
#ifndef HYPERDRONE_RENDER_MAX_OVERLAYS_PER_CAMERA
#define HYPERDRONE_RENDER_MAX_OVERLAYS_PER_CAMERA 0
#endif
#ifndef HYPERDRONE_RENDER_SEMANTIC_SEGMENTATION
#define HYPERDRONE_RENDER_SEMANTIC_SEGMENTATION 0
#endif

namespace rlt = rl_tools;
namespace rrt = rl_tools::rendering::raytracing;

namespace hyperdrone_render_impl {
    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
    using T = float;
    using TI = typename DEVICE::index_t;

    template <int SHADING_ID> struct ShadingSelector;
    template <> struct ShadingSelector<0> { using type = rrt::Low; };
    template <> struct ShadingSelector<1> { using type = rrt::Medium; };
    template <> struct ShadingSelector<2> { using type = rrt::High; };
    template <> struct ShadingSelector<3> { using type = rrt::VeryHigh; };

    // the output-mode id (0=rgb 1=rgbd 2=depth 3=segmentation 4=rgbd_segmentation) maps
    // onto the config's independent channel switches
    struct RendererConfiguration: rrt::config::Default<T, TI>{
        static constexpr TI CAM_WIDTH = HYPERDRONE_RENDER_WIDTH;
        static constexpr TI CAM_HEIGHT = HYPERDRONE_RENDER_HEIGHT;
        static constexpr TI NUM_CAMERAS = HYPERDRONE_RENDER_NUM_CAMERAS;
        static constexpr TI NUM_PROBES = HYPERDRONE_RENDER_NUM_PROBES;
        using SHADING = typename ShadingSelector<HYPERDRONE_RENDER_SHADING>::type;
        static constexpr bool OUTPUT_RGB = HYPERDRONE_RENDER_OUTPUT_MODE == 0 || HYPERDRONE_RENDER_OUTPUT_MODE == 1 || HYPERDRONE_RENDER_OUTPUT_MODE == 4;
        static constexpr bool OUTPUT_DEPTH = HYPERDRONE_RENDER_OUTPUT_MODE == 1 || HYPERDRONE_RENDER_OUTPUT_MODE == 2 || HYPERDRONE_RENDER_OUTPUT_MODE == 4;
        static constexpr bool OUTPUT_SEGMENTATION = HYPERDRONE_RENDER_OUTPUT_MODE == 3 || HYPERDRONE_RENDER_OUTPUT_MODE == 4;
        static constexpr bool SEMANTIC_SEGMENTATION = HYPERDRONE_RENDER_SEMANTIC_SEGMENTATION != 0;
        static constexpr bool ENABLE_MOTION_BLUR = HYPERDRONE_RENDER_MB_SAMPLES > 1;
        static constexpr TI MOTION_BLUR_SAMPLES = HYPERDRONE_RENDER_MB_SAMPLES;
        static constexpr bool ENABLE_ANTI_ALIASING = HYPERDRONE_RENDER_AA_GRID > 1;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = HYPERDRONE_RENDER_AA_GRID;
        static constexpr TI NUM_OVERLAYS = HYPERDRONE_RENDER_NUM_OVERLAYS;
        static constexpr TI MAX_OVERLAY_INSTANCES = HYPERDRONE_RENDER_MAX_OVERLAY_INSTANCES;
        static constexpr TI MAX_OVERLAYS_PER_CAMERA = HYPERDRONE_RENDER_MAX_OVERLAYS_PER_CAMERA;
    };

    using RENDERER_SPEC = rrt::Specification<RendererConfiguration>;

    // template so if-constexpr branches on the spec's feature flags are discarded without
    // being instantiated on configs that lack the corresponding storage/operations
    template <typename T_SPEC>
    struct RendererImpl final : hyperdrone::render::Renderer {
        using SPEC = T_SPEC;
        DEVICE device;
        rrt::Renderer<SPEC> renderer;
        bool initialized = false;

        RendererImpl(){
            rlt::init(device);
            rlt::malloc(device, renderer);
        }
        ~RendererImpl() override {
            rlt::free(device, renderer);
        }

        hyperdrone::render::Config config() const override {
            hyperdrone::render::Config c;
            c.width = HYPERDRONE_RENDER_WIDTH;
            c.height = HYPERDRONE_RENDER_HEIGHT;
            c.num_cameras = HYPERDRONE_RENDER_NUM_CAMERAS;
            c.num_probes = HYPERDRONE_RENDER_NUM_PROBES;
            c.shading = HYPERDRONE_RENDER_SHADING;
            c.output_mode = HYPERDRONE_RENDER_OUTPUT_MODE;
            c.motion_blur_samples = HYPERDRONE_RENDER_MB_SAMPLES;
            c.anti_aliasing_grid = HYPERDRONE_RENDER_AA_GRID;
            c.num_overlays = HYPERDRONE_RENDER_NUM_OVERLAYS;
            c.max_overlay_instances = HYPERDRONE_RENDER_MAX_OVERLAY_INSTANCES;
            c.max_overlays_per_camera = HYPERDRONE_RENDER_MAX_OVERLAYS_PER_CAMERA;
            c.semantic_segmentation = HYPERDRONE_RENDER_SEMANTIC_SEGMENTATION != 0;
            return c;
        }

        const char* backend() const override {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            return "optix";
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)
            return "metal";
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_VULKAN)
            return "vulkan";
#else
            return "generic";
#endif
        }

        void init(const rrt::Scene* scene, const rrt::AssetPool* pool) override {
            if(initialized){
                throw std::runtime_error("hyperdrone: renderer is already initialized; create a new Renderer for a different scene");
            }
            if(pool != nullptr){
                if constexpr (SPEC::ENABLE_OVERLAYS){
                    rlt::init(device, renderer, *scene, *pool);
                }
                else {
                    throw std::runtime_error("hyperdrone: an asset pool requires overlays (num_overlays/max_overlay_instances/max_overlays_per_camera > 0)");
                }
            }
            else {
                rlt::init(device, renderer, *scene);
            }
            initialized = true;
        }

        void require_init() const {
            if(!initialized){
                throw std::runtime_error("hyperdrone: renderer is not initialized (call init(scene) first)");
            }
        }

        void update() override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::update(device, renderer);
            }
            else {
                throw std::runtime_error("hyperdrone: update() requires overlays to be enabled");
            }
        }

        void synchronize() override {
            rlt::synchronize(device, renderer);
        }

        void set_cameras(const float* cameras) override {
            std::memcpy(rlt::data(renderer.cameras), cameras, sizeof(rrt::Camera<T>) * SPEC::NUM_CAMERAS);
            rlt::set_cameras(device, renderer, renderer.cameras);
        }

        void set_cameras_device(const float* cameras, unsigned long long producer_stream) override {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            require_init();
            rlt::set_cameras_device(device, renderer, cameras, (cudaStream_t)producer_stream);
#else
            (void)cameras; (void)producer_stream;
            throw std::runtime_error("hyperdrone: device-resident camera input is only supported on the OptiX backend");
#endif
        }

        void set_motion_blur_cameras(const float* cameras_open, const float* cameras_close) override {
            if constexpr (SPEC::ENABLE_MOTION_BLUR){
                std::memcpy(rlt::data(renderer.cameras_open), cameras_open, sizeof(rrt::Camera<T>) * SPEC::NUM_CAMERAS);
                std::memcpy(rlt::data(renderer.cameras), cameras_close, sizeof(rrt::Camera<T>) * SPEC::NUM_CAMERAS);
                rlt::set_motion_blur_cameras(device, renderer, renderer.cameras_open, renderer.cameras);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without motion blur (motion_blur_samples <= 1)");
            }
        }

        void generate_cameras(const float center[3], float radius, const float up[3], float fov) override {
            require_init();
            rlt::generate_cameras(device, renderer, center, radius, up, fov);
        }

        void generate_probe_directions() override {
            rlt::generate_probe_directions(device, renderer);
        }

        // the renderer API renders all enabled image channels together (render*) with
        // probe rays split out (probe*) uniformly across backends. Channel-specific
        // targets are validated against the spec, then map to the unified render.
        void render(hyperdrone::render::RenderTarget target, hyperdrone::render::RenderPhase phase) override {
            require_init();
            using RT = hyperdrone::render::RenderTarget;
            using RP = hyperdrone::render::RenderPhase;
            switch(target){
                case RT::RGB:
                    if constexpr (!SPEC::HAS_RGB){ throw std::runtime_error("hyperdrone: this renderer has no RGB output"); }
                    break;
                case RT::DEPTH:
                    if constexpr (!SPEC::HAS_DEPTH){ throw std::runtime_error("hyperdrone: this renderer has no depth output"); }
                    break;
                case RT::SEGMENTATION:
                    if constexpr (!SPEC::HAS_SEGMENTATION){ throw std::runtime_error("hyperdrone: this renderer has no segmentation output"); }
                    break;
                case RT::RGB_DEPTH:
                    if constexpr (!(SPEC::HAS_RGB && SPEC::HAS_DEPTH)){ throw std::runtime_error("hyperdrone: this renderer has no RGB+depth output"); }
                    break;
                default:
                    break;
            }
            const bool wants_image = target != RT::COLLISION;
            const bool wants_probes = target == RT::ALL || target == RT::COLLISION;
            if(wants_image){
                if(phase == RP::LAUNCH) rlt::render_launch(device, renderer);
                else if(phase == RP::SYNC) rlt::render_sync(device, renderer);
                else rlt::render(device, renderer);
            }
            if(wants_probes){
                if(phase == RP::LAUNCH) rlt::probe_launch(device, renderer);
                else if(phase == RP::SYNC) rlt::probe_sync(device, renderer);
                else rlt::probe(device, renderer);
            }
        }

        void read_frame_buffer(uint32_t* dst) override {
            if constexpr (SPEC::HAS_RGB){
                require_init();
                rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);
                std::memcpy(dst, rlt::data(renderer.frame_buffer), sizeof(uint32_t) * SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no RGB output");
            }
        }

        void read_depth_buffer(float* dst) override {
            if constexpr (SPEC::HAS_DEPTH){
                require_init();
                rlt::read_depth_buffer(device, renderer, renderer.depth_buffer);
                std::memcpy(dst, rlt::data(renderer.depth_buffer), sizeof(float) * SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no depth output");
            }
        }

        void read_segmentation_buffer(uint32_t* dst) override {
            if constexpr (SPEC::HAS_SEGMENTATION){
                require_init();
                rlt::read_segmentation_buffer(device, renderer, renderer.segmentation_buffer);
                std::memcpy(dst, rlt::data(renderer.segmentation_buffer), sizeof(uint32_t) * SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no segmentation output");
            }
        }

        void read_collision_results(float* distances, int32_t* hits) override {
            require_init();
            rlt::read_collision_results(device, renderer, renderer.collision_results);
            const rrt::CollisionResult* results = rlt::data(renderer.collision_results);
            for(TI i = 0; i < SPEC::NUM_CAMERAS * SPEC::NUM_PROBES; i++){
                distances[i] = results[i].distance;
                hits[i] = results[i].hit;
            }
        }

        uint32_t* framebuffer_device_ptr() override {
            if constexpr (SPEC::HAS_RGB){
                require_init();
                return rlt::get_framebuffer_device_ptr(device, renderer);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no RGB output");
            }
        }

        float* depthbuffer_device_ptr() override {
            if constexpr (SPEC::HAS_DEPTH){
                require_init();
                return rlt::get_depthbuffer_device_ptr(device, renderer);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no depth output");
            }
        }

        int buffer_device_type() const override {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            return 2;
#else
            return 1;
#endif
        }

        uint32_t* frame_buffer_live() override {
            return framebuffer_device_ptr();
        }

        float* depth_buffer_live() override {
            return depthbuffer_device_ptr();
        }

        uint32_t* frame_buffer_host(bool refresh) override {
            if constexpr (SPEC::HAS_RGB){
                require_init();
                if(refresh){
                    rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);
                }
                return rlt::data(renderer.frame_buffer);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no RGB output");
            }
        }

        float* depth_buffer_host(bool refresh) override {
            if constexpr (SPEC::HAS_DEPTH){
                require_init();
                if(refresh){
                    rlt::read_depth_buffer(device, renderer, renderer.depth_buffer);
                }
                return rlt::data(renderer.depth_buffer);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no depth output");
            }
        }

        uint32_t* segmentation_buffer_host(bool refresh) override {
            if constexpr (SPEC::HAS_SEGMENTATION){
                require_init();
                if(refresh){
                    rlt::read_segmentation_buffer(device, renderer, renderer.segmentation_buffer);
                }
                return rlt::data(renderer.segmentation_buffer);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no segmentation output");
            }
        }

        void save(hyperdrone::render::SaveTarget target, const char* path) override {
            require_init();
            using ST = hyperdrone::render::SaveTarget;
            switch(target){
                case ST::IMAGE:
                    if constexpr (SPEC::HAS_RGB){ rlt::save_image(device, renderer, path); return; }
                    break;
                case ST::DEPTH_IMAGE:
                    if constexpr (SPEC::HAS_DEPTH){ rlt::save_depth_image(device, renderer, path); return; }
                    break;
                case ST::DEPTH_RAW:
                    if constexpr (SPEC::HAS_DEPTH){ rlt::save_depth(device, renderer, path); return; }
                    break;
                case ST::SEGMENTATION_IMAGE:
                    if constexpr (SPEC::HAS_SEGMENTATION){ rlt::save_segmentation_image(device, renderer, path); return; }
                    break;
                case ST::PROBES:
                    rlt::save_probes(device, renderer, path);
                    return;
            }
            throw std::runtime_error("hyperdrone: save target not available in this renderer's output mode");
        }

        void scene_bounds(float center[3], float half_extent[3], float& camera_radius) const override {
            for(int i = 0; i < 3; i++){
                center[i] = renderer.scene_center[i];
                half_extent[i] = renderer.scene_half_extent[i];
            }
            camera_radius = renderer.camera_radius;
        }

        bool can_attach(size_t camera, size_t overlay) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                return rlt::can_attach(device, renderer, (TI)camera, rrt::OverlayIndex{overlay});
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void attach(size_t camera, size_t overlay) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::attach(device, renderer, (TI)camera, rrt::OverlayIndex{overlay});
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void detach(size_t camera, size_t overlay) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::detach(device, renderer, (TI)camera, rrt::OverlayIndex{overlay});
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        bool can_spawn(size_t overlay, size_t asset) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                return rlt::can_spawn(device, renderer, rrt::OverlayIndex{overlay}, rrt::AssetHandle{asset});
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        hyperdrone::render::OverlayPlacementData spawn(size_t overlay, size_t asset, const float transform[12]) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rrt::OverlayPlacement placement = rlt::spawn(device, renderer, rrt::OverlayIndex{overlay}, rrt::AssetHandle{asset}, transform);
                return hyperdrone::render::OverlayPlacementData{placement.first_slot, placement.num_parts, placement.first_part};
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void despawn(size_t overlay, const hyperdrone::render::OverlayPlacementData& placement) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rrt::OverlayPlacement p{placement.first_slot, placement.num_parts, placement.first_part};
                rlt::despawn(device, renderer, rrt::OverlayIndex{overlay}, p);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void set_transform(size_t overlay, const hyperdrone::render::OverlayPlacementData& placement, const float transform[12]) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rrt::OverlayPlacement p{placement.first_slot, placement.num_parts, placement.first_part};
                rlt::set_transform(device, renderer, rrt::OverlayIndex{overlay}, p, transform);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void set_part_transform(size_t overlay, const hyperdrone::render::OverlayPlacementData& placement, size_t part, const float transform[12]) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rrt::OverlayPlacement p{placement.first_slot, placement.num_parts, placement.first_part};
                rlt::set_transform(device, renderer, rrt::OverlayIndex{overlay}, p, (TI)part, transform);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
    };

    static char config_string_buffer[256];
    const char* build_config_string(){
        std::snprintf(config_string_buffer, sizeof(config_string_buffer),
            "w=%d;h=%d;nc=%d;np=%d;sh=%d;om=%d;mb=%d;aa=%d;no=%d;moi=%d;mopc=%d;ss=%d",
            (int)HYPERDRONE_RENDER_WIDTH, (int)HYPERDRONE_RENDER_HEIGHT, (int)HYPERDRONE_RENDER_NUM_CAMERAS, (int)HYPERDRONE_RENDER_NUM_PROBES,
            (int)HYPERDRONE_RENDER_SHADING, (int)HYPERDRONE_RENDER_OUTPUT_MODE, (int)HYPERDRONE_RENDER_MB_SAMPLES, (int)HYPERDRONE_RENDER_AA_GRID,
            (int)HYPERDRONE_RENDER_NUM_OVERLAYS, (int)HYPERDRONE_RENDER_MAX_OVERLAY_INSTANCES, (int)HYPERDRONE_RENDER_MAX_OVERLAYS_PER_CAMERA,
            (int)(HYPERDRONE_RENDER_SEMANTIC_SEGMENTATION != 0));
        return config_string_buffer;
    }
}

extern "C" {
    hyperdrone::render::Renderer* hyperdrone_render_create(){
        return new hyperdrone_render_impl::RendererImpl<hyperdrone_render_impl::RENDERER_SPEC>();
    }
    void hyperdrone_render_destroy(hyperdrone::render::Renderer* renderer){
        delete renderer;
    }
    const char* hyperdrone_render_config_string(){
        return hyperdrone_render_impl::build_config_string();
    }
    int hyperdrone_render_iface_version(){
        return HYPERDRONE_RENDER_IFACE_VERSION;
    }
}
