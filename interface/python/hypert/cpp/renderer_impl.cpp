#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>

#include "iface.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifndef HYPERT_WIDTH
#error "HYPERT_WIDTH must be defined"
#endif
#ifndef HYPERT_HEIGHT
#error "HYPERT_HEIGHT must be defined"
#endif
#ifndef HYPERT_NUM_CAMERAS
#error "HYPERT_NUM_CAMERAS must be defined"
#endif
#ifndef HYPERT_NUM_PROBES
#define HYPERT_NUM_PROBES 1
#endif
#ifndef HYPERT_SHADING
#define HYPERT_SHADING 2
#endif
#ifndef HYPERT_OUTPUT_MODE
#define HYPERT_OUTPUT_MODE 0
#endif
#ifndef HYPERT_MB_SAMPLES
#define HYPERT_MB_SAMPLES 1
#endif
#ifndef HYPERT_AA_GRID
#define HYPERT_AA_GRID 1
#endif
#ifndef HYPERT_NUM_OVERLAYS
#define HYPERT_NUM_OVERLAYS 0
#endif
#ifndef HYPERT_MAX_OVERLAY_INSTANCES
#define HYPERT_MAX_OVERLAY_INSTANCES 0
#endif
#ifndef HYPERT_MAX_OVERLAYS_PER_CAMERA
#define HYPERT_MAX_OVERLAYS_PER_CAMERA 0
#endif
#ifndef HYPERT_SEMANTIC_SEGMENTATION
#define HYPERT_SEMANTIC_SEGMENTATION 0
#endif

namespace rlt = rl_tools;
namespace rrt = rl_tools::rendering::raytracing;

namespace hypert_impl {
    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
    using T = float;
    using TI = typename DEVICE::index_t;

    template <int SHADING_ID> struct ShadingSelector;
    template <> struct ShadingSelector<0> { using type = rrt::Low; };
    template <> struct ShadingSelector<1> { using type = rrt::Medium; };
    template <> struct ShadingSelector<2> { using type = rrt::High; };
    template <> struct ShadingSelector<3> { using type = rrt::VeryHigh; };
    using SHADING = typename ShadingSelector<HYPERT_SHADING>::type;

    static constexpr rrt::OutputMode OUTPUT_MODE = static_cast<rrt::OutputMode>(HYPERT_OUTPUT_MODE);

    using RENDERER_SPEC = rrt::Specification<
        T, TI,
        HYPERT_WIDTH, HYPERT_HEIGHT, HYPERT_NUM_CAMERAS, HYPERT_NUM_PROBES,
        SHADING,
        (HYPERT_MB_SAMPLES > 1), HYPERT_MB_SAMPLES,
        (HYPERT_AA_GRID > 1), HYPERT_AA_GRID,
        OUTPUT_MODE,
        HYPERT_NUM_OVERLAYS, HYPERT_MAX_OVERLAY_INSTANCES, HYPERT_MAX_OVERLAYS_PER_CAMERA,
        HYPERT_SEMANTIC_SEGMENTATION != 0
    >;

    // template so if-constexpr branches on the spec's feature flags are discarded without
    // being instantiated on configs that lack the corresponding storage/operations
    template <typename T_SPEC>
    struct RendererImpl final : hypert::Renderer {
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

        hypert::Config config() const override {
            hypert::Config c;
            c.width = HYPERT_WIDTH;
            c.height = HYPERT_HEIGHT;
            c.num_cameras = HYPERT_NUM_CAMERAS;
            c.num_probes = HYPERT_NUM_PROBES;
            c.shading = HYPERT_SHADING;
            c.output_mode = HYPERT_OUTPUT_MODE;
            c.motion_blur_samples = HYPERT_MB_SAMPLES;
            c.anti_aliasing_grid = HYPERT_AA_GRID;
            c.num_overlays = HYPERT_NUM_OVERLAYS;
            c.max_overlay_instances = HYPERT_MAX_OVERLAY_INSTANCES;
            c.max_overlays_per_camera = HYPERT_MAX_OVERLAYS_PER_CAMERA;
            c.semantic_segmentation = HYPERT_SEMANTIC_SEGMENTATION != 0;
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
                throw std::runtime_error("hypert: renderer is already initialized; create a new Renderer for a different scene");
            }
            if(pool != nullptr){
                if constexpr (SPEC::ENABLE_OVERLAYS){
                    rlt::init(device, renderer, *scene, *pool);
                }
                else {
                    throw std::runtime_error("hypert: an asset pool requires overlays (num_overlays/max_overlay_instances/max_overlays_per_camera > 0)");
                }
            }
            else {
                rlt::init(device, renderer, *scene);
            }
            initialized = true;
        }

        void require_init() const {
            if(!initialized){
                throw std::runtime_error("hypert: renderer is not initialized (call init(scene) first)");
            }
        }

        void update() override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::update(device, renderer);
            }
            else {
                throw std::runtime_error("hypert: update() requires overlays to be enabled");
            }
        }

        void synchronize() override {
            rlt::synchronize(device, renderer);
        }

        void set_cameras(const float* cameras) override {
            std::memcpy(rlt::data(renderer.cameras), cameras, sizeof(rrt::Camera<T>) * SPEC::NUM_CAMERAS);
            rlt::set_cameras(device, renderer, renderer.cameras);
        }

        void set_motion_blur_cameras(const float* cameras_open, const float* cameras_close) override {
            if constexpr (SPEC::ENABLE_MOTION_BLUR){
                std::memcpy(rlt::data(renderer.cameras_open), cameras_open, sizeof(rrt::Camera<T>) * SPEC::NUM_CAMERAS);
                std::memcpy(rlt::data(renderer.cameras), cameras_close, sizeof(rrt::Camera<T>) * SPEC::NUM_CAMERAS);
                rlt::set_motion_blur_cameras(device, renderer, renderer.cameras_open, renderer.cameras);
            }
            else {
                throw std::runtime_error("hypert: this renderer was compiled without motion blur (motion_blur_samples <= 1)");
            }
        }

        void generate_cameras(const float center[3], float radius, const float up[3], float fov) override {
            require_init();
            rlt::generate_cameras(device, renderer, center, radius, up, fov);
        }

        void generate_probe_directions() override {
            rlt::generate_probe_directions(device, renderer);
        }

        void render(hypert::RenderTarget target, hypert::RenderPhase phase) override {
            require_init();
            using RT = hypert::RenderTarget;
            using RP = hypert::RenderPhase;
            switch(target){
                case RT::ALL:
                    if(phase == RP::LAUNCH) rlt::render_launch(device, renderer);
                    else if(phase == RP::SYNC) rlt::render_sync(device, renderer);
                    else rlt::render(device, renderer);
                    return;
                case RT::RGB:
                    if constexpr (SPEC::HAS_RGB){
                        if(phase == RP::LAUNCH) rlt::render_rgb_only_launch(device, renderer);
                        else if(phase == RP::SYNC) rlt::render_rgb_only_sync(device, renderer);
                        else rlt::render_rgb_only(device, renderer);
                        return;
                    }
                    break;
                case RT::DEPTH:
                    if constexpr (SPEC::HAS_DEPTH){
                        if(phase == RP::LAUNCH) rlt::render_depth_only_launch(device, renderer);
                        else if(phase == RP::SYNC) rlt::render_depth_only_sync(device, renderer);
                        else rlt::render_depth_only(device, renderer);
                        return;
                    }
                    break;
                case RT::SEGMENTATION:
                    if constexpr (SPEC::HAS_SEGMENTATION){
                        if(phase == RP::LAUNCH) rlt::render_segmentation_only_launch(device, renderer);
                        else if(phase == RP::SYNC) rlt::render_segmentation_only_sync(device, renderer);
                        else rlt::render_segmentation_only(device, renderer);
                        return;
                    }
                    break;
                case RT::RGB_DEPTH:
                    if constexpr (SPEC::HAS_RGB && SPEC::HAS_DEPTH){
                        if(phase == RP::LAUNCH) rlt::render_rgb_depth_only_launch(device, renderer);
                        else if(phase == RP::SYNC) rlt::render_rgb_depth_only_sync(device, renderer);
                        else rlt::render_rgb_depth_only(device, renderer);
                        return;
                    }
                    break;
                case RT::COLLISION:
                    if(phase == RP::LAUNCH) rlt::render_collision_only_launch(device, renderer);
                    else if(phase == RP::SYNC) rlt::render_collision_only_sync(device, renderer);
                    else rlt::render_collision_only(device, renderer);
                    return;
            }
            throw std::runtime_error("hypert: render target not available in this renderer's output mode");
        }

        void read_frame_buffer(uint32_t* dst) override {
            if constexpr (SPEC::HAS_RGB){
                require_init();
                rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);
                std::memcpy(dst, rlt::data(renderer.frame_buffer), sizeof(uint32_t) * SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);
            }
            else {
                throw std::runtime_error("hypert: this renderer has no RGB output");
            }
        }

        void read_depth_buffer(float* dst) override {
            if constexpr (SPEC::HAS_DEPTH){
                require_init();
                rlt::read_depth_buffer(device, renderer, renderer.depth_buffer);
                std::memcpy(dst, rlt::data(renderer.depth_buffer), sizeof(float) * SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);
            }
            else {
                throw std::runtime_error("hypert: this renderer has no depth output");
            }
        }

        void read_segmentation_buffer(uint32_t* dst) override {
            if constexpr (SPEC::HAS_SEGMENTATION){
                require_init();
                rlt::read_segmentation_buffer(device, renderer, renderer.segmentation_buffer);
                std::memcpy(dst, rlt::data(renderer.segmentation_buffer), sizeof(uint32_t) * SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);
            }
            else {
                throw std::runtime_error("hypert: this renderer has no segmentation output");
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
                throw std::runtime_error("hypert: this renderer has no RGB output");
            }
        }

        float* depthbuffer_device_ptr() override {
            if constexpr (SPEC::HAS_DEPTH){
                require_init();
                return rlt::get_depthbuffer_device_ptr(device, renderer);
            }
            else {
                throw std::runtime_error("hypert: this renderer has no depth output");
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
                throw std::runtime_error("hypert: this renderer has no RGB output");
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
                throw std::runtime_error("hypert: this renderer has no depth output");
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
                throw std::runtime_error("hypert: this renderer has no segmentation output");
            }
        }

        void save(hypert::SaveTarget target, const char* path) override {
            require_init();
            using ST = hypert::SaveTarget;
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
            throw std::runtime_error("hypert: save target not available in this renderer's output mode");
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
                throw std::runtime_error("hypert: this renderer was compiled without overlays");
            }
        }
        void attach(size_t camera, size_t overlay) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::attach(device, renderer, (TI)camera, rrt::OverlayIndex{overlay});
            }
            else {
                throw std::runtime_error("hypert: this renderer was compiled without overlays");
            }
        }
        void detach(size_t camera, size_t overlay) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::detach(device, renderer, (TI)camera, rrt::OverlayIndex{overlay});
            }
            else {
                throw std::runtime_error("hypert: this renderer was compiled without overlays");
            }
        }
        bool can_spawn(size_t overlay, size_t asset) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                return rlt::can_spawn(device, renderer, rrt::OverlayIndex{overlay}, rrt::AssetHandle{asset});
            }
            else {
                throw std::runtime_error("hypert: this renderer was compiled without overlays");
            }
        }
        hypert::OverlayPlacementData spawn(size_t overlay, size_t asset, const float transform[12]) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rrt::OverlayPlacement placement = rlt::spawn(device, renderer, rrt::OverlayIndex{overlay}, rrt::AssetHandle{asset}, transform);
                return hypert::OverlayPlacementData{placement.first_slot, placement.num_parts, placement.first_part};
            }
            else {
                throw std::runtime_error("hypert: this renderer was compiled without overlays");
            }
        }
        void despawn(size_t overlay, const hypert::OverlayPlacementData& placement) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rrt::OverlayPlacement p{placement.first_slot, placement.num_parts, placement.first_part};
                rlt::despawn(device, renderer, rrt::OverlayIndex{overlay}, p);
            }
            else {
                throw std::runtime_error("hypert: this renderer was compiled without overlays");
            }
        }
        void set_transform(size_t overlay, const hypert::OverlayPlacementData& placement, const float transform[12]) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rrt::OverlayPlacement p{placement.first_slot, placement.num_parts, placement.first_part};
                rlt::set_transform(device, renderer, rrt::OverlayIndex{overlay}, p, transform);
            }
            else {
                throw std::runtime_error("hypert: this renderer was compiled without overlays");
            }
        }
        void set_part_transform(size_t overlay, const hypert::OverlayPlacementData& placement, size_t part, const float transform[12]) override {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rrt::OverlayPlacement p{placement.first_slot, placement.num_parts, placement.first_part};
                rlt::set_transform(device, renderer, rrt::OverlayIndex{overlay}, p, (TI)part, transform);
            }
            else {
                throw std::runtime_error("hypert: this renderer was compiled without overlays");
            }
        }
    };

    static char config_string_buffer[256];
    const char* build_config_string(){
        std::snprintf(config_string_buffer, sizeof(config_string_buffer),
            "w=%d;h=%d;nc=%d;np=%d;sh=%d;om=%d;mb=%d;aa=%d;no=%d;moi=%d;mopc=%d;ss=%d",
            (int)HYPERT_WIDTH, (int)HYPERT_HEIGHT, (int)HYPERT_NUM_CAMERAS, (int)HYPERT_NUM_PROBES,
            (int)HYPERT_SHADING, (int)HYPERT_OUTPUT_MODE, (int)HYPERT_MB_SAMPLES, (int)HYPERT_AA_GRID,
            (int)HYPERT_NUM_OVERLAYS, (int)HYPERT_MAX_OVERLAY_INSTANCES, (int)HYPERT_MAX_OVERLAYS_PER_CAMERA,
            (int)(HYPERT_SEMANTIC_SEGMENTATION != 0));
        return config_string_buffer;
    }
}

extern "C" {
    hypert::Renderer* hypert_create(){
        return new hypert_impl::RendererImpl<hypert_impl::RENDERER_SPEC>();
    }
    void hypert_destroy(hypert::Renderer* renderer){
        delete renderer;
    }
    const char* hypert_config_string(){
        return hypert_impl::build_config_string();
    }
}
