#pragma once
#include <cstdint>
#include <cstddef>

// ABI boundary between the hyperdrone render core module and the JIT-compiled renderer
// libraries. Both sides are compiled from the same headers with the same compiler in the
// same build tree, so passing rl_tools scene types by pointer is safe. Everything
// renderer-spec dependent is hidden behind the vtable; buffer exchange uses raw pointers
// sized by config(). Bump HYPERDRONE_RENDER_IFACE_VERSION on any change to this file.
#define HYPERDRONE_RENDER_IFACE_VERSION 4

namespace rl_tools { namespace rendering { namespace raytracing {
    struct Scene;
    struct AssetPool;
}}}

namespace hyperdrone::render {
    struct Config {
        uint32_t width;
        uint32_t height;
        uint32_t num_cameras;
        uint32_t num_probes;
        int shading;                    // 0=low 1=medium 2=high 3=veryhigh
        int output_mode;                // 0=rgb 1=rgbd 2=depth 3=segmentation 4=rgbd_segmentation
        uint32_t motion_blur_samples;   // 1 = disabled
        uint32_t anti_aliasing_grid;    // 1 = disabled
        uint32_t num_overlays;
        uint32_t max_overlay_instances;
        uint32_t max_overlays_per_camera;
        bool semantic_segmentation;
        bool dynamic_motion_blur;
    };

    enum class RenderTarget : int {
        ALL = 0,
        RGB = 1,
        DEPTH = 2,
        SEGMENTATION = 3,
        RGB_DEPTH = 4,
        COLLISION = 5
    };
    enum class RenderPhase : int {
        LAUNCH = 0,
        SYNC = 1,
        FULL = 2
    };
    struct OverlayPlacementData {
        size_t first_slot;
        size_t num_parts;
        size_t first_part;
    };

    class Renderer {
    public:
        virtual ~Renderer() = default;
        virtual Config config() const = 0;
        virtual const char* backend() const = 0;

        virtual void init(const rl_tools::rendering::raytracing::Scene* scene, const rl_tools::rendering::raytracing::AssetPool* pool) = 0;
        virtual void update() = 0;
        virtual void synchronize() = 0;

        virtual void set_cameras(const float* cameras) = 0; // num_cameras * 12 floats (pos, dir_00, dir_du, dir_dv)
        // cameras in CUDA device memory, same packed layout; producer_stream (a
        // cudaStream_t handle, 0 = none) is awaited via an event on the render stream —
        // no host synchronization. OptiX backend only.
        virtual void set_cameras_device(const float* cameras, unsigned long long producer_stream) = 0;
        virtual void set_motion_blur_cameras(const float* cameras_open, const float* cameras_close) = 0;
        virtual void generate_cameras(const float center[3], float radius, const float up[3], float fov) = 0;
        virtual void generate_probe_directions() = 0;

        virtual void render(RenderTarget target, RenderPhase phase) = 0;

        virtual void read_frame_buffer(uint32_t* dst) = 0;        // num_cameras * height * width
        virtual void read_depth_buffer(float* dst) = 0;           // num_cameras * height * width
        virtual void read_segmentation_buffer(uint32_t* dst) = 0; // num_cameras * height * width
        virtual void read_collision_results(float* distances, int32_t* hits) = 0; // num_cameras * num_probes each
        virtual uint32_t* framebuffer_device_ptr() = 0;
        virtual float* depthbuffer_device_ptr() = 0;

        // zero-copy access. The live buffers are where rendering writes (CUDA device memory
        // on OptiX, CPU-visible memory elsewhere) and are valid after the corresponding
        // sync; the host buffers are the renderer-owned staging tensors, optionally
        // refreshed from the backend first.
        virtual int buffer_device_type() const = 0; // DLPack device type: 1 = kDLCPU, 2 = kDLCUDA
        virtual uint32_t* frame_buffer_live() = 0;
        virtual float* depth_buffer_live() = 0;
        virtual uint32_t* frame_buffer_host(bool refresh) = 0;
        virtual float* depth_buffer_host(bool refresh) = 0;
        virtual uint32_t* segmentation_buffer_host(bool refresh) = 0;

        virtual void scene_bounds(float center[3], float half_extent[3], float& camera_radius) const = 0;

        virtual bool can_attach(size_t camera, size_t overlay) = 0;
        virtual void attach(size_t camera, size_t overlay) = 0;
        virtual void detach(size_t camera, size_t overlay) = 0;
        virtual bool can_spawn(size_t overlay, size_t asset) = 0;
        virtual OverlayPlacementData spawn(size_t overlay, size_t asset, const float transform[12]) = 0;
        virtual void despawn(size_t overlay, const OverlayPlacementData& placement) = 0;
        virtual void set_transform(size_t overlay, const OverlayPlacementData& placement, const float transform[12]) = 0;
        virtual void set_part_transform(size_t overlay, const OverlayPlacementData& placement, size_t part, const float transform[12]) = 0;
        // dynamic motion blur: shutter-open/close transforms, slerped across the motion samples
        virtual void set_transform_pair(size_t overlay, const OverlayPlacementData& placement, const float open[12], const float close[12]) = 0;
        virtual void set_part_transform_pair(size_t overlay, const OverlayPlacementData& placement, size_t part, const float open[12], const float close[12]) = 0;
        // GPU-resident producer path: cameras_open/cameras_close in CUDA device memory (OptiX
        // only), the whole transforms_pair tensor ([2][num_overlays*max_instances][12] entries,
        // open then close), and the on-device expansion into the per-sample slabs
        virtual void set_motion_blur_cameras_device(const float* cameras_open, const float* cameras_close, unsigned long long producer_stream) = 0;
        virtual void set_transforms_pair(const float* pairs) = 0;
        virtual float* transforms_pair_device_ptr() = 0;
        virtual void expand_motion_transforms() = 0;
    };
}

extern "C" {
    hyperdrone::render::Renderer* hyperdrone_render_create();
    void hyperdrone_render_destroy(hyperdrone::render::Renderer* renderer);
    // canonical config string baked at compile time; the loader compares it against the
    // requested configuration to catch stale or mislabeled cache entries
    const char* hyperdrone_render_config_string();
    int hyperdrone_render_iface_version();
}
