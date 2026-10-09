#ifndef HYPERDRONE_RENDER_NUM_PROBES
#define HYPERDRONE_RENDER_NUM_PROBES 1
#endif
// num_probes=0 compiles the collision-probe pass out the way the C++ benchmark targets do: the
// backend macro removes the probe resources and launches, and the renderer specification keeps
// one probe slot because zero-sized tensors are not representable
#if HYPERDRONE_RENDER_NUM_PROBES == 0
#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 1
#endif

#include "verbosity.h"
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/operations_cpu.h>
#include <rendering/raytracing/camera_orbit.h>

#include "bindings.h"
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
#include "cuda_staging.h"
#endif

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <vector>

#ifndef HYPERDRONE_RENDER_WIDTH
#error "HYPERDRONE_RENDER_WIDTH must be defined"
#endif
#ifndef HYPERDRONE_RENDER_HEIGHT
#error "HYPERDRONE_RENDER_HEIGHT must be defined"
#endif
#ifndef HYPERDRONE_RENDER_NUM_CAMERAS
#error "HYPERDRONE_RENDER_NUM_CAMERAS must be defined"
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
#ifndef HYPERDRONE_RENDER_DYNAMIC_MB
#define HYPERDRONE_RENDER_DYNAMIC_MB 0
#endif

namespace rlt = rl_tools;
namespace rrt = rl_tools::rendering::raytracing;
using hyperdrone::FloatArray;

// Each configuration needs a distinct C++ type in nanobind's shared type registry.
namespace HYPERDRONE_MODULE_NAME {
    using DEVICE = rlt::devices::DEVICE_FACTORY<hyperdrone::CPUSpecification>;
    using T = float;
    using TI = typename DEVICE::index_t;

    template <int SHADING_ID> struct ShadingSelector;
    template <> struct ShadingSelector<0> { using type = rrt::Low; };
    template <> struct ShadingSelector<1> { using type = rrt::Medium; };
    template <> struct ShadingSelector<2> { using type = rrt::High; };
    template <> struct ShadingSelector<3> { using type = rrt::VeryHigh; };

    static constexpr bool PROBES_ENABLED = HYPERDRONE_RENDER_NUM_PROBES > 0;

    // the output-mode id (0=rgb 1=rgbd 2=depth 3=segmentation 4=rgbd_segmentation) maps
    // onto the config's independent channel switches
    struct RendererConfiguration: rrt::config::Default<T, TI>{
        static constexpr TI CAM_WIDTH = HYPERDRONE_RENDER_WIDTH;
        static constexpr TI CAM_HEIGHT = HYPERDRONE_RENDER_HEIGHT;
        static constexpr TI NUM_CAMERAS = HYPERDRONE_RENDER_NUM_CAMERAS;
        static constexpr TI NUM_PROBES = PROBES_ENABLED ? HYPERDRONE_RENDER_NUM_PROBES : 1;
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
        static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = HYPERDRONE_RENDER_DYNAMIC_MB != 0;
    };

    using RENDERER_SPEC = rrt::Specification<RendererConfiguration>;

    // renderer input/output tensors are backend-native (CUDA device memory on OptiX, host
    // elsewhere); rlt::copy against renderer.device crosses that boundary and orders the copy
    // after the renderer's in-flight work
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
    static constexpr bool BUFFERS_ON_HOST = false;
#else
    static constexpr bool BUFFERS_ON_HOST = true;
#endif

    // template so if-constexpr branches on the spec's feature flags are discarded without
    // being instantiated on configs that lack the corresponding storage/operations
    template <typename T_SPEC>
    struct Renderer {
        using SPEC = T_SPEC;
        using RENDERER = rrt::Renderer<SPEC>;
        using BACKEND = typename RENDERER::BACKEND;
        DEVICE device;
        RENDERER renderer;
        rlt::rendering::SceneMetadata<T> scene_metadata;
        bool initialized = false;

        // alias tensors shaped like the renderer tensor let the raw pointers of the arrays
        // cross the memory-domain copy without intermediate staging
        template <typename TENSOR, typename VALUE>
        void copy_in(const VALUE* source, TENSOR& tensor){
            using TENSOR_SPEC = typename TENSOR::SPEC;
            static_assert(std::is_same<VALUE, typename TENSOR_SPEC::T>::value);
            rlt::Tensor<rlt::tensor::Specification<VALUE, typename TENSOR_SPEC::TI, typename TENSOR_SPEC::SHAPE, true, rlt::tensor::RowMajorStride<typename TENSOR_SPEC::SHAPE>, true>> alias;
            alias._data = source;
            rlt::copy(device, renderer.device, alias, tensor);
        }
        template <typename TENSOR, typename VALUE>
        void copy_out(const TENSOR& tensor, VALUE* destination){
            using TENSOR_SPEC = typename TENSOR::SPEC;
            static_assert(std::is_same<VALUE, typename TENSOR_SPEC::T>::value);
            rlt::Tensor<rlt::tensor::Specification<VALUE, typename TENSOR_SPEC::TI, typename TENSOR_SPEC::SHAPE>> alias;
            alias._data = destination;
            rlt::copy(renderer.device, device, tensor, alias);
        }
        // host staging for the readback paths; unused (and never allocated) on backends
        // whose output tensors are host-resident
        std::vector<uint32_t> host_frame, host_segmentation;
        std::vector<float> host_depth;
        std::vector<rrt::CollisionResult> host_collisions;

        Renderer(){
            rlt::init(device);
            rlt::malloc(device, renderer);
        }
        ~Renderer(){
            rlt::free(device, renderer);
        }

        const char* backend() const {
            return rrt::backends::name<BACKEND>();
        }

        void init(rlt::rendering::Bundle<T>* bundle, const rrt::AssetPool* pool) {
            if(initialized){
                throw std::runtime_error("hyperdrone: renderer is already initialized; create a new Renderer for a different scene");
            }
            // the bundle may have been composed via add() (no loader-filled metadata) or
            // extended after load; recomputing is deterministic and idempotent for pure loads
            rlt::rendering::datasets::compute_bounds(device, *bundle);
            scene_metadata = bundle->metadata;
            if(pool != nullptr){
                if constexpr (SPEC::ENABLE_OVERLAYS){
                    rlt::init(device, renderer, *bundle, *pool);
                }
                else {
                    throw std::runtime_error("hyperdrone: an asset pool requires overlays (num_overlays/max_overlay_instances/max_overlays_per_camera > 0)");
                }
            }
            else {
                rlt::init(device, renderer, *bundle);
            }
            initialized = true;
        }

        void check_image_shape(size_t ndim, const size_t* shape) const {
            if(ndim != 3 || shape[0] != SPEC::NUM_CAMERAS || shape[1] != SPEC::CAM_HEIGHT || shape[2] != SPEC::CAM_WIDTH){
                throw std::invalid_argument("hyperdrone: output array must have shape (num_cameras, height, width)");
            }
        }
        void check_cameras_dims(size_t ndim, const int64_t* dims) const {
            const bool flat = ndim == 2 && dims[0] == SPEC::NUM_CAMERAS && dims[1] == 12;
            const bool structured = ndim == 3 && dims[0] == SPEC::NUM_CAMERAS && dims[1] == 4 && dims[2] == 3;
            if(!flat && !structured){
                throw std::invalid_argument("hyperdrone: cameras must have shape (num_cameras, 12) or (num_cameras, 4, 3) and dtype float32");
            }
        }
        void check_cameras_shape(const FloatArray& cameras) const {
            int64_t dims[3] = {0, 0, 0};
            for(size_t dimension = 0; dimension < cameras.ndim() && dimension < 3; dimension++){
                dims[dimension] = (int64_t)cameras.shape(dimension);
            }
            check_cameras_dims(cameras.ndim(), dims);
        }

        template <typename ARRAY>
        void check_rows(const ARRAY& values, std::initializer_list<size_t> leading, const char* expected) const {
            bool valid = values.ndim() == leading.size() + 2;
            size_t dimension = 0;
            for(size_t extent : leading){
                valid = valid && dimension < values.ndim() && values.shape(dimension) == extent;
                dimension++;
            }
            valid = valid && values.shape(dimension) == 3 && values.shape(dimension + 1) == 4;
            if(!valid){
                throw std::invalid_argument(std::string("hyperdrone: transforms must have shape ") + expected + " and dtype float32");
            }
        }

        void require_init() const {
            if(!initialized){
                throw std::runtime_error("hyperdrone: renderer is not initialized (call init(scene) first)");
            }
        }

        void update() {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::update(device, renderer);
            }
            else {
                throw std::runtime_error("hyperdrone: update() requires overlays to be enabled");
            }
        }

        void synchronize() {
            rlt::synchronize(device, renderer);
        }

        // camera input is a renderer-owned tensor written in place; under motion blur both
        // shutter poses must be written every step, so a blur-free update writes both
        void set_cameras(const float* cameras) {
            const rrt::Camera<T>* source = (const rrt::Camera<T>*)cameras;
            copy_in(source, rlt::cameras(device, renderer));
            if constexpr (SPEC::ENABLE_MOTION_BLUR){
                copy_in(source, rlt::cameras_open(device, renderer));
            }
        }

        void set_cameras_device(const float* cameras, unsigned long long producer_stream) {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            require_init();
            // enqueued on the render stream: orders after the previous launch (no overwrite
            // hazard) and before the next one; a foreign producer stream is awaited via an
            // event, never a host sync
            cudaStream_t render_stream = rlt::stream(device, renderer);
            cudaStream_t producer = (cudaStream_t)producer_stream;
            if(producer != nullptr && producer != render_stream){
                cudaEvent_t cameras_ready;
                cudaEventCreateWithFlags(&cameras_ready, cudaEventDisableTiming);
                cudaEventRecord(cameras_ready, producer);
                cudaStreamWaitEvent(render_stream, cameras_ready, 0);
                cudaEventDestroy(cameras_ready);
            }
            const size_t bytes = sizeof(rrt::Camera<T>) * SPEC::NUM_CAMERAS;
            cudaMemcpyAsync(rlt::data(rlt::cameras(device, renderer)), cameras, bytes, cudaMemcpyDeviceToDevice, render_stream);
            if constexpr (SPEC::ENABLE_MOTION_BLUR){
                cudaMemcpyAsync(rlt::data(rlt::cameras_open(device, renderer)), cameras, bytes, cudaMemcpyDeviceToDevice, render_stream);
            }
#else
            (void)cameras; (void)producer_stream;
            throw std::runtime_error("hyperdrone: device-resident camera input is only supported on the OptiX backend");
#endif
        }

        void set_motion_blur_cameras(const float* cameras_open, const float* cameras_close) {
            if constexpr (SPEC::ENABLE_MOTION_BLUR){
                copy_in((const rrt::Camera<T>*)cameras_open, rlt::cameras_open(device, renderer));
                copy_in((const rrt::Camera<T>*)cameras_close, rlt::cameras_close(device, renderer));
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without motion blur (motion_blur_samples <= 1)");
            }
        }

        void generate_cameras(const float center[3], float radius, const float up[3], float fov) {
            require_init();
            camera_orbit::write(device, renderer, center, radius, up, fov);
        }

        void generate_probe_directions() {
            if constexpr (PROBES_ENABLED){
                rlt::generate_probe_directions(device, renderer);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without collision probes (num_probes=0)");
            }
        }

        void render() {
            require_init();
            rlt::render(device, renderer);
        }

        void render_launch() {
            require_init();
            rlt::render_launch(device, renderer);
        }

        void render_sync() {
            require_init();
            rlt::render_sync(device, renderer);
        }

        void probe() {
            require_init();
            if constexpr (PROBES_ENABLED){
                rlt::probe(device, renderer);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without collision probes (num_probes=0)");
            }
        }

        void probe_launch() {
            require_init();
            if constexpr (PROBES_ENABLED){
                rlt::probe_launch(device, renderer);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without collision probes (num_probes=0)");
            }
        }

        void probe_sync() {
            require_init();
            if constexpr (PROBES_ENABLED){
                rlt::probe_sync(device, renderer);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without collision probes (num_probes=0)");
            }
        }

        void read_frame_buffer(uint32_t* dst) {
            if constexpr (SPEC::HAS_RGB){
                require_init();
                copy_out(rlt::frame_buffer(device, renderer), dst);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no RGB output");
            }
        }

        void read_depth_buffer(float* dst) {
            if constexpr (SPEC::HAS_DEPTH){
                require_init();
                copy_out(rlt::depth_buffer(device, renderer), dst);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no depth output");
            }
        }

        void read_segmentation_buffer(uint32_t* dst) {
            if constexpr (SPEC::HAS_SEGMENTATION){
                require_init();
                copy_out(rlt::segmentation_buffer(device, renderer), dst);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no segmentation output");
            }
        }

        void read_collision_results(float* distances, int32_t* hits) {
            if constexpr (PROBES_ENABLED){
                require_init();
                constexpr TI count = SPEC::NUM_CAMERAS * SPEC::NUM_PROBES;
                host_collisions.resize(count);
                copy_out(rlt::collision_results(device, renderer), host_collisions.data());
                for(TI i = 0; i < count; i++){
                    distances[i] = host_collisions[i].distance;
                    hits[i] = host_collisions[i].hit;
                }
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without collision probes (num_probes=0)");
            }
        }

        uint32_t* framebuffer_device_ptr() {
            if constexpr (SPEC::HAS_RGB){
                require_init();
                return rlt::data(rlt::frame_buffer(device, renderer));
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no RGB output");
            }
        }

        float* depthbuffer_device_ptr() {
            if constexpr (SPEC::HAS_DEPTH){
                require_init();
                return rlt::data(rlt::depth_buffer(device, renderer));
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no depth output");
            }
        }

        int buffer_device_type() const {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            return 2;
#else
            return 1;
#endif
        }

        uint32_t* frame_buffer_live() {
            return framebuffer_device_ptr();
        }

        float* depth_buffer_live() {
            return depthbuffer_device_ptr();
        }

        // stable host-side view of an output channel. On host-resident backends that is the
        // renderer's own tensor (zero copy, refreshed in place by rendering); on OptiX it is
        // an impl-owned staging buffer refreshed from device memory.
        template <typename VALUE, typename TENSOR>
        VALUE* host_view(std::vector<VALUE>& staging, TENSOR& tensor, bool refresh){
            if constexpr (BUFFERS_ON_HOST){
                (void)staging; (void)refresh;
                return rlt::data(tensor);
            }
            else {
                staging.resize(SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS);
                if(refresh){
                    copy_out(tensor, staging.data());
                }
                return staging.data();
            }
        }

        uint32_t* frame_buffer_host(bool refresh) {
            if constexpr (SPEC::HAS_RGB){
                require_init();
                return host_view(host_frame, rlt::frame_buffer(device, renderer), refresh);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no RGB output");
            }
        }

        float* depth_buffer_host(bool refresh) {
            if constexpr (SPEC::HAS_DEPTH){
                require_init();
                return host_view(host_depth, rlt::depth_buffer(device, renderer), refresh);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no depth output");
            }
        }

        uint32_t* segmentation_buffer_host(bool refresh) {
            if constexpr (SPEC::HAS_SEGMENTATION){
                require_init();
                return host_view(host_segmentation, rlt::segmentation_buffer(device, renderer), refresh);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no segmentation output");
            }
        }

        void scene_bounds(float center[3], float half_extent[3], float& camera_radius) const {
            for(int i = 0; i < 3; i++){
                center[i] = scene_metadata.center[i];
                half_extent[i] = scene_metadata.half_extent[i];
            }
            camera_radius = scene_metadata.max_ray_length / 2;
        }

        bool can_attach(size_t camera, size_t overlay) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                return rlt::can_attach(device, renderer, (TI)camera, rrt::OverlayIndex{overlay});
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void attach(size_t camera, size_t overlay) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::attach(device, renderer, (TI)camera, rrt::OverlayIndex{overlay});
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void detach(size_t camera, size_t overlay) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::detach(device, renderer, (TI)camera, rrt::OverlayIndex{overlay});
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        bool can_spawn(size_t overlay, size_t asset) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                return rlt::can_spawn(device, renderer, rrt::OverlayIndex{overlay}, rrt::AssetHandle{asset});
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        rrt::OverlayPlacement spawn(size_t overlay, size_t asset, const float transform[12]) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                return rlt::spawn(device, renderer, rrt::OverlayIndex{overlay}, rrt::AssetHandle{asset}, transform);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void despawn(size_t overlay, const rrt::OverlayPlacement& placement) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::despawn(device, renderer, rrt::OverlayIndex{overlay}, placement);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        // the core verbs fail fast on a placement that is not spawned; a binding throws instead
        rrt::OverlayPlacement checked_placement(size_t overlay, const rrt::OverlayPlacement& placement) const {
            require_init();
            if constexpr (SPEC::ENABLE_OVERLAYS){
                if(overlay >= SPEC::NUM_OVERLAYS || placement.first_slot + placement.num_parts > SPEC::MAX_OVERLAY_INSTANCES){
                    throw std::invalid_argument("hyperdrone: overlay or placement out of range");
                }
                for(size_t part = 0; part < placement.num_parts; part++){
                    if(!renderer.overlays[overlay].slots[placement.first_slot + part].active){
                        throw std::invalid_argument("hyperdrone: placement is not spawned (despawned or never spawned in this overlay)");
                    }
                }
            }
            return rrt::OverlayPlacement{placement.first_slot, placement.num_parts, placement.first_part};
        }
        void set_transform(size_t overlay, const rrt::OverlayPlacement& placement, const float transform[12]) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                rlt::set_transform(device, renderer, rrt::OverlayIndex{overlay}, checked_placement(overlay, placement), transform);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void set_transform_pair(size_t overlay, const rrt::OverlayPlacement& placement, const float open[12], const float close[12]) {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                rlt::set_transform_pair(device, renderer, rrt::OverlayIndex{overlay}, checked_placement(overlay, placement), open, close);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without dynamic motion blur (dynamic_motion_blur=False)");
            }
        }
        void set_transforms(size_t overlay, const rrt::OverlayPlacement& placement, const float* poses) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                rlt::set_transforms(device, renderer, rrt::OverlayIndex{overlay}, checked_placement(overlay, placement), poses);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void set_motion_transforms(size_t overlay, const rrt::OverlayPlacement& placement, const float* samples, const float* close) {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                rlt::set_motion_transforms(device, renderer, rrt::OverlayIndex{overlay}, checked_placement(overlay, placement), samples, close);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without dynamic motion blur (dynamic_motion_blur=False)");
            }
        }
        template <typename WRITE>
        void for_each_active_slot(WRITE&& write){
            for(size_t overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
                for(size_t slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                    if(renderer.overlays[overlay].slots[slot].active){
                        write(overlay, slot, overlay * SPEC::MAX_OVERLAY_INSTANCES + slot);
                    }
                }
            }
        }
        void set_all_transforms(const float* poses) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                for_each_active_slot([&](size_t overlay, size_t slot, size_t row){
                    rlt::set_transforms(device, renderer, rrt::OverlayIndex{overlay}, rrt::OverlayPlacement{slot, 1, 0}, poses + row * 12);
                });
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void set_all_motion_transforms(const float* samples, const float* close) {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                require_init();
                constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
                for_each_active_slot([&](size_t overlay, size_t slot, size_t row){
                    float rows[SPEC::MOTION_BLUR_SAMPLES][12];
                    for(size_t sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                        std::memcpy(rows[sample], samples + (sample * SLOTS + row) * 12, sizeof(rows[sample]));
                    }
                    rlt::set_motion_transforms(device, renderer, rrt::OverlayIndex{overlay}, rrt::OverlayPlacement{slot, 1, 0}, &rows[0][0], close + row * 12);
                });
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without dynamic motion blur (dynamic_motion_blur=False)");
            }
        }
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
        // device-resident rows go device-to-device into the resident tensors on the render stream
        // (sample_stride rows between the sample slabs of samples; null samples replicate poses);
        // the rows leave the host's pending state, so a later update cannot overwrite them
        void write_rows_device(size_t first_row, size_t count, const float* poses, const float* samples, size_t sample_stride, unsigned long long producer_stream){
            constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
            cudaStream_t render_stream = rlt::stream(device, renderer);
            rrt::backends::optix::await_producer(renderer.backend->ordering_event, (cudaStream_t)producer_stream, render_stream);
            cudaMemcpyAsync(rlt::data(rlt::transforms(device, renderer)) + first_row * 12, poses, count * 12 * sizeof(float), cudaMemcpyDeviceToDevice, render_stream);
            for(size_t row = first_row; row < first_row + count; row++){
                renderer.transforms_pending[row] = false;
            }
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                for(size_t sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                    const float* source = samples == nullptr ? poses : samples + sample * sample_stride * 12;
                    cudaMemcpyAsync(rlt::data(rlt::transforms_motion(device, renderer)) + (sample * SLOTS + first_row) * 12, source, count * 12 * sizeof(float), cudaMemcpyDeviceToDevice, render_stream);
                }
                for(size_t row = first_row; row < first_row + count; row++){
                    renderer.transforms_motion_pending[row] = false;
                }
            }
        }
#endif
        void set_transforms_device(size_t overlay, const rrt::OverlayPlacement& placement, const float* poses, unsigned long long producer_stream) {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            if constexpr (SPEC::ENABLE_OVERLAYS){
                const auto p = checked_placement(overlay, placement);
                write_rows_device(overlay * SPEC::MAX_OVERLAY_INSTANCES + p.first_slot, p.num_parts, poses, nullptr, 0, producer_stream);
                return;
            }
#endif
            (void)overlay; (void)placement; (void)poses; (void)producer_stream;
            throw std::runtime_error("hyperdrone: device-resident transform input requires the OptiX backend and overlays");
        }
        void set_motion_transforms_device(size_t overlay, const rrt::OverlayPlacement& placement, const float* samples, const float* close, unsigned long long producer_stream) {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                const auto p = checked_placement(overlay, placement);
                write_rows_device(overlay * SPEC::MAX_OVERLAY_INSTANCES + p.first_slot, p.num_parts, close, samples, p.num_parts, producer_stream);
                return;
            }
#endif
            (void)overlay; (void)placement; (void)samples; (void)close; (void)producer_stream;
            throw std::runtime_error("hyperdrone: device-resident motion input requires the OptiX backend and dynamic_motion_blur=True");
        }
        void set_all_transforms_device(const float* poses, unsigned long long producer_stream) {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                write_rows_device(0, (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES, poses, nullptr, 0, producer_stream);
                return;
            }
#endif
            (void)poses; (void)producer_stream;
            throw std::runtime_error("hyperdrone: device-resident transform input requires the OptiX backend and overlays");
        }
        void set_all_motion_transforms_device(const float* samples, const float* close, unsigned long long producer_stream) {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                require_init();
                constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
                write_rows_device(0, SLOTS, close, samples, SLOTS, producer_stream);
                return;
            }
#endif
            (void)samples; (void)close; (void)producer_stream;
            throw std::runtime_error("hyperdrone: device-resident motion input requires the OptiX backend and dynamic_motion_blur=True");
        }
        float* transforms_live() {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                return rlt::data(rlt::transforms(device, renderer));
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        float* transforms_motion_live() {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                require_init();
                return rlt::data(rlt::transforms_motion(device, renderer));
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without dynamic motion blur (dynamic_motion_blur=False)");
            }
        }
        unsigned long long render_stream() {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            require_init();
            return (unsigned long long)rlt::stream(device, renderer);
#else
            return 0;
#endif
        }
        void set_motion_blur_cameras_device(const float* cameras_open, const float* cameras_close, unsigned long long producer_stream) {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            if constexpr (SPEC::ENABLE_MOTION_BLUR){
                require_init();
                cudaStream_t render_stream = rlt::stream(device, renderer);
                cudaStream_t producer = (cudaStream_t)producer_stream;
                if(producer != nullptr && producer != render_stream){
                    cudaEvent_t cameras_ready;
                    cudaEventCreateWithFlags(&cameras_ready, cudaEventDisableTiming);
                    cudaEventRecord(cameras_ready, producer);
                    cudaStreamWaitEvent(render_stream, cameras_ready, 0);
                    cudaEventDestroy(cameras_ready);
                }
                const size_t bytes = sizeof(rrt::Camera<T>) * SPEC::NUM_CAMERAS;
                cudaMemcpyAsync(rlt::data(rlt::cameras_open(device, renderer)), cameras_open, bytes, cudaMemcpyDeviceToDevice, render_stream);
                cudaMemcpyAsync(rlt::data(rlt::cameras_close(device, renderer)), cameras_close, bytes, cudaMemcpyDeviceToDevice, render_stream);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without motion blur (motion_blur_samples <= 1)");
            }
#else
            (void)cameras_open; (void)cameras_close; (void)producer_stream;
            throw std::runtime_error("hyperdrone: device-resident camera input is only supported on the OptiX backend");
#endif
        }
    };
}

namespace nb = nanobind;
using hyperdrone::Transform;
using hyperdrone::Vec3;
using hyperdrone::extract_transform;
using hyperdrone::make_owned_array;

NB_MODULE(HYPERDRONE_MODULE_NAME, m){
    m.doc() = "JIT-specialized RLtools raytracer";
    m.def("_set_verbosity", &hyperdrone::set_verbosity);

    using Renderer = HYPERDRONE_MODULE_NAME::Renderer<HYPERDRONE_MODULE_NAME::RENDERER_SPEC>;
    using SPEC = Renderer::SPEC;
    auto renderer_class = nb::class_<Renderer>(m, "Renderer");
    renderer_class
        .def(nb::init<>())
        .def_prop_ro("backend", [](const Renderer& self){ return std::string(self.backend()); })
        .def("init", [](Renderer& self, rrt::Bundle<float>& bundle, const rrt::AssetPool* pool){
            nb::gil_scoped_release release;
            self.init(&bundle, pool);
        }, nb::arg("scene"), nb::arg("asset_pool").none() = nb::none(), nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def("update", &Renderer::update, nb::call_guard<nb::gil_scoped_release>())
        .def("synchronize", &Renderer::synchronize, nb::call_guard<nb::gil_scoped_release>())
        .def("set_cameras", [](Renderer& self, FloatArray cameras){
            self.check_cameras_shape(cameras);
            self.set_cameras(cameras.data());
        }, nb::arg("cameras"))
        .def("set_motion_blur_cameras", [](Renderer& self, FloatArray cameras_open, FloatArray cameras_close){
            self.check_cameras_shape(cameras_open);
            self.check_cameras_shape(cameras_close);
            self.set_motion_blur_cameras(cameras_open.data(), cameras_close.data());
        }, nb::arg("cameras_open"), nb::arg("cameras_close"))
        .def("generate_cameras", [](Renderer& self, Vec3 center, float radius, Vec3 up, float fov){
            self.generate_cameras(center.data(), radius, up.data(), fov);
        }, nb::arg("center"), nb::arg("radius"), nb::arg("up"), nb::arg("fov"))
        .def("generate_probe_directions", &Renderer::generate_probe_directions)
        .def("render", &Renderer::render, nb::call_guard<nb::gil_scoped_release>())
        .def("render_launch", &Renderer::render_launch, nb::call_guard<nb::gil_scoped_release>())
        .def("render_sync", &Renderer::render_sync, nb::call_guard<nb::gil_scoped_release>())
        .def("probe", &Renderer::probe, nb::call_guard<nb::gil_scoped_release>())
        .def("probe_launch", &Renderer::probe_launch, nb::call_guard<nb::gil_scoped_release>())
        .def("probe_sync", &Renderer::probe_sync, nb::call_guard<nb::gil_scoped_release>())
        .def("read_frame_buffer", [](Renderer& self, nb::ndarray<uint32_t, nb::c_contig, nb::device::cpu> out){
            size_t shape[3] = {(size_t)out.shape(0), out.ndim() > 1 ? (size_t)out.shape(1) : 0, out.ndim() > 2 ? (size_t)out.shape(2) : 0};
            self.check_image_shape(out.ndim(), shape);
            nb::gil_scoped_release release;
            self.read_frame_buffer(out.data());
        }, nb::arg("out"))
        .def("read_depth_buffer", [](Renderer& self, nb::ndarray<float, nb::c_contig, nb::device::cpu> out){
            size_t shape[3] = {(size_t)out.shape(0), out.ndim() > 1 ? (size_t)out.shape(1) : 0, out.ndim() > 2 ? (size_t)out.shape(2) : 0};
            self.check_image_shape(out.ndim(), shape);
            nb::gil_scoped_release release;
            self.read_depth_buffer(out.data());
        }, nb::arg("out"))
        .def("read_segmentation_buffer", [](Renderer& self, nb::ndarray<uint32_t, nb::c_contig, nb::device::cpu> out){
            size_t shape[3] = {(size_t)out.shape(0), out.ndim() > 1 ? (size_t)out.shape(1) : 0, out.ndim() > 2 ? (size_t)out.shape(2) : 0};
            self.check_image_shape(out.ndim(), shape);
            nb::gil_scoped_release release;
            self.read_segmentation_buffer(out.data());
        }, nb::arg("out"))
        .def("read_collision_results", [](Renderer& self, nb::ndarray<float, nb::c_contig, nb::device::cpu> distances, nb::ndarray<int32_t, nb::c_contig, nb::device::cpu> hits){
            const size_t expected = (size_t)SPEC::NUM_CAMERAS * HYPERDRONE_RENDER_NUM_PROBES;
            if(distances.size() != expected || hits.size() != expected){
                throw std::invalid_argument("hyperdrone: collision output arrays must have num_cameras * num_probes elements");
            }
            nb::gil_scoped_release release;
            self.read_collision_results(distances.data(), hits.data());
        }, nb::arg("distances"), nb::arg("hits"))
        .def("framebuffer_device_ptr", [](Renderer& self){ return (uintptr_t)self.framebuffer_device_ptr(); })
        .def("depthbuffer_device_ptr", [](Renderer& self){ return (uintptr_t)self.depthbuffer_device_ptr(); })
        .def("frame_view", [](Renderer& self, bool refresh){
            uint32_t* pointer;
            {
                nb::gil_scoped_release release;
                pointer = self.frame_buffer_host(refresh);
            }
            return nb::ndarray<nb::numpy, uint32_t>(pointer, {SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH}, nb::find(&self));
        }, nb::arg("refresh") = true)
        .def("depth_view", [](Renderer& self, bool refresh){
            float* pointer;
            {
                nb::gil_scoped_release release;
                pointer = self.depth_buffer_host(refresh);
            }
            return nb::ndarray<nb::numpy, float>(pointer, {SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH}, nb::find(&self));
        }, nb::arg("refresh") = true)
        .def("segmentation_view", [](Renderer& self, bool refresh){
            uint32_t* pointer;
            {
                nb::gil_scoped_release release;
                pointer = self.segmentation_buffer_host(refresh);
            }
            return nb::ndarray<nb::numpy, uint32_t>(pointer, {SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH}, nb::find(&self));
        }, nb::arg("refresh") = true)
        .def("frame_dlpack", [](Renderer& self){
            uint8_t* pointer = reinterpret_cast<uint8_t*>(self.frame_buffer_live());
            return nb::ndarray<>(pointer, {SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH, 4}, nb::find(&self), {},
                                 nb::dtype<uint8_t>(), self.buffer_device_type(), 0);
        })
        .def("frame_raw_dlpack", [](Renderer& self){
            uint32_t* pointer = self.frame_buffer_live();
            return nb::ndarray<>(pointer, {SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH}, nb::find(&self), {},
                                 nb::dtype<uint32_t>(), self.buffer_device_type(), 0);
        })
        .def("depth_dlpack", [](Renderer& self){
            float* pointer = self.depth_buffer_live();
            return nb::ndarray<>(pointer, {SPEC::NUM_CAMERAS, SPEC::CAM_HEIGHT, SPEC::CAM_WIDTH}, nb::find(&self), {},
                                 nb::dtype<float>(), self.buffer_device_type(), 0);
        })
        .def("buffer_device_type", [](Renderer& self){ return self.buffer_device_type(); })
        .def("scene_bounds", [](Renderer& self){
            float center[3], half_extent[3], camera_radius;
            self.scene_bounds(center, half_extent, camera_radius);
            return nb::make_tuple(make_owned_array(center, {3}), make_owned_array(half_extent, {3}), camera_radius);
        })
        .def("can_attach", [](Renderer& self, size_t camera, size_t overlay){ return self.can_attach(camera, overlay); })
        .def("attach", [](Renderer& self, size_t camera, size_t overlay){ self.attach(camera, overlay); })
        .def("detach", [](Renderer& self, size_t camera, size_t overlay){ self.detach(camera, overlay); })
        .def("can_spawn", [](Renderer& self, size_t overlay, size_t asset){ return self.can_spawn(overlay, asset); })
        .def("spawn", [](Renderer& self, size_t overlay, size_t asset, Transform transform){
            float values[12];
            extract_transform(transform, values);
            rrt::OverlayPlacement placement = self.spawn(overlay, asset, values);
            return nb::make_tuple(placement.first_slot, placement.num_parts, placement.first_part);
        }, nb::arg("overlay"), nb::arg("asset"), nb::arg("transform"))
        .def("despawn", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement){
            self.despawn(overlay, rrt::OverlayPlacement{placement[0], placement[1], placement[2]});
        }, nb::arg("overlay"), nb::arg("placement"))
        .def("set_transform", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement, Transform transform){
            float values[12];
            extract_transform(transform, values);
            self.set_transform(overlay, rrt::OverlayPlacement{placement[0], placement[1], placement[2]}, values);
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("transform"))
        .def("set_transform_pair", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement, Transform open, Transform close){
            float open_values[12], close_values[12];
            extract_transform(open, open_values);
            extract_transform(close, close_values);
            self.set_transform_pair(overlay, rrt::OverlayPlacement{placement[0], placement[1], placement[2]}, open_values, close_values);
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("open"), nb::arg("close"))
        .def("set_transforms", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement, FloatArray poses){
            self.check_rows(poses, {placement[1]}, "(num_parts, 3, 4)");
            self.set_transforms(overlay, rrt::OverlayPlacement{placement[0], placement[1], placement[2]}, poses.data());
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("poses"))
        .def("set_motion_transforms", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement, FloatArray samples, FloatArray close){
            self.check_rows(samples, {SPEC::MOTION_BLUR_SAMPLES, placement[1]}, "(motion_blur_samples, num_parts, 3, 4)");
            self.check_rows(close, {placement[1]}, "(num_parts, 3, 4)");
            self.set_motion_transforms(overlay, rrt::OverlayPlacement{placement[0], placement[1], placement[2]}, samples.data(), close.data());
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("samples"), nb::arg("close"))
        .def("set_all_transforms", [](Renderer& self, FloatArray poses){
            self.check_rows(poses, {SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES}, "(num_overlays, max_overlay_instances, 3, 4)");
            self.set_all_transforms(poses.data());
        }, nb::arg("poses"))
        .def("set_all_motion_transforms", [](Renderer& self, FloatArray samples, FloatArray close){
            self.check_rows(samples, {SPEC::MOTION_BLUR_SAMPLES, SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES}, "(motion_blur_samples, num_overlays, max_overlay_instances, 3, 4)");
            self.check_rows(close, {SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES}, "(num_overlays, max_overlay_instances, 3, 4)");
            self.set_all_motion_transforms(samples.data(), close.data());
        }, nb::arg("samples"), nb::arg("close"))
        .def("transforms_dlpack", [](Renderer& self){
            float* pointer = self.transforms_live();
            return nb::ndarray<>(pointer, {SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, 3, 4}, nb::find(&self), {},
                                 nb::dtype<float>(), self.buffer_device_type(), 0);
        })
        .def("transforms_motion_dlpack", [](Renderer& self){
            float* pointer = self.transforms_motion_live();
            return nb::ndarray<>(pointer, {SPEC::MOTION_BLUR_SAMPLES, SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, 3, 4}, nb::find(&self), {},
                                 nb::dtype<float>(), self.buffer_device_type(), 0);
        })
        .def("render_stream", [](Renderer& self){ return (uintptr_t)self.render_stream(); });

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
    using CudaFloatArray = nb::ndarray<const float, nb::c_contig, nb::device::cuda>;
    renderer_class
        .def("set_transforms_device", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement, CudaFloatArray poses, uintptr_t stream){
            self.check_rows(poses, {placement[1]}, "(num_parts, 3, 4)");
            self.set_transforms_device(overlay, rrt::OverlayPlacement{placement[0], placement[1], placement[2]}, poses.data(), stream);
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("poses"), nb::arg("stream") = 0)
        .def("set_motion_transforms_device", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement, CudaFloatArray samples, CudaFloatArray close, uintptr_t stream){
            self.check_rows(samples, {SPEC::MOTION_BLUR_SAMPLES, placement[1]}, "(motion_blur_samples, num_parts, 3, 4)");
            self.check_rows(close, {placement[1]}, "(num_parts, 3, 4)");
            self.set_motion_transforms_device(overlay, rrt::OverlayPlacement{placement[0], placement[1], placement[2]}, samples.data(), close.data(), stream);
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("samples"), nb::arg("close"), nb::arg("stream") = 0)
        .def("set_all_transforms_device", [](Renderer& self, CudaFloatArray poses, uintptr_t stream){
            self.check_rows(poses, {SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES}, "(num_overlays, max_overlay_instances, 3, 4)");
            self.set_all_transforms_device(poses.data(), stream);
        }, nb::arg("poses"), nb::arg("stream") = 0)
        .def("set_all_motion_transforms_device", [](Renderer& self, CudaFloatArray samples, CudaFloatArray close, uintptr_t stream){
            self.check_rows(samples, {SPEC::MOTION_BLUR_SAMPLES, SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES}, "(motion_blur_samples, num_overlays, max_overlay_instances, 3, 4)");
            self.check_rows(close, {SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES}, "(num_overlays, max_overlay_instances, 3, 4)");
            self.set_all_motion_transforms_device(samples.data(), close.data(), stream);
        }, nb::arg("samples"), nb::arg("close"), nb::arg("stream") = 0)
        .def("set_cameras_device", [](Renderer& self, nb::ndarray<const float, nb::c_contig, nb::device::cuda> cameras, uintptr_t stream){
            int64_t dims[3] = {0, 0, 0};
            for(size_t dimension = 0; dimension < cameras.ndim() && dimension < 3; dimension++){
                dims[dimension] = (int64_t)cameras.shape(dimension);
            }
            self.check_cameras_dims(cameras.ndim(), dims);
            self.set_cameras_device(cameras.data(), stream);
        }, nb::arg("cameras"), nb::arg("stream") = 0)
        // benchmark fast path: index a pre-uploaded set without any per-call DLPack traffic
        .def("set_cameras_from_cuda_buffer", [](Renderer& self, hyperdrone::CudaBuffer& buffer, size_t index, uintptr_t stream){
            if(buffer.slice_elements() != (size_t)SPEC::NUM_CAMERAS * 12){
                throw std::invalid_argument("hyperdrone: CudaBuffer slices must hold num_cameras * 12 floats");
            }
            self.set_cameras_device(buffer.slice_pointer(index), stream);
        }, nb::arg("buffer"), nb::arg("index"), nb::arg("stream") = 0)
        .def("set_motion_blur_cameras_device", [](Renderer& self, nb::ndarray<const float, nb::c_contig, nb::device::cuda> cameras_open, nb::ndarray<const float, nb::c_contig, nb::device::cuda> cameras_close, uintptr_t stream){
            int64_t dims[3] = {0, 0, 0};
            for(size_t dimension = 0; dimension < cameras_open.ndim() && dimension < 3; dimension++){
                dims[dimension] = (int64_t)cameras_open.shape(dimension);
            }
            self.check_cameras_dims(cameras_open.ndim(), dims);
            for(size_t dimension = 0; dimension < cameras_close.ndim() && dimension < 3; dimension++){
                dims[dimension] = (int64_t)cameras_close.shape(dimension);
            }
            self.check_cameras_dims(cameras_close.ndim(), dims);
            self.set_motion_blur_cameras_device(cameras_open.data(), cameras_close.data(), stream);
        }, nb::arg("cameras_open"), nb::arg("cameras_close"), nb::arg("stream") = 0);
#endif
}
