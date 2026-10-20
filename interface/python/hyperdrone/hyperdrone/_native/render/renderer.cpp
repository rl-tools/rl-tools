#include "renderer_config.h"
#include <rl_tools/rendering/datasets/operations_cpu.h>
#include <rendering/raytracing/camera_orbit.h>

#include "bindings.h"
#include "output_bindings.h"
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
#include "cuda_staging.h"
#include "set_pose_device.h"
#endif

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <stdexcept>
#include <vector>

using hyperdrone::FloatArray;
using rl_tools::data;
using rl_tools::output;
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
using rl_tools::stream;
using rl_tools::collision_stream;
#endif

namespace hyperdrone {
    inline rrt::Pose make_pose(const Transform& base, float scale){
        rrt::Pose pose;
        extract_transform(base, pose.base);
        if(!rlt::rendering::is_finite(pose.base, 12) || !rlt::rendering::is_finite(scale) || scale <= 0){
            throw std::invalid_argument("hyperdrone: a pose needs a finite base and a finite positive scale");
        }
        pose.scale = scale;
        return pose;
    }
    // the asset's joint values, borrowed from the caller's array for the duration of the call;
    // nullopt is the rest state
    inline const float* check_coordinates(const std::optional<FloatArray>& coordinates, size_t num_coordinates){
        if(!coordinates.has_value()){
            return nullptr;
        }
        if(coordinates->ndim() != 1 || coordinates->shape(0) != num_coordinates){
            throw std::invalid_argument("hyperdrone: coordinates must have shape (" + std::to_string(num_coordinates) + ",), the asset's num_coordinates");
        }
        if(!rlt::rendering::is_finite(coordinates->data(), (unsigned)num_coordinates)){
            throw std::invalid_argument("hyperdrone: coordinates must be finite");
        }
        return coordinates->data();
    }
}


namespace HYPERDRONE_MODULE_NAME {
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
        bool flow_initialized = false;
        bool images_acquired = false;
        bool probes_acquired = false;
        int device_id = 0;
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
        // the placement table of the device batches, grown on demand
        size_t* placements_device = nullptr;
        size_t placements_capacity = 0;
#endif
        const std::vector<hyperdrone::OutputMetadata> output_layouts = hyperdrone::output_metadata<SPEC>();

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
        Renderer(){
            rlt::init(device);
            rlt::malloc(device, renderer);
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            cudaPointerAttributes attributes{};
            if(cudaPointerGetAttributes(&attributes, rlt::data(rlt::cameras(device, renderer))) != cudaSuccess){
                rlt::free(device, renderer);
                throw std::runtime_error("hyperdrone: unable to determine renderer CUDA device");
            }
            device_id = attributes.device;
#endif
        }
        ~Renderer(){
            rlt::free(device, renderer);
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            cudaFree(placements_device);
#endif
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
                    for(const auto& assembly : pool->assemblies){
                        if(assembly.parts.empty() || assembly.parts.size() > SPEC::MAX_OVERLAY_INSTANCES){
                            throw std::invalid_argument("hyperdrone: asset part count must be between 1 and max_overlay_instances");
                        }
                    }
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


        void require_init() const {
            if(!initialized){
                throw std::runtime_error("hyperdrone: renderer is not initialized (call init(scene) first)");
            }
        }
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
        void await_producer(unsigned long long producer_stream, cudaStream_t render_stream){
            rlt::devices::cuda::order_after(renderer.backend->ordering_event, (cudaStream_t)producer_stream, render_stream);
        }
#endif

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
            require_init();
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
            rlt::devices::cuda::order_after(renderer.backend->ordering_event, rlt::collision_stream(device, renderer), render_stream);
            await_producer(producer_stream, render_stream);
            const size_t bytes = sizeof(rrt::Camera<T>) * SPEC::NUM_CAMERAS;
            cudaMemcpyAsync(rlt::data(rlt::cameras(device, renderer)), cameras, bytes, cudaMemcpyDeviceToDevice, render_stream);
            if constexpr (SPEC::ENABLE_MOTION_BLUR){
                cudaMemcpyAsync(rlt::data(rlt::cameras_open(device, renderer)), cameras, bytes, cudaMemcpyDeviceToDevice, render_stream);
            }
            rlt::devices::cuda::order_after(renderer.backend->ordering_event, render_stream, rlt::collision_stream(device, renderer));
#else
            (void)cameras; (void)producer_stream;
            throw std::runtime_error("hyperdrone: device-resident camera input is only supported on the OptiX backend");
#endif
        }

        void set_motion_blur_cameras(const float* cameras_open, const float* cameras_close) {
            require_init();
            if constexpr (SPEC::ENABLE_MOTION_BLUR){
                copy_in((const rrt::Camera<T>*)cameras_open, rlt::cameras_open(device, renderer));
                copy_in((const rrt::Camera<T>*)cameras_close, rlt::cameras_close(device, renderer));
            }
            else if constexpr(RendererConfiguration::ENABLE_MOTION_BLUR){
                set_cameras(cameras_close);
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

        void advance_flow() {
            require_init();
            if constexpr (SPEC::HAS_FLOW){
                using rl_tools::advance_flow;
                advance_flow(device, renderer);
                flow_initialized = true;
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer has no flow output");
            }
        }

        void prepare_flow() {
            if constexpr (SPEC::HAS_FLOW){
                if(!flow_initialized){
                    advance_flow();
                    if constexpr (SPEC::ENABLE_OVERLAYS){
                        using rl_tools::update;
                        update(device, renderer);
                    }
                }
            }
        }

        void render() {
            render_launch();
            render_sync();
        }

        void render_launch() {
            require_init();
            prepare_flow();
            rlt::render_launch(device, renderer);
            images_acquired = true;
        }

        void render_sync() {
            require_init();
            rlt::render_sync(device, renderer);
        }

        void probe() {
            require_init();
            if constexpr (PROBES_ENABLED){
                rlt::probe(device, renderer);
                probes_acquired = true;
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without collision probes (num_probes=0)");
            }
        }

        void probe_launch() {
            require_init();
            if constexpr (PROBES_ENABLED){
                rlt::probe_launch(device, renderer);
                probes_acquired = true;
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

        std::vector<hyperdrone::OutputMetadata> output_metadata() const {
            auto result = output_layouts;
            for(auto& value: result){
                value.device_type = buffer_device_type();
                value.device_id = device_id;
            }
            return result;
        }

        void require_output(size_t index) const {
            require_init();
            const auto& metadata = output_layouts.at(index);
            if(metadata.descriptor.quantity == "probes" ? !probes_acquired : !images_acquired){
                throw std::runtime_error("hyperdrone: acquire this output with render() or probe() before reading it");
            }
        }

        void* output_pointer(size_t index, const std::string& field){
            require_output(index);
            const auto& layout = hyperdrone::output_layout(output_layouts.at(index), field);
            void* pointer = nullptr;
            hyperdrone::visit_output<SPEC>(index, [&](auto tag){
                using OUTPUT = typename decltype(tag)::TYPE;
                pointer = reinterpret_cast<char*>(data(output(device, renderer, OUTPUT{}))) + layout.offset;
            });
            return pointer;
        }

        void output_order(size_t index, uintptr_t consumer){
            require_output(index);
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            const hyperdrone::CurrentCudaDevice current(device_id);
            auto producer = output_layouts.at(index).descriptor.quantity == "probes" ? collision_stream(device, renderer) : stream(device, renderer);
            rlt::devices::cuda::order_after(renderer.backend->ordering_event, producer, reinterpret_cast<cudaStream_t>(consumer));
#else
            if(output_layouts.at(index).descriptor.quantity == "probes"){
                probe_sync();
            }
            else {
                render_sync();
            }
#endif
        }

        void output_read(size_t index, void* destination, int destination_device, const std::string& field, uintptr_t consumer){
            const auto& metadata = output_layouts.at(index);
            const auto& layout = hyperdrone::output_layout(metadata, field);
            void* source = output_pointer(index, field);
            const bool probe_field = metadata.descriptor.quantity == "probes";
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            const hyperdrone::CurrentCudaDevice current(device_id);
            auto producer = probe_field ? collision_stream(device, renderer) : stream(device, renderer);
            if(destination_device == 2){
                rlt::devices::cuda::order_after(renderer.backend->ordering_event, reinterpret_cast<cudaStream_t>(consumer), producer);
            }
            const cudaMemcpyKind kind = destination_device == 2 ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
            cudaError_t status;
            if(probe_field){
                status = cudaMemcpy2DAsync(destination, layout.item_size, source, sizeof(rrt::CollisionResult), layout.item_size, layout.size(), kind, producer);
            }
            else {
                status = cudaMemcpyAsync(destination, source, layout.size() * layout.item_size, kind, producer);
            }
            if(status != cudaSuccess){
                throw std::runtime_error(std::string("hyperdrone: output transfer failed: ") + cudaGetErrorString(status));
            }
            if(destination_device == 2){
                output_order(index, consumer);
            }
            else if(cudaStreamSynchronize(producer) != cudaSuccess){
                throw std::runtime_error("hyperdrone: output transfer synchronization failed");
            }
#else
            if(destination_device != 1){
                throw std::invalid_argument("hyperdrone: CUDA output destinations require the OptiX backend");
            }
            output_order(index, 0);
            if(probe_field){
                for(size_t element = 0; element < layout.size(); element++){
                    std::memcpy(static_cast<char*>(destination) + element * layout.item_size, static_cast<char*>(source) + element * sizeof(rrt::CollisionResult), layout.item_size);
                }
            }
            else {
                std::memcpy(destination, source, layout.size() * layout.item_size);
            }
#endif
        }

        int buffer_device_type() const {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            return 2;
#else
            return 1;
#endif
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
                if(camera >= SPEC::NUM_CAMERAS || overlay >= SPEC::NUM_OVERLAYS){
                    throw std::invalid_argument("hyperdrone: camera or overlay index out of range");
                }
                if(!can_attach(camera, overlay)){
                    throw std::runtime_error("hyperdrone: camera overlay capacity exceeded");
                }
                rlt::attach(device, renderer, (TI)camera, rrt::OverlayIndex{overlay});
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void detach(size_t camera, size_t overlay) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                if(camera >= SPEC::NUM_CAMERAS || overlay >= SPEC::NUM_OVERLAYS){
                    throw std::invalid_argument("hyperdrone: camera or overlay index out of range");
                }
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
        rrt::OverlayPlacement spawn(size_t overlay, size_t asset, const rrt::Pose& pose, const float* coordinates, unsigned int segmentation_class) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                if(overlay >= SPEC::NUM_OVERLAYS || asset >= renderer.assets.size()){
                    throw std::invalid_argument("hyperdrone: overlay or asset index out of range");
                }
                if(!rlt::rendering::segmentation::placement_class_valid(segmentation_class)){
                    throw std::invalid_argument("hyperdrone: reserved segmentation class");
                }
                if(!can_spawn(overlay, asset)){
                    throw std::runtime_error("hyperdrone: overlay capacity exceeded");
                }
                return rlt::spawn(device, renderer, rrt::OverlayIndex{overlay}, rrt::AssetHandle{asset}, pose, coordinates, segmentation_class);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        size_t asset_coordinates(size_t asset) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                if(asset >= renderer.assets.size()){
                    throw std::invalid_argument("hyperdrone: asset index out of range");
                }
                return renderer.assets[asset].num_coordinates;
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        size_t placement_coordinates(size_t overlay, const rrt::OverlayPlacement& placement) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                const auto checked = checked_placement(overlay, placement);
                for(const auto& record : renderer.assets){
                    if(record.first_part == checked.first_part){
                        return record.num_coordinates;
                    }
                }
            }
            throw std::invalid_argument("hyperdrone: placement out of range or not spawned");
        }
        void despawn(size_t overlay, const rrt::OverlayPlacement& placement) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                rlt::despawn(device, renderer, rrt::OverlayIndex{overlay}, checked_placement(overlay, placement));
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        // the core verbs fail fast on a placement that is not spawned; a binding throws instead
        rrt::OverlayPlacement checked_placement(size_t overlay, const rrt::OverlayPlacement& placement) const {
            require_init();
            if constexpr (SPEC::ENABLE_OVERLAYS){
                if(!rrt::detail::placement_valid(renderer, rrt::OverlayIndex{overlay}, placement)){
                    throw std::invalid_argument("hyperdrone: placement out of range or not spawned");
                }
            }
            return rrt::OverlayPlacement{placement.first_slot, placement.num_parts, placement.first_part};
        }
        void set_pose(size_t overlay, const rrt::OverlayPlacement& placement, const rrt::Pose& pose, const float* coordinates) {
            if constexpr (SPEC::ENABLE_OVERLAYS){
                rlt::set_pose(device, renderer, rrt::OverlayIndex{overlay}, checked_placement(overlay, placement), pose, coordinates);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without overlays");
            }
        }
        void set_pose(size_t overlay, const rrt::OverlayPlacement& placement, const rrt::Pose& open, const rrt::Pose& close, const float* coordinates_open, const float* coordinates_close) {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                rlt::set_pose(device, renderer, rrt::OverlayIndex{overlay}, checked_placement(overlay, placement), open, close, coordinates_open, coordinates_close);
            }
            else if constexpr(RendererConfiguration::ENABLE_DYNAMIC_MOTION_BLUR){
                set_pose(overlay, placement, close, coordinates_close);
            }
            else {
                throw std::runtime_error("hyperdrone: this renderer was compiled without dynamic motion blur (dynamic_motion_blur=False)");
            }
        }
        // the device form of set_pose: one kernel over the batch on the render stream, ordered
        // after the producer stream; the placements' host rows stop pending (the device owns them
        // from here on) and the arrays stay the caller's until the frame is rendered
        void set_pose_device(const std::vector<size_t>& placements, const float* bases, const float* scales, const float* coordinates, const float* bases_close, const float* scales_close, const float* coordinates_close, size_t num_coordinates, unsigned long long producer_stream) {
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            if constexpr (SPEC::ENABLE_OVERLAYS){
                require_init();
                if constexpr (!RendererConfiguration::ENABLE_DYNAMIC_MOTION_BLUR){
                    if(bases_close != nullptr){
                        throw std::runtime_error("hyperdrone: this renderer was compiled without dynamic motion blur (dynamic_motion_blur=False)");
                    }
                }
                // every check precedes every change: a rejected batch leaves the pending host rows intact
                const size_t count = placements.size() / 4;
                for(size_t b = 0; b < count; b++){
                    checked_placement(placements[b * 4], rrt::OverlayPlacement{placements[b * 4 + 1], placements[b * 4 + 2], placements[b * 4 + 3]});
                }
                if(count > placements_capacity){
                    cudaFree(placements_device);
                    placements_device = nullptr;
                    placements_capacity = 0;
                    if(cudaMalloc(&placements_device, count * 4 * sizeof(size_t)) != cudaSuccess){
                        throw std::runtime_error("hyperdrone: cudaMalloc failed");
                    }
                    placements_capacity = count;
                }
                for(size_t b = 0; b < count; b++){
                    const size_t overlay = placements[b * 4];
                    for(size_t part = 0; part < placements[b * 4 + 2]; part++){
                        const size_t row = overlay * SPEC::MAX_OVERLAY_INSTANCES + placements[b * 4 + 1] + part;
                        renderer.transforms_pending[row] = false;
                        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                            renderer.transforms_motion_pending[row] = false;
                        }
                    }
                }
                cudaStream_t render_stream = rlt::stream(device, renderer);
                rlt::devices::cuda::order_after(renderer.backend->ordering_event, rlt::collision_stream(device, renderer), render_stream);
                await_producer(producer_stream, render_stream);
                cudaMemcpyAsync(placements_device, placements.data(), count * 4 * sizeof(size_t), cudaMemcpyHostToDevice, render_stream);
                PoseBatch batch;
                batch.placements = placements_device;
                batch.bases = bases;
                batch.scales = scales;
                batch.coordinates = coordinates;
                batch.count = count;
                batch.num_coordinates = num_coordinates;
                if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                    batch.pair = bases_close != nullptr;
                    batch.bases_close = bases_close;
                    batch.scales_close = scales_close;
                    batch.coordinates_close = coordinates_close;
                }
                else if(bases_close != nullptr){
                    batch.bases = bases_close;
                    batch.scales = scales_close;
                    batch.coordinates = coordinates_close;
                }
                HYPERDRONE_MODULE_NAME::set_pose_device(rlt::pose_sink(device, renderer), batch, render_stream);
                rlt::devices::cuda::order_after(renderer.backend->ordering_event, render_stream, rlt::collision_stream(device, renderer));
                return;
            }
#endif
            (void)placements; (void)bases; (void)scales; (void)coordinates; (void)bases_close; (void)scales_close; (void)coordinates_close; (void)num_coordinates; (void)producer_stream;
            throw std::runtime_error("hyperdrone: device-resident pose input requires the OptiX backend and overlays");
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
                rlt::devices::cuda::order_after(renderer.backend->ordering_event, rlt::collision_stream(device, renderer), render_stream);
                await_producer(producer_stream, render_stream);
                const size_t bytes = sizeof(rrt::Camera<T>) * SPEC::NUM_CAMERAS;
                cudaMemcpyAsync(rlt::data(rlt::cameras_open(device, renderer)), cameras_open, bytes, cudaMemcpyDeviceToDevice, render_stream);
                cudaMemcpyAsync(rlt::data(rlt::cameras_close(device, renderer)), cameras_close, bytes, cudaMemcpyDeviceToDevice, render_stream);
                rlt::devices::cuda::order_after(renderer.backend->ordering_event, render_stream, rlt::collision_stream(device, renderer));
            }
            else if constexpr(RendererConfiguration::ENABLE_MOTION_BLUR){
                set_cameras_device(cameras_close, producer_stream);
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
    hyperdrone::bind_output_blocks(m);
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
        .def("advance_flow", &Renderer::advance_flow, nb::call_guard<nb::gil_scoped_release>())
        .def("render", &Renderer::render, nb::call_guard<nb::gil_scoped_release>())
        .def("render_launch", &Renderer::render_launch, nb::call_guard<nb::gil_scoped_release>())
        .def("render_sync", &Renderer::render_sync, nb::call_guard<nb::gil_scoped_release>())
        .def("probe", &Renderer::probe, nb::call_guard<nb::gil_scoped_release>())
        .def("probe_launch", &Renderer::probe_launch, nb::call_guard<nb::gil_scoped_release>())
        .def("probe_sync", &Renderer::probe_sync, nb::call_guard<nb::gil_scoped_release>())
        .def("output_metadata", [](const Renderer& self){ return hyperdrone::output_metadata_dicts(self.output_metadata()); })
        .def("output_read", [](Renderer& self, size_t index, nb::ndarray<nb::c_contig> out, const std::string& field, uintptr_t stream){
            hyperdrone::validate_output_array(hyperdrone::output_layout(self.output_layouts.at(index), field), out);
            if(out.device_type() == 2 && out.device_id() != self.device_id){
                throw std::invalid_argument("hyperdrone: CUDA output destination must use the renderer device");
            }
            nb::gil_scoped_release release;
            self.output_read(index, out.data(), out.device_type(), field, stream);
        }, nb::arg("index"), nb::arg("out").noconvert(), nb::arg("field") = "", nb::arg("stream") = 0)
        .def("output_block", [](Renderer& self, size_t index, size_t block, const std::string& field){
            if(block != 0){
                throw std::out_of_range("hyperdrone: standalone renderer has one output block");
            }
            void* pointer = self.output_pointer(index, field);
            if(self.buffer_device_type() == 1){
                self.output_order(index, 0);
            }
            return hyperdrone::NativeOutputBlock{nb::find(&self), pointer, hyperdrone::output_layout(self.output_layouts.at(index), field), self.buffer_device_type(), self.device_id, [&self, index](uintptr_t stream){ self.output_order(index, stream); }};
        }, nb::arg("index"), nb::arg("block") = 0, nb::arg("field") = "")
        .def("buffer_device_id", [](Renderer& self){ return self.device_id; })
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
        .def("spawn", [](Renderer& self, size_t overlay, size_t asset, Transform base, float scale, std::optional<FloatArray> coordinates, std::optional<uint32_t> segmentation_class){
            rrt::OverlayPlacement placement = self.spawn(overlay, asset, hyperdrone::make_pose(base, scale), hyperdrone::check_coordinates(coordinates, self.asset_coordinates(asset)), segmentation_class.value_or(rlt::rendering::segmentation::INHERIT));
            return nb::make_tuple(placement.first_slot, placement.num_parts, placement.first_part);
        }, nb::arg("overlay"), nb::arg("asset"), nb::arg("base"), nb::arg("scale"), nb::arg("coordinates").none(), nb::arg("segmentation_class").none() = nb::none())
        .def("placement_coordinates", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement){
            return self.placement_coordinates(overlay, rrt::OverlayPlacement{placement[0], placement[1], placement[2]});
        }, nb::arg("overlay"), nb::arg("placement"))
        .def("despawn", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement){
            self.despawn(overlay, rrt::OverlayPlacement{placement[0], placement[1], placement[2]});
        }, nb::arg("overlay"), nb::arg("placement"))
        .def("set_pose", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement, Transform base, float scale, std::optional<FloatArray> coordinates){
            const rrt::OverlayPlacement native{placement[0], placement[1], placement[2]};
            self.set_pose(overlay, native, hyperdrone::make_pose(base, scale), hyperdrone::check_coordinates(coordinates, self.placement_coordinates(overlay, native)));
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("base"), nb::arg("scale"), nb::arg("coordinates").none())
        .def("set_pose_pair", [](Renderer& self, size_t overlay, std::array<size_t, 3> placement, Transform base_open, float scale_open, Transform base_close, float scale_close, std::optional<FloatArray> coordinates_open, std::optional<FloatArray> coordinates_close){
            const rrt::OverlayPlacement native{placement[0], placement[1], placement[2]};
            const size_t count = self.placement_coordinates(overlay, native);
            self.set_pose(overlay, native, hyperdrone::make_pose(base_open, scale_open), hyperdrone::make_pose(base_close, scale_close), hyperdrone::check_coordinates(coordinates_open, count), hyperdrone::check_coordinates(coordinates_close, count));
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("base_open"), nb::arg("scale_open"), nb::arg("base_close"), nb::arg("scale_close"), nb::arg("coordinates_open").none(), nb::arg("coordinates_close").none())
        .def("render_stream", [](Renderer& self){ return (uintptr_t)self.render_stream(); });

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
    using CudaFloats = nb::ndarray<const float, nb::c_contig, nb::device::cuda>;
    renderer_class
        .def("set_cameras_device", [](Renderer& self, nb::ndarray<const float, nb::c_contig, nb::device::cuda> cameras, uintptr_t stream){
            int64_t dims[3] = {0, 0, 0};
            for(size_t dimension = 0; dimension < cameras.ndim() && dimension < 3; dimension++){
                dims[dimension] = (int64_t)cameras.shape(dimension);
            }
            self.check_cameras_dims(cameras.ndim(), dims);
            self.set_cameras_device(cameras.data(), stream);
        }, nb::arg("cameras").noconvert(), nb::arg("stream") = 0)
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
        }, nb::arg("cameras_open").noconvert(), nb::arg("cameras_close").noconvert(), nb::arg("stream") = 0)
        .def("set_pose_device", [](Renderer& self, std::vector<std::array<size_t, 4>> placements, CudaFloats bases, std::optional<CudaFloats> scales, std::optional<CudaFloats> coordinates, std::optional<CudaFloats> bases_close, std::optional<CudaFloats> scales_close, std::optional<CudaFloats> coordinates_close, uintptr_t stream){
            const size_t count = placements.size();
            const auto check_bases = [count](const CudaFloats& array, const char* what){
                const bool flat = array.ndim() == 2 && array.shape(0) == count && array.shape(1) == 12;
                const bool structured = array.ndim() == 3 && array.shape(0) == count && array.shape(1) == 3 && array.shape(2) == 4;
                if(!flat && !structured){
                    throw std::invalid_argument(std::string("hyperdrone: ") + what + " must have shape (B, 3, 4) or (B, 12) for the B placements");
                }
            };
            const auto check_scales = [count](const std::optional<CudaFloats>& array, const char* what){
                if(array.has_value() && (array->ndim() != 1 || array->shape(0) != count)){
                    throw std::invalid_argument(std::string("hyperdrone: ") + what + " must have shape (B,) for the B placements");
                }
            };
            const auto coordinate_columns = [count](const std::optional<CudaFloats>& array, const char* what){
                if(!array.has_value()){
                    return (size_t)0;
                }
                if(array->ndim() != 2 || array->shape(0) != count){
                    throw std::invalid_argument(std::string("hyperdrone: ") + what + " must have shape (B, C) for the B placements");
                }
                return (size_t)array->shape(1);
            };
            check_bases(bases, "bases");
            check_scales(scales, "scales");
            size_t num_coordinates = coordinate_columns(coordinates, "coordinates");
            if(bases_close.has_value()){
                check_bases(*bases_close, "bases_close");
                check_scales(scales_close, "scales_close");
                const size_t close_columns = coordinate_columns(coordinates_close, "coordinates_close");
                if(coordinates.has_value() != coordinates_close.has_value() || close_columns != num_coordinates){
                    throw std::invalid_argument("hyperdrone: coordinates and coordinates_close must both be given with the same shape, or neither");
                }
            }
            else if(scales_close.has_value() || coordinates_close.has_value()){
                throw std::invalid_argument("hyperdrone: scales_close and coordinates_close belong to the shutter-close state, pass bases_close");
            }
            std::vector<size_t> flat;
            flat.reserve(count * 4);
            for(const auto& placement : placements){
                if(coordinates.has_value() && num_coordinates < self.placement_coordinates(placement[0], rrt::OverlayPlacement{placement[1], placement[2], placement[3]})){
                    throw std::invalid_argument("hyperdrone: coordinates need at least num_coordinates columns for every placement of the batch");
                }
                flat.insert(flat.end(), placement.begin(), placement.end());
            }
            const auto pointer = [](const std::optional<CudaFloats>& array){ return array.has_value() ? array->data() : nullptr; };
            self.set_pose_device(flat, bases.data(), pointer(scales), pointer(coordinates), pointer(bases_close), pointer(scales_close), pointer(coordinates_close), num_coordinates, stream);
        }, nb::arg("placements"), nb::arg("bases").noconvert(), nb::arg("scales").none().noconvert(), nb::arg("coordinates").none().noconvert(), nb::arg("bases_close").none().noconvert(), nb::arg("scales_close").none().noconvert(), nb::arg("coordinates_close").none().noconvert(), nb::arg("stream") = 0);
#endif
}
