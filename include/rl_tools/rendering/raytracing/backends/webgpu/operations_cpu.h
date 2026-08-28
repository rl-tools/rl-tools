#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_WEBGPU_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_WEBGPU_OPERATIONS_CPU_H

// WebGPU backend host: implements the common Renderer interface against the standard webgpu.h
// (wgpu-native is the reference runtime; Dawn/emscripten are drop-in alternatives). Standard
// WebGPU exposes no acceleration structures, so the BVHs are built on the host with the generic
// backend's deterministic builder and traversed in WGSL (device.wgsl); outputs are copied into
// staging buffers and read back into the host-resident tensors by the *_sync verbs and the
// renderer memory-domain copy (registered ReadbackTargets let the settle run on the Context
// alone). The only non-standard calls are wgpuDevicePoll (blocking wait) and
// wgpuInstanceEnumerateAdapters (RL_TOOLS_WEBGPU_DEVICE_INDEX), both isolated in the helpers
// below; the __EMSCRIPTEN__ (browser) build replaces them with wgpuInstanceRequestAdapter and
// ASYNCIFY event-loop yields.
#include "../../renderer.h"
#include "../../operations_cpu_common.h"
#include "../specialization.h"
#include "../generic/operations_generic.h"
#include "context.h"
#include "bvh_sah.h"
#include "device_source.h"

#include <webgpu/webgpu.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#include <webgpu/wgpu.h>
#endif

#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <cmath>
#include <utility>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing::backends {
        template <typename SPEC>
        struct RendererState<rendering::raytracing::backends::Webgpu, SPEC>: webgpu::Context {};

        template <typename SPEC>
        struct LibraryState<rendering::raytracing::backends::Webgpu, SPEC> {};

        template <typename SPEC>
        struct SceneState<rendering::raytracing::backends::Webgpu, SPEC> {};
    }

    namespace rendering::raytracing::backends::webgpu{
        template <typename SPEC>
        Context& context(rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
            return *renderer.backend;
        }

        inline WGPUStringView string_view(const char* string){
            return WGPUStringView{string, WGPU_STRLEN};
        }

        inline void print_string_view(WGPUStringView view){
            if(view.data != nullptr){
                const size_t length = view.length == WGPU_STRLEN ? std::strlen(view.data) : view.length;
                std::fwrite(view.data, 1, length, stderr);
            }
        }

        inline void uncaptured_error_callback(WGPUDevice const*, WGPUErrorType type, WGPUStringView message, void*, void*){
            std::fprintf(stderr, "#rl_tools::rendering::raytracing: WebGPU uncaptured error (%d): ", (int)type);
            print_string_view(message);
            std::fprintf(stderr, "\n");
            // validation errors are programming errors and would otherwise surface as silent
            // wrong output — fail fast like the Vulkan backend's check()
            std::abort();
        }

        inline void device_lost_callback(WGPUDevice const*, WGPUDeviceLostReason reason, WGPUStringView message, void*, void*){
            if(reason == WGPUDeviceLostReason_Destroyed){
                return;
            }
            std::fprintf(stderr, "#rl_tools::rendering::raytracing: WebGPU device lost (%d): ", (int)reason);
            print_string_view(message);
            std::fprintf(stderr, "\n");
        }

        // the single blocking-wait primitive: wgpu-native's wgpuDevicePoll(wait). In the browser
        // there is no blocking wait — the ASYNCIFY'd emscripten_sleep yields to the JS event loop
        // so the AllowProcessEvents callbacks can fire; call sites spin on their request flags
        inline void device_poll_blocking(Context& ctx){
#ifdef __EMSCRIPTEN__
            wgpuInstanceProcessEvents(ctx.instance);
            emscripten_sleep(1);
#else
            wgpuDevicePoll(ctx.device, true, nullptr);
#endif
        }

        template <typename DEVICE>
        BufferResource create_buffer(DEVICE& device, Context& ctx, uint64_t size, WGPUBufferUsage usage){
            BufferResource resource;
            resource.size = size < 256 ? 256 : size; // covers the largest WGSL binding stride so disabled features can bind placeholders
            if((usage & WGPUBufferUsage_Storage) != 0 && ctx.max_storage_buffer_binding_size > 0){
                utils::assert_exit(device, resource.size <= ctx.max_storage_buffer_binding_size, "WebGPU: buffer exceeds maxStorageBufferBindingSize");
            }
            WGPUBufferDescriptor descriptor{};
            descriptor.usage = usage;
            descriptor.size = resource.size;
            resource.buffer = wgpuDeviceCreateBuffer(ctx.device, &descriptor);
            utils::assert_exit(device, resource.buffer != nullptr, "WebGPU: buffer creation failed");
            return resource;
        }

        inline void destroy_buffer(BufferResource& resource){
            if(resource.buffer != nullptr){
                wgpuBufferRelease(resource.buffer);
                resource.buffer = nullptr;
            }
            resource.size = 0;
        }

        struct MapRequest{
            bool done = false;
            WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
        };

        inline void map_callback(WGPUMapAsyncStatus status, WGPUStringView, void* userdata1, void*){
            auto* request = (MapRequest*)userdata1;
            request->status = status;
            request->done = true;
        }

        template <typename DEVICE>
        void read_back(DEVICE& device, Context& ctx, BufferResource& staging, void* destination, size_t bytes){
            MapRequest request;
            WGPUBufferMapCallbackInfo callback_info{};
            callback_info.mode = WGPUCallbackMode_AllowProcessEvents;
            callback_info.callback = map_callback;
            callback_info.userdata1 = &request;
            wgpuBufferMapAsync(staging.buffer, WGPUMapMode_Read, 0, (size_t)staging.size, callback_info);
            while(!request.done){
                device_poll_blocking(ctx);
            }
            utils::assert_exit(device, request.status == WGPUMapAsyncStatus_Success, "WebGPU: staging buffer map failed");
            const void* mapped = wgpuBufferGetConstMappedRange(staging.buffer, 0, (size_t)staging.size);
            utils::assert_exit(device, mapped != nullptr, "WebGPU: mapped range unavailable");
            std::memcpy(destination, mapped, bytes);
            wgpuBufferUnmap(staging.buffer);
        }

        template <typename DEVICE>
        void settle_render(DEVICE& device, Context& ctx){
            if(!ctx.render_in_flight){
                return;
            }
            for(auto& target : ctx.render_readbacks){
                read_back(device, ctx, *target.staging, target.destination, target.bytes);
            }
            ctx.render_in_flight = false;
        }

        template <typename DEVICE>
        void settle_probe(DEVICE& device, Context& ctx){
            if(!ctx.collision_in_flight){
                return;
            }
            for(auto& target : ctx.probe_readbacks){
                read_back(device, ctx, *target.staging, target.destination, target.bytes);
            }
            ctx.collision_in_flight = false;
        }

        template <typename DEVICE>
        void wait_in_flight(DEVICE& device, Context& ctx){
            settle_render(device, ctx);
            settle_probe(device, ctx);
        }

        inline void destroy_scene_resources(Context& ctx){
            if(ctx.bind_group != nullptr){
                wgpuBindGroupRelease(ctx.bind_group);
                ctx.bind_group = nullptr;
            }
            destroy_buffer(ctx.scene_geometry);
            destroy_buffer(ctx.texture_data);
            destroy_buffer(ctx.bvh_nodes);
            destroy_buffer(ctx.triangles);
            destroy_buffer(ctx.instance_data);
            ctx.instance_classes_offset = 0;
            ctx.overlay_node_offset = 0;
            ctx.instance_data_host.clear();
            ctx.instance_classes_host.clear();
            ctx.object_classes.clear();
            ctx.object_node_offset.clear();
            ctx.object_node_count.clear();
            ctx.object_primitive_offset.clear();
            ctx.object_root_bounds_min.clear();
            ctx.object_root_bounds_max.clear();
            ctx.overlay_nodes_host.clear();
            ctx.overlay_primitives_host.clear();
            ctx.overlay_meta_host.clear();
            ctx.overlay_bounds_min.clear();
            ctx.overlay_bounds_max.clear();
            ctx.overlay_centroids.clear();
            ctx.overlay_temp_primitives.clear();
            ctx.num_scene_instances = 0;
            ctx.total_instances = 0;
        }

        inline uint32_t bvh_max_depth(const BVHNode* nodes, uint32_t num_nodes){
            if(num_nodes == 0){
                return 0;
            }
            uint32_t max_depth = 0;
            std::vector<std::pair<uint32_t, uint32_t>> stack;
            stack.push_back({0, 1});
            while(!stack.empty()){
                const auto [node_i, depth] = stack.back();
                stack.pop_back();
                max_depth = depth > max_depth ? depth : max_depth;
                const auto& node = nodes[node_i];
                if(node.count == 0){
                    stack.push_back({node.left_or_first, depth + 1});
                    stack.push_back({node.left_or_first + 1, depth + 1});
                }
            }
            return max_depth;
        }

        // world-space AABB of one instance from its object's BLAS root bounds
        inline void instance_world_bounds(Context& ctx, uint32_t object, const float object_to_world[12], bool identity, float bounds_min[3], float bounds_max[3]){
            for(int axis = 0; axis < 3; axis++){
                bounds_min[axis] = 1e30f;
                bounds_max[axis] = -1e30f;
            }
            if(ctx.object_node_count[object] == 0){
                return;
            }
            const float* root_min = &ctx.object_root_bounds_min[3 * (size_t)object];
            const float* root_max = &ctx.object_root_bounds_max[3 * (size_t)object];
            if(identity){
                for(int axis = 0; axis < 3; axis++){
                    bounds_min[axis] = root_min[axis];
                    bounds_max[axis] = root_max[axis];
                }
            }
            else{
                for(int corner = 0; corner < 8; corner++){
                    const float local[3] = {
                        (corner & 1) ? root_max[0] : root_min[0],
                        (corner & 2) ? root_max[1] : root_min[1],
                        (corner & 4) ? root_max[2] : root_min[2]
                    };
                    float world[3];
                    rendering::raytracing::detail::transform_point(object_to_world, local, world);
                    for(int axis = 0; axis < 3; axis++){
                        bounds_min[axis] = bounds_min[axis] < world[axis] ? bounds_min[axis] : world[axis];
                        bounds_max[axis] = bounds_max[axis] > world[axis] ? bounds_max[axis] : world[axis];
                    }
                }
            }
        }

        inline void fill_instance_entry(Context& ctx, size_t global, uint32_t object, const float world[12]){
            auto& entry = ctx.instance_data_host[global];
            entry = {};
            for(int element = 0; element < 12; element++){
                entry.object_to_world[element] = world[element];
            }
            const bool identity = rendering::raytracing::detail::transform_is_identity(world);
            if(identity){
                std::memcpy(entry.world_to_object, entry.object_to_world, sizeof(entry.world_to_object));
            }
            else{
                rendering::raytracing::detail::invert_transform(world, entry.world_to_object);
            }
            entry.object = object;
            entry.identity = identity ? 1 : 0;
            ctx.instance_classes_host[global] = ctx.object_classes[object];
        }

        // rebuilds one overlay region (region 0 = shutter close, 1 + s = dynamic-motion-blur
        // sample s) into the host mirrors; the caller uploads them in one batch. transforms_base
        // selects the transform slab (nullptr = the slot-mirror path via the 4-arg composer)
        template <typename DEVICE, typename SPEC, typename BACKEND_TAG>
        void rebuild_overlay_region(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND_TAG>& renderer, Context& ctx, uint32_t region, const float* transforms_base){
            using TI = typename SPEC::TI;
            constexpr size_t num_overlay_slots = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
            for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
                auto& overlay_state = renderer.overlays[overlay];
                const size_t region_overlay = (size_t)region * SPEC::NUM_OVERLAYS + overlay;
                uint32_t* primitives = ctx.overlay_primitives_host.data() + region_overlay * SPEC::MAX_OVERLAY_INSTANCES;
                uint32_t num_active = 0;
                for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                    const auto& host_slot = overlay_state.slots[slot];
                    if(!host_slot.active) continue;
                    const size_t global = (size_t)ctx.num_scene_instances + num_overlay_slots * region + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES + slot;
                    float world[12];
                    if(transforms_base == nullptr){
                        rendering::raytracing::detail::compose_overlay_slot_transform(renderer, overlay, slot, world);
                    }
                    else{
                        rendering::raytracing::detail::compose_overlay_slot_transform(renderer, transforms_base, overlay, slot, world);
                    }
                    fill_instance_entry(ctx, global, (uint32_t)host_slot.object, world);
                    float bounds_min[3], bounds_max[3];
                    instance_world_bounds(ctx, (uint32_t)host_slot.object, world, ctx.instance_data_host[global].identity != 0, bounds_min, bounds_max);
                    for(int axis = 0; axis < 3; axis++){
                        ctx.overlay_bounds_min[3 * global + axis] = bounds_min[axis];
                        ctx.overlay_bounds_max[3 * global + axis] = bounds_max[axis];
                        ctx.overlay_centroids[3 * global + axis] = (bounds_min[axis] + bounds_max[axis]) * 0.5f;
                    }
                    primitives[num_active++] = (uint32_t)global;
                }
                BVHNode* nodes = ctx.overlay_nodes_host.data() + region_overlay * 2 * SPEC::MAX_OVERLAY_INSTANCES;
                const uint32_t num_nodes = generic::build_bvh_nodes(nodes, primitives, ctx.overlay_temp_primitives.data(), ctx.overlay_bounds_min.data(), ctx.overlay_bounds_max.data(), ctx.overlay_centroids.data(), num_active);
                utils::assert_exit(device, bvh_max_depth(nodes, num_nodes) <= TRAVERSAL_STACK_SIZE, "WebGPU: overlay BVH depth exceeds the traversal stack");
                ctx.overlay_meta_host[region_overlay * 2 + 0] = num_active;
                ctx.overlay_meta_host[region_overlay * 2 + 1] = num_nodes;
            }
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer);
    template <typename DEVICE, typename SPEC>
    void probe_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer);
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        namespace wg = rendering::raytracing::backends::webgpu;
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename SPEC::T, float>, "The WebGPU raytracing backend requires T = float");

        malloc(device, renderer.cameras);
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            malloc(device, renderer.cameras_open);
        }
        if constexpr (SPEC::HAS_RGB) {
            malloc(device, renderer.frame_buffer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            malloc(device, renderer.depth_buffer);
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            malloc(device, renderer.segmentation_buffer);
        }
        if constexpr (SPEC::HAS_NORMALS) {
            malloc(device, renderer.normals_buffer);
        }
        if constexpr (SPEC::HAS_FLOW) {
            malloc(device, renderer.flow_buffer);
            if constexpr (SPEC::ENABLE_OVERLAYS) {
                malloc(device, renderer.flow_deltas);
            }
        }
        if constexpr (SPEC::HAS_OBSERVATION) {
            static_assert(utils::typing::is_same_v<typename SPEC::OBSERVATION_T, float>, "The WebGPU raytracing backend requires OBSERVATION_T = float");
            malloc(device, renderer.observation);
        }
        if constexpr (SPEC::ENABLE_OVERLAYS) {
            malloc(device, renderer.transforms);
        }
        if constexpr (SPEC::HAS_TRANSFORM_PAIR) {
            malloc(device, renderer.transforms_pair);
            std::memset(data(renderer.transforms_pair), 0, decltype(renderer.transforms_pair)::SPEC::SIZE_BYTES);
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            malloc(device, renderer.transforms_motion);
            std::memset(data(renderer.transforms_motion), 0, decltype(renderer.transforms_motion)::SPEC::SIZE_BYTES);
            renderer.transforms_motion_staging.assign((size_t)SPEC::MOTION_BLUR_SAMPLES * SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12, 0.0f);
            if constexpr (SPEC::HAS_RGB) {
                malloc(device, renderer.rgb_accumulator);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                malloc(device, renderer.depth_accumulator);
            }
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        malloc(device, renderer.collision_results);
#endif

        renderer.backend = new rendering::raytracing::backends::RendererState<rendering::raytracing::backends::Webgpu, SPEC>{};
        renderer.device.context = renderer.backend;
        auto* ctx = renderer.backend;

        ctx->instance = wgpuCreateInstance(nullptr);
        utils::assert_exit(device, ctx->instance != nullptr, "WebGPU: instance creation failed");

#ifdef __EMSCRIPTEN__
        {
            struct AdapterRequest{
                bool done = false;
                WGPUAdapter adapter = nullptr;
            } request;
            WGPURequestAdapterOptions options{};
            options.powerPreference = WGPUPowerPreference_HighPerformance;
            WGPURequestAdapterCallbackInfo callback_info{};
            callback_info.mode = WGPUCallbackMode_AllowProcessEvents;
            callback_info.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void*){
                auto* request = (AdapterRequest*)userdata1;
                if(status != WGPURequestAdapterStatus_Success){
                    std::fprintf(stderr, "#rl_tools::rendering::raytracing: WebGPU adapter request failed: ");
                    wg::print_string_view(message);
                    std::fprintf(stderr, "\n");
                }
                request->adapter = adapter;
                request->done = true;
            };
            callback_info.userdata1 = &request;
            wgpuInstanceRequestAdapter(ctx->instance, &options, callback_info);
            while(!request.done){
                wgpuInstanceProcessEvents(ctx->instance);
                emscripten_sleep(1);
            }
            ctx->adapter = request.adapter;
            utils::assert_exit(device, ctx->adapter != nullptr, "WebGPU: no suitable adapter available");
        }
#else
        // adapter selection mirrors the Vulkan backend: filter on capability (a GL adapter only
        // exposes the 8-storage-buffer minimum), prefer discrete GPUs, RL_TOOLS_WEBGPU_DEVICE_INDEX
        // forces an enumeration index. wgpuInstanceEnumerateAdapters is a wgpu-native extension;
        // the browser has no enumeration, so the emscripten path asks via wgpuInstanceRequestAdapter
        const char* device_index_env = std::getenv("RL_TOOLS_WEBGPU_DEVICE_INDEX");
        const int64_t requested_device_index = device_index_env != nullptr ? std::atoll(device_index_env) : -1;
        {
            const size_t adapter_count = wgpuInstanceEnumerateAdapters(ctx->instance, nullptr, nullptr);
            std::vector<WGPUAdapter> adapters(adapter_count);
            wgpuInstanceEnumerateAdapters(ctx->instance, nullptr, adapters.data());
            int64_t selected = -1;
            bool selected_discrete = false;
            for(size_t adapter_i = 0; adapter_i < adapter_count; adapter_i++){
                WGPULimits limits{};
                const bool qualifies = wgpuAdapterGetLimits(adapters[adapter_i], &limits) == WGPUStatus_Success
                    && limits.maxStorageBuffersPerShaderStage >= wg::bindings::STORAGE_COUNT;
                if(requested_device_index >= 0){
                    if((int64_t)adapter_i == requested_device_index){
                        utils::assert_exit(device, qualifies, "WebGPU: the requested adapter supports too few storage buffers per stage");
                        selected = (int64_t)adapter_i;
                    }
                    continue;
                }
                if(!qualifies){
                    continue;
                }
                WGPUAdapterInfo info{};
                bool discrete = false;
                if(wgpuAdapterGetInfo(adapters[adapter_i], &info) == WGPUStatus_Success){
                    discrete = info.adapterType == WGPUAdapterType_DiscreteGPU;
                    wgpuAdapterInfoFreeMembers(info);
                }
                if(selected < 0 || (discrete && !selected_discrete)){
                    selected = (int64_t)adapter_i;
                    selected_discrete = discrete;
                }
            }
            utils::assert_exit(device, selected >= 0, "WebGPU: no suitable adapter available");
            for(size_t adapter_i = 0; adapter_i < adapter_count; adapter_i++){
                if((int64_t)adapter_i == selected){
                    ctx->adapter = adapters[adapter_i];
                }
                else{
                    wgpuAdapterRelease(adapters[adapter_i]);
                }
            }
        }
#endif

        {
            WGPUAdapterInfo info{};
            if(wgpuAdapterGetInfo(ctx->adapter, &info) == WGPUStatus_Success){
                const size_t length = info.device.data == nullptr ? 0 : (info.device.length == WGPU_STRLEN ? std::strlen(info.device.data) : info.device.length);
                RL_TOOLS_RENDERING_RAYTRACING_LOG("WebGPU adapter: " << std::string(info.device.data == nullptr ? "" : info.device.data, length) << " (backend " << (int)info.backendType << ")");
                wgpuAdapterInfoFreeMembers(info);
            }
        }

        WGPULimits adapter_limits{};
        utils::assert_exit(device, wgpuAdapterGetLimits(ctx->adapter, &adapter_limits) == WGPUStatus_Success, "WebGPU: adapter limit query failed");
        utils::assert_exit(device, adapter_limits.maxStorageBuffersPerShaderStage >= wg::bindings::STORAGE_COUNT, "WebGPU: adapter supports too few storage buffers per stage");
        ctx->max_storage_buffers_per_shader_stage = adapter_limits.maxStorageBuffersPerShaderStage;
        ctx->max_storage_buffer_binding_size = adapter_limits.maxStorageBufferBindingSize;

        {
            // request the adapter's own limits: the scene buffers (geometry, textures, BVH) can
            // exceed the spec defaults (128MiB binding / 8 storage buffers per stage)
            WGPULimits required_limits = adapter_limits;
            required_limits.nextInChain = nullptr;
            WGPUDeviceDescriptor descriptor{};
            descriptor.requiredLimits = &required_limits;
            descriptor.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
            descriptor.deviceLostCallbackInfo.callback = wg::device_lost_callback;
            descriptor.uncapturedErrorCallbackInfo.callback = wg::uncaptured_error_callback;
            struct DeviceRequest{
                bool done = false;
                WGPUDevice device = nullptr;
            } request;
            WGPURequestDeviceCallbackInfo callback_info{};
            callback_info.mode = WGPUCallbackMode_AllowProcessEvents;
            callback_info.callback = [](WGPURequestDeviceStatus status, WGPUDevice wgpu_device, WGPUStringView message, void* userdata1, void*){
                auto* request = (DeviceRequest*)userdata1;
                if(status != WGPURequestDeviceStatus_Success){
                    std::fprintf(stderr, "#rl_tools::rendering::raytracing: WebGPU device request failed: ");
                    wg::print_string_view(message);
                    std::fprintf(stderr, "\n");
                }
                request->device = wgpu_device;
                request->done = true;
            };
            callback_info.userdata1 = &request;
            wgpuAdapterRequestDevice(ctx->adapter, &descriptor, callback_info);
            while(!request.done){
                wgpuInstanceProcessEvents(ctx->instance);
#ifdef __EMSCRIPTEN__
                emscripten_sleep(1);
#endif
            }
            ctx->device = request.device;
        }
        utils::assert_exit(device, ctx->device != nullptr, "WebGPU: device creation failed");
        ctx->queue = wgpuDeviceGetQueue(ctx->device);

        {
            WGPUShaderSourceWGSL wgsl{};
            wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
            wgsl.code = wg::string_view(rendering::raytracing::backends::webgpu::device_source());
            WGPUShaderModuleDescriptor descriptor{};
            descriptor.nextInChain = &wgsl.chain;
            ctx->module = wgpuDeviceCreateShaderModule(ctx->device, &descriptor);
            utils::assert_exit(device, ctx->module != nullptr, "WebGPU: shader module creation failed");
        }

        {
            WGPUBindGroupLayoutEntry layout_entries[wg::bindings::COUNT]{};
            auto set_entry = [&](uint32_t binding, WGPUBufferBindingType type){
                layout_entries[binding].binding = binding;
                layout_entries[binding].visibility = WGPUShaderStage_Compute;
                layout_entries[binding].buffer.type = type;
            };
            for(uint32_t binding_i = 0; binding_i < wg::bindings::COUNT; binding_i++){
                set_entry(binding_i, WGPUBufferBindingType_ReadOnlyStorage);
            }
            set_entry(wg::bindings::OUTPUTS, WGPUBufferBindingType_Storage);
            set_entry(wg::bindings::DISPATCH_PARAMS, WGPUBufferBindingType_Uniform);
            layout_entries[wg::bindings::DISPATCH_PARAMS].buffer.hasDynamicOffset = true;
            layout_entries[wg::bindings::DISPATCH_PARAMS].buffer.minBindingSize = sizeof(wg::DispatchParams);
            WGPUBindGroupLayoutDescriptor layout_descriptor{};
            layout_descriptor.entryCount = wg::bindings::COUNT;
            layout_descriptor.entries = layout_entries;
            ctx->bind_group_layout = wgpuDeviceCreateBindGroupLayout(ctx->device, &layout_descriptor);
            utils::assert_exit(device, ctx->bind_group_layout != nullptr, "WebGPU: bind group layout creation failed");
            WGPUPipelineLayoutDescriptor pipeline_layout_descriptor{};
            pipeline_layout_descriptor.bindGroupLayoutCount = 1;
            pipeline_layout_descriptor.bindGroupLayouts = &ctx->bind_group_layout;
            ctx->pipeline_layout = wgpuDeviceCreatePipelineLayout(ctx->device, &pipeline_layout_descriptor);
            utils::assert_exit(device, ctx->pipeline_layout != nullptr, "WebGPU: pipeline layout creation failed");
        }

        {
            const auto constant = [](const char* key, double value){
                WGPUConstantEntry entry{};
                entry.key = wg::string_view(key);
                entry.value = value;
                return entry;
            };
            using CONSTANTS = rendering::raytracing::backends::SpecializationConstants<SPEC>;
            const WGPUConstantEntry constants[] = {
                constant("fc_srgb_output", CONSTANTS::SRGB_OUTPUT ? 1.0 : 0.0),
                constant("fc_motion_blur", CONSTANTS::MOTION_BLUR ? 1.0 : 0.0),
                constant("fc_motion_samples", (double)CONSTANTS::MOTION_SAMPLES),
                constant("fc_aa_grid", (double)CONSTANTS::AA_GRID),
                constant("fc_checker_background", CONSTANTS::CHECKER_BACKGROUND ? 1.0 : 0.0),
                constant("fc_load_textures", CONSTANTS::LOAD_TEXTURES ? 1.0 : 0.0),
                constant("fc_normal_shading", CONSTANTS::NORMAL_SHADING ? 1.0 : 0.0),
                constant("fc_metallic_reflections", CONSTANTS::METALLIC_REFLECTIONS ? 1.0 : 0.0),
                constant("fc_pbr_shading", CONSTANTS::PBR_SHADING ? 1.0 : 0.0),
                constant("fc_punctual_light_shadows", CONSTANTS::PUNCTUAL_LIGHT_SHADOWS ? 1.0 : 0.0),
                constant("fc_overlay_count", (double)CONSTANTS::OVERLAY_COUNT),
                constant("fc_semantic_segmentation", CONSTANTS::SEMANTIC_SEGMENTATION ? 1.0 : 0.0),
                constant("fc_has_observation", CONSTANTS::HAS_OBSERVATION ? 1.0 : 0.0),
                constant("fc_dynamic_motion_blur", CONSTANTS::DYNAMIC_MOTION_BLUR ? 1.0 : 0.0),
                constant("fc_resolve_rgb", CONSTANTS::RESOLVE_RGB ? 1.0 : 0.0),
                constant("fc_resolve_depth", CONSTANTS::RESOLVE_DEPTH ? 1.0 : 0.0),
                constant("fc_num_overlays", (double)CONSTANTS::NUM_OVERLAYS),
                constant("fc_overlay_capacity", (double)CONSTANTS::OVERLAY_CAPACITY),
            };
            auto make_pipeline = [&](const char* entry_point) -> WGPUComputePipeline {
                WGPUComputePipelineDescriptor descriptor{};
                descriptor.layout = ctx->pipeline_layout;
                descriptor.compute.module = ctx->module;
                descriptor.compute.entryPoint = wg::string_view(entry_point);
                descriptor.compute.constantCount = sizeof(constants) / sizeof(constants[0]);
                descriptor.compute.constants = constants;
                WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(ctx->device, &descriptor);
                utils::assert_exit(device, pipeline != nullptr, "WebGPU: compute pipeline creation failed");
                return pipeline;
            };
            if constexpr (SPEC::HAS_RGB) {
                ctx->rgb_pipeline = make_pipeline("main_rgb");
            }
            if constexpr (SPEC::HAS_DEPTH) {
                ctx->depth_pipeline = make_pipeline("main_depth");
            }
            if constexpr (SPEC::HAS_SEGMENTATION) {
                ctx->segmentation_pipeline = make_pipeline("main_segmentation");
            }
            if constexpr (SPEC::HAS_NORMALS) {
                ctx->normals_pipeline = make_pipeline("main_normals");
            }
            if constexpr (SPEC::HAS_FLOW) {
                ctx->flow_pipeline = make_pipeline("main_flow");
            }
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                ctx->resolve_pipeline = make_pipeline("main_resolve");
            }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
            ctx->collision_pipeline = make_pipeline("main_collision");
#endif
            ctx->pipelines_built = true;
        }

        const WGPUBufferUsage STORAGE_UPLOAD = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        const WGPUBufferUsage STORAGE_OUTPUT = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
        const WGPUBufferUsage STAGING = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        using LAYOUT = wg::BufferLayout<SPEC>;
        ctx->launch_params = wg::create_buffer(device, *ctx, sizeof(wg::LaunchParams), STORAGE_UPLOAD);
        ctx->frame_inputs = wg::create_buffer(device, *ctx, (size_t)LAYOUT::FRAME_INPUTS_WORDS * sizeof(uint32_t), STORAGE_UPLOAD);
        ctx->outputs = wg::create_buffer(device, *ctx, (size_t)LAYOUT::OUTPUTS_WORDS * sizeof(uint32_t), STORAGE_OUTPUT);
        if constexpr (SPEC::HAS_RGB) {
            ctx->staging_frame_buffer = wg::create_buffer(device, *ctx, (size_t)LAYOUT::FRAME_BUFFER_WORDS * sizeof(uint32_t), STAGING);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            ctx->staging_depth = wg::create_buffer(device, *ctx, (size_t)LAYOUT::DEPTH_WORDS * sizeof(uint32_t), STAGING);
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            ctx->staging_segmentation = wg::create_buffer(device, *ctx, (size_t)LAYOUT::SEGMENTATION_WORDS * sizeof(uint32_t), STAGING);
        }
        if constexpr (SPEC::HAS_NORMALS) {
            ctx->staging_normals = wg::create_buffer(device, *ctx, (size_t)LAYOUT::NORMALS_WORDS * sizeof(uint32_t), STAGING);
        }
        if constexpr (SPEC::HAS_FLOW) {
            ctx->staging_flow = wg::create_buffer(device, *ctx, (size_t)LAYOUT::FLOW_WORDS * sizeof(uint32_t), STAGING);
        }
        if constexpr (SPEC::HAS_OBSERVATION) {
            ctx->staging_observation = wg::create_buffer(device, *ctx, (size_t)LAYOUT::OBSERVATION_WORDS * sizeof(uint32_t), STAGING);
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        ctx->staging_collision = wg::create_buffer(device, *ctx, (size_t)LAYOUT::COLLISION_WORDS * sizeof(uint32_t), STAGING);
#endif
        if constexpr (SPEC::HAS_RGB) {
            ctx->render_readbacks.push_back({&ctx->staging_frame_buffer, data(renderer.frame_buffer), decltype(renderer.frame_buffer)::SPEC::SIZE_BYTES});
        }
        if constexpr (SPEC::HAS_DEPTH) {
            ctx->render_readbacks.push_back({&ctx->staging_depth, data(renderer.depth_buffer), decltype(renderer.depth_buffer)::SPEC::SIZE_BYTES});
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            ctx->render_readbacks.push_back({&ctx->staging_segmentation, data(renderer.segmentation_buffer), decltype(renderer.segmentation_buffer)::SPEC::SIZE_BYTES});
        }
        if constexpr (SPEC::HAS_NORMALS) {
            ctx->render_readbacks.push_back({&ctx->staging_normals, data(renderer.normals_buffer), decltype(renderer.normals_buffer)::SPEC::SIZE_BYTES});
        }
        if constexpr (SPEC::HAS_FLOW) {
            ctx->render_readbacks.push_back({&ctx->staging_flow, data(renderer.flow_buffer), decltype(renderer.flow_buffer)::SPEC::SIZE_BYTES});
        }
        if constexpr (SPEC::HAS_OBSERVATION) {
            ctx->render_readbacks.push_back({&ctx->staging_observation, data(renderer.observation), decltype(renderer.observation)::SPEC::SIZE_BYTES});
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        ctx->probe_readbacks.push_back({&ctx->staging_collision, data(renderer.collision_results), decltype(renderer.collision_results)::SPEC::SIZE_BYTES});
#endif
        {
            // one 256-byte slot per dynamic-motion-blur sample plus slot 0 for the shutter-close
            // state; slot s + 1 carries that sample's shutter time and overlay BVH region
            constexpr uint32_t num_slots = 1 + (SPEC::ENABLE_DYNAMIC_MOTION_BLUR ? (uint32_t)SPEC::MOTION_BLUR_SAMPLES : 0);
            ctx->dispatch_params = wg::create_buffer(device, *ctx, (size_t)num_slots * wg::DISPATCH_PARAMS_STRIDE, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
            std::vector<uint8_t> slots((size_t)num_slots * wg::DISPATCH_PARAMS_STRIDE, 0);
            for(uint32_t slot = 1; slot < num_slots; slot++){
                auto* entry = (wg::DispatchParams*)(slots.data() + (size_t)slot * wg::DISPATCH_PARAMS_STRIDE);
                entry->shutter_t = ((float)(slot - 1) + 0.5f) / (float)SPEC::MOTION_BLUR_SAMPLES;
                entry->overlay_region = slot;
            }
            wgpuQueueWriteBuffer(ctx->queue, ctx->dispatch_params.buffer, 0, slots.data(), slots.size());
        }
        rendering::raytracing::detail::announce_backend(renderer);
    }

    template <typename DEVICE, typename SPEC, typename METADATA_T>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool, const rendering::SceneMetadata<METADATA_T>& metadata){
        renderer.max_ray_length = (typename SPEC::T)metadata.max_ray_length;
        rendering::raytracing::detail::announce_configuration<SPEC>();
        namespace wg = rendering::raytracing::backends::webgpu;
        namespace generic = rendering::raytracing::backends::generic;
        using TI = typename SPEC::TI;
        auto& ctx = wg::context(renderer);


        std::vector<const rendering::raytracing::Object*> all_objects;
        for(const auto& object : scene.objects){
            all_objects.push_back(&object);
        }
        if constexpr (SPEC::ENABLE_OVERLAYS){
            rendering::raytracing::detail::register_pool_assets(device, renderer, pool, all_objects);
        }
        RL_TOOLS_RENDERING_RAYTRACING_LOG("building " << all_objects.size() << " objects / " << scene.instances.size() << " instances ...");

        wg::wait_in_flight(device, ctx);
        wg::destroy_scene_resources(ctx);

        // both builders are deterministic; median is the shared generic baseline for A/B and
        // debugging, sah the default (the overlay TLAS rebuilds always use the generic builder)
        const char* bvh_builder_env = std::getenv("RL_TOOLS_WEBGPU_BVH");
        const bool median_builder = bvh_builder_env != nullptr && std::strcmp(bvh_builder_env, "median") == 0;
        RL_TOOLS_RENDERING_RAYTRACING_LOG("WebGPU BVH builder: " << (median_builder ? "median (baseline)" : "sah"));
        const auto build_bvh = [&](wg::BVHNode* build_nodes, uint32_t* build_primitives, uint32_t* build_temp, const float* build_bounds_min, const float* build_bounds_max, const float* build_centroids, uint32_t build_count) -> uint32_t {
            if(median_builder){
                return generic::build_bvh_nodes(build_nodes, build_primitives, build_temp, build_bounds_min, build_bounds_max, build_centroids, build_count);
            }
            return wg::build_bvh_nodes_sah(build_nodes, build_primitives, build_temp, build_bounds_min, build_bounds_max, build_centroids, build_count);
        };

        // meshes flattened across all objects (scene objects then pool objects); the global
        // triangle numbering ties the BLAS leaf permutations to the shading tables
        std::vector<const rendering::raytracing::Mesh*> all_meshes;
        std::vector<uint32_t> object_first_triangle;
        std::vector<uint32_t> object_triangle_count;
        std::vector<uint32_t> triangle_mesh;
        std::vector<uint32_t> triangle_local;
        ctx.object_classes.assign(all_objects.size(), 0);
        for(size_t object_i = 0; object_i < all_objects.size(); object_i++){
            ctx.object_classes[object_i] = all_objects[object_i]->segmentation_class;
            object_first_triangle.push_back((uint32_t)triangle_mesh.size());
            for(const auto& mesh : all_objects[object_i]->meshes){
                const uint32_t mesh_i = (uint32_t)all_meshes.size();
                all_meshes.push_back(&mesh);
                const uint32_t mesh_triangles = (uint32_t)(mesh.indices.size() / 3);
                for(uint32_t triangle = 0; triangle < mesh_triangles; triangle++){
                    triangle_mesh.push_back(mesh_i);
                    triangle_local.push_back(triangle);
                }
            }
            object_triangle_count.push_back((uint32_t)triangle_mesh.size() - object_first_triangle.back());
        }
        const size_t num_meshes = all_meshes.size();
        const size_t num_triangles = triangle_mesh.size();

        std::vector<float> triangle_bounds_min(num_triangles > 0 ? 3 * num_triangles : 1);
        std::vector<float> triangle_bounds_max(num_triangles > 0 ? 3 * num_triangles : 1);
        std::vector<float> centroids(num_triangles > 0 ? 3 * num_triangles : 1);
        for(size_t triangle = 0; triangle < num_triangles; triangle++){
            const auto& mesh = *all_meshes[triangle_mesh[triangle]];
            const int* index = &mesh.indices[3 * (size_t)triangle_local[triangle]];
            float bounds_min[3] = {1e30f, 1e30f, 1e30f};
            float bounds_max[3] = {-1e30f, -1e30f, -1e30f};
            for(int vertex_i = 0; vertex_i < 3; vertex_i++){
                const float* vertex = &mesh.vertices[3 * (size_t)index[vertex_i]];
                for(int axis = 0; axis < 3; axis++){
                    bounds_min[axis] = bounds_min[axis] < vertex[axis] ? bounds_min[axis] : vertex[axis];
                    bounds_max[axis] = bounds_max[axis] > vertex[axis] ? bounds_max[axis] : vertex[axis];
                }
            }
            for(int axis = 0; axis < 3; axis++){
                triangle_bounds_min[3 * triangle + axis] = bounds_min[axis];
                triangle_bounds_max[3 * triangle + axis] = bounds_max[axis];
                centroids[3 * triangle + axis] = (bounds_min[axis] + bounds_max[axis]) * 0.5f;
            }
        }

        // per-object BLAS slices at 2 * first_triangle, same arrangement as the generic backend
        std::vector<wg::BVHNode> blas_nodes(num_triangles > 0 ? 2 * num_triangles : 1);
        std::vector<uint32_t> blas_primitives(num_triangles > 0 ? num_triangles : 1);
        std::vector<uint32_t> temp_primitives(blas_primitives.size());
        ctx.object_node_offset.assign(all_objects.size(), 0);
        ctx.object_node_count.assign(all_objects.size(), 0);
        ctx.object_primitive_offset.assign(all_objects.size(), 0);
        ctx.object_root_bounds_min.assign(3 * all_objects.size() + 1, 0.0f);
        ctx.object_root_bounds_max.assign(3 * all_objects.size() + 1, 0.0f);
        for(size_t object_i = 0; object_i < all_objects.size(); object_i++){
            const uint32_t first = object_first_triangle[object_i];
            const uint32_t count = object_triangle_count[object_i];
            for(uint32_t i = 0; i < count; i++){
                blas_primitives[(size_t)first + i] = first + i;
            }
            const uint32_t num_nodes = build_bvh(blas_nodes.data() + 2 * (size_t)first, blas_primitives.data() + first, temp_primitives.data(), triangle_bounds_min.data(), triangle_bounds_max.data(), centroids.data(), count);
            utils::assert_exit(device, wg::bvh_max_depth(blas_nodes.data() + 2 * (size_t)first, num_nodes) <= wg::TRAVERSAL_STACK_SIZE, "WebGPU: BLAS depth exceeds the traversal stack");
            ctx.object_node_offset[object_i] = 2 * first;
            ctx.object_node_count[object_i] = num_nodes;
            ctx.object_primitive_offset[object_i] = first;
            if(num_nodes > 0){
                const auto& root = blas_nodes[2 * (size_t)first];
                for(int axis = 0; axis < 3; axis++){
                    ctx.object_root_bounds_min[3 * object_i + axis] = root.bounds_min[axis];
                    ctx.object_root_bounds_max[3 * object_i + axis] = root.bounds_max[axis];
                }
            }
        }

        // instance side tables: scene instances then the overlay slot pool, one extra overlay id
        // range per dynamic-motion-blur sample (mirrors the Vulkan backend's shifted ids)
        const size_t num_scene_instances = scene.instances.size();
        constexpr size_t num_overlay_slots = SPEC::ENABLE_OVERLAYS ? (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES : 0;
        constexpr size_t overlay_id_ranges = SPEC::ENABLE_DYNAMIC_MOTION_BLUR ? (size_t)1 + SPEC::MOTION_BLUR_SAMPLES : 1;
        const size_t total_instances = num_scene_instances + num_overlay_slots * overlay_id_ranges;
        ctx.num_scene_instances = (uint32_t)num_scene_instances;
        ctx.total_instances = (uint32_t)total_instances;
        ctx.instance_data_host.assign(total_instances > 0 ? total_instances : 1, wg::InstanceData{});
        ctx.instance_classes_host.assign(total_instances > 0 ? total_instances : 1, 0);
        for(size_t instance_i = 0; instance_i < num_scene_instances; instance_i++){
            const auto& instance = scene.instances[instance_i];
            float world[12];
            for(int element = 0; element < 12; element++){
                world[element] = instance.transform[element];
            }
            wg::fill_instance_entry(ctx, instance_i, (uint32_t)instance.object, world);
        }

        // scene TLAS over the shared-world instances; overlay slots live in their own TLASes
        std::vector<wg::BVHNode> tlas_nodes(num_scene_instances > 0 ? 2 * num_scene_instances : 1);
        std::vector<uint32_t> tlas_primitives(num_scene_instances > 0 ? num_scene_instances : 1);
        uint32_t tlas_node_count;
        {
            std::vector<float> instance_bounds_min(num_scene_instances > 0 ? 3 * num_scene_instances : 1);
            std::vector<float> instance_bounds_max(num_scene_instances > 0 ? 3 * num_scene_instances : 1);
            std::vector<float> instance_centroids(num_scene_instances > 0 ? 3 * num_scene_instances : 1);
            for(size_t instance_i = 0; instance_i < num_scene_instances; instance_i++){
                const auto& entry = ctx.instance_data_host[instance_i];
                float bounds_min[3], bounds_max[3];
                wg::instance_world_bounds(ctx, entry.object, entry.object_to_world, entry.identity != 0, bounds_min, bounds_max);
                for(int axis = 0; axis < 3; axis++){
                    instance_bounds_min[3 * instance_i + axis] = bounds_min[axis];
                    instance_bounds_max[3 * instance_i + axis] = bounds_max[axis];
                    instance_centroids[3 * instance_i + axis] = (bounds_min[axis] + bounds_max[axis]) * 0.5f;
                }
                tlas_primitives[instance_i] = (uint32_t)instance_i;
            }
            std::vector<uint32_t> tlas_temp(tlas_primitives.size());
            tlas_node_count = build_bvh(tlas_nodes.data(), tlas_primitives.data(), tlas_temp.data(), instance_bounds_min.data(), instance_bounds_max.data(), instance_centroids.data(), (uint32_t)num_scene_instances);
            utils::assert_exit(device, wg::bvh_max_depth(tlas_nodes.data(), tlas_node_count) <= wg::TRAVERSAL_STACK_SIZE, "WebGPU: TLAS depth exceeds the traversal stack");
        }

        // one packed u32 buffer for everything the shader addresses by element offset
        std::vector<uint32_t> geometry;
        auto append_section = [&](const void* section_data, size_t bytes) -> uint32_t {
            const uint32_t offset = (uint32_t)geometry.size();
            geometry.resize(geometry.size() + (bytes + 3) / 4);
            if(bytes > 0){
                std::memcpy(geometry.data() + offset, section_data, bytes);
            }
            geometry.resize((geometry.size() + 3) & ~(size_t)3);
            return offset;
        };
        std::vector<wg::MeshRecord> mesh_records(num_meshes > 0 ? num_meshes : 1, wg::MeshRecord{});
        std::vector<uint32_t> texture_texels;
        // content-deduplicated: meshes share maps, and the packed buffer must stay inside
        // maxStorageBufferBindingSize (128MiB on lavapipe)
        std::unordered_map<uint64_t, std::pair<wg::TextureRef, const rendering::raytracing::Texture*>> texture_cache;
        auto add_texture = [&](const rendering::raytracing::Texture& texture) -> wg::TextureRef {
            wg::TextureRef reference{wg::ABSENT, 0, 0};
            if(!texture.present()){
                return reference;
            }
            uint64_t hash = 0xcbf29ce484222325ull;
            const auto mix = [&hash](const void* bytes, size_t count){
                const auto* pointer = (const unsigned char*)bytes;
                for(size_t byte_i = 0; byte_i < count; byte_i++){
                    hash ^= pointer[byte_i];
                    hash *= 0x100000001b3ull;
                }
            };
            mix(&texture.width, sizeof(texture.width));
            mix(&texture.height, sizeof(texture.height));
            mix(texture.pixels.data(), texture.pixels.size());
            const auto hit = texture_cache.find(hash);
            if(hit != texture_cache.end() && hit->second.second->pixels == texture.pixels){
                return hit->second.first;
            }
            reference.offset = (uint32_t)texture_texels.size();
            reference.width = (uint32_t)texture.width;
            reference.height = (uint32_t)texture.height;
            const size_t texel_count = texture.pixels.size() / 4;
            texture_texels.resize(texture_texels.size() + texel_count);
            std::memcpy(texture_texels.data() + reference.offset, texture.pixels.data(), texel_count * 4);
            texture_cache.emplace(hash, std::make_pair(reference, &texture));
            return reference;
        };
        for(size_t mesh_i = 0; mesh_i < num_meshes; mesh_i++){
            const auto& mesh = *all_meshes[mesh_i];
            wg::MeshRecord& record = mesh_records[mesh_i];
            record.vertex_offset = append_section(mesh.vertices.data(), mesh.vertices.size() * sizeof(float));
            record.index_offset = append_section(mesh.indices.data(), mesh.indices.size() * sizeof(int));
            record.tex_coord_offset = mesh.tex_coords.empty() ? wg::ABSENT : append_section(mesh.tex_coords.data(), mesh.tex_coords.size() * sizeof(float));
            record.normal_offset = mesh.normals.empty() ? wg::ABSENT : append_section(mesh.normals.data(), mesh.normals.size() * sizeof(float));
            record.texture = add_texture(mesh.texture);
            record.normal_map = add_texture(mesh.normal_map);
            record.metallic_roughness_map = add_texture(mesh.metallic_roughness_map);
            record.emissive_map = add_texture(mesh.emissive_map);
            record.occlusion_map = add_texture(mesh.occlusion_map);
            record.color[0] = mesh.color[0];
            record.color[1] = mesh.color[1];
            record.color[2] = mesh.color[2];
            record.metallic = mesh.metallic;
            record.roughness = mesh.roughness;
            record.opacity = mesh.opacity;
            record.emissive[0] = mesh.emissive[0];
            record.emissive[1] = mesh.emissive[1];
            record.emissive[2] = mesh.emissive[2];
            record.alpha_cutoff = mesh.alpha_cutoff;
            record.alpha_mode = mesh.alpha_mode;
        }
        const uint32_t triangle_mesh_offset = append_section(triangle_mesh.data(), triangle_mesh.size() * sizeof(uint32_t));
        const uint32_t triangle_local_offset = append_section(triangle_local.data(), triangle_local.size() * sizeof(uint32_t));
        const uint32_t tlas_primitive_offset = append_section(tlas_primitives.data(), tlas_primitives.size() * sizeof(uint32_t));
        std::vector<uint32_t> object_records(4 * (all_objects.size() > 0 ? all_objects.size() : 1), 0);
        for(size_t object_i = 0; object_i < all_objects.size(); object_i++){
            object_records[4 * object_i + 0] = ctx.object_node_offset[object_i];
            object_records[4 * object_i + 1] = ctx.object_node_count[object_i];
            object_records[4 * object_i + 2] = ctx.object_primitive_offset[object_i];
        }
        const uint32_t object_records_offset = append_section(object_records.data(), object_records.size() * sizeof(uint32_t));
        // sRGB texel decode table: replaces the per-texel pow in the shader's bilinear filter
        // (unpack4x8unorm yields exact k/255 values, so the lookup is exact)
        float srgb_lut[256];
        for(int value = 0; value < 256; value++){
            const float x = (float)value / 255.0f;
            srgb_lut[value] = x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
        }
        const uint32_t srgb_lut_offset = append_section(srgb_lut, sizeof(srgb_lut));
        const uint32_t mesh_records_offset = append_section(mesh_records.data(), mesh_records.size() * sizeof(wg::MeshRecord));
        const auto scene_lights = rendering::raytracing::detail::effective_scene_lights<SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING>(scene);
        const uint32_t scene_lights_offset = append_section(scene_lights.data(), scene_lights.size() * sizeof(rendering::raytracing::SceneLight));
        ctx.instance_classes_offset = append_section(ctx.instance_classes_host.data(), ctx.instance_classes_host.size() * sizeof(uint32_t));

        // leaf-ordered packed triangles: the BLAS leaf ranges index this stream directly (slot =
        // first_triangle + leaf-relative index), so the hot intersection loop reads 3 contiguous
        // vec4s instead of chasing triangle table -> mesh record -> index -> vertex indirections;
        // the original global triangle id rides in the first w component for the shading tables
        std::vector<float> packed_triangles(12 * (num_triangles > 0 ? num_triangles : 1), 0.0f);
        for(size_t slot = 0; slot < num_triangles; slot++){
            const uint32_t triangle = blas_primitives[slot];
            const auto& mesh = *all_meshes[triangle_mesh[triangle]];
            const int* index = &mesh.indices[3 * (size_t)triangle_local[triangle]];
            for(int vertex_i = 0; vertex_i < 3; vertex_i++){
                const float* vertex = &mesh.vertices[3 * (size_t)index[vertex_i]];
                float* out = &packed_triangles[12 * slot + 4 * (size_t)vertex_i];
                out[0] = vertex[0];
                out[1] = vertex[1];
                out[2] = vertex[2];
            }
            std::memcpy(&packed_triangles[12 * slot + 3], &triangle, sizeof(uint32_t));
        }

        const WGPUBufferUsage STORAGE_UPLOAD = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        auto upload = [&](wg::BufferResource& resource, const void* upload_data, size_t bytes){
            resource = wg::create_buffer(device, ctx, bytes, STORAGE_UPLOAD);
            if(bytes > 0){
                wgpuQueueWriteBuffer(ctx.queue, resource.buffer, 0, upload_data, bytes);
            }
        };
        upload(ctx.scene_geometry, geometry.data(), geometry.size() * sizeof(uint32_t));
        upload(ctx.texture_data, texture_texels.data(), texture_texels.size() * sizeof(uint32_t));
        upload(ctx.triangles, packed_triangles.data(), packed_triangles.size() * sizeof(float));

        constexpr size_t regions = SPEC::ENABLE_OVERLAYS ? overlay_id_ranges : 1;
        const size_t region_overlays = regions * (SPEC::ENABLE_OVERLAYS ? SPEC::NUM_OVERLAYS : 1);
        ctx.overlay_nodes_host.assign(region_overlays * 2 * (SPEC::ENABLE_OVERLAYS ? SPEC::MAX_OVERLAY_INSTANCES : 1), wg::BVHNode{});
        ctx.overlay_primitives_host.assign(region_overlays * (SPEC::ENABLE_OVERLAYS ? SPEC::MAX_OVERLAY_INSTANCES : 1), 0);
        ctx.overlay_meta_host.assign(region_overlays * 2, 0);
        ctx.overlay_bounds_min.assign(3 * (total_instances > 0 ? total_instances : 1), 0.0f);
        ctx.overlay_bounds_max.assign(3 * (total_instances > 0 ? total_instances : 1), 0.0f);
        ctx.overlay_centroids.assign(3 * (total_instances > 0 ? total_instances : 1), 0.0f);
        ctx.overlay_temp_primitives.assign(SPEC::ENABLE_OVERLAYS ? SPEC::MAX_OVERLAY_INSTANCES : 1, 0);

        {
            using LAYOUT = wg::BufferLayout<SPEC>;
            std::vector<wg::BVHNode> all_nodes = blas_nodes;
            const uint32_t tlas_node_offset = (uint32_t)all_nodes.size();
            all_nodes.insert(all_nodes.end(), tlas_nodes.begin(), tlas_nodes.end());
            ctx.overlay_node_offset = (uint32_t)all_nodes.size();
            if constexpr (SPEC::ENABLE_OVERLAYS){
                all_nodes.insert(all_nodes.end(), ctx.overlay_nodes_host.begin(), ctx.overlay_nodes_host.end());
            }
            upload(ctx.bvh_nodes, all_nodes.data(), all_nodes.size() * sizeof(wg::BVHNode));

            wg::LaunchParams params{};
            params.fb_width = SPEC::FB_WIDTH;
            params.fb_height = SPEC::FB_HEIGHT;
            params.cam_width = SPEC::CAM_WIDTH;
            params.cam_height = SPEC::CAM_HEIGHT;
            params.grid_cols = SPEC::GRID_COLS;
            params.num_cameras = SPEC::NUM_CAMERAS;
            params.num_probes = SPEC::NUM_PROBES;
            params.num_scene_lights = (uint32_t)scene_lights.size();
            params.max_depth = renderer.max_ray_length > 0 ? renderer.max_ray_length : 1e30f;
            params.max_dist = renderer.max_ray_length;
            params.ambient_color[0] = 0.10f;
            params.ambient_color[1] = 0.10f;
            params.ambient_color[2] = 0.10f;
            if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING) {
                params.miss_color_0[0] = 0.f; params.miss_color_0[1] = 0.f; params.miss_color_0[2] = 0.f;
                params.miss_color_1[0] = 0.f; params.miss_color_1[1] = 0.f; params.miss_color_1[2] = 0.f;
            } else {
                params.miss_color_0[0] = .8f; params.miss_color_0[1] = 0.f; params.miss_color_0[2] = 0.f;
                params.miss_color_1[0] = .8f; params.miss_color_1[1] = .8f; params.miss_color_1[2] = .8f;
            }
            params.first_overlay_instance = (uint32_t)num_scene_instances;
            params.triangle_mesh_offset = triangle_mesh_offset;
            params.triangle_local_offset = triangle_local_offset;
            params.object_records_offset = object_records_offset;
            params.tlas_node_offset = tlas_node_offset;
            params.tlas_node_count = tlas_node_count;
            params.tlas_primitive_offset = tlas_primitive_offset;
            params.srgb_lut_offset = srgb_lut_offset;
            params.mesh_records_offset = mesh_records_offset;
            params.scene_lights_offset = scene_lights_offset;
            params.instance_classes_offset = ctx.instance_classes_offset;
            params.overlay_node_offset = ctx.overlay_node_offset;
            params.attachments_offset = LAYOUT::ATTACHMENTS_OFFSET;
            params.overlay_meta_offset = LAYOUT::OVERLAY_META_OFFSET;
            params.overlay_primitives_offset = LAYOUT::OVERLAY_PRIMITIVES_OFFSET;
            params.flow_deltas_offset = LAYOUT::FLOW_DELTAS_OFFSET;
            params.probe_directions_offset = LAYOUT::PROBE_DIRECTIONS_OFFSET;
            params.cameras_offset = LAYOUT::CAMERAS_OFFSET;
            params.out_frame_buffer = LAYOUT::FRAME_BUFFER_OFFSET;
            params.out_depth = LAYOUT::DEPTH_OFFSET;
            params.out_segmentation = LAYOUT::SEGMENTATION_OFFSET;
            params.out_normals = LAYOUT::NORMALS_OFFSET;
            params.out_flow = LAYOUT::FLOW_OFFSET;
            params.out_observation = LAYOUT::OBSERVATION_OFFSET;
            params.out_rgb_accumulator = LAYOUT::RGB_ACCUMULATOR_OFFSET;
            params.out_depth_accumulator = LAYOUT::DEPTH_ACCUMULATOR_OFFSET;
            params.out_collision = LAYOUT::COLLISION_OFFSET;
            wgpuQueueWriteBuffer(ctx.queue, ctx.launch_params.buffer, 0, &params, sizeof(params));
        }
        upload(ctx.instance_data, ctx.instance_data_host.data(), ctx.instance_data_host.size() * sizeof(wg::InstanceData));

        if constexpr (SPEC::ENABLE_OVERLAYS){
            using LAYOUT = wg::BufferLayout<SPEC>;
            wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)LAYOUT::OVERLAY_META_OFFSET * sizeof(uint32_t), ctx.overlay_meta_host.data(), ctx.overlay_meta_host.size() * sizeof(uint32_t));
            wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)LAYOUT::OVERLAY_PRIMITIVES_OFFSET * sizeof(uint32_t), ctx.overlay_primitives_host.data(), ctx.overlay_primitives_host.size() * sizeof(uint32_t));
            std::vector<uint32_t> attachments((size_t)LAYOUT::ATTACHMENTS_WORDS, wg::ABSENT);
            wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)LAYOUT::ATTACHMENTS_OFFSET * sizeof(uint32_t), attachments.data(), attachments.size() * sizeof(uint32_t));
            rendering::raytracing::detail::reset_overlay_state(renderer);
        }

        {
            WGPUBindGroupEntry entries[wg::bindings::COUNT]{};
            auto set_entry = [&](uint32_t binding, wg::BufferResource& resource){
                entries[binding].binding = binding;
                entries[binding].buffer = resource.buffer;
                entries[binding].offset = 0;
                entries[binding].size = WGPU_WHOLE_SIZE;
            };
            set_entry(wg::bindings::LAUNCH_PARAMS, ctx.launch_params);
            set_entry(wg::bindings::DISPATCH_PARAMS, ctx.dispatch_params);
            set_entry(wg::bindings::SCENE_GEOMETRY, ctx.scene_geometry);
            set_entry(wg::bindings::TEXTURE_DATA, ctx.texture_data);
            set_entry(wg::bindings::BVH_NODES, ctx.bvh_nodes);
            set_entry(wg::bindings::TRIANGLES, ctx.triangles);
            set_entry(wg::bindings::INSTANCE_DATA, ctx.instance_data);
            set_entry(wg::bindings::FRAME_INPUTS, ctx.frame_inputs);
            set_entry(wg::bindings::OUTPUTS, ctx.outputs);
            entries[wg::bindings::DISPATCH_PARAMS].size = sizeof(wg::DispatchParams);
            WGPUBindGroupDescriptor descriptor{};
            descriptor.layout = ctx.bind_group_layout;
            descriptor.entryCount = wg::bindings::COUNT;
            descriptor.entries = entries;
            ctx.bind_group = wgpuDeviceCreateBindGroup(ctx.device, &descriptor);
            utils::assert_exit(device, ctx.bind_group != nullptr, "WebGPU: bind group creation failed");
        }

        if constexpr (SPEC::ENABLE_OVERLAYS){
            update(device, renderer); // publish the (empty) overlays and the attachment table
        }
    }


    template <typename DEVICE, typename SPEC>
    void update_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
        namespace wg = rendering::raytracing::backends::webgpu;
        using TI = typename SPEC::TI;
        auto& ctx = wg::context(renderer);

        using LAYOUT = wg::BufferLayout<SPEC>;
        if constexpr (SPEC::HAS_FLOW){
            rendering::raytracing::detail::compose_flow_deltas(renderer, data(renderer.flow_deltas));
            wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)LAYOUT::FLOW_DELTAS_OFFSET * sizeof(uint32_t), data(renderer.flow_deltas), (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(float));
        }
        rendering::raytracing::detail::flush_overlay_transforms(renderer);

        // rebuilt unconditionally: producers may write the transforms tensor directly, which
        // leaves no host-observable dirty flag. queueWriteBuffer is ordered before subsequent
        // submits, so no synchronization with in-flight renders is needed
        wg::rebuild_overlay_region(device, renderer, ctx, 0, (const float*)nullptr);
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            rendering::raytracing::detail::flush_overlay_motion_transforms(renderer);
            constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
            for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                wg::rebuild_overlay_region(device, renderer, ctx, 1 + (uint32_t)sample, data(renderer.transforms_motion) + sample * SLAB);
            }
        }
        if(renderer.attachments_dirty){
            std::vector<uint32_t> attachments((size_t)SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA);
            for(size_t index = 0; index < attachments.size(); index++){
                attachments[index] = (uint32_t)renderer.attachments[index];
            }
            wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)LAYOUT::ATTACHMENTS_OFFSET * sizeof(uint32_t), attachments.data(), attachments.size() * sizeof(uint32_t));
            renderer.attachments_dirty = false;
        }
        const size_t overlay_tail = (size_t)ctx.total_instances - ctx.num_scene_instances;
        if(overlay_tail > 0){
            wgpuQueueWriteBuffer(ctx.queue, ctx.instance_data.buffer, (uint64_t)ctx.num_scene_instances * sizeof(wg::InstanceData), ctx.instance_data_host.data() + ctx.num_scene_instances, overlay_tail * sizeof(wg::InstanceData));
            wgpuQueueWriteBuffer(ctx.queue, ctx.scene_geometry.buffer, (uint64_t)(ctx.instance_classes_offset + ctx.num_scene_instances) * sizeof(uint32_t), ctx.instance_classes_host.data() + ctx.num_scene_instances, overlay_tail * sizeof(uint32_t));
        }
        wgpuQueueWriteBuffer(ctx.queue, ctx.bvh_nodes.buffer, (uint64_t)ctx.overlay_node_offset * sizeof(wg::BVHNode), ctx.overlay_nodes_host.data(), ctx.overlay_nodes_host.size() * sizeof(wg::BVHNode));
        wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)LAYOUT::OVERLAY_PRIMITIVES_OFFSET * sizeof(uint32_t), ctx.overlay_primitives_host.data(), ctx.overlay_primitives_host.size() * sizeof(uint32_t));
        wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)LAYOUT::OVERLAY_META_OFFSET * sizeof(uint32_t), ctx.overlay_meta_host.data(), ctx.overlay_meta_host.size() * sizeof(uint32_t));
    }

    template <typename DEVICE, typename SPEC>
    void update_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
    }

    // CPU expansion into the host-resident tensors (residency is a backend property; the
    // device-resident path is the OptiX backend)
    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        static_assert(SPEC::HAS_TRANSFORM_PAIR, "expand_motion_transforms requires a dynamic-motion-blur or flow renderer specification");
        rendering::raytracing::detail::expand_motion_transforms_host(renderer);
    }

    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        static_assert(SPEC::HAS_TRANSFORM_PAIR, "expand_motion_transforms requires a dynamic-motion-blur or flow renderer specification");
    }

    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        expand_motion_transforms_launch(device, renderer);
        expand_motion_transforms_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        update_launch(device, renderer);
        update_sync(device, renderer);
    }


    // renderer memory-domain copies: the in-flight wait settles the staging readbacks into the
    // host-resident tensors, so the transfer delegates to the host-device tensor copy
    template <typename TO_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(rendering::raytracing::backends::Device<rendering::raytracing::backends::Webgpu>& from_device, TO_DEVICE& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
        rendering::raytracing::backends::webgpu::wait_in_flight(to_device, *from_device.context);
        copy(to_device, to_device, from, to);
    }
    template <typename FROM_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(FROM_DEVICE& from_device, rendering::raytracing::backends::Device<rendering::raytracing::backends::Webgpu>& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
        rendering::raytracing::backends::webgpu::wait_in_flight(from_device, *to_device.context);
        copy(from_device, from_device, from, to);
    }

    template <typename DEVICE, typename SPEC>
    void generate_probe_directions(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Probe rays disabled (RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS=1)");
        return;
#else
        namespace wg = rendering::raytracing::backends::webgpu;
        auto& ctx = wg::context(renderer);
        std::vector<float> directions = rendering::raytracing::detail::generate_probe_direction_vectors<SPEC>();
        wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)wg::BufferLayout<SPEC>::PROBE_DIRECTIONS_OFFSET * sizeof(uint32_t), directions.data(), directions.size() * sizeof(float));
#endif
    }

    template <typename DEVICE, typename SPEC>
    void render_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        namespace wg = rendering::raytracing::backends::webgpu;
        using TI = typename SPEC::TI;
        auto& ctx = wg::context(renderer);
        render_sync(device, renderer); // single-buffered: settle a previous launch before reusing the staging buffers

        using LAYOUT = wg::BufferLayout<SPEC>;
        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)LAYOUT::CAMERAS_OFFSET * sizeof(uint32_t), data(renderer.cameras), camera_bytes);
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)LAYOUT::CAMERAS_OFFSET * sizeof(uint32_t) + camera_bytes, data(renderer.cameras_open), camera_bytes);
        }

        constexpr uint32_t fb_groups_x = (SPEC::FB_WIDTH + wg::WORKGROUP_SIZE_X - 1) / wg::WORKGROUP_SIZE_X;
        constexpr uint32_t fb_groups_y = (SPEC::FB_HEIGHT + wg::WORKGROUP_SIZE_Y - 1) / wg::WORKGROUP_SIZE_Y;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            if constexpr (SPEC::HAS_RGB){
                wgpuCommandEncoderClearBuffer(encoder, ctx.outputs.buffer, (uint64_t)LAYOUT::RGB_ACCUMULATOR_OFFSET * sizeof(uint32_t), (uint64_t)LAYOUT::RGB_ACCUMULATOR_WORDS * sizeof(uint32_t));
            }
            if constexpr (SPEC::HAS_DEPTH){
                wgpuCommandEncoderClearBuffer(encoder, ctx.outputs.buffer, (uint64_t)LAYOUT::DEPTH_ACCUMULATOR_OFFSET * sizeof(uint32_t), (uint64_t)LAYOUT::DEPTH_ACCUMULATOR_WORDS * sizeof(uint32_t));
            }
        }
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            const auto dispatch = [&](WGPUComputePipeline pipeline, uint32_t slot, uint32_t groups_x, uint32_t groups_y){
                const uint32_t dynamic_offset = slot * wg::DISPATCH_PARAMS_STRIDE;
                wgpuComputePassEncoderSetPipeline(pass, pipeline);
                wgpuComputePassEncoderSetBindGroup(pass, 0, ctx.bind_group, 1, &dynamic_offset);
                wgpuComputePassEncoderDispatchWorkgroups(pass, groups_x, groups_y, 1);
            };
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                // the launch-level motion loop: each sample's dispatches read that sample's
                // pre-built overlay BVH region and shutter time from its dispatch-params slot;
                // WebGPU orders the storage accesses between dispatches, no explicit barriers
                for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                    if constexpr (SPEC::HAS_RGB){
                        dispatch(ctx.rgb_pipeline, 1 + (uint32_t)sample, fb_groups_x, fb_groups_y);
                    }
                    if constexpr (SPEC::HAS_DEPTH){
                        dispatch(ctx.depth_pipeline, 1 + (uint32_t)sample, fb_groups_x, fb_groups_y);
                    }
                }
                if constexpr (SPEC::HAS_SEGMENTATION){
                    dispatch(ctx.segmentation_pipeline, 0, fb_groups_x, fb_groups_y);
                }
                if constexpr (SPEC::HAS_NORMALS){
                    dispatch(ctx.normals_pipeline, 0, fb_groups_x, fb_groups_y);
                }
                if constexpr (SPEC::HAS_FLOW){
                    dispatch(ctx.flow_pipeline, 0, fb_groups_x, fb_groups_y);
                }
                dispatch(ctx.resolve_pipeline, 0, fb_groups_x, fb_groups_y);
            }
            else{
                if constexpr (SPEC::HAS_RGB){
                    dispatch(ctx.rgb_pipeline, 0, fb_groups_x, fb_groups_y);
                }
                if constexpr (SPEC::HAS_DEPTH){
                    dispatch(ctx.depth_pipeline, 0, fb_groups_x, fb_groups_y);
                }
                if constexpr (SPEC::HAS_SEGMENTATION){
                    dispatch(ctx.segmentation_pipeline, 0, fb_groups_x, fb_groups_y);
                }
                if constexpr (SPEC::HAS_NORMALS){
                    dispatch(ctx.normals_pipeline, 0, fb_groups_x, fb_groups_y);
                }
                if constexpr (SPEC::HAS_FLOW){
                    dispatch(ctx.flow_pipeline, 0, fb_groups_x, fb_groups_y);
                }
            }
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }
        if constexpr (SPEC::HAS_RGB){
            wgpuCommandEncoderCopyBufferToBuffer(encoder, ctx.outputs.buffer, (uint64_t)LAYOUT::FRAME_BUFFER_OFFSET * sizeof(uint32_t), ctx.staging_frame_buffer.buffer, 0, (uint64_t)LAYOUT::FRAME_BUFFER_WORDS * sizeof(uint32_t));
        }
        if constexpr (SPEC::HAS_DEPTH){
            wgpuCommandEncoderCopyBufferToBuffer(encoder, ctx.outputs.buffer, (uint64_t)LAYOUT::DEPTH_OFFSET * sizeof(uint32_t), ctx.staging_depth.buffer, 0, (uint64_t)LAYOUT::DEPTH_WORDS * sizeof(uint32_t));
        }
        if constexpr (SPEC::HAS_SEGMENTATION){
            wgpuCommandEncoderCopyBufferToBuffer(encoder, ctx.outputs.buffer, (uint64_t)LAYOUT::SEGMENTATION_OFFSET * sizeof(uint32_t), ctx.staging_segmentation.buffer, 0, (uint64_t)LAYOUT::SEGMENTATION_WORDS * sizeof(uint32_t));
        }
        if constexpr (SPEC::HAS_NORMALS){
            wgpuCommandEncoderCopyBufferToBuffer(encoder, ctx.outputs.buffer, (uint64_t)LAYOUT::NORMALS_OFFSET * sizeof(uint32_t), ctx.staging_normals.buffer, 0, (uint64_t)LAYOUT::NORMALS_WORDS * sizeof(uint32_t));
        }
        if constexpr (SPEC::HAS_FLOW){
            wgpuCommandEncoderCopyBufferToBuffer(encoder, ctx.outputs.buffer, (uint64_t)LAYOUT::FLOW_OFFSET * sizeof(uint32_t), ctx.staging_flow.buffer, 0, (uint64_t)LAYOUT::FLOW_WORDS * sizeof(uint32_t));
        }
        if constexpr (SPEC::HAS_OBSERVATION){
            wgpuCommandEncoderCopyBufferToBuffer(encoder, ctx.outputs.buffer, (uint64_t)LAYOUT::OBSERVATION_OFFSET * sizeof(uint32_t), ctx.staging_observation.buffer, 0, (uint64_t)LAYOUT::OBSERVATION_WORDS * sizeof(uint32_t));
        }
        WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
        wgpuQueueSubmit(ctx.queue, 1, &command_buffer);
        wgpuCommandBufferRelease(command_buffer);
        ctx.render_in_flight = true;
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        namespace wg = rendering::raytracing::backends::webgpu;
        wg::settle_render(device, wg::context(renderer));
    }

    // render produces the image outputs the spec declares; the collision-probe pass is the
    // separate probe verb so it can be scheduled independently (e.g. alongside update)
    template <typename DEVICE, typename SPEC>
    void probe_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        namespace wg = rendering::raytracing::backends::webgpu;
        auto& ctx = wg::context(renderer);
        if(ctx.collision_pipeline == nullptr){
            return;
        }
        probe_sync(device, renderer);
        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        wgpuQueueWriteBuffer(ctx.queue, ctx.frame_inputs.buffer, (uint64_t)wg::BufferLayout<SPEC>::CAMERAS_OFFSET * sizeof(uint32_t), data(renderer.cameras), camera_bytes);
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
        {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
            const uint32_t dynamic_offset = 0;
            wgpuComputePassEncoderSetPipeline(pass, ctx.collision_pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, ctx.bind_group, 1, &dynamic_offset);
            wgpuComputePassEncoderDispatchWorkgroups(pass, (SPEC::NUM_CAMERAS + wg::WORKGROUP_SIZE_X - 1) / wg::WORKGROUP_SIZE_X, (SPEC::NUM_PROBES + wg::WORKGROUP_SIZE_Y - 1) / wg::WORKGROUP_SIZE_Y, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }
        wgpuCommandEncoderCopyBufferToBuffer(encoder, ctx.outputs.buffer, (uint64_t)wg::BufferLayout<SPEC>::COLLISION_OFFSET * sizeof(uint32_t), ctx.staging_collision.buffer, 0, (uint64_t)wg::BufferLayout<SPEC>::COLLISION_WORDS * sizeof(uint32_t));
        WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
        wgpuQueueSubmit(ctx.queue, 1, &command_buffer);
        wgpuCommandBufferRelease(command_buffer);
        ctx.collision_in_flight = true;
    }

    template <typename DEVICE, typename SPEC>
    void probe_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        namespace wg = rendering::raytracing::backends::webgpu;
        wg::settle_probe(device, wg::context(renderer));
    }

    template <typename DEVICE, typename SPEC>
    void synchronize(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        rendering::raytracing::backends::webgpu::wait_in_flight(device, rendering::raytracing::backends::webgpu::context(renderer));
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Webgpu>& renderer){
        namespace wg = rendering::raytracing::backends::webgpu;
        if(renderer.backend != nullptr){
            auto& ctx = wg::context(renderer);
            wg::wait_in_flight(device, ctx);
            wg::destroy_scene_resources(ctx);
            wg::destroy_buffer(ctx.launch_params);
            wg::destroy_buffer(ctx.frame_inputs);
            wg::destroy_buffer(ctx.outputs);
            wg::destroy_buffer(ctx.dispatch_params);
            wg::destroy_buffer(ctx.staging_frame_buffer);
            wg::destroy_buffer(ctx.staging_depth);
            wg::destroy_buffer(ctx.staging_segmentation);
            wg::destroy_buffer(ctx.staging_normals);
            wg::destroy_buffer(ctx.staging_flow);
            wg::destroy_buffer(ctx.staging_observation);
            wg::destroy_buffer(ctx.staging_collision);
            if(ctx.rgb_pipeline != nullptr){ wgpuComputePipelineRelease(ctx.rgb_pipeline); }
            if(ctx.depth_pipeline != nullptr){ wgpuComputePipelineRelease(ctx.depth_pipeline); }
            if(ctx.collision_pipeline != nullptr){ wgpuComputePipelineRelease(ctx.collision_pipeline); }
            if(ctx.segmentation_pipeline != nullptr){ wgpuComputePipelineRelease(ctx.segmentation_pipeline); }
            if(ctx.normals_pipeline != nullptr){ wgpuComputePipelineRelease(ctx.normals_pipeline); }
            if(ctx.flow_pipeline != nullptr){ wgpuComputePipelineRelease(ctx.flow_pipeline); }
            if(ctx.resolve_pipeline != nullptr){ wgpuComputePipelineRelease(ctx.resolve_pipeline); }
            if(ctx.pipeline_layout != nullptr){ wgpuPipelineLayoutRelease(ctx.pipeline_layout); }
            if(ctx.bind_group_layout != nullptr){ wgpuBindGroupLayoutRelease(ctx.bind_group_layout); }
            if(ctx.module != nullptr){ wgpuShaderModuleRelease(ctx.module); }
            if(ctx.queue != nullptr){ wgpuQueueRelease(ctx.queue); }
            if(ctx.device != nullptr){ wgpuDeviceRelease(ctx.device); }
            if(ctx.adapter != nullptr){ wgpuAdapterRelease(ctx.adapter); }
            if(ctx.instance != nullptr){ wgpuInstanceRelease(ctx.instance); }
            delete renderer.backend;
            renderer.backend = nullptr;
            renderer.device.context = nullptr;
        }
        free(device, renderer.cameras);
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            free(device, renderer.cameras_open);
        }
        if constexpr (SPEC::HAS_RGB) {
            free(device, renderer.frame_buffer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            free(device, renderer.depth_buffer);
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            free(device, renderer.segmentation_buffer);
        }
        if constexpr (SPEC::HAS_NORMALS) {
            free(device, renderer.normals_buffer);
        }
        if constexpr (SPEC::HAS_FLOW) {
            free(device, renderer.flow_buffer);
            if constexpr (SPEC::ENABLE_OVERLAYS) {
                free(device, renderer.flow_deltas);
            }
        }
        if constexpr (SPEC::HAS_OBSERVATION) {
            free(device, renderer.observation);
        }
        if constexpr (SPEC::ENABLE_OVERLAYS) {
            free(device, renderer.transforms);
        }
        if constexpr (SPEC::HAS_TRANSFORM_PAIR) {
            free(device, renderer.transforms_pair);
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            free(device, renderer.transforms_motion);
            if constexpr (SPEC::HAS_RGB) {
                free(device, renderer.rgb_accumulator);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                free(device, renderer.depth_accumulator);
            }
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        free(device, renderer.collision_results);
#endif
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END


#include "../../operations_cpu_post.h"

#endif
