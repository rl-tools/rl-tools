#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_METAL_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_METAL_OPERATIONS_CPU_H

#include "../../renderer.h"
#include "../../operations_cpu_common.h"
#include "context.h"
#include "device_source.h"

#include <vector>
#include <cstring>
#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing::backends {
        template <typename SPEC>
        struct RendererState<rendering::raytracing::backends::Metal, SPEC>: metal::Context {};

        template <typename SPEC>
        struct LibraryState<rendering::raytracing::backends::Metal, SPEC> {};

        template <typename SPEC>
        struct SceneState<rendering::raytracing::backends::Metal, SPEC> {};
    }

    namespace rendering::raytracing::backends::metal{
        template <typename SPEC>
        Context& context(rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
            return *renderer.backend;
        }

        inline NS::SharedPtr<MTL::Texture> make_texture(MTL::Device* device, const uint8_t* pixels, int width, int height, bool srgb){
            auto descriptor = MTL::TextureDescriptor::texture2DDescriptor(srgb ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm, width, height, false);
            descriptor->setUsage(MTL::TextureUsageShaderRead);
            descriptor->setStorageMode(MTL::StorageModeShared);
            auto texture = NS::TransferPtr(device->newTexture(descriptor));
            texture->replaceRegion(MTL::Region::Make2D(0, 0, width, height), 0, pixels, (NS::UInteger)width * 4);
            return texture;
        }

        inline void wait_in_flight(Context& ctx){
            if(ctx.in_flight.get() != nullptr){
                ctx.in_flight->waitUntilCompleted();
                ctx.in_flight.reset();
            }
            if(ctx.in_flight_collision.get() != nullptr){
                ctx.in_flight_collision->waitUntilCompleted();
                ctx.in_flight_collision.reset();
            }
            if(ctx.in_flight_update.get() != nullptr){
                ctx.in_flight_update->waitUntilCompleted();
                ctx.in_flight_update.reset();
            }
        }

        template <typename SPEC>
        void encode_fullscreen_pass(Context& ctx, MTL::CommandBuffer* command_buffer, MTL::ComputePipelineState* pipeline, MTL::Buffer* output, float shutter_t = 0.f){
            MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
            encoder->setComputePipelineState(pipeline);
            encoder->setBuffer(ctx.launch_params.get(), 0, bindings::LAUNCH_PARAMS);
            encoder->setBuffer(ctx.cameras.get(), 0, bindings::CAMERAS_CLOSE);
            encoder->setBuffer(ctx.cameras_open.get() != nullptr ? ctx.cameras_open.get() : ctx.cameras.get(), 0, bindings::CAMERAS_OPEN);
            encoder->setBuffer(output, 0, bindings::OUTPUT);
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                if(ctx.rgb_accumulator.get() != nullptr){
                    encoder->setBuffer(ctx.rgb_accumulator.get(), 0, bindings::RGB_ACCUMULATOR);
                }
                if(ctx.depth_accumulator.get() != nullptr){
                    encoder->setBuffer(ctx.depth_accumulator.get(), 0, bindings::DEPTH_ACCUMULATOR);
                }
                encoder->setBytes(&shutter_t, sizeof(float), bindings::SHUTTER);
            }
            encoder->setBuffer(ctx.mesh_records.get(), 0, bindings::MESH_RECORDS);
            if(ctx.scene_lights.get() != nullptr){
                encoder->setBuffer(ctx.scene_lights.get(), 0, bindings::SCENE_LIGHTS);
            }
            encoder->setAccelerationStructure(ctx.acceleration_structure.get(), bindings::ACCELERATION_STRUCTURE);
            encoder->setBuffer(ctx.instance_record_base.get(), 0, bindings::INSTANCE_RECORD_BASE);
            encoder->setBuffer(ctx.instance_data.get(), 0, bindings::INSTANCE_DATA);
            encoder->setBuffer(ctx.overlay_structures.get(), 0, bindings::OVERLAY_STRUCTURES);
            encoder->setBuffer(ctx.overlay_attachments.get(), 0, bindings::OVERLAY_ATTACHMENTS);
            encoder->setBuffer(ctx.instance_classes.get(), 0, bindings::INSTANCE_CLASSES);
            if(ctx.observation.get() != nullptr){
                encoder->setBuffer(ctx.observation.get(), 0, bindings::OBSERVATION);
            }
            if(ctx.flow_deltas.get() != nullptr){
                encoder->setBuffer(ctx.flow_deltas.get(), 0, bindings::FLOW_DELTAS);
            }
            for(auto& object_acceleration_structure : ctx.object_acceleration_structures){
                encoder->useResource(object_acceleration_structure.get(), MTL::ResourceUsageRead);
            }
            for(auto& overlay_acceleration_structure : ctx.overlay_acceleration_structures){
                encoder->useResource(overlay_acceleration_structure.get(), MTL::ResourceUsageRead);
            }
            for(auto& buffer : ctx.mesh_buffers){
                encoder->useResource(buffer.get(), MTL::ResourceUsageRead);
            }
            for(auto& texture : ctx.mesh_textures){
                encoder->useResource(texture.get(), MTL::ResourceUsageRead);
            }
            if(ctx.dummy_texture.get() != nullptr){
                encoder->useResource(ctx.dummy_texture.get(), MTL::ResourceUsageRead);
            }
            encoder->dispatchThreads(MTL::Size::Make(SPEC::FB_WIDTH, SPEC::FB_HEIGHT, 1), MTL::Size::Make(8, 8, 1));
            encoder->endEncoding();
        }

        template <typename SPEC>
        void encode_resolve_pass(Context& ctx, MTL::CommandBuffer* command_buffer){
            MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
            encoder->setComputePipelineState(ctx.resolve_pipeline.get());
            encoder->setBuffer(ctx.launch_params.get(), 0, bindings::LAUNCH_PARAMS);
            if constexpr (SPEC::HAS_RGB){
                encoder->setBuffer(ctx.frame_buffer.get(), 0, bindings::OUTPUT);
                encoder->setBuffer(ctx.rgb_accumulator.get(), 0, bindings::RGB_ACCUMULATOR);
            }
            if constexpr (SPEC::HAS_OBSERVATION){
                encoder->setBuffer(ctx.observation.get(), 0, bindings::OBSERVATION);
            }
            if constexpr (SPEC::HAS_DEPTH){
                encoder->setBuffer(ctx.depth_accumulator.get(), 0, bindings::DEPTH_ACCUMULATOR);
                encoder->setBuffer(ctx.depth_buffer.get(), 0, bindings::DEPTH_OUTPUT);
            }
            encoder->dispatchThreads(MTL::Size::Make(SPEC::FB_WIDTH, SPEC::FB_HEIGHT, 1), MTL::Size::Make(8, 8, 1));
            encoder->endEncoding();
        }

        template <typename SPEC>
        void encode_collision_pass(Context& ctx, MTL::CommandBuffer* command_buffer){
            MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
            encoder->setComputePipelineState(ctx.collision_pipeline.get());
            encoder->setBuffer(ctx.launch_params.get(), 0, bindings::LAUNCH_PARAMS);
            encoder->setBuffer(ctx.cameras.get(), 0, bindings::CAMERAS_CLOSE);
            encoder->setBuffer(ctx.probe_directions.get(), 0, bindings::PROBE_DIRECTIONS);
            encoder->setBuffer(ctx.collision_results.get(), 0, bindings::COLLISION_RESULTS);
            encoder->setAccelerationStructure(ctx.acceleration_structure.get(), bindings::ACCELERATION_STRUCTURE);
            encoder->setBuffer(ctx.overlay_structures.get(), 0, bindings::OVERLAY_STRUCTURES);
            encoder->setBuffer(ctx.overlay_attachments.get(), 0, bindings::OVERLAY_ATTACHMENTS);
            for(auto& object_acceleration_structure : ctx.object_acceleration_structures){
                encoder->useResource(object_acceleration_structure.get(), MTL::ResourceUsageRead);
            }
            for(auto& overlay_acceleration_structure : ctx.overlay_acceleration_structures){
                encoder->useResource(overlay_acceleration_structure.get(), MTL::ResourceUsageRead);
            }
            encoder->dispatchThreads(MTL::Size::Make(SPEC::NUM_CAMERAS, SPEC::NUM_PROBES, 1), MTL::Size::Make(8, 8, 1));
            encoder->endEncoding();
        }
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        static_assert(utils::typing::is_same_v<typename SPEC::T, float>, "The Metal raytracing backend requires T = float");

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
        }

        renderer.backend = new rendering::raytracing::backends::RendererState<rendering::raytracing::backends::Metal, SPEC>{};
        renderer.device.context = renderer.backend;
        auto* ctx = renderer.backend;
        ctx->device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        utils::assert_exit(device, ctx->device.get() != nullptr, "Metal: no default device available");
        utils::assert_exit(device, ctx->device->supportsRaytracing(), "Metal: device does not support raytracing");
        ctx->queue = NS::TransferPtr(ctx->device->newCommandQueue());

        auto compile_options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
        compile_options->setLanguageVersion(MTL::LanguageVersion3_0);
        auto source = NS::TransferPtr(NS::String::string(rendering::raytracing::backends::metal::device_source(), NS::UTF8StringEncoding));
        NS::Error* error = nullptr;
        ctx->library = NS::TransferPtr(ctx->device->newLibrary(source.get(), compile_options.get(), &error));
        if(ctx->library.get() == nullptr){
            RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Metal device source compilation failed: " << (error != nullptr ? error->localizedDescription()->utf8String() : "unknown error"));
            utils::assert_exit(device, false, "Metal device source compilation failed");
        }

        constexpr typename SPEC::TI cam_pixels = SPEC::CAM_PIXELS;
        // the output tensors alias the shared MTLBuffers the kernels write (see frame_buffer/
        // depth_buffer/segmentation_buffer/collision_results/observation accessors)
        if constexpr (SPEC::HAS_RGB) {
            ctx->frame_buffer = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * sizeof(uint32_t), MTL::ResourceStorageModeShared));
            renderer.frame_buffer._data = (uint32_t*)ctx->frame_buffer->contents();
        }
        if constexpr (SPEC::HAS_DEPTH) {
            ctx->depth_buffer = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * sizeof(float), MTL::ResourceStorageModeShared));
            renderer.depth_buffer._data = (float*)ctx->depth_buffer->contents();
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            ctx->segmentation_buffer = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * sizeof(uint32_t), MTL::ResourceStorageModeShared));
            renderer.segmentation_buffer._data = (uint32_t*)ctx->segmentation_buffer->contents();
        }
        if constexpr (SPEC::HAS_NORMALS) {
            ctx->normals_buffer = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * 3 * sizeof(float), MTL::ResourceStorageModeShared));
            renderer.normals_buffer._data = (float*)ctx->normals_buffer->contents();
        }
        if constexpr (SPEC::HAS_FLOW) {
            ctx->flow_buffer = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * 2 * sizeof(float), MTL::ResourceStorageModeShared));
            renderer.flow_buffer._data = (float*)ctx->flow_buffer->contents();
            if constexpr (SPEC::ENABLE_OVERLAYS) {
                ctx->flow_deltas = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(float), MTL::ResourceStorageModeShared));
                renderer.flow_deltas._data = (float*)ctx->flow_deltas->contents();
            }
        }
        if constexpr (SPEC::HAS_OBSERVATION) {
            static_assert(utils::typing::is_same_v<typename SPEC::OBSERVATION_T, float>, "The Metal raytracing backend requires OBSERVATION_T = float");
            ctx->observation = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * SPEC::OBSERVATION_CHANNELS * sizeof(float), MTL::ResourceStorageModeShared));
            renderer.observation._data = (float*)ctx->observation->contents();
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            // linear-radiance accumulators for the launch-level motion loop, resolved into the
            // packed frame buffer / depth buffer by resolve_outputs after the last sample pass
            if constexpr (SPEC::HAS_RGB) {
                ctx->rgb_accumulator = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * 3 * sizeof(float), MTL::ResourceStorageModeShared));
                renderer.rgb_accumulator._data = (float*)ctx->rgb_accumulator->contents();
            }
            if constexpr (SPEC::HAS_DEPTH) {
                ctx->depth_accumulator = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * sizeof(float), MTL::ResourceStorageModeShared));
                renderer.depth_accumulator._data = (float*)ctx->depth_accumulator->contents();
            }
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        ctx->collision_results = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES * sizeof(rendering::raytracing::CollisionResult), MTL::ResourceStorageModeShared));
        renderer.collision_results._data = (rendering::raytracing::CollisionResult*)ctx->collision_results->contents();
#endif
        ctx->launch_params = NS::TransferPtr(ctx->device->newBuffer(sizeof(metal::LaunchParams), MTL::ResourceStorageModeShared));

        // camera tensors alias the shared MTLBuffers the encoders bind (see cameras(device,
        // renderer)): host writes are consumed by the next launch with no staging copy
        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        ctx->cameras = NS::TransferPtr(ctx->device->newBuffer(camera_bytes, MTL::ResourceStorageModeShared));
        renderer.cameras._data = (rendering::raytracing::Camera<typename SPEC::T>*)ctx->cameras->contents();
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            ctx->cameras_open = NS::TransferPtr(ctx->device->newBuffer(camera_bytes, MTL::ResourceStorageModeShared));
            renderer.cameras_open._data = (rendering::raytracing::Camera<typename SPEC::T>*)ctx->cameras_open->contents();
        }
        rendering::raytracing::detail::announce_backend(renderer);
    }

    template <typename DEVICE, typename SPEC>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer);

    template <typename DEVICE, typename SPEC, typename METADATA_T>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool, const rendering::SceneMetadata<METADATA_T>& metadata){
        renderer.max_ray_length = (typename SPEC::T)metadata.max_ray_length;
        rendering::raytracing::detail::announce_configuration<SPEC>();
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();

        RL_TOOLS_RENDERING_RAYTRACING_LOG("building " << scene.objects.size() << " object(s), " << scene.instances.size() << " instance(s) ...");

        // scene-dependent resources from a previous init are released here; the compiled library
        // and pipelines below are spec-dependent and survive re-init
        ctx.mesh_buffers.clear();
        ctx.mesh_textures.clear();
        ctx.object_acceleration_structures.clear();
        ctx.scene_lights = NS::SharedPtr<MTL::Buffer>{};

        const uint8_t dummy_pixel[4] = {255, 255, 255, 255};
        ctx.dummy_texture = metal::make_texture(ctx.device.get(), dummy_pixel, 1, 1, false);
        const MTL::ResourceID dummy_texture_id = ctx.dummy_texture->gpuResourceID();

        std::vector<const rendering::raytracing::Object*> all_objects;
        for(const auto& object : scene.objects){
            all_objects.push_back(&object);
        }
        if constexpr (SPEC::ENABLE_OVERLAYS){
            rendering::raytracing::detail::register_pool_assets(device, renderer, pool, all_objects);
        }

        size_t total_meshes = 0;
        for(const auto* object : all_objects){
            total_meshes += object->meshes.size();
        }
        std::vector<metal::MeshRecord> mesh_records(total_meshes);
        std::vector<uint32_t> object_record_base;
        std::vector<std::vector<NS::Object*>> object_geometry_descriptors(all_objects.size());

        size_t m = 0;
        for(size_t object_i = 0; object_i < all_objects.size(); object_i++){
            object_record_base.push_back((uint32_t)m);
            for(const auto& md : all_objects[object_i]->meshes){
            auto& geometry_descriptors = object_geometry_descriptors[object_i];
            metal::MeshRecord& record = mesh_records[m];
            record = {};
            record.texture = dummy_texture_id;
            record.normal_map = dummy_texture_id;
            record.metallic_roughness_map = dummy_texture_id;
            record.emissive_map = dummy_texture_id;
            record.occlusion_map = dummy_texture_id;

            size_t num_triangles = md.indices.size() / 3;

            auto vertex_buffer = NS::TransferPtr(ctx.device->newBuffer(md.vertices.data(), md.vertices.size() * sizeof(float), MTL::ResourceStorageModeShared));
            auto index_buffer = NS::TransferPtr(ctx.device->newBuffer(md.indices.data(), md.indices.size() * sizeof(int), MTL::ResourceStorageModeShared));
            record.vertices = vertex_buffer->gpuAddress();
            record.index = index_buffer->gpuAddress();

            MTL::AccelerationStructureTriangleGeometryDescriptor* geometry = MTL::AccelerationStructureTriangleGeometryDescriptor::descriptor();
            geometry->setVertexBuffer(vertex_buffer.get());
            geometry->setVertexStride(3 * sizeof(float));
            geometry->setVertexFormat(MTL::AttributeFormatFloat3);
            geometry->setIndexBuffer(index_buffer.get());
            geometry->setIndexType(MTL::IndexTypeUInt32);
            geometry->setTriangleCount(num_triangles);
            geometry->setOpaque(true);
            geometry_descriptors.push_back(geometry);

            ctx.mesh_buffers.push_back(vertex_buffer);
            ctx.mesh_buffers.push_back(index_buffer);

            if(!md.tex_coords.empty()){
                auto tex_coord_buffer = NS::TransferPtr(ctx.device->newBuffer(md.tex_coords.data(), md.tex_coords.size() * sizeof(float), MTL::ResourceStorageModeShared));
                record.tex_coord = tex_coord_buffer->gpuAddress();
                ctx.mesh_buffers.push_back(tex_coord_buffer);
            }
            if(!md.normals.empty()){
                auto normal_buffer = NS::TransferPtr(ctx.device->newBuffer(md.normals.data(), md.normals.size() * sizeof(float), MTL::ResourceStorageModeShared));
                record.normal = normal_buffer->gpuAddress();
                ctx.mesh_buffers.push_back(normal_buffer);
            }

            record.color[0] = md.color[0];
            record.color[1] = md.color[1];
            record.color[2] = md.color[2];
            record.metallic = md.metallic;
            record.roughness = md.roughness;
            record.opacity = md.opacity;
            record.emissive[0] = md.emissive[0];
            record.emissive[1] = md.emissive[1];
            record.emissive[2] = md.emissive[2];
            record.alpha_cutoff = md.alpha_cutoff;
            record.alpha_mode = md.alpha_mode;

            if(md.texture.present()){
                auto texture = metal::make_texture(ctx.device.get(), md.texture.pixels.data(), md.texture.width, md.texture.height, true);
                record.texture = texture->gpuResourceID();
                record.has_texture = 1;
                ctx.mesh_textures.push_back(texture);
            }
            if(md.normal_map.present()){
                auto texture = metal::make_texture(ctx.device.get(), md.normal_map.pixels.data(), md.normal_map.width, md.normal_map.height, false);
                record.normal_map = texture->gpuResourceID();
                record.has_normal_map = 1;
                ctx.mesh_textures.push_back(texture);
            }
            if(md.metallic_roughness_map.present()){
                auto texture = metal::make_texture(ctx.device.get(), md.metallic_roughness_map.pixels.data(), md.metallic_roughness_map.width, md.metallic_roughness_map.height, false);
                record.metallic_roughness_map = texture->gpuResourceID();
                record.has_metallic_roughness_map = 1;
                ctx.mesh_textures.push_back(texture);
            }
            if(md.emissive_map.present()){
                auto texture = metal::make_texture(ctx.device.get(), md.emissive_map.pixels.data(), md.emissive_map.width, md.emissive_map.height, true);
                record.emissive_map = texture->gpuResourceID();
                record.has_emissive_map = 1;
                ctx.mesh_textures.push_back(texture);
            }
            if(md.occlusion_map.present()){
                auto texture = metal::make_texture(ctx.device.get(), md.occlusion_map.pixels.data(), md.occlusion_map.width, md.occlusion_map.height, false);
                record.occlusion_map = texture->gpuResourceID();
                record.has_occlusion_map = 1;
                ctx.mesh_textures.push_back(texture);
            }
            m++;
            }
        }

        // one primitive (bottom-level) acceleration structure per object, shared by its instances
        {
            MTL::CommandBuffer* command_buffer = ctx.queue->commandBuffer();
            MTL::AccelerationStructureCommandEncoder* encoder = command_buffer->accelerationStructureCommandEncoder();
            std::vector<NS::SharedPtr<MTL::Buffer>> scratch_buffers;
            for(size_t object_i = 0; object_i < all_objects.size(); object_i++){
                auto& geometry_descriptors = object_geometry_descriptors[object_i];
                NS::Array* geometry_array = NS::Array::array((const NS::Object* const*)geometry_descriptors.data(), geometry_descriptors.size());
                MTL::PrimitiveAccelerationStructureDescriptor* accel_descriptor = MTL::PrimitiveAccelerationStructureDescriptor::descriptor();
                accel_descriptor->setGeometryDescriptors(geometry_array);
                MTL::AccelerationStructureSizes sizes = ctx.device->accelerationStructureSizes(accel_descriptor);
                auto object_acceleration_structure = NS::TransferPtr(ctx.device->newAccelerationStructure(sizes.accelerationStructureSize));
                auto scratch_buffer = NS::TransferPtr(ctx.device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate));
                encoder->buildAccelerationStructure(object_acceleration_structure.get(), accel_descriptor, scratch_buffer.get(), 0);
                scratch_buffers.push_back(scratch_buffer);
                ctx.object_acceleration_structures.push_back(object_acceleration_structure);
            }
            encoder->endEncoding();
            command_buffer->commit();
            command_buffer->waitUntilCompleted();
        }

        // instance (top-level) acceleration structure over the placed objects; overlay slots get
        // their own tiny instance structures but share the global instance-id-indexed data arrays
        ctx.object_record_base = object_record_base;
        ctx.object_classes.clear();
        for(const auto* object_pointer : all_objects){
            ctx.object_classes.push_back(object_pointer->segmentation_class);
        }
        ctx.num_scene_instances = (uint32_t)scene.instances.size();
        {
            size_t total_instances = scene.instances.size();
            if constexpr (SPEC::ENABLE_OVERLAYS){
                // dynamic motion blur appends one extra overlay id range per motion sample:
                // sample s's instances carry shifted user ids (num_scene + num_overlay_slots *
                // (1 + s) + slot) so each pass's hits index a private slice of the instance side
                // tables with no shader changes; the contract ids [num_scene, num_scene +
                // num_overlay_slots) stay exclusive to the shutter-close state segmentation reads
                constexpr size_t num_overlay_slots = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
                constexpr size_t overlay_id_ranges = SPEC::ENABLE_DYNAMIC_MOTION_BLUR ? (size_t)1 + SPEC::MOTION_BLUR_SAMPLES : 1;
                total_instances += num_overlay_slots * overlay_id_ranges;
            }
            std::vector<MTL::AccelerationStructureUserIDInstanceDescriptor> instance_descriptors(scene.instances.size());
            std::vector<metal::InstanceData> instance_data(total_instances, metal::InstanceData{});
            std::vector<uint32_t> instance_record_base(total_instances, 0);
            for(size_t instance_i = 0; instance_i < scene.instances.size(); instance_i++){
                const auto& instance = scene.instances[instance_i];
                auto& descriptor = instance_descriptors[instance_i];
                descriptor = {};
                for(int column = 0; column < 4; column++){
                    descriptor.transformationMatrix.columns[column] = MTL::PackedFloat3{instance.transform[0 + column], instance.transform[4 + column], instance.transform[8 + column]};
                }
                descriptor.options = MTL::AccelerationStructureInstanceOptionOpaque;
                descriptor.mask = 0xFFFFFFFFu;
                descriptor.intersectionFunctionTableOffset = 0;
                descriptor.accelerationStructureIndex = (uint32_t)instance.object;
                descriptor.userID = (uint32_t)instance_i; // global instance id (contract: segmentation_object, operations_cpu_common.h)

                auto& data = instance_data[instance_i];
                data = {};
                for(int element = 0; element < 12; element++){
                    data.object_to_world[element] = instance.transform[element];
                }
                if(instance.identity){
                    std::memcpy(data.world_to_object, data.object_to_world, sizeof(data.world_to_object));
                }
                else{
                    rendering::raytracing::detail::invert_transform(instance.transform, data.world_to_object);
                }
                data.identity = instance.identity ? 1 : 0;
                instance_record_base[instance_i] = object_record_base[instance.object];
            }
            ctx.instance_descriptors = NS::TransferPtr(ctx.device->newBuffer(instance_descriptors.data(), instance_descriptors.size() * sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor), MTL::ResourceStorageModeShared));
            ctx.instance_data = NS::TransferPtr(ctx.device->newBuffer(instance_data.data(), instance_data.size() * sizeof(metal::InstanceData), MTL::ResourceStorageModeShared));
            ctx.instance_record_base = NS::TransferPtr(ctx.device->newBuffer(instance_record_base.data(), instance_record_base.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared));
            std::vector<uint32_t> instance_classes(total_instances > 0 ? total_instances : 1, 0);
            for(size_t instance_i = 0; instance_i < scene.instances.size(); instance_i++){
                instance_classes[instance_i] = ctx.object_classes[scene.instances[instance_i].object];
            }
            ctx.instance_classes = NS::TransferPtr(ctx.device->newBuffer(instance_classes.data(), instance_classes.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared));

            std::vector<NS::Object*> object_acceleration_structure_pointers;
            for(auto& object_acceleration_structure : ctx.object_acceleration_structures){
                object_acceleration_structure_pointers.push_back(object_acceleration_structure.get());
            }
            NS::Array* object_array = NS::Array::array((const NS::Object* const*)object_acceleration_structure_pointers.data(), object_acceleration_structure_pointers.size());
            MTL::InstanceAccelerationStructureDescriptor* instance_accel_descriptor = MTL::InstanceAccelerationStructureDescriptor::descriptor();
            instance_accel_descriptor->setInstancedAccelerationStructures(object_array);
            instance_accel_descriptor->setInstanceCount(scene.instances.size());
            instance_accel_descriptor->setInstanceDescriptorBuffer(ctx.instance_descriptors.get());
            instance_accel_descriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
            MTL::AccelerationStructureSizes sizes = ctx.device->accelerationStructureSizes(instance_accel_descriptor);
            ctx.acceleration_structure = NS::TransferPtr(ctx.device->newAccelerationStructure(sizes.accelerationStructureSize));
            auto scratch_buffer = NS::TransferPtr(ctx.device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate));
            MTL::CommandBuffer* command_buffer = ctx.queue->commandBuffer();
            MTL::AccelerationStructureCommandEncoder* encoder = command_buffer->accelerationStructureCommandEncoder();
            encoder->buildAccelerationStructure(ctx.acceleration_structure.get(), instance_accel_descriptor, scratch_buffer.get(), 0);
            encoder->endEncoding();
            command_buffer->commit();
            command_buffer->waitUntilCompleted();
        }
        ctx.overlay_acceleration_structures.clear();
        ctx.overlay_instance_descriptors.clear();
        ctx.overlay_sample_instance_descriptors.clear();
        ctx.overlay_scratch_buffers.clear();
        if constexpr (SPEC::ENABLE_OVERLAYS){
            std::vector<metal::OverlayStructureEntry> overlay_entries(SPEC::NUM_OVERLAYS, metal::OverlayStructureEntry{});
            std::vector<NS::Object*> object_acceleration_structure_pointers;
            for(auto& object_acceleration_structure : ctx.object_acceleration_structures){
                object_acceleration_structure_pointers.push_back(object_acceleration_structure.get());
            }
            NS::Array* object_array = NS::Array::array((const NS::Object* const*)object_acceleration_structure_pointers.data(), object_acceleration_structure_pointers.size());
            for(size_t overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
                auto descriptor_buffer = NS::TransferPtr(ctx.device->newBuffer((size_t)SPEC::MAX_OVERLAY_INSTANCES * sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor), MTL::ResourceStorageModeShared));
                MTL::InstanceAccelerationStructureDescriptor* overlay_descriptor = MTL::InstanceAccelerationStructureDescriptor::descriptor();
                overlay_descriptor->setInstancedAccelerationStructures(object_array);
                overlay_descriptor->setInstanceCount(SPEC::MAX_OVERLAY_INSTANCES);
                overlay_descriptor->setInstanceDescriptorBuffer(descriptor_buffer.get());
                overlay_descriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
                MTL::AccelerationStructureSizes overlay_sizes = ctx.device->accelerationStructureSizes(overlay_descriptor);
                auto overlay_acceleration_structure = NS::TransferPtr(ctx.device->newAccelerationStructure(overlay_sizes.accelerationStructureSize));
                auto overlay_scratch = NS::TransferPtr(ctx.device->newBuffer(overlay_sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate));
                ctx.overlay_instance_descriptors.push_back(descriptor_buffer);
                ctx.overlay_acceleration_structures.push_back(overlay_acceleration_structure);
                ctx.overlay_scratch_buffers.push_back(overlay_scratch);
                overlay_entries[overlay].structure = overlay_acceleration_structure->gpuResourceID();
                overlay_entries[overlay].num_active = 0;
            }
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                for(size_t sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                    for(size_t overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
                        ctx.overlay_sample_instance_descriptors.push_back(NS::TransferPtr(ctx.device->newBuffer((size_t)SPEC::MAX_OVERLAY_INSTANCES * sizeof(MTL::AccelerationStructureUserIDInstanceDescriptor), MTL::ResourceStorageModeShared)));
                    }
                }
            }
            ctx.overlay_structures = NS::TransferPtr(ctx.device->newBuffer(overlay_entries.data(), overlay_entries.size() * sizeof(metal::OverlayStructureEntry), MTL::ResourceStorageModeShared));
            std::vector<uint32_t> attachments_init((size_t)SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA, 0xFFFFFFFFu);
            ctx.overlay_attachments = NS::TransferPtr(ctx.device->newBuffer(attachments_init.data(), attachments_init.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared));
            rendering::raytracing::detail::reset_overlay_state(renderer);
        }
        else if(ctx.overlay_structures.get() == nullptr){
            // never dereferenced (fc_overlay_count == 0) but keeps the kernel bindings valid
            ctx.overlay_structures = NS::TransferPtr(ctx.device->newBuffer(sizeof(metal::OverlayStructureEntry), MTL::ResourceStorageModeShared));
            ctx.overlay_attachments = NS::TransferPtr(ctx.device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared));
        }

        ctx.mesh_records = NS::TransferPtr(ctx.device->newBuffer(mesh_records.data(), mesh_records.size() * sizeof(metal::MeshRecord), MTL::ResourceStorageModeShared));

        const auto scene_lights = rendering::raytracing::detail::effective_scene_lights<SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING>(scene);
        if(!scene_lights.empty()){
            ctx.scene_lights = NS::TransferPtr(ctx.device->newBuffer(scene_lights.data(), scene_lights.size() * sizeof(rendering::raytracing::SceneLight), MTL::ResourceStorageModeShared));
        }

        auto* params = (metal::LaunchParams*)ctx.launch_params->contents();
        *params = {};
        params->fb_width = SPEC::FB_WIDTH;
        params->fb_height = SPEC::FB_HEIGHT;
        params->cam_width = SPEC::CAM_WIDTH;
        params->cam_height = SPEC::CAM_HEIGHT;
        params->grid_cols = SPEC::GRID_COLS;
        params->num_cameras = SPEC::NUM_CAMERAS;
        params->num_probes = SPEC::NUM_PROBES;
        params->num_scene_lights = (uint32_t)scene_lights.size();
        params->max_depth = renderer.max_ray_length > 0 ? renderer.max_ray_length : 1e30f;
        params->max_dist = renderer.max_ray_length;
        params->ambient_color[0] = 0.10f;
        params->ambient_color[1] = 0.10f;
        params->ambient_color[2] = 0.10f;
        if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING) {
            params->miss_color_0[0] = 0.f; params->miss_color_0[1] = 0.f; params->miss_color_0[2] = 0.f;
            params->miss_color_1[0] = 0.f; params->miss_color_1[1] = 0.f; params->miss_color_1[2] = 0.f;
        } else {
            params->miss_color_0[0] = .8f; params->miss_color_0[1] = 0.f; params->miss_color_0[2] = 0.f;
            params->miss_color_1[0] = .8f; params->miss_color_1[1] = .8f; params->miss_color_1[2] = .8f;
        }
        params->first_overlay_instance = (uint32_t)scene.instances.size();

        if(!ctx.pipelines_built){
            auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
            bool srgb_output = SPEC::SHADING::SRGB_OUTPUT;
            bool motion_blur = SPEC::ENABLE_MOTION_BLUR;
            int motion_samples = SPEC::ENABLE_MOTION_BLUR ? (int)SPEC::MOTION_BLUR_SAMPLES : 1;
            int aa_grid = SPEC::ENABLE_ANTI_ALIASING ? (int)SPEC::ANTI_ALIASING_GRID_SIZE : 1;
            bool checker_background = SPEC::SHADING::CHECKER_BACKGROUND;
            bool load_textures = SPEC::SHADING::LOAD_TEXTURES;
            bool normal_shading = SPEC::SHADING::NORMAL_SHADING;
            bool metallic_reflections = SPEC::SHADING::METALLIC_REFLECTIONS;
            bool pbr_shading = SPEC::SHADING::PBR_SHADING;
            bool punctual_light_shadows = SPEC::SHADING::PUNCTUAL_LIGHT_SHADOWS;
            int overlay_count = SPEC::ENABLE_OVERLAYS ? (int)SPEC::MAX_OVERLAYS_PER_CAMERA : 0;
            bool semantic_segmentation = SPEC::SEMANTIC_SEGMENTATION;
            bool has_observation = SPEC::HAS_OBSERVATION;
            bool dynamic_motion_blur = SPEC::ENABLE_DYNAMIC_MOTION_BLUR;
            bool resolve_rgb = SPEC::ENABLE_DYNAMIC_MOTION_BLUR && SPEC::HAS_RGB;
            bool resolve_depth = SPEC::ENABLE_DYNAMIC_MOTION_BLUR && SPEC::HAS_DEPTH;
            constants->setConstantValue(&srgb_output, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::SRGB_OUTPUT);
            constants->setConstantValue(&motion_blur, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::MOTION_BLUR);
            constants->setConstantValue(&motion_samples, MTL::DataTypeInt, (NS::UInteger)metal::function_constants::MOTION_SAMPLES);
            constants->setConstantValue(&aa_grid, MTL::DataTypeInt, (NS::UInteger)metal::function_constants::AA_GRID);
            constants->setConstantValue(&checker_background, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::CHECKER_BACKGROUND);
            constants->setConstantValue(&load_textures, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::LOAD_TEXTURES);
            constants->setConstantValue(&normal_shading, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::NORMAL_SHADING);
            constants->setConstantValue(&metallic_reflections, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::METALLIC_REFLECTIONS);
            constants->setConstantValue(&pbr_shading, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::PBR_SHADING);
            constants->setConstantValue(&punctual_light_shadows, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::PUNCTUAL_LIGHT_SHADOWS);
            constants->setConstantValue(&overlay_count, MTL::DataTypeInt, (NS::UInteger)metal::function_constants::OVERLAY_COUNT);
            constants->setConstantValue(&semantic_segmentation, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::SEMANTIC_SEGMENTATION);
            constants->setConstantValue(&has_observation, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::HAS_OBSERVATION);
            constants->setConstantValue(&dynamic_motion_blur, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::DYNAMIC_MOTION_BLUR);
            constants->setConstantValue(&resolve_rgb, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::RESOLVE_RGB);
            constants->setConstantValue(&resolve_depth, MTL::DataTypeBool, (NS::UInteger)metal::function_constants::RESOLVE_DEPTH);

            auto make_pipeline = [&](const char* name) -> NS::SharedPtr<MTL::ComputePipelineState> {
                NS::Error* error = nullptr;
                auto function = NS::TransferPtr(ctx.library->newFunction(NS::String::string(name, NS::UTF8StringEncoding), constants.get(), &error));
                if(function.get() == nullptr){
                    RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Metal function specialization failed for " << name << ": " << (error != nullptr ? error->localizedDescription()->utf8String() : "unknown error"));
                    utils::assert_exit(device, false, "Metal function specialization failed");
                }
                error = nullptr;
                auto pipeline = NS::TransferPtr(ctx.device->newComputePipelineState(function.get(), &error));
                if(pipeline.get() == nullptr){
                    RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Metal pipeline creation failed for " << name << ": " << (error != nullptr ? error->localizedDescription()->utf8String() : "unknown error"));
                    utils::assert_exit(device, false, "Metal pipeline creation failed");
                }
                return pipeline;
            };

            if constexpr (SPEC::HAS_RGB) {
                ctx.rgb_pipeline = make_pipeline("render_rgb");
            }
            if constexpr (SPEC::HAS_DEPTH) {
                ctx.depth_pipeline = make_pipeline("render_depth");
            }
            if constexpr (SPEC::HAS_SEGMENTATION) {
                ctx.segmentation_pipeline = make_pipeline("render_segmentation");
            }
            if constexpr (SPEC::HAS_NORMALS) {
                ctx.normals_pipeline = make_pipeline("render_normals");
            }
            if constexpr (SPEC::HAS_FLOW) {
                ctx.flow_pipeline = make_pipeline("render_flow");
            }
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                ctx.resolve_pipeline = make_pipeline("resolve_outputs");
            }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
            ctx.collision_pipeline = make_pipeline("render_collision");
#endif
            ctx.pipelines_built = true;
        }

        if constexpr (SPEC::ENABLE_OVERLAYS){
            update(device, renderer); // publish the (empty) overlays and the attachment table
        }

        autorelease_pool->release();
    }


    // the host must not mutate the shared instance/descriptor buffers while renders or builds
    // are in flight, hence the wait at the top; the commit is not waited on — command buffers on
    // one queue execute in commit order and Metal's hazard tracking orders the acceleration
    // structure writes before any subsequent render pass that reads them
    template <typename DEVICE, typename SPEC>
    void update_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        using TI = typename SPEC::TI;
        auto& ctx = metal::context(renderer);
        metal::wait_in_flight(ctx);
        NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();

        auto* overlay_entries = (metal::OverlayStructureEntry*)ctx.overlay_structures->contents();
        auto* instance_data = (metal::InstanceData*)ctx.instance_data->contents();
        auto* instance_record_base = (uint32_t*)ctx.instance_record_base->contents();

        MTL::CommandBuffer* command_buffer = nullptr;
        MTL::AccelerationStructureCommandEncoder* encoder = nullptr;
        std::vector<NS::Object*> object_acceleration_structure_pointers;
        for(auto& object_acceleration_structure : ctx.object_acceleration_structures){
            object_acceleration_structure_pointers.push_back(object_acceleration_structure.get());
        }
        NS::Array* object_array = NS::Array::array((const NS::Object* const*)object_acceleration_structure_pointers.data(), object_acceleration_structure_pointers.size());

        if constexpr (SPEC::HAS_FLOW){
            rendering::raytracing::detail::compose_flow_deltas(renderer, data(renderer.flow_deltas));
        }
        rendering::raytracing::detail::flush_overlay_transforms(renderer);
        // rebuilt unconditionally: producers may write the transforms tensor directly, which
        // leaves no host-observable dirty flag
        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            auto& overlay_state = renderer.overlays[overlay];
            const size_t base = (size_t)ctx.num_scene_instances + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES;
            auto* descriptors = (MTL::AccelerationStructureUserIDInstanceDescriptor*)ctx.overlay_instance_descriptors[overlay]->contents();
            uint32_t num_active = 0;
            for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                const auto& host_slot = overlay_state.slots[slot];
                if(!host_slot.active) continue;
                const size_t global = base + slot;
                float world[12];
                rendering::raytracing::detail::compose_overlay_slot_transform(renderer, overlay, slot, world);
                auto& descriptor = descriptors[num_active];
                descriptor = {};
                for(int column = 0; column < 4; column++){
                    descriptor.transformationMatrix.columns[column] = MTL::PackedFloat3{world[0 + column], world[4 + column], world[8 + column]};
                }
                descriptor.options = MTL::AccelerationStructureInstanceOptionOpaque;
                descriptor.mask = 0xFFFFFFFFu;
                descriptor.intersectionFunctionTableOffset = 0;
                descriptor.accelerationStructureIndex = (uint32_t)host_slot.object;
                descriptor.userID = (uint32_t)global; // global instance id (contract: segmentation_object, operations_cpu_common.h)

                auto& data = instance_data[global];
                data = {};
                for(int element = 0; element < 12; element++){
                    data.object_to_world[element] = world[element];
                }
                const bool identity = rendering::raytracing::detail::transform_is_identity(world);
                if(identity){
                    std::memcpy(data.world_to_object, data.object_to_world, sizeof(data.world_to_object));
                }
                else{
                    rendering::raytracing::detail::invert_transform(world, data.world_to_object);
                }
                data.identity = identity ? 1 : 0;
                instance_record_base[global] = ctx.object_record_base[host_slot.object];
                ((uint32_t*)ctx.instance_classes->contents())[global] = ctx.object_classes[host_slot.object];
                num_active++;
            }
            overlay_entries[overlay].num_active = num_active;
            if(num_active > 0){
                if(command_buffer == nullptr){
                    command_buffer = ctx.queue->commandBuffer();
                    encoder = command_buffer->accelerationStructureCommandEncoder();
                }
                MTL::InstanceAccelerationStructureDescriptor* overlay_descriptor = MTL::InstanceAccelerationStructureDescriptor::descriptor();
                overlay_descriptor->setInstancedAccelerationStructures(object_array);
                overlay_descriptor->setInstanceCount(num_active);
                overlay_descriptor->setInstanceDescriptorBuffer(ctx.overlay_instance_descriptors[overlay].get());
                overlay_descriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
                encoder->buildAccelerationStructure(ctx.overlay_acceleration_structures[overlay].get(), overlay_descriptor, ctx.overlay_scratch_buffers[overlay].get(), 0);
            }
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            // per-sample overlay instance descriptors + shifted-id side-table slices consumed by
            // render_launch's per-sample builds; layout mirrors the shutter-close path above
            rendering::raytracing::detail::flush_overlay_motion_transforms(renderer);
            constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
            constexpr size_t num_overlay_slots = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
            for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                const float* transforms_base = data(renderer.transforms_motion) + sample * SLAB;
                for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
                    auto& overlay_state = renderer.overlays[overlay];
                    auto* descriptors = (MTL::AccelerationStructureUserIDInstanceDescriptor*)ctx.overlay_sample_instance_descriptors[(size_t)sample * SPEC::NUM_OVERLAYS + overlay]->contents();
                    uint32_t num_active = 0;
                    for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                        const auto& host_slot = overlay_state.slots[slot];
                        if(!host_slot.active) continue;
                        const size_t global = (size_t)ctx.num_scene_instances + num_overlay_slots * (1 + (size_t)sample) + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES + slot;
                        float world[12];
                        rendering::raytracing::detail::compose_overlay_slot_transform(renderer, transforms_base, overlay, slot, world);
                        auto& descriptor = descriptors[num_active];
                        descriptor = {};
                        for(int column = 0; column < 4; column++){
                            descriptor.transformationMatrix.columns[column] = MTL::PackedFloat3{world[0 + column], world[4 + column], world[8 + column]};
                        }
                        descriptor.options = MTL::AccelerationStructureInstanceOptionOpaque;
                        descriptor.mask = 0xFFFFFFFFu;
                        descriptor.intersectionFunctionTableOffset = 0;
                        descriptor.accelerationStructureIndex = (uint32_t)host_slot.object;
                        descriptor.userID = (uint32_t)global;

                        auto& data_entry = instance_data[global];
                        data_entry = {};
                        for(int element = 0; element < 12; element++){
                            data_entry.object_to_world[element] = world[element];
                        }
                        const bool identity = rendering::raytracing::detail::transform_is_identity(world);
                        if(identity){
                            std::memcpy(data_entry.world_to_object, data_entry.object_to_world, sizeof(data_entry.world_to_object));
                        }
                        else{
                            rendering::raytracing::detail::invert_transform(world, data_entry.world_to_object);
                        }
                        data_entry.identity = identity ? 1 : 0;
                        instance_record_base[global] = ctx.object_record_base[host_slot.object];
                        ((uint32_t*)ctx.instance_classes->contents())[global] = ctx.object_classes[host_slot.object];
                        num_active++;
                    }
                }
            }
        }
        if(renderer.attachments_dirty){
            auto* attachments = (uint32_t*)ctx.overlay_attachments->contents();
            for(size_t index = 0; index < (size_t)SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA; index++){
                attachments[index] = (uint32_t)renderer.attachments[index];
            }
            renderer.attachments_dirty = false;
        }
        if(command_buffer != nullptr){
            encoder->endEncoding();
            command_buffer->commit();
            ctx.in_flight_update = NS::RetainPtr(command_buffer);
        }
        autorelease_pool->release();
    }

    template <typename DEVICE, typename SPEC>
    void update_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.in_flight_update.get() != nullptr){
            ctx.in_flight_update->waitUntilCompleted();
            ctx.in_flight_update.reset();
        }
    }

    template <typename DEVICE, typename SPEC>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        update_launch(device, renderer);
        update_sync(device, renderer);
    }

    // CPU expansion into the host-resident tensors (residency is a backend property; the
    // device-resident path is the OptiX backend)
    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        static_assert(SPEC::HAS_TRANSFORM_PAIR, "expand_motion_transforms requires a dynamic-motion-blur or flow renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        metal::wait_in_flight(metal::context(renderer));
        rendering::raytracing::detail::expand_motion_transforms_host(renderer);
    }

    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        static_assert(SPEC::HAS_TRANSFORM_PAIR, "expand_motion_transforms requires a dynamic-motion-blur or flow renderer specification");
    }

    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        expand_motion_transforms_launch(device, renderer);
        expand_motion_transforms_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void generate_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer,
                          const typename SPEC::T center[3], typename SPEC::T radius,
                          const typename SPEC::T up[3], typename SPEC::T fov){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        metal::wait_in_flight(ctx);
        rendering::raytracing::detail::generate_camera_poses<SPEC>(device, data(renderer.cameras), center, radius, up, fov);
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            std::memcpy(data(renderer.cameras_open), data(renderer.cameras), (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>));
        }
    }

    // renderer memory-domain copies: shared-storage buffers are host-addressable after the
    // in-flight wait, so the transfer delegates to the host-device tensor copy
    template <typename TO_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(rendering::raytracing::backends::Device<rendering::raytracing::backends::Metal>& from_device, TO_DEVICE& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
        rendering::raytracing::backends::metal::wait_in_flight(*from_device.context);
        copy(to_device, to_device, from, to);
    }
    template <typename FROM_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(FROM_DEVICE& from_device, rendering::raytracing::backends::Device<rendering::raytracing::backends::Metal>& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
        rendering::raytracing::backends::metal::wait_in_flight(*to_device.context);
        copy(from_device, from_device, from, to);
    }

    template <typename DEVICE, typename SPEC>
    void generate_probe_directions(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Probe rays disabled (RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS=1)");
        return;
#else
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        std::vector<float> dirs = rendering::raytracing::detail::generate_probe_direction_vectors<SPEC>();
        ctx.probe_directions = NS::TransferPtr(ctx.device->newBuffer(dirs.data(), dirs.size() * sizeof(float), MTL::ResourceStorageModeShared));
#endif
    }

    // render produces the image outputs the spec declares; the collision-probe pass is the
    // separate probe verb so it can be scheduled independently (e.g. alongside update)
    template <typename DEVICE, typename SPEC>
    void render_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        using TI = typename SPEC::TI;
        auto& ctx = metal::context(renderer);
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* command_buffer = ctx.queue->commandBuffer();
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            // one command buffer per frame: zero the accumulators, then per motion sample build
            // the overlay structures in place from that sample's descriptors and run the
            // accumulate dispatches, then restore the shutter-close state (segmentation, probes,
            // steady state) and resolve — encoder order plus Metal hazard tracking on the
            // acceleration structures and accumulators serializes builds against dispatches
            uint32_t num_active_per_overlay[SPEC::NUM_OVERLAYS];
            uint32_t total_active = 0;
            for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
                uint32_t num_active = 0;
                for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                    num_active += renderer.overlays[overlay].slots[slot].active ? 1 : 0;
                }
                num_active_per_overlay[overlay] = num_active;
                total_active += num_active;
            }
            {
                MTL::BlitCommandEncoder* blit = command_buffer->blitCommandEncoder();
                if constexpr (SPEC::HAS_RGB){
                    blit->fillBuffer(ctx.rgb_accumulator.get(), NS::Range::Make(0, ctx.rgb_accumulator->length()), 0);
                }
                if constexpr (SPEC::HAS_DEPTH){
                    blit->fillBuffer(ctx.depth_accumulator.get(), NS::Range::Make(0, ctx.depth_accumulator->length()), 0);
                }
                blit->endEncoding();
            }
            std::vector<NS::Object*> object_acceleration_structure_pointers;
            for(auto& object_acceleration_structure : ctx.object_acceleration_structures){
                object_acceleration_structure_pointers.push_back(object_acceleration_structure.get());
            }
            NS::Array* object_array = NS::Array::array((const NS::Object* const*)object_acceleration_structure_pointers.data(), object_acceleration_structure_pointers.size());
            const auto encode_overlay_builds = [&](auto&& descriptor_buffer_for_overlay){
                if(total_active == 0){
                    return;
                }
                MTL::AccelerationStructureCommandEncoder* encoder = command_buffer->accelerationStructureCommandEncoder();
                for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
                    if(num_active_per_overlay[overlay] == 0) continue;
                    MTL::InstanceAccelerationStructureDescriptor* overlay_descriptor = MTL::InstanceAccelerationStructureDescriptor::descriptor();
                    overlay_descriptor->setInstancedAccelerationStructures(object_array);
                    overlay_descriptor->setInstanceCount(num_active_per_overlay[overlay]);
                    overlay_descriptor->setInstanceDescriptorBuffer(descriptor_buffer_for_overlay(overlay));
                    overlay_descriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeUserID);
                    encoder->buildAccelerationStructure(ctx.overlay_acceleration_structures[overlay].get(), overlay_descriptor, ctx.overlay_scratch_buffers[overlay].get(), 0);
                }
                encoder->endEncoding();
            };
            for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                encode_overlay_builds([&](TI overlay){ return ctx.overlay_sample_instance_descriptors[(size_t)sample * SPEC::NUM_OVERLAYS + overlay].get(); });
                const float shutter_t = ((float)sample + 0.5f) / (float)SPEC::MOTION_BLUR_SAMPLES;
                if constexpr (SPEC::HAS_RGB) {
                    metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.rgb_pipeline.get(), ctx.frame_buffer.get(), shutter_t);
                }
                if constexpr (SPEC::HAS_DEPTH) {
                    metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.depth_pipeline.get(), ctx.depth_buffer.get(), shutter_t);
                }
            }
            encode_overlay_builds([&](TI overlay){ return ctx.overlay_instance_descriptors[overlay].get(); });
            if constexpr (SPEC::HAS_SEGMENTATION) {
                metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.segmentation_pipeline.get(), ctx.segmentation_buffer.get());
            }
            if constexpr (SPEC::HAS_NORMALS) {
                metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.normals_pipeline.get(), ctx.normals_buffer.get());
            }
            if constexpr (SPEC::HAS_FLOW) {
                metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.flow_pipeline.get(), ctx.flow_buffer.get());
            }
            metal::encode_resolve_pass<SPEC>(ctx, command_buffer);
            command_buffer->commit();
            ctx.in_flight = NS::RetainPtr(command_buffer);
            pool->release();
            return;
        }
        if constexpr (SPEC::HAS_RGB) {
            metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.rgb_pipeline.get(), ctx.frame_buffer.get());
        }
        if constexpr (SPEC::HAS_DEPTH) {
            metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.depth_pipeline.get(), ctx.depth_buffer.get());
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.segmentation_pipeline.get(), ctx.segmentation_buffer.get());
        }
        if constexpr (SPEC::HAS_NORMALS) {
            metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.normals_pipeline.get(), ctx.normals_buffer.get());
        }
        if constexpr (SPEC::HAS_FLOW) {
            metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.flow_pipeline.get(), ctx.flow_buffer.get());
        }
        command_buffer->commit();
        ctx.in_flight = NS::RetainPtr(command_buffer);
        pool->release();
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.in_flight.get() != nullptr){
            ctx.in_flight->waitUntilCompleted();
            ctx.in_flight.reset();
        }
    }

    template <typename DEVICE, typename SPEC>
    void render(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        render_launch(device, renderer);
        render_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void probe_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.collision_pipeline.get() != nullptr){
            NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
            MTL::CommandBuffer* command_buffer = ctx.queue->commandBuffer();
            metal::encode_collision_pass<SPEC>(ctx, command_buffer);
            command_buffer->commit();
            ctx.in_flight_collision = NS::RetainPtr(command_buffer);
            pool->release();
        }
    }

    template <typename DEVICE, typename SPEC>
    void probe_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.in_flight_collision.get() != nullptr){
            ctx.in_flight_collision->waitUntilCompleted();
            ctx.in_flight_collision.reset();
        }
    }

    template <typename DEVICE, typename SPEC>
    void probe(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        probe_launch(device, renderer);
        probe_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void synchronize(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        metal::wait_in_flight(metal::context(renderer));
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        if(renderer.backend != nullptr){
            metal::wait_in_flight(metal::context(renderer));
            delete renderer.backend;
            renderer.backend = nullptr;
            renderer.device.context = nullptr;
        }
        // the input and output tensors alias shared MTLBuffers destroyed with the context
        renderer.cameras._data = nullptr;
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            renderer.cameras_open._data = nullptr;
        }
        if constexpr (SPEC::HAS_RGB) {
            renderer.frame_buffer._data = nullptr;
        }
        if constexpr (SPEC::HAS_DEPTH) {
            renderer.depth_buffer._data = nullptr;
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            renderer.segmentation_buffer._data = nullptr;
        }
        if constexpr (SPEC::HAS_NORMALS) {
            renderer.normals_buffer._data = nullptr;
        }
        if constexpr (SPEC::HAS_FLOW) {
            renderer.flow_buffer._data = nullptr;
            if constexpr (SPEC::ENABLE_OVERLAYS) {
                renderer.flow_deltas._data = nullptr;
            }
        }
        if constexpr (SPEC::HAS_OBSERVATION) {
            renderer.observation._data = nullptr;
        }
        renderer.collision_results._data = nullptr;
        if constexpr (SPEC::ENABLE_OVERLAYS) {
            free(device, renderer.transforms);
        }
        if constexpr (SPEC::HAS_TRANSFORM_PAIR) {
            free(device, renderer.transforms_pair);
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            free(device, renderer.transforms_motion);
            if constexpr (SPEC::HAS_RGB) {
                renderer.rgb_accumulator._data = nullptr;
            }
            if constexpr (SPEC::HAS_DEPTH) {
                renderer.depth_accumulator._data = nullptr;
            }
        }
    }

    // shared-asset-library fallbacks: this backend has no cross-renderer sharing, so the
    // library is empty and every renderer builds its own copy — the API stays uniform
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Metal>& library){
        library.backend = new rendering::raytracing::backends::LibraryState<rendering::raytracing::backends::Metal, SPEC>{};
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Metal>& library){
        for(auto* assets : library.assets){
            delete assets;
        }
        library.assets.clear();
        library.scenes.clear();
        library.metadata.clear();
        delete library.backend;
        library.backend = nullptr;
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Metal>& renderer, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Metal>& library){
        malloc(device, renderer);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END


#include "../../operations_cpu_post.h"

#endif
