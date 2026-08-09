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
        struct RendererState<devices::rendering::Metal, SPEC>: metal::Context {};

        template <typename SPEC>
        struct LibraryState<devices::rendering::Metal, SPEC> {};

        template <typename SPEC>
        struct SceneState<devices::rendering::Metal, SPEC> {};
    }

    namespace rendering::raytracing::backends::metal{
        template <typename SPEC>
        Context& context(rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
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
        void encode_fullscreen_pass(Context& ctx, MTL::CommandBuffer* command_buffer, MTL::ComputePipelineState* pipeline, MTL::Buffer* output){
            MTL::ComputeCommandEncoder* encoder = command_buffer->computeCommandEncoder();
            encoder->setComputePipelineState(pipeline);
            encoder->setBuffer(ctx.launch_params.get(), 0, bindings::LAUNCH_PARAMS);
            encoder->setBuffer(ctx.cameras.get(), 0, bindings::CAMERAS_CLOSE);
            encoder->setBuffer(ctx.cameras_open.get() != nullptr ? ctx.cameras_open.get() : ctx.cameras.get(), 0, bindings::CAMERAS_OPEN);
            encoder->setBuffer(output, 0, bindings::OUTPUT);
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
    void malloc(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        static_assert(utils::typing::is_same_v<typename SPEC::T, float>, "The Metal raytracing backend requires T = float");

        if constexpr (SPEC::ENABLE_OVERLAYS) {
            malloc(device, renderer.transforms);
        }

        renderer.backend = new rendering::raytracing::backends::RendererState<devices::rendering::Metal, SPEC>{};
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
        if constexpr (SPEC::HAS_OBSERVATION) {
            static_assert(utils::typing::is_same_v<typename SPEC::OBSERVATION_T, float>, "The Metal raytracing backend requires OBSERVATION_T = float");
            ctx->observation = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * SPEC::OBSERVATION_CHANNELS * sizeof(float), MTL::ResourceStorageModeShared));
            renderer.observation._data = (float*)ctx->observation->contents();
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
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            ctx->cameras_open = NS::TransferPtr(ctx->device->newBuffer(camera_bytes, MTL::ResourceStorageModeShared));
            renderer.cameras_open._data = (rendering::raytracing::Camera<typename SPEC::T>*)ctx->cameras_open->contents();
        }
    }

    template <typename DEVICE, typename SPEC>
    void update(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer);

    template <typename DEVICE, typename SPEC>
    void init(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        NS::AutoreleasePool* autorelease_pool = NS::AutoreleasePool::alloc()->init();

        rendering::raytracing::detail::compute_scene_bounds(renderer, scene);
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
                total_instances += (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
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
        params->max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
        params->max_dist = renderer.camera_radius * 2.0f;
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
            int overlay_count = (int)SPEC::MAX_OVERLAYS_PER_CAMERA;
            bool semantic_segmentation = SPEC::SEMANTIC_SEGMENTATION;
            bool has_observation = SPEC::HAS_OBSERVATION;
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
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
            ctx.collision_pipeline = make_pipeline("render_collision");
#endif
            ctx.pipelines_built = true;
        }

        if constexpr (SPEC::ENABLE_OVERLAYS){
            update(render_device, device, renderer); // publish the (empty) overlays and the attachment table
        }

        autorelease_pool->release();
    }

    template <typename DEVICE, typename SPEC>
    void init(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, const rendering::raytracing::Scene& scene){
        static const rendering::raytracing::AssetPool empty_pool{};
        init(render_device, device, renderer, scene, empty_pool);
    }

    // the host must not mutate the shared instance/descriptor buffers while renders or builds
    // are in flight, hence the wait at the top; the commit is not waited on — command buffers on
    // one queue execute in commit order and Metal's hazard tracking orders the acceleration
    // structure writes before any subsequent render pass that reads them
    template <typename DEVICE, typename SPEC>
    void update_launch(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
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
    void update_sync(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.in_flight_update.get() != nullptr){
            ctx.in_flight_update->waitUntilCompleted();
            ctx.in_flight_update.reset();
        }
    }

    template <typename DEVICE, typename SPEC>
    void update(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
        update_launch(render_device, device, renderer);
        update_sync(render_device, device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void generate_cameras(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer,
                          const typename SPEC::T center[3], typename SPEC::T radius,
                          const typename SPEC::T up[3], typename SPEC::T fov){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        metal::wait_in_flight(ctx);
        rendering::raytracing::detail::generate_camera_poses<SPEC>(device, data(renderer.cameras), center, radius, up, fov);
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            std::memcpy(data(renderer.cameras_open), data(renderer.cameras), (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>));
        }
    }

    template <typename DEVICE, typename SPEC, typename T>
    void copy_to_renderer(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, const T* source, T* destination, size_t count){
        rendering::raytracing::backends::metal::wait_in_flight(rendering::raytracing::backends::metal::context(renderer));
        std::memcpy(destination, source, count * sizeof(T));
    }

    template <typename DEVICE, typename SPEC, typename T>
    void copy_from_renderer(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, const T* source, T* destination, size_t count){
        rendering::raytracing::backends::metal::wait_in_flight(rendering::raytracing::backends::metal::context(renderer));
        std::memcpy(destination, source, count * sizeof(T));
    }

    template <typename DEVICE, typename SPEC>
    void generate_probe_directions(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
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
    void render_launch(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* command_buffer = ctx.queue->commandBuffer();
        if constexpr (SPEC::HAS_RGB) {
            metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.rgb_pipeline.get(), ctx.frame_buffer.get());
        }
        if constexpr (SPEC::HAS_DEPTH) {
            metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.depth_pipeline.get(), ctx.depth_buffer.get());
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.segmentation_pipeline.get(), ctx.segmentation_buffer.get());
        }
        command_buffer->commit();
        ctx.in_flight = NS::RetainPtr(command_buffer);
        pool->release();
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.in_flight.get() != nullptr){
            ctx.in_flight->waitUntilCompleted();
            ctx.in_flight.reset();
        }
    }

    template <typename DEVICE, typename SPEC>
    void render(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
        render_launch(render_device, device, renderer);
        render_sync(render_device, device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void probe_launch(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
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
    void probe_sync(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.in_flight_collision.get() != nullptr){
            ctx.in_flight_collision->waitUntilCompleted();
            ctx.in_flight_collision.reset();
        }
    }

    template <typename DEVICE, typename SPEC>
    void probe(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
        probe_launch(render_device, device, renderer);
        probe_sync(render_device, device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void save_image(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, const char* filename){
        static_assert(SPEC::HAS_RGB, "save_image requires an RGB-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        rendering::raytracing::detail::write_grid_png<SPEC>(data(renderer.frame_buffer), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_segmentation_image(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, const char* filename){
        static_assert(SPEC::HAS_SEGMENTATION, "save_segmentation_image requires a segmentation-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        rendering::raytracing::detail::write_segmentation_grid_png<SPEC>(data(renderer.segmentation_buffer), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth_image(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth_image requires a depth-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        rendering::raytracing::detail::write_depth_grid_png<SPEC>(data(renderer.depth_buffer), renderer.camera_radius, filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth requires a depth-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        rendering::raytracing::detail::write_depth_bin<SPEC>(data(renderer.depth_buffer), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_probes(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, const char* filename){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("save_probes skipped: probe rays are disabled.");
        (void)filename;
        return;
#else
        namespace metal = rendering::raytracing::backends::metal;
        rendering::raytracing::detail::write_probes_bin_and_log<SPEC>(data(renderer.collision_results), filename);
#endif
    }

    template <typename DEVICE, typename SPEC>
    void synchronize(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        metal::wait_in_flight(metal::context(renderer));
    }

    template <typename DEVICE, typename SPEC>
    void free(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        if(renderer.backend != nullptr){
            metal::wait_in_flight(metal::context(renderer));
            delete renderer.backend;
            renderer.backend = nullptr;
        }
        // the input and output tensors alias shared MTLBuffers destroyed with the context
        renderer.cameras._data = nullptr;
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
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
        if constexpr (SPEC::HAS_OBSERVATION) {
            renderer.observation._data = nullptr;
        }
        renderer.collision_results._data = nullptr;
        if constexpr (SPEC::ENABLE_OVERLAYS) {
            free(device, renderer.transforms);
        }
    }

    // shared-asset-library fallbacks: this backend has no cross-renderer sharing, so the
    // library is empty and every renderer builds its own copy — the API stays uniform
    template <typename DEVICE, typename SPEC>
    void malloc(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, devices::rendering::Metal>& library){
        library.backend = new rendering::raytracing::backends::LibraryState<devices::rendering::Metal, SPEC>{};
    }

    template <typename DEVICE, typename SPEC>
    void free(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, devices::rendering::Metal>& library){
        for(auto* assets : library.assets){
            delete assets;
        }
        library.assets.clear();
        delete library.backend;
        library.backend = nullptr;
    }

    template <typename DEVICE, typename SPEC>
    void malloc(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, rendering::raytracing::AssetLibrary<SPEC, devices::rendering::Metal>& library){
        malloc(render_device, device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    typename SPEC::TI init(devices::rendering::Metal& render_device, DEVICE& device, rendering::raytracing::Renderer<SPEC, devices::rendering::Metal>& renderer, rendering::raytracing::AssetLibrary<SPEC, devices::rendering::Metal>& library, const char* scene_path){
        bool is_new = false;
        const auto scene_id = rendering::raytracing::detail::library_lookup_or_load(device, library, scene_path, is_new);
        init(render_device, device, renderer, library.scenes[scene_id], library.pool);
        return scene_id;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
