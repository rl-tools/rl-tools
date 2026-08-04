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
    namespace rendering::raytracing::backends::metal{
        template <typename SPEC>
        Context& context(rendering::raytracing::Renderer<SPEC>& renderer){
            return *(Context*)renderer.backend.context;
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
            encoder->dispatchThreads(MTL::Size::Make(SPEC::NUM_CAMERAS, SPEC::NUM_PROBES, 1), MTL::Size::Make(8, 8, 1));
            encoder->endEncoding();
        }
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        static_assert(utils::typing::is_same_v<typename SPEC::T, float>, "The Metal raytracing backend requires T = float");

        malloc(device, renderer.cameras);
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            malloc(device, renderer.cameras_open);
        }
        if constexpr (SPEC::HAS_RGB) {
            malloc(device, renderer.frame_buffer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            malloc(device, renderer.depth_buffer);
        }
        malloc(device, renderer.collision_results);

        auto* ctx = new metal::Context{};
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
        if constexpr (SPEC::HAS_RGB) {
            ctx->frame_buffer = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * sizeof(uint32_t), MTL::ResourceStorageModeShared));
            renderer.backend.frame_buffer_handle = ctx->frame_buffer.get();
        }
        if constexpr (SPEC::HAS_DEPTH) {
            ctx->depth_buffer = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * cam_pixels * sizeof(float), MTL::ResourceStorageModeShared));
            renderer.backend.depth_buffer_handle = ctx->depth_buffer.get();
        }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        ctx->collision_results = NS::TransferPtr(ctx->device->newBuffer((size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES * sizeof(rendering::raytracing::CollisionResult), MTL::ResourceStorageModeShared));
        renderer.backend.collision_results_buffer = ctx->collision_results.get();
#endif
        ctx->launch_params = NS::TransferPtr(ctx->device->newBuffer(sizeof(metal::LaunchParams), MTL::ResourceStorageModeShared));

        renderer.backend.context = ctx;
        renderer.backend.module = ctx->library.get();
    }

    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const rendering::raytracing::Scene& scene){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        rendering::raytracing::detail::compute_scene_bounds(renderer, scene);
        RL_TOOLS_RENDERING_RAYTRACING_LOG("building " << scene.meshes.size() << " geometries ...");

        // scene-dependent resources from a previous init are released here; the compiled library
        // and pipelines below are spec-dependent and survive re-init
        ctx.mesh_buffers.clear();
        ctx.mesh_textures.clear();
        ctx.scene_lights = NS::SharedPtr<MTL::Buffer>{};

        const uint8_t dummy_pixel[4] = {255, 255, 255, 255};
        ctx.dummy_texture = metal::make_texture(ctx.device.get(), dummy_pixel, 1, 1, false);
        const MTL::ResourceID dummy_texture_id = ctx.dummy_texture->gpuResourceID();

        std::vector<metal::MeshRecord> mesh_records(scene.meshes.size());
        std::vector<NS::Object*> geometry_descriptors;
        geometry_descriptors.reserve(scene.meshes.size());

        for(size_t m = 0; m < scene.meshes.size(); m++){
            const auto& md = scene.meshes[m];
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
        }

        NS::Array* geometry_array = NS::Array::array((const NS::Object* const*)geometry_descriptors.data(), geometry_descriptors.size());
        MTL::PrimitiveAccelerationStructureDescriptor* accel_descriptor = MTL::PrimitiveAccelerationStructureDescriptor::descriptor();
        accel_descriptor->setGeometryDescriptors(geometry_array);
        MTL::AccelerationStructureSizes sizes = ctx.device->accelerationStructureSizes(accel_descriptor);
        ctx.acceleration_structure = NS::TransferPtr(ctx.device->newAccelerationStructure(sizes.accelerationStructureSize));
        auto scratch_buffer = NS::TransferPtr(ctx.device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate));
        MTL::CommandBuffer* command_buffer = ctx.queue->commandBuffer();
        MTL::AccelerationStructureCommandEncoder* encoder = command_buffer->accelerationStructureCommandEncoder();
        encoder->buildAccelerationStructure(ctx.acceleration_structure.get(), accel_descriptor, scratch_buffer.get(), 0);
        encoder->endEncoding();
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        renderer.backend.world = ctx.acceleration_structure.get();

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
                renderer.backend.ray_gen = ctx.rgb_pipeline.get();
            }
            if constexpr (SPEC::HAS_DEPTH) {
                ctx.depth_pipeline = make_pipeline("render_depth");
                renderer.backend.depth_ray_gen = ctx.depth_pipeline.get();
            }
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
            ctx.collision_pipeline = make_pipeline("render_collision");
            renderer.backend.collision_ray_gen = ctx.collision_pipeline.get();
#endif
            ctx.pipelines_built = true;
        }

        pool->release();
    }

    template <typename DEVICE, typename SPEC>
    void generate_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer,
                          const typename SPEC::T center[3], typename SPEC::T radius,
                          const typename SPEC::T up[3], typename SPEC::T fov){
        namespace metal = rendering::raytracing::backends::metal;
        rendering::raytracing::detail::generate_camera_poses(device, renderer, center, radius, up, fov);

        auto& ctx = metal::context(renderer);
        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        ctx.cameras = NS::TransferPtr(ctx.device->newBuffer(data(renderer.cameras), camera_bytes, MTL::ResourceStorageModeShared));
        renderer.backend.cameras_buffer = ctx.cameras.get();
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            ctx.cameras_open = NS::TransferPtr(ctx.device->newBuffer(data(renderer.cameras), camera_bytes, MTL::ResourceStorageModeShared));
            renderer.backend.cameras_open_buffer = ctx.cameras_open.get();
        }
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void set_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        static_assert(utils::typing::is_same_v<typename CAMERAS_SPEC::T, rendering::raytracing::Camera<typename SPEC::T>>);
        static_assert(get<0>(typename CAMERAS_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        metal::wait_in_flight(ctx);

        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        if(ctx.cameras.get() == nullptr){
            ctx.cameras = NS::TransferPtr(ctx.device->newBuffer(data(cameras), camera_bytes, MTL::ResourceStorageModeShared));
            renderer.backend.cameras_buffer = ctx.cameras.get();
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                ctx.cameras_open = NS::TransferPtr(ctx.device->newBuffer(data(cameras), camera_bytes, MTL::ResourceStorageModeShared));
                renderer.backend.cameras_open_buffer = ctx.cameras_open.get();
            }
        }
        else{
            std::memcpy(ctx.cameras->contents(), data(cameras), camera_bytes);
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                std::memcpy(ctx.cameras_open->contents(), data(cameras), camera_bytes);
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void set_cameras_async(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        set_cameras(device, renderer, cameras);
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_OPEN_SPEC, typename CAMERAS_CLOSE_SPEC>
    void set_motion_blur_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_OPEN_SPEC>& cameras_open, const Tensor<CAMERAS_CLOSE_SPEC>& cameras_close){
        static_assert(SPEC::ENABLE_MOTION_BLUR, "set_motion_blur_cameras requires a motion-blur renderer specification");
        static_assert(utils::typing::is_same_v<typename CAMERAS_OPEN_SPEC::T, rendering::raytracing::Camera<typename SPEC::T>>);
        static_assert(utils::typing::is_same_v<typename CAMERAS_CLOSE_SPEC::T, rendering::raytracing::Camera<typename SPEC::T>>);
        static_assert(get<0>(typename CAMERAS_OPEN_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<0>(typename CAMERAS_CLOSE_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        metal::wait_in_flight(ctx);

        constexpr size_t camera_bytes = (size_t)SPEC::NUM_CAMERAS * sizeof(rendering::raytracing::Camera<typename SPEC::T>);
        if(ctx.cameras.get() == nullptr){
            ctx.cameras = NS::TransferPtr(ctx.device->newBuffer(data(cameras_close), camera_bytes, MTL::ResourceStorageModeShared));
            ctx.cameras_open = NS::TransferPtr(ctx.device->newBuffer(data(cameras_open), camera_bytes, MTL::ResourceStorageModeShared));
            renderer.backend.cameras_buffer = ctx.cameras.get();
            renderer.backend.cameras_open_buffer = ctx.cameras_open.get();
        }
        else{
            std::memcpy(ctx.cameras_open->contents(), data(cameras_open), camera_bytes);
            std::memcpy(ctx.cameras->contents(), data(cameras_close), camera_bytes);
        }
    }

    template <typename DEVICE, typename SPEC>
    void generate_probe_directions(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Probe rays disabled (RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS=1)");
        return;
#else
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        std::vector<float> dirs = rendering::raytracing::detail::generate_probe_direction_vectors<SPEC>();
        ctx.probe_directions = NS::TransferPtr(ctx.device->newBuffer(dirs.data(), dirs.size() * sizeof(float), MTL::ResourceStorageModeShared));
        renderer.backend.probe_dirs_buffer = ctx.probe_directions.get();
#endif
    }

    template <typename DEVICE, typename SPEC>
    void render_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
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
        command_buffer->commit();
        ctx.in_flight = NS::RetainPtr(command_buffer);
        if(renderer.backend.collision_ray_gen != nullptr){
            MTL::CommandBuffer* collision_command_buffer = ctx.queue->commandBuffer();
            metal::encode_collision_pass<SPEC>(ctx, collision_command_buffer);
            collision_command_buffer->commit();
            ctx.in_flight_collision = NS::RetainPtr(collision_command_buffer);
        }
        pool->release();
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        metal::wait_in_flight(metal::context(renderer));
    }

    template <typename DEVICE, typename SPEC>
    void render(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_launch(device, renderer);
        render_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(renderer.backend.collision_ray_gen != nullptr){
            NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
            MTL::CommandBuffer* command_buffer = ctx.queue->commandBuffer();
            metal::encode_collision_pass<SPEC>(ctx, command_buffer);
            command_buffer->commit();
            ctx.in_flight_collision = NS::RetainPtr(command_buffer);
            pool->release();
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.in_flight_collision.get() != nullptr){
            ctx.in_flight_collision->waitUntilCompleted();
            ctx.in_flight_collision.reset();
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_collision_only_launch(device, renderer);
        render_collision_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "render_rgb_only requires an RGB-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* command_buffer = ctx.queue->commandBuffer();
        metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.rgb_pipeline.get(), ctx.frame_buffer.get());
        command_buffer->commit();
        ctx.in_flight = NS::RetainPtr(command_buffer);
        pool->release();
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "render_rgb_only requires an RGB-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.in_flight.get() != nullptr){
            ctx.in_flight->waitUntilCompleted();
            ctx.in_flight.reset();
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_rgb_only_launch(device, renderer);
        render_rgb_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "render_depth_only requires a depth-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* command_buffer = ctx.queue->commandBuffer();
        metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.depth_pipeline.get(), ctx.depth_buffer.get());
        command_buffer->commit();
        ctx.in_flight = NS::RetainPtr(command_buffer);
        pool->release();
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "render_depth_only requires a depth-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.in_flight.get() != nullptr){
            ctx.in_flight->waitUntilCompleted();
            ctx.in_flight.reset();
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_depth_only_launch(device, renderer);
        render_depth_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB && SPEC::HAS_DEPTH, "render_rgb_depth_only requires an RGBD renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer* command_buffer = ctx.queue->commandBuffer();
        metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.rgb_pipeline.get(), ctx.frame_buffer.get());
        metal::encode_fullscreen_pass<SPEC>(ctx, command_buffer, ctx.depth_pipeline.get(), ctx.depth_buffer.get());
        command_buffer->commit();
        ctx.in_flight = NS::RetainPtr(command_buffer);
        pool->release();
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB && SPEC::HAS_DEPTH, "render_rgb_depth_only requires an RGBD renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        auto& ctx = metal::context(renderer);
        if(ctx.in_flight.get() != nullptr){
            ctx.in_flight->waitUntilCompleted();
            ctx.in_flight.reset();
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_rgb_depth_only_launch(device, renderer);
        render_rgb_depth_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void render_rgb_only_async(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        set_cameras_async(device, renderer, cameras);
        render_rgb_only_launch(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename FB_SPEC>
    void read_frame_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<FB_SPEC>& out_pixels){
        static_assert(SPEC::HAS_RGB, "read_frame_buffer requires an RGB-capable renderer specification");
        static_assert(utils::typing::is_same_v<typename FB_SPEC::T, uint32_t>);
        static_assert(get<0>(typename FB_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        namespace metal = rendering::raytracing::backends::metal;
        constexpr typename SPEC::TI expected = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::memcpy(data(out_pixels), metal::context(renderer).frame_buffer->contents(), expected * sizeof(uint32_t));
    }

    template <typename DEVICE, typename SPEC, typename DEPTH_SPEC>
    void read_depth_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<DEPTH_SPEC>& out_depth){
        static_assert(SPEC::HAS_DEPTH, "read_depth_buffer requires a depth-capable renderer specification");
        static_assert(utils::typing::is_same_v<typename DEPTH_SPEC::T, float>);
        static_assert(get<0>(typename DEPTH_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<1>(typename DEPTH_SPEC::SHAPE{}) == SPEC::CAM_HEIGHT);
        static_assert(get<2>(typename DEPTH_SPEC::SHAPE{}) == SPEC::CAM_WIDTH);
        namespace metal = rendering::raytracing::backends::metal;
        constexpr typename SPEC::TI expected = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::memcpy(data(out_depth), metal::context(renderer).depth_buffer->contents(), expected * sizeof(float));
    }

    template <typename DEVICE, typename SPEC>
    void save_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_RGB, "save_image requires an RGB-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        rendering::raytracing::detail::write_grid_png<SPEC>((const uint32_t*)metal::context(renderer).frame_buffer->contents(), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth_image requires a depth-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        rendering::raytracing::detail::write_depth_grid_png<SPEC>((const float*)metal::context(renderer).depth_buffer->contents(), renderer.camera_radius, filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth requires a depth-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        rendering::raytracing::detail::write_depth_bin<SPEC>((const float*)metal::context(renderer).depth_buffer->contents(), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_probes(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("save_probes skipped: probe rays are disabled.");
        (void)filename;
        return;
#else
        namespace metal = rendering::raytracing::backends::metal;
        const auto* probe_results = (const rendering::raytracing::CollisionResult*)metal::context(renderer).collision_results->contents();
        rendering::raytracing::detail::write_probes_bin_and_log<SPEC>(probe_results, filename);
#endif
    }

    template <typename DEVICE, typename SPEC, typename COLL_SPEC>
    void read_collision_results(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<COLL_SPEC>& out){
        static_assert(utils::typing::is_same_v<typename COLL_SPEC::T, rendering::raytracing::CollisionResult>);
        static_assert(get<0>(typename COLL_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<1>(typename COLL_SPEC::SHAPE{}) == SPEC::NUM_PROBES);
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        namespace metal = rendering::raytracing::backends::metal;
        if(renderer.backend.collision_results_buffer != nullptr){
            std::memcpy(data(out), metal::context(renderer).collision_results->contents(),
                        SPEC::NUM_CAMERAS * SPEC::NUM_PROBES * sizeof(rendering::raytracing::CollisionResult));
        }
#endif
    }

    template <typename DEVICE, typename SPEC>
    const rendering::raytracing::CollisionResult* read_collision_results_raw(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        return nullptr;
#else
        namespace metal = rendering::raytracing::backends::metal;
        if(renderer.backend.collision_results_buffer == nullptr){
            return nullptr;
        }
        return (const rendering::raytracing::CollisionResult*)metal::context(renderer).collision_results->contents();
#endif
    }

    template <typename DEVICE, typename SPEC>
    uint32_t* get_framebuffer_device_ptr(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "get_framebuffer_device_ptr requires an RGB-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        return (uint32_t*)metal::context(renderer).frame_buffer->contents();
    }

    template <typename DEVICE, typename SPEC>
    float* get_depthbuffer_device_ptr(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "get_depthbuffer_device_ptr requires a depth-capable renderer specification");
        namespace metal = rendering::raytracing::backends::metal;
        return (float*)metal::context(renderer).depth_buffer->contents();
    }

    template <typename DEVICE, typename SPEC>
    void synchronize(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        metal::wait_in_flight(metal::context(renderer));
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        namespace metal = rendering::raytracing::backends::metal;
        if(renderer.backend.context != nullptr){
            metal::wait_in_flight(metal::context(renderer));
            delete (metal::Context*)renderer.backend.context;
            renderer.backend.context = nullptr;
        }
        free(device, renderer.cameras);
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            free(device, renderer.cameras_open);
        }
        if constexpr (SPEC::HAS_RGB) {
            free(device, renderer.frame_buffer);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            free(device, renderer.depth_buffer);
        }
        free(device, renderer.collision_results);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
