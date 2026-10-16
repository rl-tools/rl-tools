#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_OPERATIONS_CUDA_H

#include "../../renderer.h"
#include "../../operations_cpu_common.h"
#include "device.h"
#include "overlay_accel.h"
#include "../../../../devices/cuda_streams.h"

#include "owl/owl.h"


#include <vector>
#include <limits>
#include <algorithm>
#include <functional>
#include <string>
#include <map>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    extern "C" char device_ptx[];
    extern "C" char device_normals_ptx[];

    namespace rendering::raytracing::backends::optix {
        // inactive overlay slots point at a shared degenerate-triangle BLAS, so every overlay
        // TLAS keeps a fixed instance count and instance-id layout across rebuilds
        struct OverlayState {
            std::vector<OWLGroup> object_groups;
            OWLGroup filler_group = nullptr;
            OverlayAccelState* accel = nullptr;
            OWLBuffer traversables_buffer = nullptr;
            OWLBuffer attachments_buffer = nullptr;
            // pageable staging for host-verb writes: cudaMemcpyAsync returns only after pageable
            // sources are consumed, so the lists can be refilled right away. The pending rows of a
            // staging mirror compact into row_indices/rows and scatter on the render stream.
            std::vector<OverlaySlotStructure> structure_staging;
            std::vector<unsigned int> row_indices;
            std::vector<float> rows;
            unsigned int* row_indices_device = nullptr;
            float* rows_device = nullptr;
            OWLBuffer traversables_previous_buffer = nullptr; // flow: the previous instant's TLAS handles
            size_t num_scene_instances = 0;
        };

        using Slot = rendering::raytracing::outputs::Slot;

        struct State {
            OWLBuffer outputs[(size_t)Slot::COUNT] = {};
            OWLBuffer instance_classes_buffer = nullptr;
            OWLContext context = nullptr;
            OWLModule module = nullptr;
            OWLMissProg miss_prog = nullptr;
            OWLBuffer cameras_buffer = nullptr;
            OWLBuffer cameras_open_buffer = nullptr;
            OWLGroup world = nullptr;
            OWLParams launch_params = nullptr;
            OWLRayGen collision_ray_gen = nullptr;
            OWLBuffer probe_dirs_buffer = nullptr;
            OWLParams coll_launch_params = nullptr;
            OverlayState* overlay_state = nullptr;
            OWLRayGen ray_gen = nullptr;
            OWLRayGen auxiliary_ray_gen = nullptr;
            OWLRayGen flow_forward_ray_gen = nullptr;
            OWLBuffer flow_deltas_buffer = nullptr; // internal [2][slots][12] delta table, composed by update_launch
            OWLBuffer cameras_previous_buffer = nullptr;
            OWLBuffer rgb_accumulator_buffer = nullptr;
            OWLBuffer camera_table_buffer = nullptr; // outputs::fill_camera_table
            float* shutter_device = nullptr; // per-pass shutter time, written by the overlay fill kernel
            cudaEvent_t ordering_event = nullptr; // reused for every stream-to-stream ordering (record/wait pairs are consumed on the spot)
        };
        inline const void* output_pointer(State& state, Slot slot){
            OWLBuffer buffer = state.outputs[(size_t)slot];
            return buffer != nullptr ? owlBufferGetPointer(buffer, 0) : nullptr;
        }
    }

    namespace rendering::raytracing::backends {
        template <typename SPEC>
        struct RendererState<rendering::raytracing::backends::Optix, SPEC>: optix::State {
            rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Optix>* library = nullptr;
        };

        template <typename SPEC>
        struct LibraryState<rendering::raytracing::backends::Optix, SPEC> {
            OWLContext context = nullptr;
            OWLModule module = nullptr;
            OWLMissProg miss_prog = nullptr;
            OWLGeomType geom_type = nullptr;
        };

        template <typename SPEC>
        struct SceneState<rendering::raytracing::backends::Optix, SPEC> {
            OWLGroup world = nullptr;
            OWLGroup filler_group = nullptr;
            std::vector<OWLGroup> object_groups;
            size_t num_scene_instances = 0;
        };
    }

    namespace rendering::raytracing::detail {
        template <typename SPEC>
        const char* ray_gen_program_name(const char* srgb_name, const char* linear_name) {
            if constexpr (SPEC::SHADING::SRGB_OUTPUT) {
                return srgb_name;
            }
            else {
                return linear_name;
            }
        }

        template <typename SPEC>
        const char* ray_gen_program_name() {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                // the motion loop runs at launch level (one pass per sample), so only the AA grid
                // is compiled in; accumulation is linear, srgb/linear resolve happens later
                if constexpr (SPEC::ENABLE_ANTI_ALIASING) {
                    if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                        return "accumRayGenAA2";
                    }
                    else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                        return "accumRayGenAA3";
                    }
                    else {
                        return "accumRayGenAA4";
                    }
                }
                else {
                    return "accumRayGen";
                }
            }
            else if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                if constexpr (SPEC::ENABLE_ANTI_ALIASING) {
                    if constexpr (SPEC::MOTION_BLUR_SAMPLES == 2) {
                        if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur2AA2", "linearRayGenMotionBlur2AA2");
                        }
                        else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur2AA3", "linearRayGenMotionBlur2AA3");
                        }
                        else {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur2AA4", "linearRayGenMotionBlur2AA4");
                        }
                    }
                    else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 4) {
                        if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur4AA2", "linearRayGenMotionBlur4AA2");
                        }
                        else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur4AA3", "linearRayGenMotionBlur4AA3");
                        }
                        else {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur4AA4", "linearRayGenMotionBlur4AA4");
                        }
                    }
                    else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 8) {
                        if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur8AA2", "linearRayGenMotionBlur8AA2");
                        }
                        else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur8AA3", "linearRayGenMotionBlur8AA3");
                        }
                        else {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur8AA4", "linearRayGenMotionBlur8AA4");
                        }
                    }
                    else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 16) {
                        if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur16AA2", "linearRayGenMotionBlur16AA2");
                        }
                        else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur16AA3", "linearRayGenMotionBlur16AA3");
                        }
                        else {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur16AA4", "linearRayGenMotionBlur16AA4");
                        }
                    }
                    else {
                        if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur32AA2", "linearRayGenMotionBlur32AA2");
                        }
                        else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur32AA3", "linearRayGenMotionBlur32AA3");
                        }
                        else {
                            return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur32AA4", "linearRayGenMotionBlur32AA4");
                        }
                    }
                }
                else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 2) {
                    return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur2", "linearRayGenMotionBlur2");
                }
                else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 4) {
                    return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur4", "linearRayGenMotionBlur4");
                }
                else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 8) {
                    return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur8", "linearRayGenMotionBlur8");
                }
                else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 16) {
                    return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur16", "linearRayGenMotionBlur16");
                }
                else {
                    return ray_gen_program_name<SPEC>("simpleRayGenMotionBlur32", "linearRayGenMotionBlur32");
                }
            }
            else if constexpr (SPEC::ENABLE_ANTI_ALIASING) {
                if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                    return ray_gen_program_name<SPEC>("simpleRayGenAA2", "linearRayGenAA2");
                }
                else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                    return ray_gen_program_name<SPEC>("simpleRayGenAA3", "linearRayGenAA3");
                }
                else {
                    return ray_gen_program_name<SPEC>("simpleRayGenAA4", "linearRayGenAA4");
                }
            }
            else {
                return ray_gen_program_name<SPEC>("simpleRayGen", "linearRayGen");
            }
        }

        template <typename SPEC>
        const char* closest_hit_program_name() {
            if constexpr (SPEC::SHADING::PBR_SHADING) {
                return SPEC::SHADING::PUNCTUAL_LIGHT_SHADOWS ? "TriangleMeshPBRShadows" : "TriangleMeshPBR";
            }
            else if constexpr (SPEC::SHADING::LOAD_TEXTURES) {
                if constexpr (SPEC::SHADING::NORMAL_SHADING) {
                    return SPEC::SHADING::METALLIC_REFLECTIONS ? "TriangleMeshBasicTTT" : "TriangleMeshBasicTTF";
                }
                else {
                    return SPEC::SHADING::METALLIC_REFLECTIONS ? "TriangleMeshBasicTFT" : "TriangleMeshBasicTFF";
                }
            }
            else {
                if constexpr (SPEC::SHADING::NORMAL_SHADING) {
                    return SPEC::SHADING::METALLIC_REFLECTIONS ? "TriangleMeshBasicFTT" : "TriangleMeshBasicFTF";
                }
                else {
                    return SPEC::SHADING::METALLIC_REFLECTIONS ? "TriangleMeshBasicFFT" : "TriangleMeshBasicFFF";
                }
            }
        }

        template <typename SPEC>
        struct MediumShadingUsage {
            static constexpr bool USES_INDEX = SPEC::SHADING::LOAD_TEXTURES || SPEC::SHADING::NORMAL_SHADING || SPEC::SHADING::METALLIC_REFLECTIONS;
            static constexpr bool USES_VERTEX = SPEC::SHADING::NORMAL_SHADING || SPEC::SHADING::METALLIC_REFLECTIONS;
            static constexpr bool USES_TEXTURE = SPEC::SHADING::LOAD_TEXTURES;
            static constexpr bool USES_WORLD = SPEC::SHADING::METALLIC_REFLECTIONS;
        };
    }

    // =========================================================================
    // malloc: create OWL context and allocate buffers
    // =========================================================================
    namespace rendering::raytracing::backends::optix{
        // context-scoped resources (context, module, miss programs): created once per renderer
        // in standalone mode, once per AssetLibrary in shared mode. The context is bound to the
        // CUDA device that is current at creation (multi-GPU targets cudaSetDevice before malloc);
        // owlContextCreate(nullptr, 1) would always pick ordinal 0
        template <typename SPEC>
        void create_context_resources(OWLContext& context_out, OWLModule& module_out, OWLMissProg& miss_prog_out){
            int32_t cuda_device_id = 0;
            cudaGetDevice(&cuda_device_id);
            OWLContext context = owlContextCreate(&cuda_device_id, 1);
            // ray type 2 (normals) only exists in the normals-enabled PTX variants; the count
            // must match the NUM_RAY_TYPES the selected PTX was compiled with (SBT stride)
            owlContextSetRayTypeCount(context, SPEC::HAS_NORMALS ? 3 : 2);
            owlContextSetNumPayloadValues(context, SPEC::HAS_NORMALS ? 5 : 4);
            owlContextSetMaxRegisterCount(context, 64);
            owlContextSetPrimitiveTypeFlags(context, OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
            // OWL retains specialization value pointers until owlBuildPrograms.
            static const unsigned int output_mask = SPEC::PLAN::MASK;
            static const int all_cameras = rendering::raytracing::outputs::Layout<SPEC>::ALL_CAMERAS ? 1 : 0;
            static const int overlay_count = (int)SPEC::MAX_OVERLAYS_PER_CAMERA;
            static const int cam_width = (int)SPEC::CAM_WIDTH;
            static const int cam_height = (int)SPEC::CAM_HEIGHT;
            static const int grid_cols = (int)SPEC::GRID_COLS;
            OWLBoundValueDecl bound_values[] = {
                {{"output_mask", OWL_UINT, OWL_OFFSETOF(OverlayLaunchParams, output_mask)}, (void*)&output_mask},
                {{"all_cameras", OWL_INT, OWL_OFFSETOF(OverlayLaunchParams, all_cameras)}, (void*)&all_cameras},
                {{"overlay_count", OWL_INT, OWL_OFFSETOF(OverlayLaunchParams, overlay_count)}, (void*)&overlay_count},
                {{"cam_width", OWL_INT, OWL_OFFSETOF(OverlayLaunchParams, cam_width)}, (void*)&cam_width},
                {{"cam_height", OWL_INT, OWL_OFFSETOF(OverlayLaunchParams, cam_height)}, (void*)&cam_height},
                {{"grid_cols", OWL_INT, OWL_OFFSETOF(OverlayLaunchParams, grid_cols)}, (void*)&grid_cols},
            };
            owlContextSetBoundLaunchParamValues(context, bound_values, 6);
            const char* ptx = SPEC::HAS_NORMALS ? device_normals_ptx : device_ptx;
            OWLModule module = owlModuleCreate(context, ptx);

            // RGB miss program (ray type 0)
            OWLVarDecl miss_prog_vars[] = {
                { "color_0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color_0)},
                { "color_1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color_1)},
                { /* sentinel */ }
            };
            const char* miss_program_name = "miss";
            if constexpr (SPEC::HAS_RGB && !SPEC::SHADING::CHECKER_BACKGROUND) {
                miss_program_name = "missConstant";
            }
            OWLMissProg miss_prog = owlMissProgCreate(context, module, miss_program_name,
                                                        sizeof(MissProgData), miss_prog_vars, -1);
            // colors are scene content (Scene::Environment), resolved and set at init before the
            // SBT build; the creation-time values are placeholders
            owlMissProgSet3f(miss_prog, "color_0", owl3f{0.f, 0.f, 0.f});
            owlMissProgSet3f(miss_prog, "color_1", owl3f{0.f, 0.f, 0.f});
            miss_prog_out = miss_prog;

            // Collision miss program (ray type 1) — always registered to keep SBT consistent
            OWLVarDecl collision_miss_vars[] = {
                { "dummy", OWL_INT, OWL_OFFSETOF(CollisionMissData, dummy)},
                { /* sentinel */ }
            };
            OWLMissProg collision_miss_prog = owlMissProgCreate(context, module, "collisionMiss",
                                                                 sizeof(CollisionMissData), collision_miss_vars, -1);
            (void)collision_miss_prog;

            context_out = context;
            module_out = module;
        }
    }

    namespace rendering::raytracing::backends::optix::detail{
        // everything renderer-private: framebuffers, cameras, ray gens, collision resources
        template <typename DEVICE, typename SPEC>
        void malloc_renderer(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, OWLContext context, OWLModule module){
        using TI = typename SPEC::TI;

        if constexpr (SPEC::ENABLE_OVERLAYS) {
            // device-resident transform input (see transforms(device, renderer)): producers write
            // it directly, update_launch consumes it on the render stream
            using TRANSFORMS_SPEC = typename decltype(renderer.transforms)::SPEC;
            float* transforms_device = nullptr;
            cudaMalloc(&transforms_device, TRANSFORMS_SPEC::SIZE_BYTES);
            cudaMemset(transforms_device, 0, TRANSFORMS_SPEC::SIZE_BYTES);
            renderer.transforms._data = transforms_device;
            rendering::raytracing::detail::malloc_overlay_staging(renderer);
        }
        if constexpr (SPEC::HAS_FLOW_OVERLAYS) {
            using TRANSFORMS_PREVIOUS_SPEC = typename decltype(renderer.transforms_previous)::SPEC;
            float* transforms_previous_device = nullptr;
            cudaMalloc(&transforms_previous_device, TRANSFORMS_PREVIOUS_SPEC::SIZE_BYTES);
            cudaMemset(transforms_previous_device, 0, TRANSFORMS_PREVIOUS_SPEC::SIZE_BYTES);
            renderer.transforms_previous._data = transforms_previous_device;
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            using TRANSFORMS_MOTION_SPEC = typename decltype(renderer.transforms_motion)::SPEC;
            float* transforms_motion_device = nullptr;
            cudaMalloc(&transforms_motion_device, TRANSFORMS_MOTION_SPEC::SIZE_BYTES);
            cudaMemset(transforms_motion_device, 0, TRANSFORMS_MOTION_SPEC::SIZE_BYTES);
            renderer.transforms_motion._data = transforms_motion_device;
            cudaMalloc(&renderer.backend->shutter_device, sizeof(float));
            cudaMemset(renderer.backend->shutter_device, 0, sizeof(float));
        }

        constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        using Slot = rendering::raytracing::outputs::Slot;
        float depth_scale = 10.f, flow_scale = 20.f, flow_forward_scale = 20.f;
        for_each_output(device, renderer.outputs, [&](auto& buffer){
            using BUFFER = utils::typing::remove_reference_t<decltype(buffer)>;
            using OUTPUT = typename BUFFER::OUTPUT;
            constexpr auto SLOT = rendering::raytracing::outputs::Traits<OUTPUT>::SLOT;
            auto& resource = renderer.backend->outputs[(size_t)SLOT];
            resource = owlDeviceBufferCreate(context, OWL_UINT, BUFFER::TENSOR_SPEC::SIZE_BYTES / sizeof(unsigned int), nullptr);
            buffer.tensor._data = (typename BUFFER::TENSOR_SPEC::T*)owlBufferGetPointer(resource, 0);
            if constexpr(SLOT == Slot::DEPTH_RGBA8){
                depth_scale = OUTPUT::PARAMETERS::SCALE;
            }
            if constexpr(SLOT == Slot::FLOW_RGBA8){
                flow_scale = OUTPUT::PARAMETERS::SCALE;
            }
            if constexpr(SLOT == Slot::FLOW_FORWARD_RGBA8){
                flow_forward_scale = OUTPUT::PARAMETERS::SCALE;
            }
        });
        const auto pointer = [&](Slot slot){ return output_pointer(*renderer.backend, slot); };
        std::vector<unsigned int> camera_table((size_t)SPEC::NUM_CAMERAS * rendering::raytracing::outputs::CAMERA_TABLE_STRIDE);
        fill_camera_table<SPEC>(device, camera_table.data());
        renderer.backend->camera_table_buffer = owlDeviceBufferCreate(context, OWL_UINT, camera_table.size(), camera_table.data());
        OWLBuffer flow_deltas_buffer = nullptr;
        if constexpr(SPEC::HAS_FLOW_OVERLAYS){
            flow_deltas_buffer = owlDeviceBufferCreate(context, OWL_FLOAT, (size_t)2 * SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12, nullptr);
            renderer.backend->flow_deltas_buffer = flow_deltas_buffer;
        }
        OWLBuffer rgb_accumulator_buffer = nullptr;
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            if constexpr (SPEC::HAS_RGB) {
                rgb_accumulator_buffer = owlDeviceBufferCreate(context, OWL_FLOAT,
                                                               (size_t)SPEC::NUM_CAMERAS * cam_pixels * 3, nullptr);
                renderer.backend->rgb_accumulator_buffer = rgb_accumulator_buffer;
                renderer.rgb_accumulator._data = (float*)owlBufferGetPointer(rgb_accumulator_buffer, 0);
            }
        }

        OWLRayGen ray_gen = nullptr;
        if constexpr (SPEC::HAS_RGB) {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                OWLVarDecl ray_gen_vars[] = {
                    { "accum_ptr",     OWL_RAW_POINTER, OWL_OFFSETOF(AccumulateRayGenData, accum_ptr)},
                    { "shutter_ptr",   OWL_RAW_POINTER, OWL_OFFSETOF(AccumulateRayGenData, shutter_ptr)},
                    { "num_cameras",   OWL_INT,    OWL_OFFSETOF(AccumulateRayGenData, num_cameras)},
                    { "world",         OWL_GROUP,  OWL_OFFSETOF(AccumulateRayGenData, world)},
                    { "cameras_open",  OWL_BUFPTR, OWL_OFFSETOF(AccumulateRayGenData, cameras_open)},
                    { "cameras_close", OWL_BUFPTR, OWL_OFFSETOF(AccumulateRayGenData, cameras_close)},
                    { /* sentinel */ }
                };
                const char* ray_gen_name = rendering::raytracing::detail::ray_gen_program_name<SPEC>();
                ray_gen = owlRayGenCreate(context, module, ray_gen_name,
                                          sizeof(AccumulateRayGenData), ray_gen_vars, -1);
            }
            else if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                OWLVarDecl ray_gen_vars[] = {
                    { "fb_ptr",        OWL_RAW_POINTER, OWL_OFFSETOF(MotionBlurRayGenData, fb_ptr)},
                    { "obs_ptr",       OWL_RAW_POINTER, OWL_OFFSETOF(MotionBlurRayGenData, obs_ptr)},
                    { "num_cameras",   OWL_INT,    OWL_OFFSETOF(MotionBlurRayGenData, num_cameras)},
                    { "world",         OWL_GROUP,  OWL_OFFSETOF(MotionBlurRayGenData, world)},
                    { "cameras_open",  OWL_BUFPTR, OWL_OFFSETOF(MotionBlurRayGenData, cameras_open)},
                    { "cameras_close", OWL_BUFPTR, OWL_OFFSETOF(MotionBlurRayGenData, cameras_close)},
                    { /* sentinel */ }
                };
                const char* ray_gen_name = rendering::raytracing::detail::ray_gen_program_name<SPEC>();
                ray_gen = owlRayGenCreate(context, module, ray_gen_name,
                                          sizeof(MotionBlurRayGenData), ray_gen_vars, -1);
            }
            else {
                OWLVarDecl ray_gen_vars[] = {
                    { "fb_ptr",       OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData, fb_ptr)},
                    { "obs_ptr",      OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData, obs_ptr)},
                    { "num_cameras",  OWL_INT,    OWL_OFFSETOF(RayGenData, num_cameras)},
                    { "world",       OWL_GROUP,  OWL_OFFSETOF(RayGenData, world)},
                    { "cameras",     OWL_BUFPTR, OWL_OFFSETOF(RayGenData, cameras)},
                    { /* sentinel */ }
                };
                const char* ray_gen_name = rendering::raytracing::detail::ray_gen_program_name<SPEC>();
                ray_gen = owlRayGenCreate(context, module, ray_gen_name,
                                          sizeof(RayGenData), ray_gen_vars, -1);
            }
        }

        OWLRayGen flow_forward_ray_gen = nullptr;
        if constexpr (SPEC::HAS_FLOW_FORWARD) {
            OWLVarDecl flow_ray_gen_vars[] = {
                { "flow_ptr",       OWL_RAW_POINTER, OWL_OFFSETOF(FlowRayGenData, flow_ptr)},
                { "color_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(FlowRayGenData, color_ptr)},
                { "status_color_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(FlowRayGenData, status_color_ptr)},
                { "color_scale", OWL_FLOAT, OWL_OFFSETOF(FlowRayGenData, color_scale)},
                { "status_ptr",     OWL_RAW_POINTER, OWL_OFFSETOF(FlowRayGenData, status_ptr)},
                { "num_cameras",    OWL_INT,    OWL_OFFSETOF(FlowRayGenData, num_cameras)},
                { "world",          OWL_GROUP,  OWL_OFFSETOF(FlowRayGenData, world)},
                { "cameras_source", OWL_BUFPTR, OWL_OFFSETOF(FlowRayGenData, cameras_source)},
                { "cameras_target", OWL_BUFPTR, OWL_OFFSETOF(FlowRayGenData, cameras_target)},
                { "flow_deltas",    OWL_RAW_POINTER, OWL_OFFSETOF(FlowRayGenData, flow_deltas)},
                { "source_first_overlay_instance", OWL_UINT, OWL_OFFSETOF(FlowRayGenData, source_first_overlay_instance)},
                { "source_previous", OWL_INT,   OWL_OFFSETOF(FlowRayGenData, source_previous)},
                { /* sentinel */ }
            };
            flow_forward_ray_gen = owlRayGenCreate(context, module, "flowForwardRayGen", sizeof(FlowRayGenData), flow_ray_gen_vars, -1);
        }

        OWLRayGen auxiliary_ray_gen = nullptr;
        if constexpr (SPEC::HAS_AUXILIARY) {
            OWLVarDecl auxiliary_vars[] = {
                { "flow_ptr",       OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, flow_ptr)},
                { "color_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, color_ptr)},
                { "status_color_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, status_color_ptr)},
                { "color_scale", OWL_FLOAT, OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, color_scale)},
                { "status_ptr",     OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, status_ptr)},
                { "num_cameras",    OWL_INT,    OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, num_cameras)},
                { "world",          OWL_GROUP,  OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, world)},
                { "cameras_source", OWL_BUFPTR, OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, cameras_source)},
                { "cameras_target", OWL_BUFPTR, OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, cameras_target)},
                { "flow_deltas",    OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, flow_deltas)},
                { "source_first_overlay_instance", OWL_UINT, OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, source_first_overlay_instance)},
                { "source_previous", OWL_INT,   OWL_OFFSETOF(AuxiliaryRayGenData, flow) + OWL_OFFSETOF(FlowRayGenData, source_previous)},
                { "depth_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, depth_ptr)},
                { "depth_color_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, depth_color_ptr)},
                { "seg_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, seg_ptr)},
                { "seg_color_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, seg_color_ptr)},
                { "semantic_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, semantic_ptr)},
                { "semantic_color_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, semantic_color_ptr)},
                { "normals_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, normals_ptr)},
                { "normals_color_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AuxiliaryRayGenData, normals_color_ptr)},
                { "depth_scale", OWL_FLOAT, OWL_OFFSETOF(AuxiliaryRayGenData, depth_scale)},
                { /* sentinel */ }
            };
            auxiliary_ray_gen = owlRayGenCreate(context, module, "auxiliaryRayGen", sizeof(AuxiliaryRayGenData), auxiliary_vars, -1);
            renderer.backend->auxiliary_ray_gen = auxiliary_ray_gen;
        }


        if constexpr (SPEC::HAS_RGB) {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                owlRayGenSetPointer(ray_gen, "accum_ptr", owlBufferGetPointer(rgb_accumulator_buffer, 0));
                owlRayGenSetPointer(ray_gen, "shutter_ptr", renderer.backend->shutter_device);
            }
            else {
                owlRayGenSetPointer(ray_gen, "fb_ptr", pointer(Slot::RGB_RGBA8));
                owlRayGenSetPointer(ray_gen, "obs_ptr", pointer(Slot::RGB_FLOAT32));
            }
            owlRayGenSet1i    (ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
        }
        if constexpr (SPEC::HAS_FLOW_FORWARD) {
            constexpr size_t FLOW_SLOTS = SPEC::ENABLE_OVERLAYS ? (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES : 0;
            owlRayGenSetPointer(flow_forward_ray_gen, "flow_ptr", pointer(Slot::FLOW_FORWARD_FLOAT32));
            owlRayGenSetPointer(flow_forward_ray_gen, "color_ptr", pointer(Slot::FLOW_FORWARD_RGBA8));
            owlRayGenSetPointer(flow_forward_ray_gen, "status_ptr", pointer(Slot::FLOW_FORWARD_STATUS_UINT32));
            owlRayGenSetPointer(flow_forward_ray_gen, "status_color_ptr", pointer(Slot::FLOW_FORWARD_STATUS_RGBA8));
            owlRayGenSet1f(flow_forward_ray_gen, "color_scale", flow_forward_scale);
            owlRayGenSet1i(flow_forward_ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
            owlRayGenSetPointer(flow_forward_ray_gen, "flow_deltas", flow_deltas_buffer != nullptr ? (const float*)owlBufferGetPointer(flow_deltas_buffer, 0) + FLOW_SLOTS * 12 : nullptr);
            owlRayGenSet1ui(flow_forward_ray_gen, "source_first_overlay_instance", 0u);
            owlRayGenSet1i(flow_forward_ray_gen, "source_previous", 1);
            renderer.backend->flow_forward_ray_gen = flow_forward_ray_gen;
        }
        if constexpr (SPEC::HAS_AUXILIARY) {
            owlRayGenSetPointer(auxiliary_ray_gen, "flow_ptr", pointer(Slot::FLOW_FLOAT32));
            owlRayGenSetPointer(auxiliary_ray_gen, "color_ptr", pointer(Slot::FLOW_RGBA8));
            owlRayGenSetPointer(auxiliary_ray_gen, "status_ptr", pointer(Slot::FLOW_STATUS_UINT32));
            owlRayGenSetPointer(auxiliary_ray_gen, "status_color_ptr", pointer(Slot::FLOW_STATUS_RGBA8));
            owlRayGenSetPointer(auxiliary_ray_gen, "depth_ptr", pointer(Slot::DEPTH_FLOAT32));
            owlRayGenSetPointer(auxiliary_ray_gen, "depth_color_ptr", pointer(Slot::DEPTH_RGBA8));
            owlRayGenSetPointer(auxiliary_ray_gen, "seg_ptr", pointer(Slot::SEGMENTATION_UINT32));
            owlRayGenSetPointer(auxiliary_ray_gen, "seg_color_ptr", pointer(Slot::SEGMENTATION_RGBA8));
            owlRayGenSetPointer(auxiliary_ray_gen, "semantic_ptr", pointer(Slot::SEMANTIC_UINT32));
            owlRayGenSetPointer(auxiliary_ray_gen, "semantic_color_ptr", pointer(Slot::SEMANTIC_RGBA8));
            owlRayGenSetPointer(auxiliary_ray_gen, "normals_ptr", pointer(Slot::NORMALS_FLOAT32));
            owlRayGenSetPointer(auxiliary_ray_gen, "normals_color_ptr", pointer(Slot::NORMALS_RGBA8));
            owlRayGenSet1f(auxiliary_ray_gen, "depth_scale", depth_scale);
            owlRayGenSet1f(auxiliary_ray_gen, "color_scale", flow_scale);
            owlRayGenSet1i(auxiliary_ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
            owlRayGenSetPointer(auxiliary_ray_gen, "flow_deltas", flow_deltas_buffer != nullptr ? owlBufferGetPointer(flow_deltas_buffer, 0) : nullptr);
            owlRayGenSet1ui(auxiliary_ray_gen, "source_first_overlay_instance", 0u);
            owlRayGenSet1i(auxiliary_ray_gen, "source_previous", 0);
        }

        // The raygen-referenced buffers are created here so every raygen variable is set before
        // init builds the SBT: OWL serializes raygen variables into the SBT records only during
        // owlBuildSBT, so anything set later would be baked in as null.
        OWLBuffer cameras_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, nullptr);
        // device-resident camera input (see cameras(device, renderer)): the tensors alias the
        // OWL buffers the ray gens bind, so producer writes are consumed with no copy
        renderer.cameras._data = (rendering::raytracing::Camera<typename SPEC::T>*)owlBufferGetPointer(cameras_buffer, 0);
        OWLBuffer cameras_open_buffer = nullptr;
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            cameras_open_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, nullptr);
            renderer.backend->cameras_open_buffer = cameras_open_buffer;
            renderer.cameras_open._data = (rendering::raytracing::Camera<typename SPEC::T>*)owlBufferGetPointer(cameras_open_buffer, 0);
        }
        if constexpr (SPEC::HAS_RGB) {
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                owlRayGenSetBuffer(ray_gen, "cameras_open", cameras_open_buffer);
                owlRayGenSetBuffer(ray_gen, "cameras_close", cameras_buffer);
            }
            else {
                owlRayGenSetBuffer(ray_gen, "cameras", cameras_buffer);
            }
        }
        if constexpr (SPEC::HAS_FLOW) {
            OWLBuffer cameras_previous_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, nullptr);
            renderer.backend->cameras_previous_buffer = cameras_previous_buffer;
            renderer.cameras_previous._data = (rendering::raytracing::Camera<typename SPEC::T>*)owlBufferGetPointer(cameras_previous_buffer, 0);
            if constexpr (SPEC::HAS_FLOW_FORWARD) {
                owlRayGenSetBuffer(flow_forward_ray_gen, "cameras_source", cameras_previous_buffer);
                owlRayGenSetBuffer(flow_forward_ray_gen, "cameras_target", cameras_buffer);
            }
        }

        if constexpr (SPEC::HAS_AUXILIARY) {
            owlRayGenSetBuffer(auxiliary_ray_gen, "cameras_source", cameras_buffer);
            owlRayGenSetBuffer(auxiliary_ray_gen, "cameras_target", SPEC::HAS_FLOW ? renderer.backend->cameras_previous_buffer : cameras_buffer);
        }

        renderer.backend->context = context;
        renderer.backend->module = module;
        renderer.backend->cameras_buffer = cameras_buffer;
        if constexpr (SPEC::HAS_RGB) {
            renderer.backend->ray_gen = ray_gen;
        }
        if constexpr(SPEC::HAS_PROBES){
            OWLVarDecl collision_ray_gen_vars[] = {
                { "results",         OWL_BUFPTR, OWL_OFFSETOF(CollisionRayGenData, results)},
                { "probe_directions", OWL_BUFPTR, OWL_OFFSETOF(CollisionRayGenData, probe_directions)},
                { "cameras",         OWL_BUFPTR, OWL_OFFSETOF(CollisionRayGenData, cameras)},
                { "world",           OWL_GROUP,  OWL_OFFSETOF(CollisionRayGenData, world)},
                { "num_probes",       OWL_INT,    OWL_OFFSETOF(CollisionRayGenData, num_probes)},
                { "num_cameras",      OWL_INT,    OWL_OFFSETOF(CollisionRayGenData, num_cameras)},
                { "max_dist",         OWL_FLOAT,  OWL_OFFSETOF(CollisionRayGenData, max_dist)},
                { /* sentinel */ }
            };
            OWLRayGen collision_ray_gen = owlRayGenCreate(context, module, "collisionRayGen",
                                                           sizeof(CollisionRayGenData),
                                                           collision_ray_gen_vars, -1);

            OWLBuffer probe_dirs_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(owl::vec3f), SPEC::NUM_PROBES, nullptr);

            owlRayGenSetBuffer(collision_ray_gen, "results", renderer.backend->outputs[(size_t)Slot::PROBES]);
            owlRayGenSetBuffer(collision_ray_gen, "probe_directions", probe_dirs_buffer);
            owlRayGenSetBuffer(collision_ray_gen, "cameras", cameras_buffer);
            owlRayGenSet1i    (collision_ray_gen, "num_probes", SPEC::NUM_PROBES);
            owlRayGenSet1i    (collision_ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
            owlRayGenSet1f    (collision_ray_gen, "max_dist", 1e30f);

            renderer.backend->collision_ray_gen = collision_ray_gen;
            renderer.backend->probe_dirs_buffer = probe_dirs_buffer;
        }
        rendering::raytracing::detail::announce_backend(device, renderer);
        }
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        namespace optix = rendering::raytracing::backends::optix;
        renderer.backend = new rendering::raytracing::backends::RendererState<rendering::raytracing::backends::Optix, SPEC>{};
        renderer.device.state = renderer.backend;
        cudaEventCreateWithFlags(&renderer.backend->ordering_event, cudaEventDisableTiming);
        OWLContext context = nullptr;
        OWLModule module = nullptr;
        OWLMissProg miss_prog = nullptr;
        optix::create_context_resources<SPEC>(context, module, miss_prog);
        renderer.backend->miss_prog = miss_prog;
        optix::detail::malloc_renderer(device, renderer, context, module);
    }

    // shared-library mode: the library owns the context; the renderer allocates only its own
    // cameras/outputs/ray gens against it
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Optix>& library){
        namespace optix = rendering::raytracing::backends::optix;
        renderer.backend = new rendering::raytracing::backends::RendererState<rendering::raytracing::backends::Optix, SPEC>{};
        renderer.device.state = renderer.backend;
        cudaEventCreateWithFlags(&renderer.backend->ordering_event, cudaEventDisableTiming);
        optix::detail::malloc_renderer(device, renderer, library.backend->context, library.backend->module);
        renderer.backend->miss_prog = library.backend->miss_prog;
        renderer.backend->library = &library;
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Optix>& library){
        library.backend = new rendering::raytracing::backends::LibraryState<rendering::raytracing::backends::Optix, SPEC>{};
        rendering::raytracing::backends::optix::create_context_resources<SPEC>(library.backend->context, library.backend->module, library.backend->miss_prog);
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Optix>& library){
        if(library.backend != nullptr && library.backend->context != nullptr){
            owlContextDestroy(library.backend->context);
        }
        for(auto* assets : library.assets){
            delete assets;
        }
        library.assets.clear();
        library.scenes.clear();
        library.metadata.clear();
        delete library.backend;
        library.backend = nullptr;
    }

    // =========================================================================
    // init: upload meshes, build single shared BVH, build programs/pipeline/SBT
    // =========================================================================
    namespace rendering::raytracing::backends::optix::detail{
        template <typename SPEC>
        OWLGeomType create_geom_type(OWLContext context, OWLModule module){
        OWLGeomType triangles_geom_type;
            if constexpr (SPEC::HAS_RGB) {
                if constexpr (SPEC::SHADING::PBR_SHADING) {
                    OWLVarDecl triangles_geom_vars[] = {
                        { "index",      OWL_BUFPTR,  OWL_OFFSETOF(TrianglesGeomData, index)},
                        { "vertex",     OWL_BUFPTR,  OWL_OFFSETOF(TrianglesGeomData, vertex)},
                        { "tex_coord",   OWL_BUFPTR,  OWL_OFFSETOF(TrianglesGeomData, tex_coord)},
                        { "color",      OWL_FLOAT3,  OWL_OFFSETOF(TrianglesGeomData, color)},
                        { "texture",    OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData, texture)},
                        { "has_texture", OWL_INT,     OWL_OFFSETOF(TrianglesGeomData, has_texture)},
                        { "unlit", OWL_INT, OWL_OFFSETOF(TrianglesGeomData, unlit)},
                        { "metallic",    OWL_FLOAT,   OWL_OFFSETOF(TrianglesGeomData, metallic)},
                        { "world",       OWL_GROUP,   OWL_OFFSETOF(TrianglesGeomData, world)},
                        { "normal",      OWL_BUFPTR,  OWL_OFFSETOF(TrianglesGeomData, normal)},
                        { "roughness",   OWL_FLOAT,   OWL_OFFSETOF(TrianglesGeomData, roughness)},
                        { "normal_map",  OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData, normal_map)},
                        { "has_normal_map", OWL_INT,  OWL_OFFSETOF(TrianglesGeomData, has_normal_map)},
                        { "metallic_roughness_map", OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData, metallic_roughness_map)},
                        { "has_metallic_roughness_map", OWL_INT, OWL_OFFSETOF(TrianglesGeomData, has_metallic_roughness_map)},
                        { "emissive",      OWL_FLOAT3,  OWL_OFFSETOF(TrianglesGeomData, emissive)},
                        { "emissive_map",  OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData, emissive_map)},
                        { "has_emissive_map", OWL_INT,  OWL_OFFSETOF(TrianglesGeomData, has_emissive_map)},
                        { "occlusion_map", OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData, occlusion_map)},
                        { "has_occlusion_map", OWL_INT, OWL_OFFSETOF(TrianglesGeomData, has_occlusion_map)},
                        { "opacity",       OWL_FLOAT,   OWL_OFFSETOF(TrianglesGeomData, opacity)},
                        { "alpha_mode",    OWL_INT,     OWL_OFFSETOF(TrianglesGeomData, alpha_mode)},
                        { "alpha_cutoff",  OWL_FLOAT,   OWL_OFFSETOF(TrianglesGeomData, alpha_cutoff)},
                        { "scene_lights",  OWL_BUFPTR,  OWL_OFFSETOF(TrianglesGeomData, scene_lights)},
                        { "num_scene_lights", OWL_INT,  OWL_OFFSETOF(TrianglesGeomData, num_scene_lights)},
                        { "ambient_color", OWL_FLOAT3,  OWL_OFFSETOF(TrianglesGeomData, ambient_color)},
                        { /* sentinel */ }
                    };
                    triangles_geom_type = owlGeomTypeCreate(context, OWL_TRIANGLES,
                                                             sizeof(TrianglesGeomData),
                                                             triangles_geom_vars, -1);
                    owlGeomTypeSetClosestHit(triangles_geom_type, 0, module, rendering::raytracing::detail::closest_hit_program_name<SPEC>());
                } else {
                    using SHADING_USAGE = rendering::raytracing::detail::MediumShadingUsage<SPEC>;
                    std::vector<OWLVarDecl> triangles_geom_vars;
                    if constexpr (SHADING_USAGE::USES_INDEX || SPEC::HAS_NORMALS) {
                        triangles_geom_vars.push_back({ "index", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index)});
                    }
                    if constexpr (SHADING_USAGE::USES_VERTEX || SPEC::HAS_NORMALS) {
                        triangles_geom_vars.push_back({ "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex)});
                    }
                    if constexpr (SHADING_USAGE::USES_TEXTURE) {
                        triangles_geom_vars.push_back({ "tex_coord", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, tex_coord)});
                    }
                    triangles_geom_vars.push_back({ "color", OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData, color)});
                    triangles_geom_vars.push_back({ "unlit", OWL_INT, OWL_OFFSETOF(TrianglesGeomData, unlit)});
                    if constexpr (SHADING_USAGE::USES_TEXTURE) {
                        triangles_geom_vars.push_back({ "texture", OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData, texture)});
                        triangles_geom_vars.push_back({ "has_texture", OWL_INT, OWL_OFFSETOF(TrianglesGeomData, has_texture)});
                    }
                    if constexpr (SPEC::SHADING::METALLIC_REFLECTIONS) {
                        triangles_geom_vars.push_back({ "metallic", OWL_FLOAT, OWL_OFFSETOF(TrianglesGeomData, metallic)});
                    }
                    if constexpr (SHADING_USAGE::USES_WORLD) {
                        triangles_geom_vars.push_back({ "world", OWL_GROUP, OWL_OFFSETOF(TrianglesGeomData, world)});
                    }
                    triangles_geom_vars.push_back({});
                    triangles_geom_type = owlGeomTypeCreate(context, OWL_TRIANGLES,
                                                             sizeof(TrianglesGeomData),
                                                             triangles_geom_vars.data(), -1);
                    owlGeomTypeSetClosestHit(triangles_geom_type, 0, module, rendering::raytracing::detail::closest_hit_program_name<SPEC>());
                }
            }
            else if constexpr (SPEC::HAS_NORMALS) {
                OWLVarDecl triangles_geom_vars[] = {
                    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index)},
                    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex)},
                    { /* sentinel */ }
                };
                triangles_geom_type = owlGeomTypeCreate(context, OWL_TRIANGLES,
                                                         sizeof(TrianglesGeomData),
                                                         triangles_geom_vars, -1);
            }
            else {
                OWLVarDecl triangles_geom_vars[] = {
                    { /* sentinel */ }
                };
                triangles_geom_type = owlGeomTypeCreate(context, OWL_TRIANGLES,
                                                         sizeof(CollisionGeomData),
                                                         triangles_geom_vars, -1);
            }
            owlGeomTypeSetClosestHit(triangles_geom_type, 1, module, "collisionHit");
            if constexpr (SPEC::HAS_NORMALS) {
                owlGeomTypeSetClosestHit(triangles_geom_type, 2, module, "auxiliaryHit");
            }
            return triangles_geom_type;
        }

        // scene objects (and pool assets) all become the same all_objects order everywhere —
        // part of the deterministic global instance-id layout
        inline void collect_all_objects(const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool, std::vector<const rendering::raytracing::Object*>& all_objects){
            all_objects.clear();
            for(const auto& object : scene.objects){
                all_objects.push_back(&object);
            }
            for(const auto& assembly : pool.assemblies){
                for(const auto& object : assembly.objects){
                    all_objects.push_back(&object);
                }
            }
        }

        // one geometry/texture/BLAS build per unique scene: standalone renderers build into
        // their own context, shared-library renderers reference the library's single build
        template <typename DEVICE, typename SPEC>
        void build_scene_assets(DEVICE& device, OWLContext context, OWLGeomType triangles_geom_type, const rendering::raytracing::Scene& scene, const std::vector<const rendering::raytracing::Object*>& all_objects, rendering::raytracing::backends::SceneState<rendering::raytracing::backends::Optix, SPEC>& assets){
            namespace optix = rendering::raytracing::backends::optix;
            log(device, device.logger, devices::logging::Verbosity::debug, rendering::raytracing::LOG_PREFIX, "building ", scene.objects.size(), " object(s), ", scene.instances.size(), " instance(s), ", (all_objects.size() - scene.objects.size()), " pool object(s) ...");
            // ambient is baked per geometry, so it resolves at build time from the scene the
            // build belongs to (per unique scene in library mode)
            typename SPEC::T environment_ambient[3], environment_miss_0[3], environment_miss_1[3];
            rendering::raytracing::detail::resolve_environment<SPEC>(scene.environment, environment_ambient, environment_miss_0, environment_miss_1);
            (void)environment_miss_0; (void)environment_miss_1;

            std::vector<OWLGeom> geoms;
            std::vector<OWLGroup> object_groups;
            for(size_t object_i = 0; object_i < all_objects.size(); object_i++){
                std::vector<OWLGeom> object_geoms;
                for(const auto& md : all_objects[object_i]->meshes){
            size_t num_vertices = md.vertices.size() / 3;
                size_t num_indices = md.indices.size() / 3;

                OWLBuffer vb = owlDeviceBufferCreate(context, OWL_FLOAT3, num_vertices, md.vertices.data());
                OWLBuffer ib = owlDeviceBufferCreate(context, OWL_INT3, num_indices, md.indices.data());

                OWLGeom geom = owlGeomCreate(context, triangles_geom_type);
                owlTrianglesSetVertices(geom, vb, num_vertices, sizeof(owl::vec3f), 0);
                owlTrianglesSetIndices(geom, ib, num_indices, sizeof(owl::vec3i), 0);
                {
                    using SHADING_USAGE = rendering::raytracing::detail::MediumShadingUsage<SPEC>;
                    if constexpr ((SPEC::HAS_RGB && (SPEC::SHADING::PBR_SHADING || SHADING_USAGE::USES_VERTEX)) || SPEC::HAS_NORMALS) {
                        owlGeomSetBuffer(geom, "vertex", vb);
                    }
                    if constexpr ((SPEC::HAS_RGB && (SPEC::SHADING::PBR_SHADING || SHADING_USAGE::USES_INDEX)) || SPEC::HAS_NORMALS) {
                        owlGeomSetBuffer(geom, "index", ib);
                    }
                }
                if constexpr (SPEC::HAS_RGB) {
                    owlGeomSet3f(geom, "color", owl3f{md.color[0], md.color[1], md.color[2]});
                    owlGeomSet1i(geom, "unlit", md.unlit);

                    if constexpr (SPEC::SHADING::PBR_SHADING || SPEC::SHADING::LOAD_TEXTURES) {
                    if(!md.tex_coords.empty()){
                        size_t num_tc = md.tex_coords.size() / 2;
                        OWLBuffer tcb = owlDeviceBufferCreate(context, OWL_FLOAT2, num_tc, md.tex_coords.data());
                        owlGeomSetBuffer(geom, "tex_coord", tcb);
                    }

                    if(md.texture.present()){
                        OWLTexture tex = owlTexture2DCreate(context,
                                                             OWL_TEXEL_FORMAT_RGBA8,
                                                             md.texture.width, md.texture.height,
                                                             md.texture.pixels->data(),
                                                             OWL_TEXTURE_LINEAR,
                                                             OWL_TEXTURE_WRAP,
                                                             OWL_TEXTURE_WRAP,
                                                             OWL_COLOR_SPACE_SRGB);
                        owlGeomSetTexture(geom, "texture", tex);
                        owlGeomSet1i(geom, "has_texture", 1);
                    } else {
                        owlGeomSet1i(geom, "has_texture", 0);
                    }
                    }

                    if constexpr (SPEC::SHADING::PBR_SHADING || SPEC::SHADING::METALLIC_REFLECTIONS) {
                    owlGeomSet1f(geom, "metallic", md.metallic);
                    }

                    if constexpr (SPEC::SHADING::PBR_SHADING) {
                        if (!md.normals.empty()) {
                            size_t num_normals = md.normals.size() / 3;
                            OWLBuffer nb = owlDeviceBufferCreate(context, OWL_FLOAT3, num_normals, md.normals.data());
                            owlGeomSetBuffer(geom, "normal", nb);
                        }

                        owlGeomSet1f(geom, "roughness", md.roughness);

                    if (md.normal_map.present()) {
                        OWLTexture nm_tex = owlTexture2DCreate(context,
                                                               OWL_TEXEL_FORMAT_RGBA8,
                                                               md.normal_map.width, md.normal_map.height,
                                                               md.normal_map.pixels->data(),
                                                               OWL_TEXTURE_LINEAR,
                                                               OWL_TEXTURE_WRAP,
                                                               OWL_TEXTURE_WRAP,
                                                               OWL_COLOR_SPACE_LINEAR);
                        owlGeomSetTexture(geom, "normal_map", nm_tex);
                        owlGeomSet1i(geom, "has_normal_map", 1);
                    } else {
                        owlGeomSet1i(geom, "has_normal_map", 0);
                    }

                    if (md.metallic_roughness_map.present()) {
                        OWLTexture mr_tex = owlTexture2DCreate(context,
                                                               OWL_TEXEL_FORMAT_RGBA8,
                                                               md.metallic_roughness_map.width, md.metallic_roughness_map.height,
                                                               md.metallic_roughness_map.pixels->data(),
                                                               OWL_TEXTURE_LINEAR,
                                                               OWL_TEXTURE_WRAP,
                                                               OWL_TEXTURE_WRAP,
                                                               OWL_COLOR_SPACE_LINEAR);
                        owlGeomSetTexture(geom, "metallic_roughness_map", mr_tex);
                        owlGeomSet1i(geom, "has_metallic_roughness_map", 1);
                    } else {
                        owlGeomSet1i(geom, "has_metallic_roughness_map", 0);
                    }

                    owlGeomSet3f(geom, "emissive", owl3f{md.emissive[0], md.emissive[1], md.emissive[2]});
                    if (md.emissive_map.present()) {
                        OWLTexture em_tex = owlTexture2DCreate(context,
                                                               OWL_TEXEL_FORMAT_RGBA8,
                                                               md.emissive_map.width, md.emissive_map.height,
                                                               md.emissive_map.pixels->data(),
                                                               OWL_TEXTURE_LINEAR,
                                                               OWL_TEXTURE_WRAP,
                                                               OWL_TEXTURE_WRAP,
                                                               OWL_COLOR_SPACE_SRGB);
                        owlGeomSetTexture(geom, "emissive_map", em_tex);
                        owlGeomSet1i(geom, "has_emissive_map", 1);
                    } else {
                        owlGeomSet1i(geom, "has_emissive_map", 0);
                    }

                    if (md.occlusion_map.present()) {
                        OWLTexture ao_tex = owlTexture2DCreate(context,
                                                               OWL_TEXEL_FORMAT_RGBA8,
                                                               md.occlusion_map.width, md.occlusion_map.height,
                                                               md.occlusion_map.pixels->data(),
                                                               OWL_TEXTURE_LINEAR,
                                                               OWL_TEXTURE_WRAP,
                                                               OWL_TEXTURE_WRAP,
                                                               OWL_COLOR_SPACE_LINEAR);
                        owlGeomSetTexture(geom, "occlusion_map", ao_tex);
                        owlGeomSet1i(geom, "has_occlusion_map", 1);
                    } else {
                        owlGeomSet1i(geom, "has_occlusion_map", 0);
                    }

                    owlGeomSet1f(geom, "opacity", md.opacity);
                    owlGeomSet1i(geom, "alpha_mode", md.alpha_mode);
                    owlGeomSet1f(geom, "alpha_cutoff", md.alpha_cutoff);
                    owlGeomSet3f(geom, "ambient_color", owl3f{(float)environment_ambient[0], (float)environment_ambient[1], (float)environment_ambient[2]});
                    }
                }

                geoms.push_back(geom);
                object_geoms.push_back(geom);
                }
            OWLGroup triangles_group = owlTrianglesGeomGroupCreate(context, object_geoms.size(), object_geoms.data());
                owlGroupBuildAccel(triangles_group);
                object_groups.push_back(triangles_group);
            }

            if constexpr (SPEC::ENABLE_OVERLAYS){
                // degenerate triangle: valid to build, never reported as a hit
                const float filler_vertices[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                const int filler_indices[3] = {0, 1, 2};
                OWLBuffer filler_vertex_buffer = owlDeviceBufferCreate(context, OWL_FLOAT3, 3, filler_vertices);
                OWLBuffer filler_index_buffer = owlDeviceBufferCreate(context, OWL_INT3, 1, filler_indices);
                OWLGeom filler_geom = owlGeomCreate(context, triangles_geom_type);
                owlTrianglesSetVertices(filler_geom, filler_vertex_buffer, 3, sizeof(owl::vec3f), 0);
                owlTrianglesSetIndices(filler_geom, filler_index_buffer, 1, sizeof(owl::vec3i), 0);
                if constexpr (SPEC::HAS_RGB){
                    owlGeomSet3f(filler_geom, "color", owl3f{0.f, 0.f, 0.f});
                }
                geoms.push_back(filler_geom);
                OWLGroup filler_group = owlTrianglesGeomGroupCreate(context, 1, &filler_geom);
                owlGroupBuildAccel(filler_group);
                assets.filler_group = filler_group;
            }

            std::vector<OWLGroup> instance_children;
            std::vector<uint32_t> instance_ids; // user instance ids == global instance ids (contract: segmentation_object, operations_cpu_common.h)
            std::vector<float> instance_transforms; // 12 per instance: owl affine3f (linear columns vx, vy, vz, then translation)
            for(const auto& instance : scene.instances){
                instance_children.push_back(object_groups[instance.object]);
                instance_ids.push_back((uint32_t)instance_ids.size());
                const float* transform = instance.transform; // 3x4 row-major [R|t]
                const float owl_transform[12] = {
                    transform[0], transform[4], transform[8],
                    transform[1], transform[5], transform[9],
                    transform[2], transform[6], transform[10],
                    transform[3], transform[7], transform[11]
                };
                instance_transforms.insert(instance_transforms.end(), owl_transform, owl_transform + 12);
            }
            OWLGroup world = owlInstanceGroupCreate(context, instance_children.size(), instance_children.data(), instance_ids.data(), instance_transforms.data(), OWL_MATRIX_FORMAT_OWL);
            owlGroupBuildAccel(world);

            if constexpr (SPEC::HAS_RGB && (SPEC::SHADING::PBR_SHADING || SPEC::SHADING::METALLIC_REFLECTIONS)) {
                for(size_t m = 0; m < geoms.size(); m++){
                    owlGeomSetGroup(geoms[m], "world", world);
                }
            }

            if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING) {
                const auto scene_lights = rendering::raytracing::detail::effective_scene_lights<true>(device, scene);
                OWLBuffer light_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(rendering::raytracing::SceneLight),
                                                                scene_lights.size(), scene_lights.data());
                for (size_t m = 0; m < geoms.size(); m++) {
                    owlGeomSetBuffer(geoms[m], "scene_lights", light_buffer);
                    owlGeomSet1i(geoms[m], "num_scene_lights", (int)scene_lights.size());
                }
            }

            assets.world = world;
            assets.num_scene_instances = scene.instances.size();
            assets.object_groups.clear();
            for(OWLGroup object_group : object_groups){
                assets.object_groups.push_back(object_group);
            }
        }

        // wires one renderer to a scene build: bounds, overlay state, ray gen vars, launch
        // params, and the context-level programs/pipeline/SBT rebuild
        template <typename DEVICE, typename SPEC>
        void init_renderer_scene(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const rendering::raytracing::Scene& scene, const std::vector<const rendering::raytracing::Object*>& all_objects, const rendering::raytracing::backends::SceneState<rendering::raytracing::backends::Optix, SPEC>& assets){
            namespace optix = rendering::raytracing::backends::optix;
            OWLContext context = (OWLContext)renderer.backend->context;
            OWLGroup world = (OWLGroup)assets.world;
            // per renderer: the overlay rows carry each renderer's own placement classes
            std::vector<unsigned int> host_instance_classes(scene.instances.size() + (SPEC::ENABLE_OVERLAYS ? (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES : 0), 0);
            for(size_t instance_i = 0; instance_i < scene.instances.size(); instance_i++){
                host_instance_classes[instance_i] = all_objects[scene.instances[instance_i].object]->segmentation_class;
            }
            if(host_instance_classes.empty()){
                host_instance_classes.push_back(0);
            }
            if(renderer.backend->instance_classes_buffer != nullptr){
                owlBufferDestroy(renderer.backend->instance_classes_buffer);
            }
            OWLBuffer instance_classes_buffer = owlDeviceBufferCreate(context, OWL_UINT, host_instance_classes.size(), host_instance_classes.data());
            renderer.backend->instance_classes_buffer = instance_classes_buffer;

            if(renderer.backend->overlay_state != nullptr){
                auto* previous_overlay_state = (optix::OverlayState*)renderer.backend->overlay_state;
                optix::overlay_accel_destroy(previous_overlay_state->accel);
                cudaFree(previous_overlay_state->row_indices_device);
                cudaFree(previous_overlay_state->rows_device);
                delete previous_overlay_state;
                renderer.backend->overlay_state = nullptr;
            }
            if constexpr (SPEC::ENABLE_OVERLAYS){
                auto* overlay_state = new optix::OverlayState{};
                for(OWLGroup object_group : assets.object_groups){
                    overlay_state->object_groups.push_back(object_group);
                }
                overlay_state->filler_group = (OWLGroup)assets.filler_group;
                renderer.backend->overlay_state = overlay_state;
            }

            if constexpr (SPEC::HAS_RGB) {
                owlRayGenSetGroup((OWLRayGen)renderer.backend->ray_gen, "world", world);
            }
            if constexpr (SPEC::HAS_FLOW_FORWARD) {
                constexpr unsigned int FLOW_SLOTS = SPEC::ENABLE_OVERLAYS ? (unsigned int)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES : 0u;
                owlRayGenSetGroup((OWLRayGen)renderer.backend->flow_forward_ray_gen, "world", world);
                owlRayGenSet1ui((OWLRayGen)renderer.backend->flow_forward_ray_gen, "source_first_overlay_instance", (unsigned int)scene.instances.size() + FLOW_SLOTS);
            }
            if constexpr (SPEC::HAS_AUXILIARY) {
                owlRayGenSetGroup(renderer.backend->auxiliary_ray_gen, "world", world);
                owlRayGenSet1ui(renderer.backend->auxiliary_ray_gen, "source_first_overlay_instance", (unsigned int)scene.instances.size());
            }
            if(renderer.backend->collision_ray_gen){
                owlRayGenSetGroup((OWLRayGen)renderer.backend->collision_ray_gen, "world", world);
                owlRayGenSet1f((OWLRayGen)renderer.backend->collision_ray_gen, "max_dist", renderer.max_ray_length);
            }
            renderer.backend->world = world;

            // programs/pipeline/SBT must be (re)built after the geometry set changes; the launch
            // params are spec-dependent and created once. In shared-library mode later scene
            // builds only append SBT records, so earlier renderers' baked offsets stay valid.
            owlBuildPrograms(context);
            owlBuildPipeline(context);
            owlBuildSBT(context);

            if constexpr (SPEC::ENABLE_OVERLAYS){
                auto* overlay_state = (optix::OverlayState*)renderer.backend->overlay_state;
                // per-object BLAS traversables and SBT offsets are only final after owlBuildSBT above;
                // baked into a device table the fill kernel joins against slot structure
                std::vector<optix::OverlayObjectEntry> object_entries(overlay_state->object_groups.size());
                for(size_t object_i = 0; object_i < overlay_state->object_groups.size(); object_i++){
                    object_entries[object_i].traversable = (unsigned long long)owlGroupGetTraversable(overlay_state->object_groups[object_i], 0);
                    object_entries[object_i].sbt_offset = owlGroupGetSBTOffset(overlay_state->object_groups[object_i]);
                    object_entries[object_i].segmentation_class = all_objects[object_i]->segmentation_class;
                }
                overlay_state->accel = optix::overlay_accel_create(owlContextGetOptixContext(context, 0), SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, (unsigned int)scene.instances.size(), (unsigned long long)owlGroupGetTraversable(overlay_state->filler_group, 0), SPEC::HAS_FLOW ? 2u : 1u, SPEC::ENABLE_DYNAMIC_MOTION_BLUR);
                optix::overlay_accel_upload_objects(overlay_state->accel, object_entries.data(), (unsigned int)object_entries.size());
                constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
                overlay_state->structure_staging.assign(SLOTS, optix::OverlaySlotStructure{});
                overlay_state->row_indices.assign(SLOTS, 0u);
                overlay_state->rows.assign((size_t)(SPEC::ENABLE_DYNAMIC_MOTION_BLUR ? SPEC::MOTION_BLUR_SAMPLES : 1) * SLOTS * 12, 0.0f);
                cudaMalloc(&overlay_state->row_indices_device, overlay_state->row_indices.size() * sizeof(unsigned int));
                cudaMalloc(&overlay_state->rows_device, overlay_state->rows.size() * sizeof(float));
                // published once: raw builds into fixed per-group buffers keep the handles stable,
                // so no per-rebuild re-publication is needed
                overlay_state->traversables_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(unsigned long long), SPEC::NUM_OVERLAYS, optix::overlay_accel_traversables(overlay_state->accel, 0));
                if constexpr (SPEC::HAS_FLOW) {
                    overlay_state->traversables_previous_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(unsigned long long), SPEC::NUM_OVERLAYS, optix::overlay_accel_traversables(overlay_state->accel, 1));
                }
                overlay_state->attachments_buffer = owlDeviceBufferCreate(context, OWL_UINT, (size_t)SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA, nullptr);
                rendering::raytracing::detail::reset_overlay_state(renderer);
                overlay_state->num_scene_instances = scene.instances.size();
            }

            if(renderer.backend->launch_params == nullptr){
                OWLVarDecl launch_params_vars[] = {
                    { "overlays",      OWL_BUFPTR, OWL_OFFSETOF(OverlayLaunchParams, overlays)},
                    { "overlays_previous", OWL_BUFPTR, OWL_OFFSETOF(OverlayLaunchParams, overlays_previous)},
                    { "attachments",   OWL_BUFPTR, OWL_OFFSETOF(OverlayLaunchParams, attachments)},
                    { "overlay_count", OWL_INT,    OWL_OFFSETOF(OverlayLaunchParams, overlay_count)},
                    { "cam_width",     OWL_INT,    OWL_OFFSETOF(OverlayLaunchParams, cam_width)},
                    { "cam_height",    OWL_INT,    OWL_OFFSETOF(OverlayLaunchParams, cam_height)},
                    { "grid_cols",     OWL_INT,    OWL_OFFSETOF(OverlayLaunchParams, grid_cols)},
                    { "instance_classes", OWL_BUFPTR, OWL_OFFSETOF(OverlayLaunchParams, instance_classes)},
                    { "camera_table", OWL_BUFPTR, OWL_OFFSETOF(OverlayLaunchParams, camera_table)},
                    { "output_mask",   OWL_UINT,   OWL_OFFSETOF(OverlayLaunchParams, output_mask)},
                    { "all_cameras",   OWL_INT,    OWL_OFFSETOF(OverlayLaunchParams, all_cameras)},
                    { /* sentinel */ }
                };
                OWLParams launch_params = owlParamsCreate(context, sizeof(OverlayLaunchParams), launch_params_vars, -1);
                renderer.backend->launch_params = launch_params;
                if(renderer.backend->collision_ray_gen){
                    OWLParams coll_lp = owlParamsCreate(context, sizeof(OverlayLaunchParams), launch_params_vars, -1);
                    renderer.backend->coll_launch_params = coll_lp;
                }
            }
            OWLParams all_launch_params[2] = {(OWLParams)renderer.backend->launch_params, (OWLParams)renderer.backend->coll_launch_params};
            for(OWLParams launch_params : all_launch_params){
                if(launch_params == nullptr) continue;
                owlParamsSet1i(launch_params, "overlay_count", (int)SPEC::MAX_OVERLAYS_PER_CAMERA);
                owlParamsSet1i(launch_params, "cam_width", (int)SPEC::CAM_WIDTH);
                owlParamsSet1i(launch_params, "cam_height", (int)SPEC::CAM_HEIGHT);
                owlParamsSet1i(launch_params, "grid_cols", (int)SPEC::GRID_COLS);
                owlParamsSetBuffer(launch_params, "instance_classes", instance_classes_buffer);
                owlParamsSetBuffer(launch_params, "camera_table", renderer.backend->camera_table_buffer);
                owlParamsSet1ui(launch_params, "output_mask", SPEC::PLAN::MASK);
                owlParamsSet1i(launch_params, "all_cameras", rendering::raytracing::outputs::Layout<SPEC>::ALL_CAMERAS ? 1 : 0);
                if constexpr (SPEC::ENABLE_OVERLAYS){
                    auto* overlay_state = (optix::OverlayState*)renderer.backend->overlay_state;
                    owlParamsSetBuffer(launch_params, "overlays", overlay_state->traversables_buffer);
                    owlParamsSetBuffer(launch_params, "attachments", overlay_state->attachments_buffer);
                    if constexpr (SPEC::HAS_FLOW){
                        owlParamsSetBuffer(launch_params, "overlays_previous", overlay_state->traversables_previous_buffer);
                    }
                }
            }

            if constexpr (SPEC::ENABLE_OVERLAYS){
                update(device, renderer); // publish the (empty) overlays and the attachment table
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename METADATA_T>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool, const rendering::SceneMetadata<METADATA_T>& metadata){
        namespace optix = rendering::raytracing::backends::optix;
        validate_semantic_segmentation<SPEC>(device, scene, pool);
        renderer.max_ray_length = (typename SPEC::T)metadata.max_ray_length;
        rendering::raytracing::detail::announce_configuration<SPEC>(device);
        rendering::raytracing::detail::resolve_environment<SPEC>(scene.environment, renderer.ambient_color, renderer.miss_color_0, renderer.miss_color_1);
        owlMissProgSet3f((OWLMissProg)renderer.backend->miss_prog, "color_0", owl3f{(float)renderer.miss_color_0[0], (float)renderer.miss_color_0[1], (float)renderer.miss_color_0[2]});
        owlMissProgSet3f((OWLMissProg)renderer.backend->miss_prog, "color_1", owl3f{(float)renderer.miss_color_1[0], (float)renderer.miss_color_1[1], (float)renderer.miss_color_1[2]});
        OWLContext context = (OWLContext)renderer.backend->context;
        OWLModule module = (OWLModule)renderer.backend->module;
        OWLGeomType triangles_geom_type = optix::detail::create_geom_type<SPEC>(context, module);

        std::vector<const rendering::raytracing::Object*> all_objects;
        for(const auto& object : scene.objects){
            all_objects.push_back(&object);
        }
        if constexpr (SPEC::ENABLE_OVERLAYS){
            rendering::raytracing::detail::register_pool_assets(device, renderer, pool, all_objects);
        }

        rendering::raytracing::backends::SceneState<rendering::raytracing::backends::Optix, SPEC> assets;
        optix::detail::build_scene_assets<DEVICE, SPEC>(device, context, triangles_geom_type, scene, all_objects, assets);
        optix::detail::init_renderer_scene(device, renderer, scene, all_objects, assets);
    }

    // shared-library mode: one geometry/texture/BLAS build per unique scene (content-hash
    // dedup inside the library), then init(device, renderer, library, scene_id) wires a
    // renderer to the shared build. Returns the unique-scene index so callers can key their
    // own per-scene data.
    template <typename DEVICE, typename SPEC, typename T>
    typename SPEC::TI insert(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Optix>& library, rendering::Bundle<T>& bundle){
        namespace optix = rendering::raytracing::backends::optix;
        using TI = typename SPEC::TI;
        bool is_new = false;
        const TI scene_id = rendering::raytracing::detail::library_insert(device, library, bundle, is_new);
        if(is_new){
            const rendering::raytracing::Scene& scene = library.scenes[scene_id];
            if(library.backend->geom_type == nullptr){
                library.backend->geom_type = optix::detail::create_geom_type<SPEC>(library.backend->context, library.backend->module);
            }
            std::vector<const rendering::raytracing::Object*> build_objects;
            optix::detail::collect_all_objects(scene, library.pool, build_objects);
            library.assets[scene_id] = new rendering::raytracing::backends::SceneState<rendering::raytracing::backends::Optix, SPEC>{};
            optix::detail::build_scene_assets<DEVICE, SPEC>(device, library.backend->context, library.backend->geom_type, scene, build_objects, *library.assets[scene_id]);
        }
        return scene_id;
    }

    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Optix>& library, typename SPEC::TI scene_id){
        namespace optix = rendering::raytracing::backends::optix;
        utils::assert_exit(device, renderer.backend->library == &library, "init: the renderer was not malloc'd against this library");
        utils::assert_exit(device, scene_id < (typename SPEC::TI)library.scenes.size(), "init: scene_id out of range");
        const rendering::raytracing::Scene& scene = library.scenes[scene_id];
        validate_semantic_segmentation<SPEC>(device, scene, library.pool);
        renderer.max_ray_length = library.metadata[scene_id].max_ray_length;
        rendering::raytracing::detail::announce_configuration<SPEC>(device);
        rendering::raytracing::detail::resolve_environment<SPEC>(scene.environment, renderer.ambient_color, renderer.miss_color_0, renderer.miss_color_1);
        owlMissProgSet3f((OWLMissProg)renderer.backend->miss_prog, "color_0", owl3f{(float)renderer.miss_color_0[0], (float)renderer.miss_color_0[1], (float)renderer.miss_color_0[2]});
        owlMissProgSet3f((OWLMissProg)renderer.backend->miss_prog, "color_1", owl3f{(float)renderer.miss_color_1[0], (float)renderer.miss_color_1[1], (float)renderer.miss_color_1[2]});

        std::vector<const rendering::raytracing::Object*> all_objects;
        for(const auto& object : scene.objects){
            all_objects.push_back(&object);
        }
        if constexpr (SPEC::ENABLE_OVERLAYS){
            rendering::raytracing::detail::register_pool_assets(device, renderer, library.pool, all_objects);
        }
        else{
            optix::detail::collect_all_objects(scene, library.pool, all_objects);
        }
        optix::detail::init_renderer_scene(device, renderer, scene, all_objects, *library.assets[scene_id]);
    }


    namespace rendering::raytracing::backends::optix{
        // the launch verbs are stream-ordered on a CUDA producer device's stream: the backend stream
        // waits for the work enqueued there so far (fork) and that stream waits for the launch
        // (join), so callers see one stream; a host producer has no stream and writes synchronously
        template <typename DEVICE>
        auto fork(DEVICE& device, cudaEvent_t event, cudaStream_t backend_stream, int) -> decltype(device.stream, void()){
            devices::cuda::order_after(event, device.stream, backend_stream);
        }
        template <typename DEVICE>
        void fork(DEVICE& device, cudaEvent_t event, cudaStream_t backend_stream, long){}
        template <typename DEVICE>
        auto join(DEVICE& device, cudaEvent_t event, cudaStream_t backend_stream, int) -> decltype(device.stream, void()){
            devices::cuda::order_after(event, backend_stream, device.stream);
        }
        template <typename DEVICE>
        void join(DEVICE& device, cudaEvent_t event, cudaStream_t backend_stream, long){}

        // publishes the pending rows of one host staging mirror into its resident tensor: the
        // rows compact into the pageable index/row lists and one scatter kernel writes them —
        // two copies and one kernel per tensor and update, however sparse the host writes were.
        // num_slabs > 1 publishes the sample-major motion mirror (slabs SLOTS rows apart).
        template <typename SPEC>
        void publish_rows(OverlayState& overlay_state, bool* pending, const float* staging, float* resident, unsigned int num_slabs, cudaStream_t stream){
            constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
            unsigned int count = 0;
            for(size_t slot = 0; slot < SLOTS; slot++){
                if(pending[slot]){
                    overlay_state.row_indices[count++] = (unsigned int)slot;
                    pending[slot] = false;
                }
            }
            if(count == 0){
                return;
            }
            for(unsigned int slab = 0; slab < num_slabs; slab++){
                for(unsigned int row = 0; row < count; row++){
                    std::memcpy(overlay_state.rows.data() + ((size_t)slab * count + row) * 12, staging + ((size_t)slab * SLOTS + overlay_state.row_indices[row]) * 12, 12 * sizeof(float));
                }
            }
            cudaMemcpyAsync(overlay_state.row_indices_device, overlay_state.row_indices.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(overlay_state.rows_device, overlay_state.rows.data(), (size_t)num_slabs * count * 12 * sizeof(float), cudaMemcpyHostToDevice, stream);
            overlay_accel_scatter_rows(overlay_state.row_indices_device, count, overlay_state.rows_device, resident, num_slabs, (unsigned int)SLOTS, stream);
        }
    }

    template <typename DEVICE, typename SPEC>
    cudaStream_t stream(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        return (cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0);
    }

    template <typename DEVICE, typename SPEC>
    cudaStream_t collision_stream(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        return renderer.backend->coll_launch_params != nullptr ? (cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->coll_launch_params, 0) : stream(device, renderer);
    }

    // fully stream-ordered on the render stream: the pending host rows publish by scatter, then
    // the device-side instance fill and per-group raw optixAccelBuild — no host synchronization
    template <typename DEVICE, typename SPEC>
    void update_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
        namespace optix = rendering::raytracing::backends::optix;
        using TI = typename SPEC::TI;
        auto* overlay_state = (optix::OverlayState*)renderer.backend->overlay_state;
        cudaStream_t cuda_stream = (cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0);
        cudaStream_t coll_stream = collision_stream(device, renderer);
        optix::fork(device, renderer.backend->ordering_event, cuda_stream, 0);
        // probe launches trace the overlay TLASes on their own stream: order this rebuild after
        // in-flight probes and subsequent probes after this rebuild, without host syncs
        devices::cuda::order_after(renderer.backend->ordering_event, coll_stream, cuda_stream);

        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            auto& overlay_host = renderer.overlays[overlay];
            if(!overlay_host.dirty){
                continue;
            }
            const size_t base = (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES;
            for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                auto& row = overlay_state->structure_staging[base + slot];
                row.active = overlay_host.slots[slot].active ? 1u : 0u;
                row.object = (unsigned int)overlay_host.slots[slot].object;
                row.segmentation_class = overlay_host.slots[slot].segmentation_class;
            }
            cudaMemcpyAsync(optix::overlay_accel_structure(overlay_state->accel) + base, &overlay_state->structure_staging[base], SPEC::MAX_OVERLAY_INSTANCES * sizeof(optix::OverlaySlotStructure), cudaMemcpyHostToDevice, cuda_stream);
            overlay_host.dirty = false;
        }
        // producer writes stay in place: the scatter touches only the rows the host marked, and
        // the accel build below consumes the tensors unconditionally (overlay-producer contract tests)
        optix::publish_rows<SPEC>(*overlay_state, renderer.transforms_pending, renderer.transforms_staging.data(), data(renderer.transforms), 1u, cuda_stream);
        if constexpr (SPEC::HAS_FLOW_OVERLAYS){
            optix::publish_rows<SPEC>(*overlay_state, renderer.transforms_previous_pending, renderer.transforms_previous_staging.data(), data(renderer.transforms_previous), 1u, cuda_stream);
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            optix::publish_rows<SPEC>(*overlay_state, renderer.transforms_motion_pending, renderer.transforms_motion_staging.data(), data(renderer.transforms_motion), (unsigned int)SPEC::MOTION_BLUR_SAMPLES, cuda_stream);
        }
        if(renderer.attachments_dirty){
            std::vector<uint32_t> attachments((size_t)SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA);
            for(size_t index = 0; index < attachments.size(); index++){
                attachments[index] = (uint32_t)renderer.attachments[index];
            }
            owlBufferUpload(overlay_state->attachments_buffer, attachments.data(), 0, attachments.size());
            renderer.attachments_dirty = false;
        }
        // only semantic segmentation reads the overlay classes
        unsigned int* instance_classes = SPEC::HAS_SEMANTIC_SEGMENTATION ? (unsigned int*)owlBufferGetPointer(renderer.backend->instance_classes_buffer, 0) : nullptr;
        optix::overlay_accel_build(overlay_state->accel, data(renderer.transforms), instance_classes, 1.0f, nullptr, cuda_stream, 0u);
        if constexpr (SPEC::HAS_FLOW){
            // the previous instant's TLASes (shifted id range, no segmentation classes) and the
            // per-slot deltas between the instants, both from the tensors, both stream-ordered
            optix::overlay_accel_build(overlay_state->accel, data(renderer.transforms_previous), nullptr, 1.0f, nullptr, cuda_stream, 1u);
            optix::overlay_accel_compose_flow_deltas(optix::overlay_accel_structure(overlay_state->accel), data(renderer.transforms), data(renderer.transforms_previous), (float*)owlBufferGetPointer(renderer.backend->flow_deltas_buffer, 0),
                                                     (unsigned int)SPEC::NUM_OVERLAYS, (unsigned int)SPEC::MAX_OVERLAY_INSTANCES, cuda_stream);
        }
        devices::cuda::order_after(renderer.backend->ordering_event, cuda_stream, coll_stream);
        optix::join(device, renderer.backend->ordering_event, cuda_stream, 0);
    }

    template <typename DEVICE, typename SPEC>
    void update_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
        cudaStreamSynchronize((cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0));
    }

    // the previous instant becomes the current one, entirely on the render stream: the camera
    // and transform tensors copy device-to-device, rows the host has not yet published reach
    // transforms_previous through their staging mirror at the next update
    template <typename DEVICE, typename SPEC>
    void advance_flow_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        static_assert(SPEC::HAS_FLOW, "advance_flow requires a flow-capable renderer specification");
        namespace optix = rendering::raytracing::backends::optix;
        cudaStream_t cuda_stream = (cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0);
        optix::fork(device, renderer.backend->ordering_event, cuda_stream, 0);
        cudaMemcpyAsync(data(renderer.cameras_previous), data(renderer.cameras), decltype(renderer.cameras)::SPEC::SIZE_BYTES, cudaMemcpyDeviceToDevice, cuda_stream);
        if constexpr (SPEC::HAS_FLOW_OVERLAYS){
            rendering::raytracing::detail::advance_flow_staging(renderer);
            cudaMemcpyAsync(data(renderer.transforms_previous), data(renderer.transforms), decltype(renderer.transforms)::SPEC::SIZE_BYTES, cudaMemcpyDeviceToDevice, cuda_stream);
        }
        optix::join(device, renderer.backend->ordering_event, cuda_stream, 0);
    }

    template <typename DEVICE, typename SPEC>
    void advance_flow_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        static_assert(SPEC::HAS_FLOW, "advance_flow requires a flow-capable renderer specification");
        cudaStreamSynchronize((cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0));
    }

    template <typename DEVICE, typename SPEC>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        update_launch(device, renderer);
        update_sync(device, renderer);
    }


    namespace rendering::raytracing::backends::optix {
        inline void synchronize(rendering::raytracing::backends::Device<rendering::raytracing::backends::Optix>& device){
            if(device.state->launch_params != nullptr){
                cudaStreamSynchronize((cudaStream_t)owlParamsGetCudaStream((OWLParams)device.state->launch_params, 0));
            }
            if(device.state->coll_launch_params != nullptr){
                cudaStreamSynchronize((cudaStream_t)owlParamsGetCudaStream((OWLParams)device.state->coll_launch_params, 0));
            }
        }
    }

    // renderer memory-domain copies: ordered after the renderer's in-flight work on both
    // renderer-owned streams, so no separate synchronize() is needed at readback/upload boundaries
    template <typename TO_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(rendering::raytracing::backends::Device<rendering::raytracing::backends::Optix>& from_device, TO_DEVICE& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
        static_assert(tensor::same_dimensions_shape<typename FROM_SPEC::SHAPE, typename TO_SPEC::SHAPE>());
        static_assert(tensor::same_dimensions_shape<typename FROM_SPEC::STRIDE, typename TO_SPEC::STRIDE>() && tensor::dense_row_major_layout<FROM_SPEC>(), "renderer copies require matching dense row-major layouts");
        static_assert(utils::typing::is_same_v<typename FROM_SPEC::T, typename TO_SPEC::T>);
        rendering::raytracing::backends::optix::synchronize(from_device);
        cudaMemcpy(to._data, from._data, FROM_SPEC::SIZE_BYTES, cudaMemcpyDeviceToHost);
    }
    template <typename FROM_DEVICE, typename FROM_SPEC, typename TO_SPEC>
    void copy(FROM_DEVICE& from_device, rendering::raytracing::backends::Device<rendering::raytracing::backends::Optix>& to_device, const Tensor<FROM_SPEC>& from, Tensor<TO_SPEC>& to){
        static_assert(tensor::same_dimensions_shape<typename FROM_SPEC::SHAPE, typename TO_SPEC::SHAPE>());
        static_assert(tensor::same_dimensions_shape<typename FROM_SPEC::STRIDE, typename TO_SPEC::STRIDE>() && tensor::dense_row_major_layout<FROM_SPEC>(), "renderer copies require matching dense row-major layouts");
        static_assert(utils::typing::is_same_v<typename FROM_SPEC::T, typename TO_SPEC::T>);
        rendering::raytracing::backends::optix::synchronize(to_device);
        cudaMemcpy(to._data, from._data, FROM_SPEC::SIZE_BYTES, cudaMemcpyHostToDevice);
    }

    // =========================================================================
    // generate_probe_directions: Fibonacci probe directions + upload
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void generate_probe_directions(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        if constexpr(SPEC::HAS_PROBES){
            std::vector<float> dirs = rendering::raytracing::detail::generate_probe_direction_vectors<SPEC>(device);
            owlBufferUpload((OWLBuffer)renderer.backend->probe_dirs_buffer, dirs.data(), 0, SPEC::NUM_PROBES);
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        namespace optix = rendering::raytracing::backends::optix;
        using TI = typename SPEC::TI;
        OWLParams launch_params = (OWLParams)renderer.backend->launch_params;
        cudaStream_t cuda_stream = (cudaStream_t)owlParamsGetCudaStream(launch_params, 0);
        optix::fork(device, renderer.backend->ordering_event, cuda_stream, 0);
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            // The first sample rebuilds to consume device-written slot structures; subsequent samples only change transforms.
            auto* overlay_state = (optix::OverlayState*)renderer.backend->overlay_state;
            unsigned int* instance_classes = SPEC::HAS_SEMANTIC_SEGMENTATION ? (unsigned int*)owlBufferGetPointer(renderer.backend->instance_classes_buffer, 0) : nullptr;
            cudaStream_t coll_stream = collision_stream(device, renderer);
            devices::cuda::order_after(renderer.backend->ordering_event, coll_stream, cuda_stream);
            if constexpr (SPEC::HAS_RGB){
                cudaMemsetAsync(data(renderer.rgb_accumulator), 0, decltype(renderer.rgb_accumulator)::SPEC::SIZE_BYTES, cuda_stream);
            }
            constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
            for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                const float shutter_t = ((float)sample + 0.5f) / (float)SPEC::MOTION_BLUR_SAMPLES;
                if(sample == 0){
                    optix::overlay_accel_build(overlay_state->accel, data(renderer.transforms_motion) + sample * SLAB, instance_classes, shutter_t, renderer.backend->shutter_device, cuda_stream, 0u);
                }
                else{
                    optix::overlay_accel_refit(overlay_state->accel, data(renderer.transforms_motion) + sample * SLAB, instance_classes, shutter_t, renderer.backend->shutter_device, cuda_stream, 0u);
                }
                if constexpr (SPEC::HAS_RGB){
                    owlAsyncLaunch2D((OWLRayGen)renderer.backend->ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
                }
            }
            optix::overlay_accel_refit(overlay_state->accel, data(renderer.transforms), instance_classes, 1.0f, nullptr, cuda_stream, 0u);
            devices::cuda::order_after(renderer.backend->ordering_event, cuda_stream, coll_stream);
            if constexpr (SPEC::HAS_AUXILIARY) {
                owlAsyncLaunch2D(renderer.backend->auxiliary_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
            }
            if constexpr (SPEC::HAS_FLOW_FORWARD){
                owlAsyncLaunch2D((OWLRayGen)renderer.backend->flow_forward_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
            }
            const float* rgb_accumulation = nullptr;
            unsigned int* frame_buffer = nullptr;
            float* observation = nullptr;
            if constexpr (SPEC::HAS_RGB){
                rgb_accumulation = data(renderer.rgb_accumulator);
            }
            if constexpr (SPEC::HAS_RGBA8){
                frame_buffer = data(output(device, renderer, rendering::raytracing::outputs::RGB<>{}));
            }
            if constexpr (SPEC::HAS_RGB32F){
                observation = data(output(device, renderer, rendering::raytracing::outputs::RGB<rendering::raytracing::outputs::format::Float32>{}));
            }
            optix::overlay_accel_resolve(rgb_accumulation, frame_buffer, observation, (const unsigned int*)owlBufferGetPointer(renderer.backend->camera_table_buffer, 0), SPEC::SHADING::SRGB_OUTPUT ? 1 : 0,
                                         (unsigned int)SPEC::NUM_CAMERAS, (unsigned int)SPEC::CAM_PIXELS,
                                         (unsigned int)SPEC::MOTION_BLUR_SAMPLES, cuda_stream);
            optix::join(device, renderer.backend->ordering_event, cuda_stream, 0);
            return;
        }
        if constexpr (SPEC::HAS_RGB) {
            OWLRayGen ray_gen = (OWLRayGen)renderer.backend->ray_gen;
            owlAsyncLaunch2D(ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
        }
        if constexpr (SPEC::HAS_AUXILIARY) {
            owlAsyncLaunch2D(renderer.backend->auxiliary_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
        }
        if constexpr (SPEC::HAS_FLOW_FORWARD) {
            owlAsyncLaunch2D((OWLRayGen)renderer.backend->flow_forward_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
        }
        optix::join(device, renderer.backend->ordering_event, cuda_stream, 0);
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        owlLaunchSync((OWLParams)renderer.backend->launch_params);
    }

    // render produces the image outputs the spec declares; the collision-probe pass is the
    // separate probe verb so it can be scheduled independently (e.g. alongside update)
    template <typename DEVICE, typename SPEC>
    void probe_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        namespace optix = rendering::raytracing::backends::optix;
        if(renderer.backend->collision_ray_gen){
            OWLRayGen collision_ray_gen = (OWLRayGen)renderer.backend->collision_ray_gen;
            OWLParams coll_lp = (OWLParams)renderer.backend->coll_launch_params;
            cudaStream_t coll_stream = (cudaStream_t)owlParamsGetCudaStream(coll_lp, 0);
            optix::fork(device, renderer.backend->ordering_event, coll_stream, 0);
            owlAsyncLaunch2D(collision_ray_gen, SPEC::NUM_CAMERAS, SPEC::NUM_PROBES, coll_lp);
            optix::join(device, renderer.backend->ordering_event, coll_stream, 0);
        }
    }

    template <typename DEVICE, typename SPEC>
    void probe_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        if(renderer.backend->coll_launch_params)
            owlLaunchSync((OWLParams)renderer.backend->coll_launch_params);
    }

    // the device rows {active, object} every overlay build reads: a producer may rewrite the object
    // of an active row, and update() overwrites a row from host truth only when the host marked
    // its overlay dirty
    template <typename DEVICE, typename SPEC>
    rendering::raytracing::backends::optix::OverlaySlotStructure* overlay_structure(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "overlay_structure requires an overlay-enabled renderer specification");
        auto* overlay_state = (rendering::raytracing::backends::optix::OverlayState*)renderer.backend->overlay_state;
        return rendering::raytracing::backends::optix::overlay_accel_structure(overlay_state->accel);
    }

    template <typename DEVICE, typename SPEC>
    std::string device_name(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        cudaPointerAttributes attributes{};
        utils::assert_exit(device, cudaPointerGetAttributes(&attributes, data(renderer.cameras)) == cudaSuccess, "OptiX: cannot identify renderer device");
        cudaDeviceProp properties{};
        utils::assert_exit(device, cudaGetDeviceProperties(&properties, attributes.device) == cudaSuccess, "OptiX: cannot query renderer device");
        return properties.name;
    }

    // scoped to the renderer's own streams — never a whole-device barrier
    template <typename DEVICE, typename SPEC>
    void synchronize(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        rendering::raytracing::backends::optix::synchronize(renderer.device);
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        log(device, device.logger, devices::logging::Verbosity::debug, rendering::raytracing::LOG_PREFIX, "destroying devicegroups ...");
        if(renderer.backend != nullptr){
            if(renderer.backend->instance_classes_buffer != nullptr){
                owlBufferDestroy(renderer.backend->instance_classes_buffer);
                renderer.backend->instance_classes_buffer = nullptr;
            }
            if(renderer.backend->context != nullptr && renderer.backend->library == nullptr){
                owlContextDestroy((OWLContext)renderer.backend->context); // library-backed renderers borrow the context — the library destroys it
            }
            if(renderer.backend->overlay_state != nullptr){
                auto* overlay_state = (rendering::raytracing::backends::optix::OverlayState*)renderer.backend->overlay_state;
                rendering::raytracing::backends::optix::overlay_accel_destroy(overlay_state->accel);
                cudaFree(overlay_state->row_indices_device);
                cudaFree(overlay_state->rows_device);
                delete overlay_state;
            }
            if(renderer.backend->shutter_device != nullptr){
                cudaFree(renderer.backend->shutter_device);
            }
            if(renderer.backend->ordering_event != nullptr){
                cudaEventDestroy(renderer.backend->ordering_event);
            }
            delete renderer.backend;
            renderer.backend = nullptr;
            renderer.device.state = nullptr;
        }
        if constexpr (SPEC::ENABLE_OVERLAYS){
            if(renderer.transforms._data != nullptr){
                cudaFree(renderer.transforms._data);
                renderer.transforms._data = nullptr;
            }
        }
        if constexpr (SPEC::HAS_FLOW_OVERLAYS){
            if(renderer.transforms_previous._data != nullptr){
                cudaFree(renderer.transforms_previous._data);
                renderer.transforms_previous._data = nullptr;
            }
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            if(renderer.transforms_motion._data != nullptr){
                cudaFree(renderer.transforms_motion._data);
                renderer.transforms_motion._data = nullptr;
            }
            if constexpr (SPEC::HAS_RGB){
                renderer.rgb_accumulator._data = nullptr;
            }
        }
        // the input and output tensors alias OWL buffers destroyed with the context
        renderer.cameras._data = nullptr;
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            renderer.cameras_open._data = nullptr;
        }
        for_each_output(device, renderer.outputs, [&](auto& buffer){
            buffer.tensor._data = nullptr;
        });
        if constexpr(SPEC::HAS_FLOW){
            renderer.cameras_previous._data = nullptr;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#include "../../operations_cpu_post.h"

#endif
