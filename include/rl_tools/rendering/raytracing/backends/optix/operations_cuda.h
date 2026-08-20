#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_OPERATIONS_CUDA_H

#include "../../renderer.h"
#include "../../operations_cpu_common.h"
#include "device.h"
#include "overlay_accel.h"

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
    extern "C" char device_depth_ptx[];
    extern "C" char device_segmentation_ptx[];
    extern "C" char device_depth_segmentation_ptx[];
    extern "C" char device_normals_ptx[];
    extern "C" char device_depth_normals_ptx[];
    extern "C" char device_segmentation_normals_ptx[];
    extern "C" char device_depth_segmentation_normals_ptx[];

    namespace rendering::raytracing::backends::optix {
        // inactive overlay slots point at a shared degenerate-triangle BLAS, so every overlay
        // TLAS keeps a fixed instance count and instance-id layout across rebuilds
        struct OverlayState {
            std::vector<OWLGroup> object_groups;
            OWLGroup filler_group = nullptr;
            OverlayAccelState* accel = nullptr;
            OWLBuffer traversables_buffer = nullptr;
            OWLBuffer attachments_buffer = nullptr;
            OWLBuffer instance_classes_buffer = nullptr;
            // pageable staging for host-verb writes: cudaMemcpyAsync returns only after pageable
            // sources are consumed, so rows can be rewritten on the next update without hazards
            std::vector<OverlaySlotStructure> structure_staging;
            std::vector<float> transforms_staging;
            std::vector<float> flow_deltas_staging;
            size_t num_scene_instances = 0;
        };

        struct State {
            OWLContext context = nullptr;
            OWLModule module = nullptr;
            OWLBuffer cameras_buffer = nullptr;
            OWLBuffer cameras_open_buffer = nullptr;
            OWLGroup world = nullptr;
            OWLParams launch_params = nullptr;
            OWLRayGen collision_ray_gen = nullptr;
            OWLBuffer collision_results_buffer = nullptr;
            OWLBuffer probe_dirs_buffer = nullptr;
            OWLParams coll_launch_params = nullptr;
            OverlayState* overlay_state = nullptr;
            OWLRayGen ray_gen = nullptr;
            OWLBuffer frame_buffer = nullptr;
            OWLRayGen depth_ray_gen = nullptr;
            OWLBuffer depth_buffer = nullptr;
            OWLRayGen segmentation_ray_gen = nullptr;
            OWLBuffer segmentation_buffer = nullptr;
            OWLRayGen normals_ray_gen = nullptr;
            OWLBuffer normals_buffer = nullptr;
            OWLRayGen flow_ray_gen = nullptr;
            OWLBuffer flow_buffer = nullptr;
            OWLBuffer flow_deltas_buffer = nullptr;
            OWLBuffer observation_buffer = nullptr;
            OWLBuffer rgb_accumulator_buffer = nullptr;
            OWLBuffer depth_accumulator_buffer = nullptr;
            float* shutter_device = nullptr; // per-pass shutter time, written by the overlay fill kernel
        };
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
            OWLGeomType geom_type = nullptr;
        };

        template <typename SPEC>
        struct SceneState<rendering::raytracing::backends::Optix, SPEC> {
            OWLGroup world = nullptr;
            OWLBuffer instance_classes_buffer = nullptr;
            OWLGroup filler_group = nullptr;
            std::vector<OWLGroup> object_groups;
            size_t num_scene_instances = 0;
        };
    }

    namespace rendering::raytracing::detail {
        template <bool T_DEPTH, typename SPEC>
        const char* ray_gen_program_name(const char* depth_name, const char* srgb_name, const char* linear_name) {
            if constexpr (T_DEPTH) {
                return depth_name;
            }
            else if constexpr (SPEC::SHADING::SRGB_OUTPUT) {
                return srgb_name;
            }
            else {
                return linear_name;
            }
        }

        template <bool T_DEPTH, typename SPEC>
        const char* ray_gen_program_name() {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                // the motion loop runs at launch level (one pass per sample), so only the AA grid
                // is compiled in; accumulation is linear, srgb/linear resolve happens later
                if constexpr (SPEC::ENABLE_ANTI_ALIASING) {
                    if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                        return ray_gen_program_name<T_DEPTH, SPEC>("depthAccumRayGenAA2", "accumRayGenAA2", "accumRayGenAA2");
                    }
                    else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                        return ray_gen_program_name<T_DEPTH, SPEC>("depthAccumRayGenAA3", "accumRayGenAA3", "accumRayGenAA3");
                    }
                    else {
                        return ray_gen_program_name<T_DEPTH, SPEC>("depthAccumRayGenAA4", "accumRayGenAA4", "accumRayGenAA4");
                    }
                }
                else {
                    return ray_gen_program_name<T_DEPTH, SPEC>("depthAccumRayGen", "accumRayGen", "accumRayGen");
                }
            }
            else if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                if constexpr (SPEC::ENABLE_ANTI_ALIASING) {
                    if constexpr (SPEC::MOTION_BLUR_SAMPLES == 2) {
                        if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur2AA2", "simpleRayGenMotionBlur2AA2", "linearRayGenMotionBlur2AA2");
                        }
                        else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur2AA3", "simpleRayGenMotionBlur2AA3", "linearRayGenMotionBlur2AA3");
                        }
                        else {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur2AA4", "simpleRayGenMotionBlur2AA4", "linearRayGenMotionBlur2AA4");
                        }
                    }
                    else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 4) {
                        if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur4AA2", "simpleRayGenMotionBlur4AA2", "linearRayGenMotionBlur4AA2");
                        }
                        else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur4AA3", "simpleRayGenMotionBlur4AA3", "linearRayGenMotionBlur4AA3");
                        }
                        else {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur4AA4", "simpleRayGenMotionBlur4AA4", "linearRayGenMotionBlur4AA4");
                        }
                    }
                    else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 8) {
                        if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur8AA2", "simpleRayGenMotionBlur8AA2", "linearRayGenMotionBlur8AA2");
                        }
                        else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur8AA3", "simpleRayGenMotionBlur8AA3", "linearRayGenMotionBlur8AA3");
                        }
                        else {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur8AA4", "simpleRayGenMotionBlur8AA4", "linearRayGenMotionBlur8AA4");
                        }
                    }
                    else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 16) {
                        if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur16AA2", "simpleRayGenMotionBlur16AA2", "linearRayGenMotionBlur16AA2");
                        }
                        else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur16AA3", "simpleRayGenMotionBlur16AA3", "linearRayGenMotionBlur16AA3");
                        }
                        else {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur16AA4", "simpleRayGenMotionBlur16AA4", "linearRayGenMotionBlur16AA4");
                        }
                    }
                    else {
                        if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur32AA2", "simpleRayGenMotionBlur32AA2", "linearRayGenMotionBlur32AA2");
                        }
                        else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur32AA3", "simpleRayGenMotionBlur32AA3", "linearRayGenMotionBlur32AA3");
                        }
                        else {
                            return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur32AA4", "simpleRayGenMotionBlur32AA4", "linearRayGenMotionBlur32AA4");
                        }
                    }
                }
                else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 2) {
                    return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur2", "simpleRayGenMotionBlur2", "linearRayGenMotionBlur2");
                }
                else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 4) {
                    return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur4", "simpleRayGenMotionBlur4", "linearRayGenMotionBlur4");
                }
                else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 8) {
                    return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur8", "simpleRayGenMotionBlur8", "linearRayGenMotionBlur8");
                }
                else if constexpr (SPEC::MOTION_BLUR_SAMPLES == 16) {
                    return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur16", "simpleRayGenMotionBlur16", "linearRayGenMotionBlur16");
                }
                else {
                    return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenMotionBlur32", "simpleRayGenMotionBlur32", "linearRayGenMotionBlur32");
                }
            }
            else if constexpr (SPEC::ENABLE_ANTI_ALIASING) {
                if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 2) {
                    return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenAA2", "simpleRayGenAA2", "linearRayGenAA2");
                }
                else if constexpr (SPEC::ANTI_ALIASING_GRID_SIZE == 3) {
                    return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenAA3", "simpleRayGenAA3", "linearRayGenAA3");
                }
                else {
                    return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGenAA4", "simpleRayGenAA4", "linearRayGenAA4");
                }
            }
            else {
                return ray_gen_program_name<T_DEPTH, SPEC>("depthRayGen", "simpleRayGen", "linearRayGen");
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
        // in standalone mode, once per AssetLibrary in shared mode
        template <typename SPEC>
        void create_context_resources(OWLContext& context_out, OWLModule& module_out){
            OWLContext context = owlContextCreate(nullptr, 1);
            // ray type 2 (normals) only exists in the normals-enabled PTX variants; the count
            // must match the NUM_RAY_TYPES the selected PTX was compiled with (SBT stride)
            owlContextSetRayTypeCount(context, SPEC::HAS_NORMALS ? 3 : 2);
            owlContextSetNumPayloadValues(context, 4); // color ptr (2), recursion depth, hit distance
            const char* ptx = nullptr;
            if constexpr (SPEC::HAS_NORMALS) {
                if constexpr (SPEC::HAS_SEGMENTATION && SPEC::HAS_DEPTH) {
                    ptx = device_depth_segmentation_normals_ptx;
                }
                else if constexpr (SPEC::HAS_SEGMENTATION) {
                    ptx = device_segmentation_normals_ptx;
                }
                else if constexpr (SPEC::HAS_DEPTH) {
                    ptx = device_depth_normals_ptx;
                }
                else {
                    ptx = device_normals_ptx;
                }
            }
            else if constexpr (SPEC::HAS_SEGMENTATION && SPEC::HAS_DEPTH) {
                ptx = device_depth_segmentation_ptx;
            }
            else if constexpr (SPEC::HAS_SEGMENTATION) {
                ptx = device_segmentation_ptx;
            }
            else if constexpr (SPEC::HAS_DEPTH) {
                ptx = device_depth_ptx;
            }
            else {
                ptx = device_ptx;
            }
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
            if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING) {
                owlMissProgSet3f(miss_prog, "color_0", owl3f{0.f, 0.f, 0.f});
                owlMissProgSet3f(miss_prog, "color_1", owl3f{0.f, 0.f, 0.f});
            } else {
                owlMissProgSet3f(miss_prog, "color_0", owl3f{.8f, 0.f, 0.f});
                owlMissProgSet3f(miss_prog, "color_1", owl3f{.8f, .8f, .8f});
            }

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
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            using TRANSFORMS_MOTION_SPEC = typename decltype(renderer.transforms_motion)::SPEC;
            float* transforms_motion_device = nullptr;
            cudaMalloc(&transforms_motion_device, TRANSFORMS_MOTION_SPEC::SIZE_BYTES);
            cudaMemset(transforms_motion_device, 0, TRANSFORMS_MOTION_SPEC::SIZE_BYTES);
            renderer.transforms_motion._data = transforms_motion_device;
            using TRANSFORMS_PAIR_SPEC = typename decltype(renderer.transforms_pair)::SPEC;
            float* transforms_pair_device = nullptr;
            cudaMalloc(&transforms_pair_device, TRANSFORMS_PAIR_SPEC::SIZE_BYTES);
            cudaMemset(transforms_pair_device, 0, TRANSFORMS_PAIR_SPEC::SIZE_BYTES);
            renderer.transforms_pair._data = transforms_pair_device;
            renderer.transforms_motion_staging.assign((size_t)SPEC::MOTION_BLUR_SAMPLES * SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12, 0.0f);
            cudaMalloc(&renderer.backend->shutter_device, sizeof(float));
            cudaMemset(renderer.backend->shutter_device, 0, sizeof(float));
        }

        constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        // device-resident outputs (see frame_buffer/depth_buffer/segmentation_buffer/observation
        // accessors): the tensors alias the OWL buffers the ray gens write, so device consumers
        // read them in place and host readers stage through an explicit copy
        OWLBuffer frame_buffer = nullptr;
        if constexpr (SPEC::HAS_RGB) {
            frame_buffer = owlDeviceBufferCreate(context, OWL_INT,
                                                 (size_t)SPEC::NUM_CAMERAS * cam_pixels, nullptr);
            renderer.frame_buffer._data = (uint32_t*)owlBufferGetPointer(frame_buffer, 0);
        }
        OWLBuffer observation_buffer = nullptr;
        if constexpr (SPEC::HAS_OBSERVATION) {
            static_assert(utils::typing::is_same_v<typename SPEC::OBSERVATION_T, float>, "The OptiX raytracing backend requires OBSERVATION_T = float");
            observation_buffer = owlDeviceBufferCreate(context, OWL_FLOAT,
                                                       (size_t)SPEC::NUM_CAMERAS * cam_pixels * SPEC::OBSERVATION_CHANNELS, nullptr);
            renderer.backend->observation_buffer = observation_buffer;
            renderer.observation._data = (float*)owlBufferGetPointer(observation_buffer, 0);
        }
        OWLBuffer depth_buffer = nullptr;
        if constexpr (SPEC::HAS_DEPTH) {
            depth_buffer = owlDeviceBufferCreate(context, OWL_FLOAT,
                                                (size_t)SPEC::NUM_CAMERAS * cam_pixels, nullptr);
            renderer.depth_buffer._data = (float*)owlBufferGetPointer(depth_buffer, 0);
        }
        OWLBuffer segmentation_buffer = nullptr;
        if constexpr (SPEC::HAS_SEGMENTATION) {
            segmentation_buffer = owlDeviceBufferCreate(context, OWL_UINT,
                                                        (size_t)SPEC::NUM_CAMERAS * cam_pixels, nullptr);
            renderer.segmentation_buffer._data = (uint32_t*)owlBufferGetPointer(segmentation_buffer, 0);
        }
        OWLBuffer normals_buffer = nullptr;
        if constexpr (SPEC::HAS_NORMALS) {
            normals_buffer = owlDeviceBufferCreate(context, OWL_FLOAT,
                                                   (size_t)SPEC::NUM_CAMERAS * cam_pixels * 3, nullptr);
            renderer.normals_buffer._data = (float*)owlBufferGetPointer(normals_buffer, 0);
        }
        OWLBuffer flow_buffer = nullptr;
        OWLBuffer flow_deltas_buffer = nullptr;
        if constexpr (SPEC::HAS_FLOW) {
            flow_buffer = owlDeviceBufferCreate(context, OWL_FLOAT,
                                                (size_t)SPEC::NUM_CAMERAS * cam_pixels * 2, nullptr);
            renderer.flow_buffer._data = (float*)owlBufferGetPointer(flow_buffer, 0);
            if constexpr (SPEC::ENABLE_OVERLAYS) {
                flow_deltas_buffer = owlDeviceBufferCreate(context, OWL_FLOAT,
                                                           (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12, nullptr);
                renderer.backend->flow_deltas_buffer = flow_deltas_buffer;
                renderer.flow_deltas._data = (float*)owlBufferGetPointer(flow_deltas_buffer, 0);
            }
        }
        OWLBuffer rgb_accumulator_buffer = nullptr;
        OWLBuffer depth_accumulator_buffer = nullptr;
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
            if constexpr (SPEC::HAS_RGB) {
                rgb_accumulator_buffer = owlDeviceBufferCreate(context, OWL_FLOAT,
                                                               (size_t)SPEC::NUM_CAMERAS * cam_pixels * 3, nullptr);
                renderer.backend->rgb_accumulator_buffer = rgb_accumulator_buffer;
                renderer.rgb_accumulator._data = (float*)owlBufferGetPointer(rgb_accumulator_buffer, 0);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                depth_accumulator_buffer = owlDeviceBufferCreate(context, OWL_FLOAT,
                                                                 (size_t)SPEC::NUM_CAMERAS * cam_pixels, nullptr);
                renderer.backend->depth_accumulator_buffer = depth_accumulator_buffer;
                renderer.depth_accumulator._data = (float*)owlBufferGetPointer(depth_accumulator_buffer, 0);
            }
        }

        OWLRayGen ray_gen = nullptr;
        if constexpr (SPEC::HAS_RGB) {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                OWLVarDecl ray_gen_vars[] = {
                    { "accum_ptr",     OWL_RAW_POINTER, OWL_OFFSETOF(AccumulateRayGenData, accum_ptr)},
                    { "shutter_ptr",   OWL_RAW_POINTER, OWL_OFFSETOF(AccumulateRayGenData, shutter_ptr)},
                    { "fb_size",       OWL_INT2,   OWL_OFFSETOF(AccumulateRayGenData, fb_size)},
                    { "cam_size",      OWL_INT2,   OWL_OFFSETOF(AccumulateRayGenData, cam_size)},
                    { "grid_cols",     OWL_INT,    OWL_OFFSETOF(AccumulateRayGenData, grid_cols)},
                    { "num_cameras",   OWL_INT,    OWL_OFFSETOF(AccumulateRayGenData, num_cameras)},
                    { "world",         OWL_GROUP,  OWL_OFFSETOF(AccumulateRayGenData, world)},
                    { "cameras_open",  OWL_BUFPTR, OWL_OFFSETOF(AccumulateRayGenData, cameras_open)},
                    { "cameras_close", OWL_BUFPTR, OWL_OFFSETOF(AccumulateRayGenData, cameras_close)},
                    { /* sentinel */ }
                };
                const char* ray_gen_name = rendering::raytracing::detail::ray_gen_program_name<false, SPEC>();
                ray_gen = owlRayGenCreate(context, module, ray_gen_name,
                                          sizeof(AccumulateRayGenData), ray_gen_vars, -1);
            }
            else if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                OWLVarDecl ray_gen_vars[] = {
                    { "fb_ptr",        OWL_BUFPTR, OWL_OFFSETOF(MotionBlurRayGenData, fb_ptr)},
                    { "obs_ptr",       OWL_RAW_POINTER, OWL_OFFSETOF(MotionBlurRayGenData, obs_ptr)},
                    { "fb_size",       OWL_INT2,   OWL_OFFSETOF(MotionBlurRayGenData, fb_size)},
                    { "cam_size",      OWL_INT2,   OWL_OFFSETOF(MotionBlurRayGenData, cam_size)},
                    { "grid_cols",     OWL_INT,    OWL_OFFSETOF(MotionBlurRayGenData, grid_cols)},
                    { "num_cameras",   OWL_INT,    OWL_OFFSETOF(MotionBlurRayGenData, num_cameras)},
                    { "world",         OWL_GROUP,  OWL_OFFSETOF(MotionBlurRayGenData, world)},
                    { "cameras_open",  OWL_BUFPTR, OWL_OFFSETOF(MotionBlurRayGenData, cameras_open)},
                    { "cameras_close", OWL_BUFPTR, OWL_OFFSETOF(MotionBlurRayGenData, cameras_close)},
                    { /* sentinel */ }
                };
                const char* ray_gen_name = rendering::raytracing::detail::ray_gen_program_name<false, SPEC>();
                ray_gen = owlRayGenCreate(context, module, ray_gen_name,
                                          sizeof(MotionBlurRayGenData), ray_gen_vars, -1);
            }
            else {
                OWLVarDecl ray_gen_vars[] = {
                    { "fb_ptr",       OWL_BUFPTR, OWL_OFFSETOF(RayGenData, fb_ptr)},
                    { "obs_ptr",      OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData, obs_ptr)},
                    { "fb_size",      OWL_INT2,   OWL_OFFSETOF(RayGenData, fb_size)},
                    { "cam_size",     OWL_INT2,   OWL_OFFSETOF(RayGenData, cam_size)},
                    { "grid_cols",    OWL_INT,    OWL_OFFSETOF(RayGenData, grid_cols)},
                    { "num_cameras",  OWL_INT,    OWL_OFFSETOF(RayGenData, num_cameras)},
                    { "world",       OWL_GROUP,  OWL_OFFSETOF(RayGenData, world)},
                    { "cameras",     OWL_BUFPTR, OWL_OFFSETOF(RayGenData, cameras)},
                    { /* sentinel */ }
                };
                const char* ray_gen_name = rendering::raytracing::detail::ray_gen_program_name<false, SPEC>();
                ray_gen = owlRayGenCreate(context, module, ray_gen_name,
                                          sizeof(RayGenData), ray_gen_vars, -1);
            }
        }

        OWLRayGen depth_ray_gen = nullptr;
        if constexpr (SPEC::HAS_DEPTH) {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                OWLVarDecl depth_ray_gen_vars[] = {
                    { "depth_accum_ptr", OWL_RAW_POINTER, OWL_OFFSETOF(AccumulateDepthRayGenData, depth_accum_ptr)},
                    { "shutter_ptr",   OWL_RAW_POINTER, OWL_OFFSETOF(AccumulateDepthRayGenData, shutter_ptr)},
                    { "fb_size",       OWL_INT2,   OWL_OFFSETOF(AccumulateDepthRayGenData, fb_size)},
                    { "cam_size",      OWL_INT2,   OWL_OFFSETOF(AccumulateDepthRayGenData, cam_size)},
                    { "grid_cols",     OWL_INT,    OWL_OFFSETOF(AccumulateDepthRayGenData, grid_cols)},
                    { "num_cameras",   OWL_INT,    OWL_OFFSETOF(AccumulateDepthRayGenData, num_cameras)},
                    { "world",         OWL_GROUP,  OWL_OFFSETOF(AccumulateDepthRayGenData, world)},
                    { "cameras_open",  OWL_BUFPTR, OWL_OFFSETOF(AccumulateDepthRayGenData, cameras_open)},
                    { "cameras_close", OWL_BUFPTR, OWL_OFFSETOF(AccumulateDepthRayGenData, cameras_close)},
                    { "max_depth",     OWL_FLOAT,  OWL_OFFSETOF(AccumulateDepthRayGenData, max_depth)},
                    { /* sentinel */ }
                };
                depth_ray_gen = owlRayGenCreate(context, module, rendering::raytracing::detail::ray_gen_program_name<true, SPEC>(),
                                                sizeof(AccumulateDepthRayGenData), depth_ray_gen_vars, -1);
            }
            else if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                OWLVarDecl depth_ray_gen_vars[] = {
                    { "depth_ptr",     OWL_BUFPTR, OWL_OFFSETOF(MotionBlurDepthRayGenData, depth_ptr)},
                    { "fb_size",       OWL_INT2,   OWL_OFFSETOF(MotionBlurDepthRayGenData, fb_size)},
                    { "cam_size",      OWL_INT2,   OWL_OFFSETOF(MotionBlurDepthRayGenData, cam_size)},
                    { "grid_cols",     OWL_INT,    OWL_OFFSETOF(MotionBlurDepthRayGenData, grid_cols)},
                    { "num_cameras",   OWL_INT,    OWL_OFFSETOF(MotionBlurDepthRayGenData, num_cameras)},
                    { "world",         OWL_GROUP,  OWL_OFFSETOF(MotionBlurDepthRayGenData, world)},
                    { "cameras_open",  OWL_BUFPTR, OWL_OFFSETOF(MotionBlurDepthRayGenData, cameras_open)},
                    { "cameras_close", OWL_BUFPTR, OWL_OFFSETOF(MotionBlurDepthRayGenData, cameras_close)},
                    { "max_depth",     OWL_FLOAT,  OWL_OFFSETOF(MotionBlurDepthRayGenData, max_depth)},
                    { /* sentinel */ }
                };
                depth_ray_gen = owlRayGenCreate(context, module, rendering::raytracing::detail::ray_gen_program_name<true, SPEC>(),
                                                sizeof(MotionBlurDepthRayGenData), depth_ray_gen_vars, -1);
            }
            else {
                OWLVarDecl depth_ray_gen_vars[] = {
                    { "depth_ptr",   OWL_BUFPTR, OWL_OFFSETOF(DepthRayGenData, depth_ptr)},
                    { "fb_size",     OWL_INT2,   OWL_OFFSETOF(DepthRayGenData, fb_size)},
                    { "cam_size",    OWL_INT2,   OWL_OFFSETOF(DepthRayGenData, cam_size)},
                    { "grid_cols",   OWL_INT,    OWL_OFFSETOF(DepthRayGenData, grid_cols)},
                    { "num_cameras", OWL_INT,    OWL_OFFSETOF(DepthRayGenData, num_cameras)},
                    { "world",       OWL_GROUP,  OWL_OFFSETOF(DepthRayGenData, world)},
                    { "cameras",     OWL_BUFPTR, OWL_OFFSETOF(DepthRayGenData, cameras)},
                    { "max_depth",   OWL_FLOAT,  OWL_OFFSETOF(DepthRayGenData, max_depth)},
                    { /* sentinel */ }
                };
                depth_ray_gen = owlRayGenCreate(context, module, rendering::raytracing::detail::ray_gen_program_name<true, SPEC>(),
                                                sizeof(DepthRayGenData), depth_ray_gen_vars, -1);
            }
        }

        OWLRayGen segmentation_ray_gen = nullptr;
        if constexpr (SPEC::HAS_SEGMENTATION) {
            OWLVarDecl segmentation_ray_gen_vars[] = {
                { "seg_ptr",     OWL_BUFPTR, OWL_OFFSETOF(SegmentationRayGenData, seg_ptr)},
                { "fb_size",     OWL_INT2,   OWL_OFFSETOF(SegmentationRayGenData, fb_size)},
                { "cam_size",    OWL_INT2,   OWL_OFFSETOF(SegmentationRayGenData, cam_size)},
                { "grid_cols",   OWL_INT,    OWL_OFFSETOF(SegmentationRayGenData, grid_cols)},
                { "num_cameras", OWL_INT,    OWL_OFFSETOF(SegmentationRayGenData, num_cameras)},
                { "world",       OWL_GROUP,  OWL_OFFSETOF(SegmentationRayGenData, world)},
                { "cameras",     OWL_BUFPTR, OWL_OFFSETOF(SegmentationRayGenData, cameras)},
                { /* sentinel */ }
            };
            segmentation_ray_gen = owlRayGenCreate(context, module, "segmentationRayGen",
                                                   sizeof(SegmentationRayGenData), segmentation_ray_gen_vars, -1);
        }

        OWLRayGen normals_ray_gen = nullptr;
        if constexpr (SPEC::HAS_NORMALS) {
            OWLVarDecl normals_ray_gen_vars[] = {
                { "normals_ptr", OWL_BUFPTR, OWL_OFFSETOF(NormalsRayGenData, normals_ptr)},
                { "fb_size",     OWL_INT2,   OWL_OFFSETOF(NormalsRayGenData, fb_size)},
                { "cam_size",    OWL_INT2,   OWL_OFFSETOF(NormalsRayGenData, cam_size)},
                { "grid_cols",   OWL_INT,    OWL_OFFSETOF(NormalsRayGenData, grid_cols)},
                { "num_cameras", OWL_INT,    OWL_OFFSETOF(NormalsRayGenData, num_cameras)},
                { "world",       OWL_GROUP,  OWL_OFFSETOF(NormalsRayGenData, world)},
                { "cameras",     OWL_BUFPTR, OWL_OFFSETOF(NormalsRayGenData, cameras)},
                { /* sentinel */ }
            };
            normals_ray_gen = owlRayGenCreate(context, module, "normalsRayGen",
                                              sizeof(NormalsRayGenData), normals_ray_gen_vars, -1);
        }

        OWLRayGen flow_ray_gen = nullptr;
        if constexpr (SPEC::HAS_FLOW) {
            OWLVarDecl flow_ray_gen_vars[] = {
                { "flow_ptr",      OWL_BUFPTR, OWL_OFFSETOF(FlowRayGenData, flow_ptr)},
                { "fb_size",       OWL_INT2,   OWL_OFFSETOF(FlowRayGenData, fb_size)},
                { "cam_size",      OWL_INT2,   OWL_OFFSETOF(FlowRayGenData, cam_size)},
                { "grid_cols",     OWL_INT,    OWL_OFFSETOF(FlowRayGenData, grid_cols)},
                { "num_cameras",   OWL_INT,    OWL_OFFSETOF(FlowRayGenData, num_cameras)},
                { "world",         OWL_GROUP,  OWL_OFFSETOF(FlowRayGenData, world)},
                { "cameras_open",  OWL_BUFPTR, OWL_OFFSETOF(FlowRayGenData, cameras_open)},
                { "cameras_close", OWL_BUFPTR, OWL_OFFSETOF(FlowRayGenData, cameras_close)},
                { "flow_deltas",   OWL_RAW_POINTER, OWL_OFFSETOF(FlowRayGenData, flow_deltas)},
                { "first_overlay_instance", OWL_UINT, OWL_OFFSETOF(FlowRayGenData, first_overlay_instance)},
                { /* sentinel */ }
            };
            flow_ray_gen = owlRayGenCreate(context, module, "flowRayGen",
                                           sizeof(FlowRayGenData), flow_ray_gen_vars, -1);
        }

        const owl2i fb_size  = {(int)SPEC::FB_WIDTH, (int)SPEC::FB_HEIGHT};
        const owl2i cam_size = {(int)SPEC::CAM_WIDTH, (int)SPEC::CAM_HEIGHT};

        if constexpr (SPEC::HAS_RGB) {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                owlRayGenSetPointer(ray_gen, "accum_ptr", owlBufferGetPointer(rgb_accumulator_buffer, 0));
                owlRayGenSetPointer(ray_gen, "shutter_ptr", renderer.backend->shutter_device);
            }
            else {
                owlRayGenSetBuffer(ray_gen, "fb_ptr", frame_buffer);
                owlRayGenSetPointer(ray_gen, "obs_ptr", SPEC::HAS_OBSERVATION ? owlBufferGetPointer(observation_buffer, 0) : nullptr);
            }
            owlRayGenSet2i    (ray_gen, "fb_size", fb_size);
            owlRayGenSet2i    (ray_gen, "cam_size", cam_size);
            owlRayGenSet1i    (ray_gen, "grid_cols", SPEC::GRID_COLS);
            owlRayGenSet1i    (ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            owlRayGenSetBuffer(segmentation_ray_gen, "seg_ptr", segmentation_buffer);
            owlRayGenSet2i    (segmentation_ray_gen, "fb_size", fb_size);
            owlRayGenSet2i    (segmentation_ray_gen, "cam_size", cam_size);
            owlRayGenSet1i    (segmentation_ray_gen, "grid_cols", SPEC::GRID_COLS);
            owlRayGenSet1i    (segmentation_ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
            renderer.backend->segmentation_ray_gen = segmentation_ray_gen;
            renderer.backend->segmentation_buffer = segmentation_buffer;
        }
        if constexpr (SPEC::HAS_NORMALS) {
            owlRayGenSetBuffer(normals_ray_gen, "normals_ptr", normals_buffer);
            owlRayGenSet2i    (normals_ray_gen, "fb_size", fb_size);
            owlRayGenSet2i    (normals_ray_gen, "cam_size", cam_size);
            owlRayGenSet1i    (normals_ray_gen, "grid_cols", SPEC::GRID_COLS);
            owlRayGenSet1i    (normals_ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
            renderer.backend->normals_ray_gen = normals_ray_gen;
            renderer.backend->normals_buffer = normals_buffer;
        }
        if constexpr (SPEC::HAS_FLOW) {
            owlRayGenSetBuffer(flow_ray_gen, "flow_ptr", flow_buffer);
            owlRayGenSet2i    (flow_ray_gen, "fb_size", fb_size);
            owlRayGenSet2i    (flow_ray_gen, "cam_size", cam_size);
            owlRayGenSet1i    (flow_ray_gen, "grid_cols", SPEC::GRID_COLS);
            owlRayGenSet1i    (flow_ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
            owlRayGenSetPointer(flow_ray_gen, "flow_deltas", flow_deltas_buffer != nullptr ? owlBufferGetPointer(flow_deltas_buffer, 0) : nullptr);
            owlRayGenSet1ui   (flow_ray_gen, "first_overlay_instance", 0u); // scene-dependent: rebound in init_renderer_scene before the SBT build
            renderer.backend->flow_ray_gen = flow_ray_gen;
            renderer.backend->flow_buffer = flow_buffer;
        }
        if constexpr (SPEC::HAS_DEPTH) {
            if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR) {
                owlRayGenSetPointer(depth_ray_gen, "depth_accum_ptr", owlBufferGetPointer(depth_accumulator_buffer, 0));
                owlRayGenSetPointer(depth_ray_gen, "shutter_ptr", renderer.backend->shutter_device);
            }
            else {
                owlRayGenSetBuffer(depth_ray_gen, "depth_ptr", depth_buffer);
            }
            owlRayGenSet2i    (depth_ray_gen, "fb_size", fb_size);
            owlRayGenSet2i    (depth_ray_gen, "cam_size", cam_size);
            owlRayGenSet1i    (depth_ray_gen, "grid_cols", SPEC::GRID_COLS);
            owlRayGenSet1i    (depth_ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
            owlRayGenSet1f    (depth_ray_gen, "max_depth", 1e30f);
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
        if constexpr (SPEC::HAS_DEPTH) {
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                owlRayGenSetBuffer(depth_ray_gen, "cameras_open", cameras_open_buffer);
                owlRayGenSetBuffer(depth_ray_gen, "cameras_close", cameras_buffer);
            }
            else {
                owlRayGenSetBuffer(depth_ray_gen, "cameras", cameras_buffer);
            }
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            owlRayGenSetBuffer(segmentation_ray_gen, "cameras", cameras_buffer);
        }
        if constexpr (SPEC::HAS_NORMALS) {
            owlRayGenSetBuffer(normals_ray_gen, "cameras", cameras_buffer);
        }
        if constexpr (SPEC::HAS_FLOW) {
            owlRayGenSetBuffer(flow_ray_gen, "cameras_open", cameras_open_buffer);
            owlRayGenSetBuffer(flow_ray_gen, "cameras_close", cameras_buffer);
        }

        renderer.backend->context = context;
        renderer.backend->module = module;
        renderer.backend->cameras_buffer = cameras_buffer;
        if constexpr (SPEC::HAS_RGB) {
            renderer.backend->ray_gen = ray_gen;
            renderer.backend->frame_buffer = frame_buffer;
        }
        if constexpr (SPEC::HAS_DEPTH) {
            renderer.backend->depth_ray_gen = depth_ray_gen;
            renderer.backend->depth_buffer = depth_buffer;
        }

#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
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

        // device-resident (was host-pinned, which cost a PCIe write per probe launch): device
        // consumers read collision_results(device, renderer) in place, host readers copy
        OWLBuffer collision_results_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(CollisionResult),
                                                                   (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES, nullptr);
        renderer.collision_results._data = (rendering::raytracing::CollisionResult*)owlBufferGetPointer(collision_results_buffer, 0);
        OWLBuffer probe_dirs_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(owl::vec3f), SPEC::NUM_PROBES, nullptr);

        owlRayGenSetBuffer(collision_ray_gen, "results", collision_results_buffer);
        owlRayGenSetBuffer(collision_ray_gen, "probe_directions", probe_dirs_buffer);
        owlRayGenSetBuffer(collision_ray_gen, "cameras", cameras_buffer);
        owlRayGenSet1i    (collision_ray_gen, "num_probes", SPEC::NUM_PROBES);
        owlRayGenSet1i    (collision_ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
        owlRayGenSet1f    (collision_ray_gen, "max_dist", 1e30f);

        renderer.backend->collision_ray_gen = collision_ray_gen;
        renderer.backend->collision_results_buffer = collision_results_buffer;
        renderer.backend->probe_dirs_buffer = probe_dirs_buffer;
#endif
        rendering::raytracing::detail::announce_backend(renderer);
        }
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        namespace optix = rendering::raytracing::backends::optix;
        renderer.backend = new rendering::raytracing::backends::RendererState<rendering::raytracing::backends::Optix, SPEC>{};
        renderer.device.state = renderer.backend;
        OWLContext context = nullptr;
        OWLModule module = nullptr;
        optix::create_context_resources<SPEC>(context, module);
        optix::detail::malloc_renderer(device, renderer, context, module);
    }

    // shared-library mode: the library owns the context; the renderer allocates only its own
    // cameras/outputs/ray gens against it
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Optix>& library){
        namespace optix = rendering::raytracing::backends::optix;
        renderer.backend = new rendering::raytracing::backends::RendererState<rendering::raytracing::backends::Optix, SPEC>{};
        renderer.device.state = renderer.backend;
        optix::detail::malloc_renderer(device, renderer, library.backend->context, library.backend->module);
        renderer.backend->library = &library;
    }

    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Optix>& library){
        library.backend = new rendering::raytracing::backends::LibraryState<rendering::raytracing::backends::Optix, SPEC>{};
        rendering::raytracing::backends::optix::create_context_resources<SPEC>(library.backend->context, library.backend->module);
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
        library.hashes.clear();
        delete library.backend;
        library.backend = nullptr;
    }

    // =========================================================================
    // init: upload meshes, build single shared BVH, build programs/pipeline/SBT
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer);

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
                owlGeomTypeSetClosestHit(triangles_geom_type, 2, module, "normalsHit");
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
            RL_TOOLS_RENDERING_RAYTRACING_LOG("building " << scene.objects.size() << " object(s), " << scene.instances.size() << " instance(s), " << (all_objects.size() - scene.objects.size()) << " pool object(s) ...");

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
                                                             md.texture.pixels.data(),
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
                                                               md.normal_map.pixels.data(),
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
                                                               md.metallic_roughness_map.pixels.data(),
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
                                                               md.emissive_map.pixels.data(),
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
                                                               md.occlusion_map.pixels.data(),
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
                    owlGeomSet3f(geom, "ambient_color", owl3f{0.10f, 0.10f, 0.10f});
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
                const auto scene_lights = rendering::raytracing::detail::effective_scene_lights<true>(scene);
                OWLBuffer light_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(rendering::raytracing::SceneLight),
                                                                scene_lights.size(), scene_lights.data());
                for (size_t m = 0; m < geoms.size(); m++) {
                    owlGeomSetBuffer(geoms[m], "scene_lights", light_buffer);
                    owlGeomSet1i(geoms[m], "num_scene_lights", (int)scene_lights.size());
                }
            }

            std::vector<unsigned int> host_instance_classes(scene.instances.size() + (SPEC::ENABLE_OVERLAYS ? (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES : 0), 0);
            for(size_t instance_i = 0; instance_i < scene.instances.size(); instance_i++){
                host_instance_classes[instance_i] = all_objects[scene.instances[instance_i].object]->segmentation_class;
            }
            if(host_instance_classes.empty()){
                host_instance_classes.push_back(0);
            }
            OWLBuffer instance_classes_buffer = owlDeviceBufferCreate(context, OWL_UINT, host_instance_classes.size(), host_instance_classes.data());

            assets.world = world;
            assets.instance_classes_buffer = instance_classes_buffer;
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
            rendering::raytracing::detail::compute_scene_bounds(renderer, scene);
            OWLGroup world = (OWLGroup)assets.world;
            OWLBuffer instance_classes_buffer = (OWLBuffer)assets.instance_classes_buffer;

            delete (optix::OverlayState*)renderer.backend->overlay_state;
            renderer.backend->overlay_state = nullptr;
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
            if constexpr (SPEC::HAS_DEPTH) {
                const float max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
                owlRayGenSetGroup((OWLRayGen)renderer.backend->depth_ray_gen, "world", world);
                owlRayGenSet1f((OWLRayGen)renderer.backend->depth_ray_gen, "max_depth", max_depth);
            }
            if constexpr (SPEC::HAS_SEGMENTATION) {
                owlRayGenSetGroup((OWLRayGen)renderer.backend->segmentation_ray_gen, "world", world);
            }
            if constexpr (SPEC::HAS_NORMALS) {
                owlRayGenSetGroup((OWLRayGen)renderer.backend->normals_ray_gen, "world", world);
            }
            if constexpr (SPEC::HAS_FLOW) {
                owlRayGenSetGroup((OWLRayGen)renderer.backend->flow_ray_gen, "world", world);
                owlRayGenSet1ui((OWLRayGen)renderer.backend->flow_ray_gen, "first_overlay_instance", (unsigned int)scene.instances.size());
            }
            if(renderer.backend->collision_ray_gen){
                owlRayGenSetGroup((OWLRayGen)renderer.backend->collision_ray_gen, "world", world);
                owlRayGenSet1f((OWLRayGen)renderer.backend->collision_ray_gen, "max_dist", renderer.camera_radius * 2.0f);
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
                overlay_state->accel = optix::overlay_accel_create(owlContextGetOptixContext(context, 0), SPEC::NUM_OVERLAYS, SPEC::MAX_OVERLAY_INSTANCES, (unsigned int)scene.instances.size(), (unsigned long long)owlGroupGetTraversable(overlay_state->filler_group, 0));
                optix::overlay_accel_upload_objects(overlay_state->accel, object_entries.data(), (unsigned int)object_entries.size());
                overlay_state->structure_staging.assign((size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES, optix::OverlaySlotStructure{});
                overlay_state->transforms_staging.assign((size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12, 0.0f);
                if constexpr (SPEC::HAS_FLOW) {
                    overlay_state->flow_deltas_staging.assign((size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12, 0.0f);
                }
                // published once: raw builds into fixed per-overlay buffers keep the handles stable,
                // so no per-rebuild re-publication is needed
                overlay_state->traversables_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(unsigned long long), SPEC::NUM_OVERLAYS, optix::overlay_accel_traversables(overlay_state->accel));
                overlay_state->attachments_buffer = owlDeviceBufferCreate(context, OWL_UINT, (size_t)SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA, nullptr);
                rendering::raytracing::detail::reset_overlay_state(renderer);
                overlay_state->instance_classes_buffer = instance_classes_buffer;
                overlay_state->num_scene_instances = scene.instances.size();
            }

            if(renderer.backend->launch_params == nullptr){
                OWLVarDecl launch_params_vars[] = {
                    { "overlays",      OWL_BUFPTR, OWL_OFFSETOF(OverlayLaunchParams, overlays)},
                    { "attachments",   OWL_BUFPTR, OWL_OFFSETOF(OverlayLaunchParams, attachments)},
                    { "overlay_count", OWL_INT,    OWL_OFFSETOF(OverlayLaunchParams, overlay_count)},
                    { "cam_width",     OWL_INT,    OWL_OFFSETOF(OverlayLaunchParams, cam_width)},
                    { "cam_height",    OWL_INT,    OWL_OFFSETOF(OverlayLaunchParams, cam_height)},
                    { "grid_cols",     OWL_INT,    OWL_OFFSETOF(OverlayLaunchParams, grid_cols)},
                    { "instance_classes", OWL_BUFPTR, OWL_OFFSETOF(OverlayLaunchParams, instance_classes)},
                    { "semantic_segmentation", OWL_INT, OWL_OFFSETOF(OverlayLaunchParams, semantic_segmentation)},
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
                owlParamsSet1i(launch_params, "semantic_segmentation", SPEC::SEMANTIC_SEGMENTATION ? 1 : 0);
                owlParamsSetBuffer(launch_params, "instance_classes", instance_classes_buffer);
                if constexpr (SPEC::ENABLE_OVERLAYS){
                    auto* overlay_state = (optix::OverlayState*)renderer.backend->overlay_state;
                    owlParamsSetBuffer(launch_params, "overlays", overlay_state->traversables_buffer);
                    owlParamsSetBuffer(launch_params, "attachments", overlay_state->attachments_buffer);
                }
            }

            if constexpr (SPEC::ENABLE_OVERLAYS){
                update(device, renderer); // publish the (empty) overlays and the attachment table
            }
        }
    }

    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool){
        namespace optix = rendering::raytracing::backends::optix;
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
    // dedup inside the library), then the renderer is wired to the shared build. Returns the
    // unique-scene index so callers can key their own per-scene data.
    template <typename DEVICE, typename SPEC>
    typename SPEC::TI init(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, rendering::raytracing::AssetLibrary<SPEC, rendering::raytracing::backends::Optix>& library, const char* scene_path){
        namespace optix = rendering::raytracing::backends::optix;
        using TI = typename SPEC::TI;
        utils::assert_exit(device, renderer.backend->library == &library, "init: the renderer was not malloc'd against this library");
        bool is_new = false;
        const TI scene_id = rendering::raytracing::detail::library_lookup_or_load(device, library, scene_path, is_new);
        const rendering::raytracing::Scene& scene = library.scenes[scene_id];
        if(is_new){
            if(library.backend->geom_type == nullptr){
                library.backend->geom_type = optix::detail::create_geom_type<SPEC>(library.backend->context, library.backend->module);
            }
            std::vector<const rendering::raytracing::Object*> build_objects;
            optix::detail::collect_all_objects(scene, library.pool, build_objects);
            library.assets[scene_id] = new rendering::raytracing::backends::SceneState<rendering::raytracing::backends::Optix, SPEC>{};
            optix::detail::build_scene_assets<DEVICE, SPEC>(device, library.backend->context, library.backend->geom_type, scene, build_objects, *library.assets[scene_id]);
        }
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
        return scene_id;
    }

    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const rendering::raytracing::Scene& scene){
        static const rendering::raytracing::AssetPool empty_pool{};
        init(device, renderer, scene, empty_pool);
    }

    namespace rendering::raytracing::backends::optix{
        // producers write the renderer's device tensors on the caller's stream (device.stream);
        // the launch verbs bridge it to the backend-owned streams with an event so producer
        // writes are ordered before consumption — callers never handle streams. A CPU device has
        // no stream member and resolves to the null overload (host writes are synchronous).
        template <typename DEVICE>
        auto producer_stream(DEVICE& device, int) -> decltype(device.stream) { return device.stream; }
        template <typename DEVICE>
        cudaStream_t producer_stream(DEVICE& device, long){ return nullptr; }

        inline void await_producer(cudaStream_t producer, cudaStream_t consumer){
            if(producer == nullptr || producer == consumer){
                return;
            }
            cudaEvent_t ready;
            cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
            cudaEventRecord(ready, producer);
            cudaStreamWaitEvent(consumer, ready, 0);
            cudaEventDestroy(ready);
        }
    }

    // fully stream-ordered on the render stream: staging uploads for host-verb writes, then the
    // device-side instance fill and per-overlay raw optixAccelBuild — no host synchronization
    template <typename DEVICE, typename SPEC>
    void update_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
        namespace optix = rendering::raytracing::backends::optix;
        using TI = typename SPEC::TI;
        auto* overlay_state = (optix::OverlayState*)renderer.backend->overlay_state;
        cudaStream_t cuda_stream = (cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0);
        cudaStream_t coll_stream = renderer.backend->coll_launch_params != nullptr ? (cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->coll_launch_params, 0) : cuda_stream;
        optix::await_producer(optix::producer_stream(device, 0), cuda_stream);
        if(coll_stream != cuda_stream){
            // probe launches trace the overlay TLASes on their own stream: order this rebuild
            // after in-flight probes and subsequent probes after this rebuild, without host syncs
            cudaEvent_t probes_done;
            cudaEventCreateWithFlags(&probes_done, cudaEventDisableTiming);
            cudaEventRecord(probes_done, coll_stream);
            cudaStreamWaitEvent(cuda_stream, probes_done, 0);
            cudaEventDestroy(probes_done);
        }

        if constexpr (SPEC::HAS_FLOW){
            // dirty-gated internally; must run before the loop below consumes the dirty flags
            rendering::raytracing::detail::compose_flow_deltas(renderer, overlay_state->flow_deltas_staging.data());
        }
        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            auto& overlay_host = renderer.overlays[overlay];
            if(!overlay_host.dirty) continue;
            const size_t base = (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES;
            for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                const auto& host_slot = overlay_host.slots[slot];
                auto& row = overlay_state->structure_staging[base + slot];
                row.active = host_slot.active ? 1u : 0u;
                row.pose_slot = (unsigned int)host_slot.pose_slot;
                row.object = (unsigned int)host_slot.object;
                std::memcpy(row.part_local, host_slot.part_local, sizeof(row.part_local));
                std::memcpy(&overlay_state->transforms_staging[(base + slot) * 12], host_slot.transform_entry, 12 * sizeof(float));
            }
            cudaMemcpyAsync(optix::overlay_accel_structure(overlay_state->accel) + base, &overlay_state->structure_staging[base], SPEC::MAX_OVERLAY_INSTANCES * sizeof(optix::OverlaySlotStructure), cudaMemcpyHostToDevice, cuda_stream);
            cudaMemcpyAsync(data(renderer.transforms) + base * 12, &overlay_state->transforms_staging[base * 12], SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
            if constexpr (SPEC::HAS_FLOW){
                cudaMemcpyAsync(data(renderer.flow_deltas) + base * 12, &overlay_state->flow_deltas_staging[base * 12], SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
            }
            overlay_host.dirty = false;
        }
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            bool motion_dirty = false;
            for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
                motion_dirty |= renderer.transforms_motion_dirty[overlay];
                renderer.transforms_motion_dirty[overlay] = false;
            }
            if(motion_dirty){
                cudaMemcpyAsync(data(renderer.transforms_motion), renderer.transforms_motion_staging.data(), renderer.transforms_motion_staging.size() * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
            }
        }
        if(renderer.attachments_dirty){
            std::vector<uint32_t> attachments((size_t)SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA);
            for(size_t index = 0; index < attachments.size(); index++){
                attachments[index] = (uint32_t)renderer.attachments[index];
            }
            owlBufferUpload(overlay_state->attachments_buffer, attachments.data(), 0, attachments.size());
            renderer.attachments_dirty = false;
        }
        optix::overlay_accel_build(overlay_state->accel, data(renderer.transforms), (unsigned int*)owlBufferGetPointer(overlay_state->instance_classes_buffer, 0), 1.0f, nullptr, cuda_stream);
        if(coll_stream != cuda_stream){
            cudaEvent_t built;
            cudaEventCreateWithFlags(&built, cudaEventDisableTiming);
            cudaEventRecord(built, cuda_stream);
            cudaStreamWaitEvent(coll_stream, built, 0);
            cudaEventDestroy(built);
        }
    }

    template <typename DEVICE, typename SPEC>
    void update_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "update requires an overlay-enabled renderer specification");
        cudaStreamSynchronize((cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0));
    }

    // expands the device-resident transforms_pair tensor into the per-sample transforms_motion
    // slabs (and the close state into transforms) entirely on the render stream — with a CUDA
    // device argument the kernel is ordered behind the producer stream via an event, so a sim
    // kernel writing the pair tensor needs no host synchronization
    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "expand_motion_transforms requires a dynamic-motion-blur renderer specification");
        namespace optix = rendering::raytracing::backends::optix;
        cudaStream_t cuda_stream = (cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0);
        optix::await_producer(optix::producer_stream(device, 0), cuda_stream);
        if constexpr (SPEC::HAS_FLOW){
            auto* overlay_state = (optix::OverlayState*)renderer.backend->overlay_state;
            optix::overlay_accel_expand_flow_deltas(data(renderer.transforms_pair), optix::overlay_accel_structure(overlay_state->accel), data(renderer.flow_deltas),
                                                    (unsigned int)SPEC::NUM_OVERLAYS, (unsigned int)SPEC::MAX_OVERLAY_INSTANCES,
                                                    (cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0));
        }
        optix::overlay_accel_expand_motion(data(renderer.transforms_pair), data(renderer.transforms_motion), data(renderer.transforms),
                                           (unsigned int)((size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES), (unsigned int)SPEC::MOTION_BLUR_SAMPLES, cuda_stream);
    }

    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "expand_motion_transforms requires a dynamic-motion-blur renderer specification");
        cudaStreamSynchronize((cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0));
    }

    template <typename DEVICE, typename SPEC>
    void expand_motion_transforms(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        expand_motion_transforms_launch(device, renderer);
        expand_motion_transforms_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        update_launch(device, renderer);
        update_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void generate_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer,
                          const typename SPEC::T center[3], typename SPEC::T radius,
                          const typename SPEC::T up[3], typename SPEC::T fov){
        std::vector<rendering::raytracing::Camera<typename SPEC::T>> staging(SPEC::NUM_CAMERAS);
        rendering::raytracing::detail::generate_camera_poses<SPEC>(device, staging.data(), center, radius, up, fov);

        owlBufferUpload((OWLBuffer)renderer.backend->cameras_buffer, staging.data(), 0, SPEC::NUM_CAMERAS);
        if constexpr (SPEC::HAS_CAMERA_PAIR) {
            owlBufferUpload((OWLBuffer)renderer.backend->cameras_open_buffer, staging.data(), 0, SPEC::NUM_CAMERAS);
        }
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
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Probe rays disabled (RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS=1)");
        return;
#else
        std::vector<float> dirs = rendering::raytracing::detail::generate_probe_direction_vectors<SPEC>();
        owlBufferUpload((OWLBuffer)renderer.backend->probe_dirs_buffer, dirs.data(), 0, SPEC::NUM_PROBES);
#endif
    }

    template <typename DEVICE, typename SPEC>
    void render_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        namespace optix = rendering::raytracing::backends::optix;
        using TI = typename SPEC::TI;
        OWLParams launch_params = (OWLParams)renderer.backend->launch_params;
        cudaStream_t cuda_stream = (cudaStream_t)owlParamsGetCudaStream(launch_params, 0);
        optix::await_producer(optix::producer_stream(device, 0), cuda_stream);
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            // one enqueue-only pass per motion sample: overlay TLAS rebuild from the per-sample
            // transforms slab (the fill kernel also publishes the pass's shutter time), then the
            // accumulate ray gens; the shutter-close rebuild restores the steady state for
            // segmentation/probes before the resolve kernel averages the accumulators
            auto* overlay_state = (optix::OverlayState*)renderer.backend->overlay_state;
            unsigned int* instance_classes = (unsigned int*)owlBufferGetPointer(overlay_state->instance_classes_buffer, 0);
            cudaStream_t coll_stream = renderer.backend->coll_launch_params != nullptr ? (cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->coll_launch_params, 0) : cuda_stream;
            if(coll_stream != cuda_stream){
                cudaEvent_t probes_done;
                cudaEventCreateWithFlags(&probes_done, cudaEventDisableTiming);
                cudaEventRecord(probes_done, coll_stream);
                cudaStreamWaitEvent(cuda_stream, probes_done, 0);
                cudaEventDestroy(probes_done);
            }
            if constexpr (SPEC::HAS_RGB){
                cudaMemsetAsync(data(renderer.rgb_accumulator), 0, decltype(renderer.rgb_accumulator)::SPEC::SIZE_BYTES, cuda_stream);
            }
            if constexpr (SPEC::HAS_DEPTH){
                cudaMemsetAsync(data(renderer.depth_accumulator), 0, decltype(renderer.depth_accumulator)::SPEC::SIZE_BYTES, cuda_stream);
            }
            constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
            for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                const float shutter_t = ((float)sample + 0.5f) / (float)SPEC::MOTION_BLUR_SAMPLES;
                optix::overlay_accel_build(overlay_state->accel, data(renderer.transforms_motion) + sample * SLAB, instance_classes, shutter_t, renderer.backend->shutter_device, cuda_stream);
                if constexpr (SPEC::HAS_RGB){
                    owlAsyncLaunch2D((OWLRayGen)renderer.backend->ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
                }
                if constexpr (SPEC::HAS_DEPTH){
                    owlAsyncLaunch2D((OWLRayGen)renderer.backend->depth_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
                }
            }
            optix::overlay_accel_build(overlay_state->accel, data(renderer.transforms), instance_classes, 1.0f, nullptr, cuda_stream);
            if(coll_stream != cuda_stream){
                cudaEvent_t built;
                cudaEventCreateWithFlags(&built, cudaEventDisableTiming);
                cudaEventRecord(built, cuda_stream);
                cudaStreamWaitEvent(coll_stream, built, 0);
                cudaEventDestroy(built);
            }
            if constexpr (SPEC::HAS_SEGMENTATION){
                owlAsyncLaunch2D((OWLRayGen)renderer.backend->segmentation_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
            }
            if constexpr (SPEC::HAS_NORMALS){
                owlAsyncLaunch2D((OWLRayGen)renderer.backend->normals_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
            }
            if constexpr (SPEC::HAS_FLOW){
                owlAsyncLaunch2D((OWLRayGen)renderer.backend->flow_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
            }
            const float* rgb_accumulation = nullptr;
            unsigned int* frame_buffer = nullptr;
            float* observation = nullptr;
            const float* depth_accumulation = nullptr;
            float* depth_buffer = nullptr;
            if constexpr (SPEC::HAS_RGB){
                rgb_accumulation = data(renderer.rgb_accumulator);
                frame_buffer = data(renderer.frame_buffer);
            }
            if constexpr (SPEC::HAS_OBSERVATION){
                observation = data(renderer.observation);
            }
            if constexpr (SPEC::HAS_DEPTH){
                depth_accumulation = data(renderer.depth_accumulator);
                depth_buffer = data(renderer.depth_buffer);
            }
            optix::overlay_accel_resolve(rgb_accumulation, frame_buffer, observation, SPEC::SHADING::SRGB_OUTPUT ? 1 : 0,
                                         depth_accumulation, depth_buffer,
                                         (unsigned int)((size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS),
                                         (unsigned int)SPEC::MOTION_BLUR_SAMPLES, cuda_stream);
            return;
        }
        if constexpr (SPEC::HAS_RGB) {
            OWLRayGen ray_gen = (OWLRayGen)renderer.backend->ray_gen;
            owlAsyncLaunch2D(ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            OWLRayGen depth_ray_gen = (OWLRayGen)renderer.backend->depth_ray_gen;
            owlAsyncLaunch2D(depth_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
        }
        if constexpr (SPEC::HAS_SEGMENTATION) {
            owlAsyncLaunch2D((OWLRayGen)renderer.backend->segmentation_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
        }
        if constexpr (SPEC::HAS_NORMALS) {
            owlAsyncLaunch2D((OWLRayGen)renderer.backend->normals_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
        }
        if constexpr (SPEC::HAS_FLOW) {
            owlAsyncLaunch2D((OWLRayGen)renderer.backend->flow_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        owlLaunchSync((OWLParams)renderer.backend->launch_params);
    }

    template <typename DEVICE, typename SPEC>
    void render(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        render_launch(device, renderer);
        render_sync(device, renderer);
    }

    // render produces the image outputs the spec declares; the collision-probe pass is the
    // separate probe verb so it can be scheduled independently (e.g. alongside update)
    template <typename DEVICE, typename SPEC>
    void probe_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        namespace optix = rendering::raytracing::backends::optix;
        if(renderer.backend->collision_ray_gen){
            OWLRayGen collision_ray_gen = (OWLRayGen)renderer.backend->collision_ray_gen;
            OWLParams coll_lp = (OWLParams)renderer.backend->coll_launch_params;
            optix::await_producer(optix::producer_stream(device, 0), (cudaStream_t)owlParamsGetCudaStream(coll_lp, 0));
            owlAsyncLaunch2D(collision_ray_gen, SPEC::NUM_CAMERAS, SPEC::NUM_PROBES, coll_lp);
        }
    }

    template <typename DEVICE, typename SPEC>
    void probe_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        if(renderer.backend->coll_launch_params)
            owlLaunchSync((OWLParams)renderer.backend->coll_launch_params);
    }

    template <typename DEVICE, typename SPEC>
    void probe(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        probe_launch(device, renderer);
        probe_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void save_segmentation_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const char* filename){
        static_assert(SPEC::HAS_SEGMENTATION, "save_segmentation_image requires a segmentation-capable renderer specification");
        const size_t segmentation_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::vector<uint32_t> segmentation_host(segmentation_count);
        Tensor<typename decltype(renderer.segmentation_buffer)::SPEC> segmentation_alias;
        segmentation_alias._data = segmentation_host.data();
        copy(renderer.device, device, renderer.segmentation_buffer, segmentation_alias);
        rendering::raytracing::detail::write_segmentation_grid_png<SPEC>(segmentation_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_normals_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const char* filename){
        static_assert(SPEC::HAS_NORMALS, "save_normals_image requires a normals-capable renderer specification");
        const size_t normals_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS * 3;
        std::vector<float> normals_host(normals_count);
        Tensor<typename decltype(renderer.normals_buffer)::SPEC> normals_alias;
        normals_alias._data = normals_host.data();
        copy(renderer.device, device, renderer.normals_buffer, normals_alias);
        rendering::raytracing::detail::write_normals_grid_png<SPEC>(normals_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_flow_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const char* filename){
        static_assert(SPEC::HAS_FLOW, "save_flow_image requires a flow-capable renderer specification");
        const size_t flow_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS * 2;
        std::vector<float> flow_host(flow_count);
        Tensor<typename decltype(renderer.flow_buffer)::SPEC> flow_alias;
        flow_alias._data = flow_host.data();
        copy(renderer.device, device, renderer.flow_buffer, flow_alias);
        rendering::raytracing::detail::write_flow_grid_png<SPEC>(flow_host.data(), filename);
    }

    // =========================================================================
    // save_image: readback framebuffer, rearrange to grid, write PNG
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void save_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const char* filename){
        static_assert(SPEC::HAS_RGB, "save_image requires an RGB-capable renderer specification");
        using TI = typename SPEC::TI;
        constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        const size_t fb_count = (size_t)SPEC::NUM_CAMERAS * cam_pixels;

        std::vector<uint32_t> fb_host(fb_count);
        Tensor<typename decltype(renderer.frame_buffer)::SPEC> fb_alias;
        fb_alias._data = fb_host.data();
        copy(renderer.device, device, renderer.frame_buffer, fb_alias);
        rendering::raytracing::detail::write_grid_png<SPEC>(fb_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth_image requires a depth-capable renderer specification");
        using TI = typename SPEC::TI;
        constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        const size_t depth_count = (size_t)SPEC::NUM_CAMERAS * cam_pixels;

        std::vector<float> depth_host(depth_count);
        Tensor<typename decltype(renderer.depth_buffer)::SPEC> depth_alias;
        depth_alias._data = depth_host.data();
        copy(renderer.device, device, renderer.depth_buffer, depth_alias);

        rendering::raytracing::detail::write_depth_grid_png<SPEC>(depth_host.data(), renderer.camera_radius, filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth requires a depth-capable renderer specification");
        constexpr size_t depth_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::vector<float> depth_host(depth_count);
        Tensor<typename decltype(renderer.depth_buffer)::SPEC> depth_alias;
        depth_alias._data = depth_host.data();
        copy(renderer.device, device, renderer.depth_buffer, depth_alias);
        rendering::raytracing::detail::write_depth_bin<SPEC>(depth_host.data(), filename);
    }

    // =========================================================================
    // save_probes: readback collision results, write binary
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void save_probes(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer, const char* filename){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("save_probes skipped: probe rays are disabled.");
        (void)filename;
        return;
#else
        constexpr size_t probe_count = (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES;
        std::vector<CollisionResult> probes_host(probe_count);
        Tensor<typename decltype(renderer.collision_results)::SPEC> probes_alias;
        probes_alias._data = probes_host.data();
        copy(renderer.device, device, renderer.collision_results, probes_alias);
        rendering::raytracing::detail::write_probes_bin_and_log<SPEC>(probes_host.data(), filename);
#endif
    }

    // render stream shared by render/probe/update launches; producers writing renderer inputs
    // from their own kernels can run on it to get ordering without events or host syncs
    template <typename DEVICE, typename SPEC>
    cudaStream_t stream(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        return (cudaStream_t)owlParamsGetCudaStream((OWLParams)renderer.backend->launch_params, 0);
    }

    // scoped to the renderer's own streams — never a whole-device barrier
    template <typename DEVICE, typename SPEC>
    void synchronize(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        rendering::raytracing::backends::optix::synchronize(renderer.device);
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC, rendering::raytracing::backends::Optix>& renderer){
        RL_TOOLS_RENDERING_RAYTRACING_LOG("destroying devicegroups ...");
        if(renderer.backend != nullptr){
            if(renderer.backend->context != nullptr && renderer.backend->library == nullptr){
                owlContextDestroy((OWLContext)renderer.backend->context); // library-backed renderers borrow the context — the library destroys it
            }
            if(renderer.backend->overlay_state != nullptr){
                auto* overlay_state = (rendering::raytracing::backends::optix::OverlayState*)renderer.backend->overlay_state;
                rendering::raytracing::backends::optix::overlay_accel_destroy(overlay_state->accel);
                delete overlay_state;
            }
            if(renderer.backend->shutter_device != nullptr){
                cudaFree(renderer.backend->shutter_device);
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
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            if(renderer.transforms_motion._data != nullptr){
                cudaFree(renderer.transforms_motion._data);
                renderer.transforms_motion._data = nullptr;
            }
            if(renderer.transforms_pair._data != nullptr){
                cudaFree(renderer.transforms_pair._data);
                renderer.transforms_pair._data = nullptr;
            }
            if constexpr (SPEC::HAS_RGB){
                renderer.rgb_accumulator._data = nullptr;
            }
            if constexpr (SPEC::HAS_DEPTH){
                renderer.depth_accumulator._data = nullptr;
            }
        }
        // the input and output tensors alias OWL buffers destroyed with the context
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
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
