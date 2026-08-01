#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_OPERATIONS_CUDA_H

#include "../../renderer.h"
#include "../../operations_cpu_common.h"
#include "device.h"

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
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
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
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        using TI = typename SPEC::TI;

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

        OWLContext context = owlContextCreate(nullptr, 1);
        owlContextSetRayTypeCount(context, 2);
        owlContextSetNumPayloadValues(context, 3);
        const char* ptx = nullptr;
        if constexpr (SPEC::HAS_DEPTH) {
            ptx = device_depth_ptx;
        }
        else {
            ptx = device_ptx;
        }
        OWLModule module = owlModuleCreate(context, ptx);

        constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        OWLBuffer frame_buffer = nullptr;
        if constexpr (SPEC::HAS_RGB) {
            frame_buffer = owlDeviceBufferCreate(context, OWL_INT,
                                                 (size_t)SPEC::NUM_CAMERAS * cam_pixels, nullptr);
        }
        OWLBuffer depth_buffer = nullptr;
        if constexpr (SPEC::HAS_DEPTH) {
            depth_buffer = owlDeviceBufferCreate(context, OWL_FLOAT,
                                                (size_t)SPEC::NUM_CAMERAS * cam_pixels, nullptr);
        }

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

        OWLRayGen ray_gen = nullptr;
        if constexpr (SPEC::HAS_RGB) {
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                OWLVarDecl ray_gen_vars[] = {
                    { "fb_ptr",        OWL_BUFPTR, OWL_OFFSETOF(MotionBlurRayGenData, fb_ptr)},
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
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
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

        const owl2i fb_size  = {(int)SPEC::FB_WIDTH, (int)SPEC::FB_HEIGHT};
        const owl2i cam_size = {(int)SPEC::CAM_WIDTH, (int)SPEC::CAM_HEIGHT};

        if constexpr (SPEC::HAS_RGB) {
            owlRayGenSetBuffer(ray_gen, "fb_ptr", frame_buffer);
            owlRayGenSet2i    (ray_gen, "fb_size", fb_size);
            owlRayGenSet2i    (ray_gen, "cam_size", cam_size);
            owlRayGenSet1i    (ray_gen, "grid_cols", SPEC::GRID_COLS);
            owlRayGenSet1i    (ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            owlRayGenSetBuffer(depth_ray_gen, "depth_ptr", depth_buffer);
            owlRayGenSet2i    (depth_ray_gen, "fb_size", fb_size);
            owlRayGenSet2i    (depth_ray_gen, "cam_size", cam_size);
            owlRayGenSet1i    (depth_ray_gen, "grid_cols", SPEC::GRID_COLS);
            owlRayGenSet1i    (depth_ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
            owlRayGenSet1f    (depth_ray_gen, "max_depth", 1e30f);
        }

        renderer.backend.context = context;
        renderer.backend.module = module;
        if constexpr (SPEC::HAS_RGB) {
            renderer.backend.ray_gen = ray_gen;
            renderer.backend.frame_buffer_handle = frame_buffer;
        }
        if constexpr (SPEC::HAS_DEPTH) {
            renderer.backend.depth_ray_gen = depth_ray_gen;
            renderer.backend.depth_buffer_handle = depth_buffer;
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

        OWLBuffer collision_results_buffer = owlHostPinnedBufferCreate(context, OWL_USER_TYPE(CollisionResult),
                                                                        (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES);

        renderer.backend.collision_ray_gen = collision_ray_gen;
        renderer.backend.collision_results_buffer = collision_results_buffer;
#endif
    }

    // =========================================================================
    // upload_geometry: upload meshes + build single shared BVH
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void upload_geometry(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        OWLContext context = (OWLContext)renderer.backend.context;
        OWLModule module = (OWLModule)renderer.backend.module;

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
                if constexpr (SHADING_USAGE::USES_INDEX) {
                    triangles_geom_vars.push_back({ "index", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index)});
                }
                if constexpr (SHADING_USAGE::USES_VERTEX) {
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
        else {
            OWLVarDecl triangles_geom_vars[] = {
                { /* sentinel */ }
            };
            triangles_geom_type = owlGeomTypeCreate(context, OWL_TRIANGLES,
                                                     sizeof(CollisionGeomData),
                                                     triangles_geom_vars, -1);
        }
        owlGeomTypeSetClosestHit(triangles_geom_type, 1, module, "collisionHit");

        RL_TOOLS_RENDERING_RAYTRACING_LOG("building " << renderer.meshes.size() << " geometries ...");

        std::vector<OWLGeom> geoms;
        for(size_t m = 0; m < renderer.meshes.size(); m++){
            auto& md = renderer.meshes[m];
            size_t num_vertices = md.vertices.size() / 3;
            size_t num_indices = md.indices.size() / 3;

            OWLBuffer vb = owlDeviceBufferCreate(context, OWL_FLOAT3, num_vertices, md.vertices.data());
            OWLBuffer ib = owlDeviceBufferCreate(context, OWL_INT3, num_indices, md.indices.data());

            OWLGeom geom = owlGeomCreate(context, triangles_geom_type);
            owlTrianglesSetVertices(geom, vb, num_vertices, sizeof(owl::vec3f), 0);
            owlTrianglesSetIndices(geom, ib, num_indices, sizeof(owl::vec3i), 0);
            if constexpr (SPEC::HAS_RGB) {
                using SHADING_USAGE = rendering::raytracing::detail::MediumShadingUsage<SPEC>;
                if constexpr (SPEC::SHADING::PBR_SHADING || SHADING_USAGE::USES_VERTEX) {
                    owlGeomSetBuffer(geom, "vertex", vb);
                }
                if constexpr (SPEC::SHADING::PBR_SHADING || SHADING_USAGE::USES_INDEX) {
                    owlGeomSetBuffer(geom, "index", ib);
                }
                owlGeomSet3f(geom, "color", owl3f{md.color[0], md.color[1], md.color[2]});

                if constexpr (SPEC::SHADING::PBR_SHADING || SPEC::SHADING::LOAD_TEXTURES) {
                if(!md.tex_coords.empty()){
                    size_t num_tc = md.tex_coords.size() / 2;
                    OWLBuffer tcb = owlDeviceBufferCreate(context, OWL_FLOAT2, num_tc, md.tex_coords.data());
                    owlGeomSetBuffer(geom, "tex_coord", tcb);
                }

                if(md.has_texture && md.tex_width > 0 && md.tex_height > 0){
                    OWLTexture tex = owlTexture2DCreate(context,
                                                         OWL_TEXEL_FORMAT_RGBA8,
                                                         md.tex_width, md.tex_height,
                                                         md.tex_pixels.data(),
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

                if (md.has_normal_map && md.normal_tex_width > 0 && md.normal_tex_height > 0) {
                    OWLTexture nm_tex = owlTexture2DCreate(context,
                                                           OWL_TEXEL_FORMAT_RGBA8,
                                                           md.normal_tex_width, md.normal_tex_height,
                                                           md.normal_tex_pixels.data(),
                                                           OWL_TEXTURE_LINEAR,
                                                           OWL_TEXTURE_WRAP,
                                                           OWL_TEXTURE_WRAP,
                                                           OWL_COLOR_SPACE_LINEAR);
                    owlGeomSetTexture(geom, "normal_map", nm_tex);
                    owlGeomSet1i(geom, "has_normal_map", 1);
                } else {
                    owlGeomSet1i(geom, "has_normal_map", 0);
                }

                if (md.has_metallic_roughness_map && md.mr_tex_width > 0 && md.mr_tex_height > 0) {
                    OWLTexture mr_tex = owlTexture2DCreate(context,
                                                           OWL_TEXEL_FORMAT_RGBA8,
                                                           md.mr_tex_width, md.mr_tex_height,
                                                           md.metallic_roughness_tex_pixels.data(),
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
                if (md.has_emissive_map && md.emissive_tex_width > 0 && md.emissive_tex_height > 0) {
                    OWLTexture em_tex = owlTexture2DCreate(context,
                                                           OWL_TEXEL_FORMAT_RGBA8,
                                                           md.emissive_tex_width, md.emissive_tex_height,
                                                           md.emissive_tex_pixels.data(),
                                                           OWL_TEXTURE_LINEAR,
                                                           OWL_TEXTURE_WRAP,
                                                           OWL_TEXTURE_WRAP,
                                                           OWL_COLOR_SPACE_SRGB);
                    owlGeomSetTexture(geom, "emissive_map", em_tex);
                    owlGeomSet1i(geom, "has_emissive_map", 1);
                } else {
                    owlGeomSet1i(geom, "has_emissive_map", 0);
                }

                if (md.has_occlusion_map && md.occlusion_tex_width > 0 && md.occlusion_tex_height > 0) {
                    OWLTexture ao_tex = owlTexture2DCreate(context,
                                                           OWL_TEXEL_FORMAT_RGBA8,
                                                           md.occlusion_tex_width, md.occlusion_tex_height,
                                                           md.occlusion_tex_pixels.data(),
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
        }

        OWLGroup triangles_group = owlTrianglesGeomGroupCreate(context, geoms.size(), geoms.data());
        owlGroupBuildAccel(triangles_group);
        OWLGroup world = owlInstanceGroupCreate(context, 1, &triangles_group);
        owlGroupBuildAccel(world);

        if constexpr (SPEC::HAS_RGB && (SPEC::SHADING::PBR_SHADING || SPEC::SHADING::METALLIC_REFLECTIONS)) {
            for(size_t m = 0; m < geoms.size(); m++){
                owlGeomSetGroup(geoms[m], "world", world);
            }
        }

        if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING) {
            OWLBuffer light_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(rendering::raytracing::SceneLight),
                                                            renderer.scene_lights.size(), renderer.scene_lights.data());
            for (size_t m = 0; m < geoms.size(); m++) {
                owlGeomSetBuffer(geoms[m], "scene_lights", light_buffer);
                owlGeomSet1i(geoms[m], "num_scene_lights", (int)renderer.scene_lights.size());
            }
        }

        if constexpr (SPEC::HAS_RGB) {
            owlRayGenSetGroup((OWLRayGen)renderer.backend.ray_gen, "world", world);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            const float max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
            owlRayGenSetGroup((OWLRayGen)renderer.backend.depth_ray_gen, "world", world);
            owlRayGenSet1f((OWLRayGen)renderer.backend.depth_ray_gen, "max_depth", max_depth);
        }
        if(renderer.backend.collision_ray_gen)
            owlRayGenSetGroup((OWLRayGen)renderer.backend.collision_ray_gen, "world", world);
        renderer.backend.world = world;
    }

    template <typename DEVICE, typename SPEC>
    void generate_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer,
                          const typename SPEC::T center[3], typename SPEC::T radius,
                          const typename SPEC::T up[3], typename SPEC::T fov){
        rendering::raytracing::detail::generate_camera_poses(device, renderer, center, radius, up, fov);

        OWLContext context = (OWLContext)renderer.backend.context;
        OWLBuffer cameras_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData),
                                                          SPEC::NUM_CAMERAS, data(renderer.cameras));
        if constexpr (SPEC::ENABLE_MOTION_BLUR) {
            OWLBuffer cameras_open_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData),
                                                                  SPEC::NUM_CAMERAS, data(renderer.cameras));
            if constexpr (SPEC::HAS_RGB) {
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_open", cameras_open_buffer);
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_close", cameras_buffer);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_open", cameras_open_buffer);
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_close", cameras_buffer);
            }
            renderer.backend.cameras_open_buffer = cameras_open_buffer;
        }
        else {
            if constexpr (SPEC::HAS_RGB) {
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras", cameras_buffer);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras", cameras_buffer);
            }
        }
        if(renderer.backend.collision_ray_gen)
            owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "cameras", cameras_buffer);
        renderer.backend.cameras_buffer = cameras_buffer;
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void set_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        static_assert(utils::typing::is_same_v<typename CAMERAS_SPEC::T, rendering::raytracing::CameraData<typename SPEC::T>>);
        static_assert(get<0>(typename CAMERAS_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);

        OWLContext context = (OWLContext)renderer.backend.context;

        if(renderer.backend.cameras_buffer == nullptr){
            renderer.backend.cameras_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras));
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                renderer.backend.cameras_open_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras));
                if constexpr (SPEC::HAS_RGB) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_open", (OWLBuffer)renderer.backend.cameras_open_buffer);
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_close", (OWLBuffer)renderer.backend.cameras_buffer);
                }
                if constexpr (SPEC::HAS_DEPTH) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_open", (OWLBuffer)renderer.backend.cameras_open_buffer);
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_close", (OWLBuffer)renderer.backend.cameras_buffer);
                }
            }
            else {
                if constexpr (SPEC::HAS_RGB) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras", (OWLBuffer)renderer.backend.cameras_buffer);
                }
                if constexpr (SPEC::HAS_DEPTH) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras", (OWLBuffer)renderer.backend.cameras_buffer);
                }
            }
            if(renderer.backend.collision_ray_gen)
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "cameras", (OWLBuffer)renderer.backend.cameras_buffer);
        }
        else{
            owlBufferUpload((OWLBuffer)renderer.backend.cameras_buffer, data(cameras), 0, SPEC::NUM_CAMERAS);
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                owlBufferUpload((OWLBuffer)renderer.backend.cameras_open_buffer, data(cameras), 0, SPEC::NUM_CAMERAS);
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_OPEN_SPEC, typename CAMERAS_CLOSE_SPEC>
    void set_motion_blur_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_OPEN_SPEC>& cameras_open, const Tensor<CAMERAS_CLOSE_SPEC>& cameras_close){
        static_assert(SPEC::ENABLE_MOTION_BLUR, "set_motion_blur_cameras requires a motion-blur renderer specification");
        static_assert(utils::typing::is_same_v<typename CAMERAS_OPEN_SPEC::T, rendering::raytracing::CameraData<typename SPEC::T>>);
        static_assert(utils::typing::is_same_v<typename CAMERAS_CLOSE_SPEC::T, rendering::raytracing::CameraData<typename SPEC::T>>);
        static_assert(get<0>(typename CAMERAS_OPEN_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<0>(typename CAMERAS_CLOSE_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);

        OWLContext context = (OWLContext)renderer.backend.context;

        if(renderer.backend.cameras_buffer == nullptr){
            renderer.backend.cameras_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras_close));
            renderer.backend.cameras_open_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras_open));
            if constexpr (SPEC::HAS_RGB) {
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_open", (OWLBuffer)renderer.backend.cameras_open_buffer);
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_close", (OWLBuffer)renderer.backend.cameras_buffer);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_open", (OWLBuffer)renderer.backend.cameras_open_buffer);
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_close", (OWLBuffer)renderer.backend.cameras_buffer);
            }
            if(renderer.backend.collision_ray_gen)
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "cameras", (OWLBuffer)renderer.backend.cameras_buffer);
        }
        else{
            owlBufferUpload((OWLBuffer)renderer.backend.cameras_open_buffer, data(cameras_open), 0, SPEC::NUM_CAMERAS);
            owlBufferUpload((OWLBuffer)renderer.backend.cameras_buffer, data(cameras_close), 0, SPEC::NUM_CAMERAS);
        }
    }

    // =========================================================================
    // generate_probe_directions: Fibonacci probe directions + upload
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void generate_probe_directions(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Probe rays disabled (RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS=1)");
        return;
#else
        std::vector<float> dirs = rendering::raytracing::detail::generate_probe_direction_vectors<SPEC>();

        OWLContext context = (OWLContext)renderer.backend.context;
        OWLBuffer probe_dirs_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(owl::vec3f),
                                                              SPEC::NUM_PROBES, dirs.data());
        owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "probe_directions", probe_dirs_buffer);
        renderer.backend.probe_dirs_buffer = probe_dirs_buffer;

        owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "results", (OWLBuffer)renderer.backend.collision_results_buffer);
        owlRayGenSet1i    ((OWLRayGen)renderer.backend.collision_ray_gen, "num_probes", SPEC::NUM_PROBES);
        owlRayGenSet1i    ((OWLRayGen)renderer.backend.collision_ray_gen, "num_cameras", SPEC::NUM_CAMERAS);
        owlRayGenSet1f    ((OWLRayGen)renderer.backend.collision_ray_gen, "max_dist", renderer.camera_radius * 2.0f);
#endif
    }

    // =========================================================================
    // build_pipeline: build programs, pipeline, and SBT for both contexts
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void build_pipeline(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        OWLContext context = (OWLContext)renderer.backend.context;

        owlBuildPrograms(context);
        owlBuildPipeline(context);
        owlBuildSBT(context);

        OWLParams launch_params = owlParamsCreate(context, 0, nullptr, 0);
        renderer.backend.launch_params = launch_params;
        if(renderer.backend.collision_ray_gen){
            OWLParams coll_lp = owlParamsCreate(context, 0, nullptr, 0);
            renderer.backend.coll_launch_params = coll_lp;
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        OWLParams launch_params = (OWLParams)renderer.backend.launch_params;
        if constexpr (SPEC::HAS_RGB) {
            OWLRayGen ray_gen = (OWLRayGen)renderer.backend.ray_gen;
            owlAsyncLaunch2D(ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
        }
        if constexpr (SPEC::HAS_DEPTH) {
            OWLRayGen depth_ray_gen = (OWLRayGen)renderer.backend.depth_ray_gen;
            owlAsyncLaunch2D(depth_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
        }
        if(renderer.backend.collision_ray_gen){
            OWLRayGen collision_ray_gen = (OWLRayGen)renderer.backend.collision_ray_gen;
            OWLParams coll_lp = (OWLParams)renderer.backend.coll_launch_params;
            owlAsyncLaunch2D(collision_ray_gen, SPEC::NUM_CAMERAS, SPEC::NUM_PROBES, coll_lp);
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        owlLaunchSync((OWLParams)renderer.backend.launch_params);
        if(renderer.backend.coll_launch_params)
            owlLaunchSync((OWLParams)renderer.backend.coll_launch_params);
    }

    template <typename DEVICE, typename SPEC>
    void render(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_launch(device, renderer);
        render_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        if(renderer.backend.collision_ray_gen){
            OWLRayGen collision_ray_gen = (OWLRayGen)renderer.backend.collision_ray_gen;
            OWLParams coll_lp = (OWLParams)renderer.backend.coll_launch_params;
            owlAsyncLaunch2D(collision_ray_gen, SPEC::NUM_CAMERAS, SPEC::NUM_PROBES, coll_lp);
        }
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        if(renderer.backend.coll_launch_params)
            owlLaunchSync((OWLParams)renderer.backend.coll_launch_params);
    }

    template <typename DEVICE, typename SPEC>
    void render_collision_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_collision_only_launch(device, renderer);
        render_collision_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "render_rgb_only requires an RGB-capable renderer specification");
        OWLRayGen ray_gen = (OWLRayGen)renderer.backend.ray_gen;
        OWLParams launch_params = (OWLParams)renderer.backend.launch_params;
        owlAsyncLaunch2D(ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "render_rgb_only requires an RGB-capable renderer specification");
        owlLaunchSync((OWLParams)renderer.backend.launch_params);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_rgb_only_launch(device, renderer);
        render_rgb_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "render_depth_only requires a depth-capable renderer specification");
        OWLRayGen depth_ray_gen = (OWLRayGen)renderer.backend.depth_ray_gen;
        OWLParams launch_params = (OWLParams)renderer.backend.launch_params;
        owlAsyncLaunch2D(depth_ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, launch_params);
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "render_depth_only requires a depth-capable renderer specification");
        owlLaunchSync((OWLParams)renderer.backend.launch_params);
    }

    template <typename DEVICE, typename SPEC>
    void render_depth_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_depth_only_launch(device, renderer);
        render_depth_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB && SPEC::HAS_DEPTH, "render_rgb_depth_only requires an RGBD renderer specification");
        render_rgb_only_launch(device, renderer);
        render_depth_only_launch(device, renderer);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB && SPEC::HAS_DEPTH, "render_rgb_depth_only requires an RGBD renderer specification");
        owlLaunchSync((OWLParams)renderer.backend.launch_params);
    }

    template <typename DEVICE, typename SPEC>
    void render_rgb_depth_only(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        render_rgb_depth_only_launch(device, renderer);
        render_rgb_depth_only_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void set_cameras_async(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        static_assert(utils::typing::is_same_v<typename CAMERAS_SPEC::T, rendering::raytracing::CameraData<typename SPEC::T>>);
        static_assert(get<0>(typename CAMERAS_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);

        OWLContext context = (OWLContext)renderer.backend.context;

        if(renderer.backend.cameras_buffer == nullptr){
            renderer.backend.cameras_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras));
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                renderer.backend.cameras_open_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras));
                if constexpr (SPEC::HAS_RGB) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_open", (OWLBuffer)renderer.backend.cameras_open_buffer);
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_close", (OWLBuffer)renderer.backend.cameras_buffer);
                }
                if constexpr (SPEC::HAS_DEPTH) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_open", (OWLBuffer)renderer.backend.cameras_open_buffer);
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_close", (OWLBuffer)renderer.backend.cameras_buffer);
                }
            }
            else {
                if constexpr (SPEC::HAS_RGB) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras", (OWLBuffer)renderer.backend.cameras_buffer);
                }
                if constexpr (SPEC::HAS_DEPTH) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras", (OWLBuffer)renderer.backend.cameras_buffer);
                }
            }
            if(renderer.backend.collision_ray_gen)
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "cameras", (OWLBuffer)renderer.backend.cameras_buffer);
        } else {
            OWLParams launch_params = (OWLParams)renderer.backend.launch_params;
            cudaStream_t stream = (cudaStream_t)owlParamsGetCudaStream(launch_params, 0);
            void* d_ptr = (void*)owlBufferGetPointer((OWLBuffer)renderer.backend.cameras_buffer, 0);
            cudaMemcpyAsync(d_ptr, data(cameras), SPEC::NUM_CAMERAS * sizeof(OptixCameraData), cudaMemcpyHostToDevice, stream);
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                void* d_open_ptr = (void*)owlBufferGetPointer((OWLBuffer)renderer.backend.cameras_open_buffer, 0);
                cudaMemcpyAsync(d_open_ptr, data(cameras), SPEC::NUM_CAMERAS * sizeof(OptixCameraData), cudaMemcpyHostToDevice, stream);
            }
        }
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
        constexpr typename SPEC::TI expected = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        cudaMemcpy(data(out_pixels), owlBufferGetPointer((OWLBuffer)renderer.backend.frame_buffer_handle, 0), expected * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    template <typename DEVICE, typename SPEC, typename DEPTH_SPEC>
    void read_depth_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<DEPTH_SPEC>& out_depth){
        static_assert(SPEC::HAS_DEPTH, "read_depth_buffer requires a depth-capable renderer specification");
        static_assert(utils::typing::is_same_v<typename DEPTH_SPEC::T, float>);
        static_assert(get<0>(typename DEPTH_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<1>(typename DEPTH_SPEC::SHAPE{}) == SPEC::CAM_HEIGHT);
        static_assert(get<2>(typename DEPTH_SPEC::SHAPE{}) == SPEC::CAM_WIDTH);
        constexpr typename SPEC::TI expected = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        cudaMemcpy(data(out_depth), owlBufferGetPointer((OWLBuffer)renderer.backend.depth_buffer_handle, 0), expected * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // =========================================================================
    // save_image: readback framebuffer, rearrange to grid, write PNG
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void save_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_RGB, "save_image requires an RGB-capable renderer specification");
        using TI = typename SPEC::TI;
        constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        const size_t fb_count = (size_t)SPEC::NUM_CAMERAS * cam_pixels;

        std::vector<uint32_t> fb_host(fb_count);
        cudaMemcpy(fb_host.data(),
                   owlBufferGetPointer((OWLBuffer)renderer.backend.frame_buffer_handle, 0),
                   fb_count * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        rendering::raytracing::detail::write_grid_png<SPEC>(fb_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth_image requires a depth-capable renderer specification");
        using TI = typename SPEC::TI;
        constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        const size_t depth_count = (size_t)SPEC::NUM_CAMERAS * cam_pixels;

        std::vector<float> depth_host(depth_count);
        cudaMemcpy(depth_host.data(),
                   owlBufferGetPointer((OWLBuffer)renderer.backend.depth_buffer_handle, 0),
                   depth_count * sizeof(float),
                   cudaMemcpyDeviceToHost);

        rendering::raytracing::detail::write_depth_grid_png<SPEC>(depth_host.data(), renderer.camera_radius, filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth requires a depth-capable renderer specification");
        constexpr size_t depth_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::vector<float> depth_host(depth_count);
        cudaMemcpy(depth_host.data(),
                   owlBufferGetPointer((OWLBuffer)renderer.backend.depth_buffer_handle, 0),
                   depth_count * sizeof(float),
                   cudaMemcpyDeviceToHost);
        rendering::raytracing::detail::write_depth_bin<SPEC>(depth_host.data(), filename);
    }

    // =========================================================================
    // save_probes: readback collision results, write binary
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void save_probes(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        RL_TOOLS_RENDERING_RAYTRACING_LOG("save_probes skipped: probe rays are disabled.");
        (void)filename;
        return;
#else
        const CollisionResult* probe_results =
            (const CollisionResult*)owlBufferGetPointer((OWLBuffer)renderer.backend.collision_results_buffer, 0);

        rendering::raytracing::detail::write_probes_bin_and_log<SPEC>(probe_results, filename);
#endif
    }


    // =========================================================================
    // read_collision_results: typed access to collision probe buffer
    // =========================================================================
    template <typename DEVICE, typename SPEC, typename COLL_SPEC>
    void read_collision_results(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<COLL_SPEC>& out){
        static_assert(utils::typing::is_same_v<typename COLL_SPEC::T, rendering::raytracing::CollisionResult>);
        static_assert(get<0>(typename COLL_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<1>(typename COLL_SPEC::SHAPE{}) == SPEC::NUM_PROBES);
#if !RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        if(renderer.backend.collision_results_buffer != nullptr){
            memcpy(data(out),
                   owlBufferGetPointer((OWLBuffer)renderer.backend.collision_results_buffer, 0),
                   SPEC::NUM_CAMERAS * SPEC::NUM_PROBES * sizeof(rendering::raytracing::CollisionResult));
        }
#endif
    }

    template <typename DEVICE, typename SPEC>
    const rendering::raytracing::CollisionResult* read_collision_results_raw(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        return nullptr;
#else
        if(renderer.backend.collision_results_buffer == nullptr){
            return nullptr;
        }
        return (const rendering::raytracing::CollisionResult*)owlBufferGetPointer((OWLBuffer)renderer.backend.collision_results_buffer, 0);
#endif
    }

    template <typename DEVICE, typename SPEC>
    uint32_t* get_framebuffer_device_ptr(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "get_framebuffer_device_ptr requires an RGB-capable renderer specification");
        return (uint32_t*)owlBufferGetPointer((OWLBuffer)renderer.backend.frame_buffer_handle, 0);
    }

    template <typename DEVICE, typename SPEC>
    float* get_depthbuffer_device_ptr(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "get_depthbuffer_device_ptr requires a depth-capable renderer specification");
        return (float*)owlBufferGetPointer((OWLBuffer)renderer.backend.depth_buffer_handle, 0);
    }

    template <typename DEVICE, typename SPEC>
    void synchronize(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        cudaDeviceSynchronize();
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        RL_TOOLS_RENDERING_RAYTRACING_LOG("destroying devicegroups ...");
        if(renderer.backend.context) owlContextDestroy((OWLContext)renderer.backend.context);
        renderer.backend.context = nullptr;
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
