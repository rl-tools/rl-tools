#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_OPERATIONS_CUDA_H

#include "../../renderer.h"
#include "device.h"

#include "owl/owl.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <nlohmann/json.hpp>

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

#define RL_TOOLS_RENDERING_RAYTRACING_LOG(message)                                            \
std::cout << OWL_TERMINAL_BLUE;                               \
std::cout << "#rl_tools::rendering::raytracing: " << message << std::endl;   \
std::cout << OWL_TERMINAL_DEFAULT;
#define RL_TOOLS_RENDERING_RAYTRACING_LOG_OK(message)                                         \
std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
std::cout << "#rl_tools::rendering::raytracing: " << message << std::endl;   \
std::cout << OWL_TERMINAL_DEFAULT;
#define RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR(message)                                        \
std::cerr << OWL_TERMINAL_RED;                                \
std::cerr << "#rl_tools::rendering::raytracing: " << message << std::endl;   \
std::cerr << OWL_TERMINAL_DEFAULT;

#ifndef RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 0
#endif

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
                return "TriangleMeshPBR";
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
        struct BasicShadingUsage {
            static constexpr bool USES_INDEX = SPEC::SHADING::LOAD_TEXTURES || SPEC::SHADING::NORMAL_SHADING || SPEC::SHADING::METALLIC_REFLECTIONS;
            static constexpr bool USES_VERTEX = SPEC::SHADING::NORMAL_SHADING || SPEC::SHADING::METALLIC_REFLECTIONS;
            static constexpr bool USES_TEXTURE = SPEC::SHADING::LOAD_TEXTURES;
            static constexpr bool USES_WORLD = SPEC::SHADING::METALLIC_REFLECTIONS;
        };
    }

    // =========================================================================
    // Default cube geometry
    // =========================================================================
    namespace rendering::raytracing::constants{
        const int NUM_VERTICES = 8;
        const owl::vec3f default_vertices[8] = {
            { -1.f,-1.f,-1.f },
            { +1.f,-1.f,-1.f },
            { -1.f,+1.f,-1.f },
            { +1.f,+1.f,-1.f },
            { -1.f,-1.f,+1.f },
            { +1.f,-1.f,+1.f },
            { -1.f,+1.f,+1.f },
            { +1.f,+1.f,+1.f }
        };
        const int NUM_INDICES = 12;
        const owl::vec3i default_indices[12] = {
            { 0,1,3 }, { 2,3,0 },
            { 5,7,6 }, { 5,6,4 },
            { 0,4,5 }, { 0,5,1 },
            { 2,3,7 }, { 2,7,6 },
            { 1,5,7 }, { 1,7,3 },
            { 4,0,2 }, { 4,2,6 }
        };
    }

    // =========================================================================
    // Decode an embedded texture from assimp into RGBA8 pixels
    // =========================================================================
    namespace rendering::raytracing{
        static bool decode_embedded_texture(const aiTexture* tex,
                                            std::vector<uint8_t>& pixels,
                                            int& w, int& h){
            if(tex->mHeight != 0){
                w = tex->mWidth;
                h = tex->mHeight;
                pixels.resize(w * h * 4);
                for(int i = 0; i < w * h; i++){
                    pixels[i*4+0] = tex->pcData[i].r;
                    pixels[i*4+1] = tex->pcData[i].g;
                    pixels[i*4+2] = tex->pcData[i].b;
                    pixels[i*4+3] = tex->pcData[i].a;
                }
                return true;
            } else {
                int channels;
                unsigned char* data = stbi_load_from_memory(
                    reinterpret_cast<const unsigned char*>(tex->pcData),
                    tex->mWidth, &w, &h, &channels, 4);
                if(!data) return false;
                pixels.assign(data, data + w * h * 4);
                stbi_image_free(data);
                return true;
            }
        }
    }

    namespace rendering::raytracing::glb{
        struct MaterialMeta {
            int alpha_mode = 0;
            float alpha_cutoff = 0.5f;
            float base_color_factor_alpha = 1.0f;
        };

        struct ParsedMetadata {
            std::vector<rendering::raytracing::SceneLight> lights;
            std::vector<MaterialMeta> materials;
        };

        static void node_world_transform(const nlohmann::json& nodes, int node_idx, const std::vector<int>& parent_map, float out[16]) {
            float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
            std::memcpy(out, identity, sizeof(identity));

            std::vector<int> chain;
            for (int cur = node_idx; cur >= 0; cur = parent_map[cur]) chain.push_back(cur);

            for (int i = (int)chain.size() - 1; i >= 0; i--) {
                const auto& node = nodes[chain[i]];
                float local[16];
                if (node.contains("matrix")) {
                    auto& m = node["matrix"];
                    for (int j = 0; j < 16; j++) local[j] = m[j].get<float>();
                } else {
                    float tx = 0, ty = 0, tz = 0;
                    float qx = 0, qy = 0, qz = 0, qw = 1;
                    float sx = 1, sy = 1, sz = 1;
                    if (node.contains("translation")) { auto& t = node["translation"]; tx = t[0]; ty = t[1]; tz = t[2]; }
                    if (node.contains("rotation")) { auto& r = node["rotation"]; qx = r[0]; qy = r[1]; qz = r[2]; qw = r[3]; }
                    if (node.contains("scale")) { auto& s = node["scale"]; sx = s[0]; sy = s[1]; sz = s[2]; }
                    float r00 = (1 - 2*(qy*qy + qz*qz)) * sx, r01 = 2*(qx*qy - qw*qz) * sy, r02 = 2*(qx*qz + qw*qy) * sz;
                    float r10 = 2*(qx*qy + qw*qz) * sx, r11 = (1 - 2*(qx*qx + qz*qz)) * sy, r12 = 2*(qy*qz - qw*qx) * sz;
                    float r20 = 2*(qx*qz - qw*qy) * sx, r21 = 2*(qy*qz + qw*qx) * sy, r22 = (1 - 2*(qx*qx + qy*qy)) * sz;
                    local[0]=r00; local[4]=r01; local[8]=r02;  local[12]=tx;
                    local[1]=r10; local[5]=r11; local[9]=r12;  local[13]=ty;
                    local[2]=r20; local[6]=r21; local[10]=r22; local[14]=tz;
                    local[3]=0;   local[7]=0;   local[11]=0;   local[15]=1;
                }
                float tmp[16];
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        tmp[r + c*4] = out[r]*local[c*4] + out[r+4]*local[c*4+1] + out[r+8]*local[c*4+2] + out[r+12]*local[c*4+3];
                std::memcpy(out, tmp, sizeof(tmp));
            }
        }

        static ParsedMetadata parse(const std::string& filename) {
            ParsedMetadata result;
            FILE* f = fopen(filename.c_str(), "rb");
            if (!f) return result;

            uint32_t header[3];
            if (fread(header, 4, 3, f) != 3 || header[0] != 0x46546C67u) { fclose(f); return result; }

            uint32_t chunk_header[2];
            if (fread(chunk_header, 4, 2, f) != 2) { fclose(f); return result; }
            uint32_t json_len = chunk_header[0];

            std::string json_str(json_len, '\0');
            if (fread(&json_str[0], 1, json_len, f) != json_len) { fclose(f); return result; }
            fclose(f);

            nlohmann::json gltf = nlohmann::json::parse(json_str, nullptr, false);
            if (gltf.is_discarded()) return result;

            auto& nodes = gltf["nodes"];
            std::vector<int> parent_map(nodes.size(), -1);
            for (int i = 0; i < (int)nodes.size(); i++) {
                if (nodes[i].contains("children")) {
                    for (auto& child : nodes[i]["children"]) parent_map[child.get<int>()] = i;
                }
            }

            if (gltf.contains("extensions") && gltf["extensions"].contains("KHR_lights_punctual")) {
                auto& light_defs = gltf["extensions"]["KHR_lights_punctual"]["lights"];
                for (int ni = 0; ni < (int)nodes.size(); ni++) {
                    auto& node = nodes[ni];
                    if (!node.contains("extensions") || !node["extensions"].contains("KHR_lights_punctual")) continue;
                    int light_idx = node["extensions"]["KHR_lights_punctual"]["light"].get<int>();
                    auto& ldef = light_defs[light_idx];

                    float intensity = ldef.value("intensity", 1.0f);
                    std::string type_str = ldef.value("type", "point");
                    float color_r = 1, color_g = 1, color_b = 1;
                    if (ldef.contains("color")) { color_r = ldef["color"][0]; color_g = ldef["color"][1]; color_b = ldef["color"][2]; }

                    float world[16];
                    node_world_transform(nodes, ni, parent_map, world);
                    float px = world[12], py = world[13], pz = world[14];

                    rendering::raytracing::SceneLight sl{};
                    if (type_str == "directional") sl.type = 0;
                    else if (type_str == "spot") sl.type = 2;
                    else sl.type = 1;

                    sl.position[0] = px; sl.position[1] = -pz; sl.position[2] = py;
                    sl.color[0] = color_r * intensity; sl.color[1] = color_g * intensity; sl.color[2] = color_b * intensity;
                    sl.attenuation_constant = 0.f; sl.attenuation_linear = 0.f; sl.attenuation_quadratic = 1.f;

                    if (type_str == "spot" && ldef.contains("spot")) {
                        float inner = ldef["spot"].value("innerConeAngle", 0.0f);
                        float outer = ldef["spot"].value("outerConeAngle", 0.7854f);
                        sl.cos_inner_cone = cosf(inner);
                        sl.cos_outer_cone = cosf(outer);
                        float dx = world[8], dy = world[9], dz = world[10];
                        float len = sqrtf(dx*dx + dy*dy + dz*dz);
                        if (len > 1e-6f) { dx /= len; dy /= len; dz /= len; }
                        sl.direction[0] = dx; sl.direction[1] = -dz; sl.direction[2] = dy;
                    }

                    result.lights.push_back(sl);
                }
            }

            if (gltf.contains("materials")) {
                for (auto& mat : gltf["materials"]) {
                    MaterialMeta mm;
                    std::string am = mat.value("alphaMode", "OPAQUE");
                    if (am == "MASK") mm.alpha_mode = 1;
                    else if (am == "BLEND") mm.alpha_mode = 2;
                    mm.alpha_cutoff = mat.value("alphaCutoff", 0.5f);
                    if (mat.contains("pbrMetallicRoughness") && mat["pbrMetallicRoughness"].contains("baseColorFactor")) {
                        auto& bcf = mat["pbrMetallicRoughness"]["baseColorFactor"];
                        if (bcf.size() >= 4) mm.base_color_factor_alpha = bcf[3].get<float>();
                    }
                    result.materials.push_back(mm);
                }
            }

            return result;
        }
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
        owlMissProgSet3f(miss_prog, "color_0", owl3f{.8f, 0.f, 0.f});
        owlMissProgSet3f(miss_prog, "color_1", owl3f{.8f, .8f, .8f});

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
            renderer.backend.owl_frame_buffer = frame_buffer;
        }
        if constexpr (SPEC::HAS_DEPTH) {
            renderer.backend.depth_ray_gen = depth_ray_gen;
            renderer.backend.owl_depth_buffer = depth_buffer;
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
        renderer.backend.owl_collision_results_buffer = collision_results_buffer;
#endif
    }

    // =========================================================================
    // load_model: Assimp model loading
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    bool load_model(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const std::string& filename){
        using T = typename SPEC::T;

        Assimp::Importer importer;
        unsigned int import_flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality;
        if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING) {
            import_flags |= aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace;
        } else if constexpr (SPEC::HAS_RGB && SPEC::SHADING::NORMAL_SHADING) {
            import_flags |= aiProcess_GenNormals;
        }
        const aiScene* scene = importer.ReadFile(filename, import_flags);

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode){
            RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Assimp error: " << importer.GetErrorString());
            return false;
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Loaded model with " << scene->mNumMeshes << " mesh(es)");

        owl::vec3f bbox_min(std::numeric_limits<float>::max());
        owl::vec3f bbox_max(std::numeric_limits<float>::lowest());

        [[maybe_unused]] std::map<std::string, size_t> tex_cache;
        struct DecodedTex { std::vector<uint8_t> pixels; int w, h; };
        [[maybe_unused]] std::vector<DecodedTex> decoded_textures;

        size_t total_verts = 0, total_tris = 0;

        renderer.meshes.clear();
        std::vector<unsigned int> mesh_source_indices;

        // Build mesh-index → global transform map by walking the node tree
        std::vector<std::vector<aiMatrix4x4>> mesh_transforms(scene->mNumMeshes);
        std::function<void(const aiNode*, const aiMatrix4x4&)> collect_transforms = [&](const aiNode* node, const aiMatrix4x4& parent_transform){
            aiMatrix4x4 global_transform = parent_transform * node->mTransformation;
            for(unsigned int i = 0; i < node->mNumMeshes; i++){
                mesh_transforms[node->mMeshes[i]].push_back(global_transform);
            }
            for(unsigned int i = 0; i < node->mNumChildren; i++){
                collect_transforms(node->mChildren[i], global_transform);
            }
        };
        collect_transforms(scene->mRootNode, aiMatrix4x4());

        for(unsigned int m = 0; m < scene->mNumMeshes; m++){
            const aiMesh* mesh = scene->mMeshes[m];
            auto& transforms = mesh_transforms[m];
            if(transforms.empty()){
                transforms.push_back(aiMatrix4x4());
            }
            for(const auto& global_transform : transforms){
            rendering::raytracing::MeshData<SPEC> md;
            [[maybe_unused]] const aiMaterial* mat = nullptr;
            if constexpr (SPEC::HAS_RGB) {
                if(mesh->mMaterialIndex < scene->mNumMaterials){
                    mat = scene->mMaterials[mesh->mMaterialIndex];
                }
            }

            // vertices: apply node transform, then GLB (Y-up) → FLU (Z-up)
            for(unsigned int v = 0; v < mesh->mNumVertices; v++){
                aiVector3D pos = mesh->mVertices[v];
                pos = global_transform * pos;
                float flu_x = pos.x, flu_y = -pos.z, flu_z = pos.y;
                md.vertices.push_back(flu_x);
                md.vertices.push_back(flu_y);
                md.vertices.push_back(flu_z);
                owl::vec3f vertex(flu_x, flu_y, flu_z);
                bbox_min = min(bbox_min, vertex);
                bbox_max = max(bbox_max, vertex);
            }

            if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING) {
                if (mesh->mNormals) {
                    aiMatrix3x3 normal_matrix(global_transform);
                    for (unsigned int v = 0; v < mesh->mNumVertices; v++) {
                        aiVector3D n = normal_matrix * mesh->mNormals[v];
                        n.Normalize();
                        md.normals.push_back(n.x);
                        md.normals.push_back(-n.z);
                        md.normals.push_back(n.y);
                    }
                }
            }

            // indices
            for(unsigned int f = 0; f < mesh->mNumFaces; f++){
                const aiFace& face = mesh->mFaces[f];
                if(face.mNumIndices == 3){
                    md.indices.push_back(face.mIndices[0]);
                    md.indices.push_back(face.mIndices[1]);
                    md.indices.push_back(face.mIndices[2]);
                }
            }

            if constexpr (SPEC::HAS_RGB && SPEC::SHADING::LOAD_TEXTURES) {
                unsigned int uv_channel = 0;
                if(mat != nullptr){
                    int uv_candidate = 0;
                    if(mat->Get(AI_MATKEY_UVWSRC(aiTextureType_BASE_COLOR, 0), uv_candidate) == AI_SUCCESS && uv_candidate >= 0){
                        uv_channel = (unsigned int)uv_candidate;
                    }
                    else if(mat->Get(AI_MATKEY_UVWSRC(aiTextureType_DIFFUSE, 0), uv_candidate) == AI_SUCCESS && uv_candidate >= 0){
                        uv_channel = (unsigned int)uv_candidate;
                    }
                }
                if(uv_channel >= AI_MAX_NUMBER_OF_TEXTURECOORDS || !mesh->mTextureCoords[uv_channel]){
                    for(unsigned int channel_i = 0; channel_i < AI_MAX_NUMBER_OF_TEXTURECOORDS; channel_i++){
                        if(mesh->mTextureCoords[channel_i]){
                            uv_channel = channel_i;
                            break;
                        }
                    }
                }
                if(mesh->mTextureCoords[uv_channel]){
                    for(unsigned int v = 0; v < mesh->mNumVertices; v++){
                        const aiVector3D& tc = mesh->mTextureCoords[uv_channel][v];
                        md.tex_coords.push_back(tc.x);
                        if constexpr (SPEC::SHADING::PBR_SHADING) {
                            md.tex_coords.push_back(1.0f - tc.y);
                        } else {
                            md.tex_coords.push_back(tc.y);
                        }
                    }
                }
            }

            // material / texture
            if constexpr (SPEC::SHADING::PBR_SHADING) {
                md.color[0] = 1.0f; md.color[1] = 1.0f; md.color[2] = 1.0f;
            } else {
                md.color[0] = 0.8f; md.color[1] = 0.8f; md.color[2] = 0.8f;
            }
            if constexpr (SPEC::HAS_RGB) {
            if(mat != nullptr){

                if constexpr (SPEC::SHADING::PBR_SHADING) {
                    aiColor4D base_color(1.0f, 1.0f, 1.0f, 1.0f);
                    if (aiGetMaterialColor(mat, AI_MATKEY_BASE_COLOR, &base_color) == AI_SUCCESS) {
                        md.color[0] = base_color.r; md.color[1] = base_color.g; md.color[2] = base_color.b;
                    } else {
                        aiColor4D diffuse;
                        if (aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diffuse) == AI_SUCCESS) {
                            md.color[0] = diffuse.r; md.color[1] = diffuse.g; md.color[2] = diffuse.b;
                        }
                    }
                } else {
                    aiColor4D diffuse;
                    if(aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diffuse) == AI_SUCCESS){
                        md.color[0] = diffuse.r; md.color[1] = diffuse.g; md.color[2] = diffuse.b;
                    }
                    else if constexpr (!SPEC::SHADING::LOAD_TEXTURES) {
                        aiColor4D base_color;
                        if (aiGetMaterialColor(mat, AI_MATKEY_BASE_COLOR, &base_color) == AI_SUCCESS) {
                            md.color[0] = base_color.r; md.color[1] = base_color.g; md.color[2] = base_color.b;
                        }
                    }
                }

                if constexpr (SPEC::SHADING::LOAD_TEXTURES) {
                if(mat->GetTextureCount(aiTextureType_DIFFUSE) > 0){
                    aiString tex_path;
                    if(mat->GetTexture(aiTextureType_DIFFUSE, 0, &tex_path) == AI_SUCCESS){
                        std::string path_str(tex_path.C_Str());

                        auto it = tex_cache.find(path_str);
                        if(it != tex_cache.end()){
                            auto& cached = decoded_textures[it->second];
                            md.tex_pixels = cached.pixels;
                            md.tex_width = cached.w;
                            md.tex_height = cached.h;
                            md.has_texture = true;
                        } else {
                            const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                            if(emb_tex){
                                int w, h;
                                std::vector<uint8_t> pixels;
                                if(rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)){
                                    md.tex_pixels = pixels;
                                    md.tex_width = w;
                                    md.tex_height = h;
                                    md.has_texture = true;
                                    tex_cache[path_str] = decoded_textures.size();
                                    decoded_textures.push_back({std::move(pixels), w, h});
                                }
                            } else if(!path_str.empty()){
                                int w, h, channels;
                                unsigned char* data = stbi_load(path_str.c_str(), &w, &h, &channels, 4);
                                if(data){
                                    md.tex_pixels.assign(data, data + w * h * 4);
                                    md.tex_width = w;
                                    md.tex_height = h;
                                    md.has_texture = true;
                                    stbi_image_free(data);
                                    tex_cache[path_str] = decoded_textures.size();
                                    decoded_textures.push_back({md.tex_pixels, w, h});
                                }
                            }
                        }
                    }
                }
                }

                if constexpr (SPEC::SHADING::METALLIC_REFLECTIONS || SPEC::SHADING::PBR_SHADING) {
                float metallic_factor = 0.0f;
                mat->Get(AI_MATKEY_METALLIC_FACTOR, metallic_factor);
                md.metallic = metallic_factor;
                }

                if constexpr (SPEC::SHADING::PBR_SHADING) {
                    float metallic_factor_pbr = 1.0f;
                    mat->Get(AI_MATKEY_METALLIC_FACTOR, metallic_factor_pbr);
                    md.metallic = metallic_factor_pbr;

                    float roughness_factor = 1.0f;
                    mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness_factor);
                    md.roughness = roughness_factor;

                    if constexpr (SPEC::SHADING::LOAD_TEXTURES) {
                    if (mat->GetTextureCount(aiTextureType_NORMALS) > 0) {
                        aiString tex_path;
                        if (mat->GetTexture(aiTextureType_NORMALS, 0, &tex_path) == AI_SUCCESS) {
                            const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                            if (emb_tex) {
                                int w, h;
                                std::vector<uint8_t> pixels;
                                if (rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)) {
                                    md.normal_tex_pixels = std::move(pixels);
                                    md.normal_tex_width = w;
                                    md.normal_tex_height = h;
                                    md.has_normal_map = true;
                                }
                            }
                        }
                    }

                    if (mat->GetTextureCount(aiTextureType_UNKNOWN) > 0) {
                        aiString tex_path;
                        if (mat->GetTexture(aiTextureType_UNKNOWN, 0, &tex_path) == AI_SUCCESS) {
                            const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                            if (emb_tex) {
                                int w, h;
                                std::vector<uint8_t> pixels;
                                if (rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)) {
                                    md.metallic_roughness_tex_pixels = std::move(pixels);
                                    md.mr_tex_width = w;
                                    md.mr_tex_height = h;
                                    md.has_metallic_roughness_map = true;
                                }
                            }
                        }
                    }
                    }

                    aiColor3D emissive_color(0.f, 0.f, 0.f);
                    mat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive_color);
                    md.emissive[0] = emissive_color.r; md.emissive[1] = emissive_color.g; md.emissive[2] = emissive_color.b;

                    if constexpr (SPEC::SHADING::LOAD_TEXTURES) {
                    if (mat->GetTextureCount(aiTextureType_EMISSIVE) > 0) {
                        aiString tex_path;
                        if (mat->GetTexture(aiTextureType_EMISSIVE, 0, &tex_path) == AI_SUCCESS) {
                            const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                            if (emb_tex) {
                                int w, h;
                                std::vector<uint8_t> pixels;
                                if (rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)) {
                                    md.emissive_tex_pixels = std::move(pixels);
                                    md.emissive_tex_width = w;
                                    md.emissive_tex_height = h;
                                    md.has_emissive_map = true;
                                }
                            }
                        }
                    }

                    if (mat->GetTextureCount(aiTextureType_AMBIENT_OCCLUSION) > 0) {
                        aiString tex_path;
                        if (mat->GetTexture(aiTextureType_AMBIENT_OCCLUSION, 0, &tex_path) == AI_SUCCESS) {
                            const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                            if (emb_tex) {
                                int w, h;
                                std::vector<uint8_t> pixels;
                                if (rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)) {
                                    md.occlusion_tex_pixels = std::move(pixels);
                                    md.occlusion_tex_width = w;
                                    md.occlusion_tex_height = h;
                                    md.has_occlusion_map = true;
                                }
                            }
                        }
                    } else if (mat->GetTextureCount(aiTextureType_LIGHTMAP) > 0) {
                        aiString tex_path;
                        if (mat->GetTexture(aiTextureType_LIGHTMAP, 0, &tex_path) == AI_SUCCESS) {
                            const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                            if (emb_tex) {
                                int w, h;
                                std::vector<uint8_t> pixels;
                                if (rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)) {
                                    md.occlusion_tex_pixels = std::move(pixels);
                                    md.occlusion_tex_width = w;
                                    md.occlusion_tex_height = h;
                                    md.has_occlusion_map = true;
                                }
                            }
                        }
                    }
                    }
                    float opacity_val = 1.0f;
                    mat->Get(AI_MATKEY_OPACITY, opacity_val);
                    float transmission_factor = 0.0f;
                    mat->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmission_factor);
                    if (transmission_factor > 0.0f) {
                        opacity_val = fminf(opacity_val, 1.0f - transmission_factor);
                    }
                    md.opacity = opacity_val;
                }

                if constexpr (SPEC::SHADING::LOAD_TEXTURES) {
                if(!md.has_texture && mat->GetTextureCount(aiTextureType_BASE_COLOR) > 0){
                    aiString tex_path;
                    if(mat->GetTexture(aiTextureType_BASE_COLOR, 0, &tex_path) == AI_SUCCESS){
                        const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                        if(emb_tex){
                            int w, h;
                            std::vector<uint8_t> pixels;
                            if(rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)){
                                md.tex_pixels = pixels;
                                md.tex_width = w;
                                md.tex_height = h;
                                md.has_texture = true;
                            }
                        }
                    }
                }
                }
            }
            }

            total_verts += md.vertices.size() / 3;
            total_tris += md.indices.size() / 3;
            mesh_source_indices.push_back(m);
            renderer.meshes.push_back(std::move(md));
            } // end for global_transform
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Total vertices: " << total_verts << ", triangles: " << total_tris);
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Bounding box: [" << bbox_min.x << "," << bbox_min.y << "," << bbox_min.z << "] - ["
              << bbox_max.x << "," << bbox_max.y << "," << bbox_max.z << "]");

        int textured_count = 0;
        int metallic_count = 0;
        for(auto& m : renderer.meshes){
            if(m.has_texture) textured_count++;
            if(m.metallic > 0.f) metallic_count++;
        }
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Meshes with textures: " << textured_count << "/" << renderer.meshes.size()
              << ", metallic: " << metallic_count << "/" << renderer.meshes.size());

        if constexpr (SPEC::HAS_RGB && SPEC::SHADING::PBR_SHADING) {
            renderer.scene_lights.clear();
            float inv_sqrt2 = 0.70710678f;
            renderer.scene_lights.push_back({0, {0,0,0}, {-inv_sqrt2, 0.f, inv_sqrt2}, {0.4f, 0.4f, 0.4f}, 0,0,0, 0,0});
            renderer.scene_lights.push_back({0, {0,0,0}, {0.f, -inv_sqrt2, inv_sqrt2}, {0.3f, 0.3f, 0.3f}, 0,0,0, 0,0});
            renderer.scene_lights.push_back({0, {0,0,0}, {0.f, inv_sqrt2, inv_sqrt2}, {0.2f, 0.2f, 0.2f}, 0,0,0, 0,0});

            auto glb_meta = rendering::raytracing::glb::parse(filename);
            for (auto& sl : glb_meta.lights) {
                renderer.scene_lights.push_back(sl);
            }
            RL_TOOLS_RENDERING_RAYTRACING_LOG("Scene lights: " << glb_meta.lights.size() << " from GLB + 3 directional fill");
            for (size_t li = 0; li < glb_meta.lights.size(); li++) {
                auto& sl = glb_meta.lights[li];
                RL_TOOLS_RENDERING_RAYTRACING_LOG("  light " << li << ": pos=(" << sl.position[0] << "," << sl.position[1] << "," << sl.position[2]
                    << ") color=(" << sl.color[0] << "," << sl.color[1] << "," << sl.color[2] << ")");
            }

            for (size_t mi = 0; mi < renderer.meshes.size(); mi++) {
                auto& md = renderer.meshes[mi];
                unsigned int mat_idx = scene->mMeshes[mesh_source_indices[mi]]->mMaterialIndex;
                if (mat_idx < glb_meta.materials.size()) {
                    auto& mm = glb_meta.materials[mat_idx];
                    md.alpha_mode = mm.alpha_mode;
                    md.alpha_cutoff = mm.alpha_cutoff;
                    if (mm.alpha_mode == 2) {
                        md.opacity = fminf(md.opacity, mm.base_color_factor_alpha);
                    }
                }
            }
        }

        // Adjust camera based on bounding box
        owl::vec3f center = 0.5f * (bbox_min + bbox_max);
        owl::vec3f size = bbox_max - bbox_min;
        float max_dim = std::max({size.x, size.y, size.z});
        owl::vec3f look_from = center + owl::vec3f(max_dim * 1.5f, max_dim * 1.5f, max_dim * 0.8f);
        renderer.scene_center[0] = center.x;
        renderer.scene_center[1] = center.y;
        renderer.scene_center[2] = center.z;
        renderer.scene_half_extent[0] = size.x * 0.5f;
        renderer.scene_half_extent[1] = size.y * 0.5f;
        renderer.scene_half_extent[2] = size.z * 0.5f;
        renderer.camera_radius = length(look_from - center);
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Camera positioned at [" << look_from.x << "," << look_from.y << "," << look_from.z << "]");

        return true;
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
                owlGeomTypeSetClosestHit(triangles_geom_type, 0, module, "TriangleMeshPBR");
            } else {
                using SHADING_USAGE = rendering::raytracing::detail::BasicShadingUsage<SPEC>;
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
                using SHADING_USAGE = rendering::raytracing::detail::BasicShadingUsage<SPEC>;
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
                owlGeomSet3f(geom, "ambient_color", owl3f{0.5f, 0.5f, 0.5f});
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

    namespace rendering::raytracing::vec3{
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void sub(const T a[3], const T b[3], T out[3]){
            out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T dot(const T a[3], const T b[3]){
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void cross(const T a[3], const T b[3], T out[3]){
            out[0] = a[1]*b[2] - a[2]*b[1];
            out[1] = a[2]*b[0] - a[0]*b[2];
            out[2] = a[0]*b[1] - a[1]*b[0];
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T length(const T v[3]){
            return sqrtf(dot(v, v));
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void normalize(const T v[3], T out[3]){
            T len = length(v);
            out[0] = v[0]/len; out[1] = v[1]/len; out[2] = v[2]/len;
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void scale(const T v[3], T s, T out[3]){
            out[0] = v[0]*s; out[1] = v[1]*s; out[2] = v[2]*s;
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void add(const T a[3], const T b[3], T out[3]){
            out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT void cross_normalized(const T a[3], const T b[3], T out[3]){
            T tmp[3];
            cross(a, b, tmp);
            normalize(tmp, out);
        }
    }

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::CameraData<T> make_camera_data(const T position[3], const T look_at[3], const T up[3], T fov, T aspect){
        namespace v3 = rendering::raytracing::vec3;
        T raw_dir[3], dir[3];
        v3::sub(look_at, position, raw_dir);
        v3::normalize(raw_dir, dir);

        T image_plane_scale = T{2} * tanf(fov / T{2});

        T du_dir[3], du[3];
        v3::cross_normalized(dir, up, du_dir);
        v3::scale(du_dir, image_plane_scale, du);

        T dv_dir[3], dv[3];
        v3::cross_normalized(du, dir, dv_dir);
        v3::scale(dv_dir, image_plane_scale / aspect, dv);

        T half_du[3], half_dv[3], tmp[3];
        v3::scale(du, T{-0.5}, half_du);
        v3::scale(dv, T{0.5}, half_dv);
        v3::add(dir, half_du, tmp);
        rendering::raytracing::CameraData<T> cam;
        v3::add(tmp, half_dv, cam.dir_00);
        cam.pos[0] = position[0]; cam.pos[1] = position[1]; cam.pos[2] = position[2];
        cam.dir_du[0] = du[0]; cam.dir_du[1] = du[1]; cam.dir_du[2] = du[2];
        v3::scale(dv, T{-1}, cam.dir_dv);
        return cam;
    }

    template <typename DEVICE, typename SPEC>
    void generate_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer,
                          const typename SPEC::T center[3], typename SPEC::T radius,
                          const typename SPEC::T up[3], typename SPEC::T fov){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        const T golden_ratio = (T{1} + sqrtf(T{5})) / T{2};
        const T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;

        for(TI i = 0; i < SPEC::NUM_CAMERAS; i++){
            T theta = T{2} * (T)M_PI * i / golden_ratio;
            T cos_inc = T{1} - T{2} * (i + T{0.5}) / SPEC::NUM_CAMERAS;
            cos_inc = cos_inc * T{0.85};
            T sin_inc = sqrtf(T{1} - cos_inc * cos_inc);

            T cam_pos[3] = {
                center[0] + radius * sin_inc * cosf(theta),
                center[1] + radius * sin_inc * sinf(theta),
                center[2] + radius * cos_inc
            };

            if(cam_pos[2] < center[2] - radius * T{0.1})
                cam_pos[2] = center[2] + radius * T{0.3};

            set(device, renderer.cameras, make_camera_data(cam_pos, center, up, fov, aspect), i);
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Generated " << SPEC::NUM_CAMERAS << " camera positions");
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Per-camera resolution: " << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT);

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
            renderer.backend.owl_cameras_open_buffer = cameras_open_buffer;
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
        renderer.backend.owl_cameras_buffer = cameras_buffer;
    }

    template <typename DEVICE, typename SPEC, typename CAMERAS_SPEC>
    void set_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const Tensor<CAMERAS_SPEC>& cameras){
        static_assert(utils::typing::is_same_v<typename CAMERAS_SPEC::T, rendering::raytracing::CameraData<typename SPEC::T>>);
        static_assert(get<0>(typename CAMERAS_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);

        OWLContext context = (OWLContext)renderer.backend.context;

        if(renderer.backend.owl_cameras_buffer == nullptr){
            renderer.backend.owl_cameras_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras));
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                renderer.backend.owl_cameras_open_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras));
                if constexpr (SPEC::HAS_RGB) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_open", (OWLBuffer)renderer.backend.owl_cameras_open_buffer);
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_close", (OWLBuffer)renderer.backend.owl_cameras_buffer);
                }
                if constexpr (SPEC::HAS_DEPTH) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_open", (OWLBuffer)renderer.backend.owl_cameras_open_buffer);
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_close", (OWLBuffer)renderer.backend.owl_cameras_buffer);
                }
            }
            else {
                if constexpr (SPEC::HAS_RGB) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras", (OWLBuffer)renderer.backend.owl_cameras_buffer);
                }
                if constexpr (SPEC::HAS_DEPTH) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras", (OWLBuffer)renderer.backend.owl_cameras_buffer);
                }
            }
            if(renderer.backend.collision_ray_gen)
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "cameras", (OWLBuffer)renderer.backend.owl_cameras_buffer);
        }
        else{
            owlBufferUpload((OWLBuffer)renderer.backend.owl_cameras_buffer, data(cameras), 0, SPEC::NUM_CAMERAS);
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                owlBufferUpload((OWLBuffer)renderer.backend.owl_cameras_open_buffer, data(cameras), 0, SPEC::NUM_CAMERAS);
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

        if(renderer.backend.owl_cameras_buffer == nullptr){
            renderer.backend.owl_cameras_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras_close));
            renderer.backend.owl_cameras_open_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras_open));
            if constexpr (SPEC::HAS_RGB) {
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_open", (OWLBuffer)renderer.backend.owl_cameras_open_buffer);
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_close", (OWLBuffer)renderer.backend.owl_cameras_buffer);
            }
            if constexpr (SPEC::HAS_DEPTH) {
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_open", (OWLBuffer)renderer.backend.owl_cameras_open_buffer);
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_close", (OWLBuffer)renderer.backend.owl_cameras_buffer);
            }
            if(renderer.backend.collision_ray_gen)
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "cameras", (OWLBuffer)renderer.backend.owl_cameras_buffer);
        }
        else{
            owlBufferUpload((OWLBuffer)renderer.backend.owl_cameras_open_buffer, data(cameras_open), 0, SPEC::NUM_CAMERAS);
            owlBufferUpload((OWLBuffer)renderer.backend.owl_cameras_buffer, data(cameras_close), 0, SPEC::NUM_CAMERAS);
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
        using TI = typename SPEC::TI;

        std::vector<owl::vec3f> dirs;
        dirs.reserve(SPEC::NUM_PROBES);

        const float golden_ratio = (1.0f + sqrtf(5.0f)) / 2.0f;

        for(int i = 0; i < (int)SPEC::NUM_PROBES; i++){
            float theta = 2.0f * (float)M_PI * i / golden_ratio;
            float cos_inc = 1.0f - 2.0f * (i + 0.5f) / SPEC::NUM_PROBES;
            float sin_inc = sqrtf(1.0f - cos_inc * cos_inc);

            dirs.push_back(normalize(owl::vec3f(sin_inc * cosf(theta),
                                           sin_inc * sinf(theta),
                                           cos_inc)));
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Generated " << dirs.size() << " probe directions per camera");

        OWLContext context = (OWLContext)renderer.backend.context;
        OWLBuffer probe_dirs_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(owl::vec3f),
                                                              dirs.size(), dirs.data());
        owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "probe_directions", probe_dirs_buffer);
        renderer.backend.probe_dirs_buffer = probe_dirs_buffer;

        owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "results", (OWLBuffer)renderer.backend.owl_collision_results_buffer);
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

        if(renderer.backend.owl_cameras_buffer == nullptr){
            renderer.backend.owl_cameras_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras));
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                renderer.backend.owl_cameras_open_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixCameraData), SPEC::NUM_CAMERAS, data(cameras));
                if constexpr (SPEC::HAS_RGB) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_open", (OWLBuffer)renderer.backend.owl_cameras_open_buffer);
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras_close", (OWLBuffer)renderer.backend.owl_cameras_buffer);
                }
                if constexpr (SPEC::HAS_DEPTH) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_open", (OWLBuffer)renderer.backend.owl_cameras_open_buffer);
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras_close", (OWLBuffer)renderer.backend.owl_cameras_buffer);
                }
            }
            else {
                if constexpr (SPEC::HAS_RGB) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.ray_gen, "cameras", (OWLBuffer)renderer.backend.owl_cameras_buffer);
                }
                if constexpr (SPEC::HAS_DEPTH) {
                    owlRayGenSetBuffer((OWLRayGen)renderer.backend.depth_ray_gen, "cameras", (OWLBuffer)renderer.backend.owl_cameras_buffer);
                }
            }
            if(renderer.backend.collision_ray_gen)
                owlRayGenSetBuffer((OWLRayGen)renderer.backend.collision_ray_gen, "cameras", (OWLBuffer)renderer.backend.owl_cameras_buffer);
        } else {
            OWLParams launch_params = (OWLParams)renderer.backend.launch_params;
            cudaStream_t stream = (cudaStream_t)owlParamsGetCudaStream(launch_params, 0);
            void* d_ptr = (void*)owlBufferGetPointer((OWLBuffer)renderer.backend.owl_cameras_buffer, 0);
            cudaMemcpyAsync(d_ptr, data(cameras), SPEC::NUM_CAMERAS * sizeof(OptixCameraData), cudaMemcpyHostToDevice, stream);
            if constexpr (SPEC::ENABLE_MOTION_BLUR) {
                void* d_open_ptr = (void*)owlBufferGetPointer((OWLBuffer)renderer.backend.owl_cameras_open_buffer, 0);
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
        cudaMemcpy(data(out_pixels), owlBufferGetPointer((OWLBuffer)renderer.backend.owl_frame_buffer, 0), expected * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    template <typename DEVICE, typename SPEC, typename DEPTH_SPEC>
    void read_depth_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, Tensor<DEPTH_SPEC>& out_depth){
        static_assert(SPEC::HAS_DEPTH, "read_depth_buffer requires a depth-capable renderer specification");
        static_assert(utils::typing::is_same_v<typename DEPTH_SPEC::T, float>);
        static_assert(get<0>(typename DEPTH_SPEC::SHAPE{}) == SPEC::NUM_CAMERAS);
        static_assert(get<1>(typename DEPTH_SPEC::SHAPE{}) == SPEC::CAM_HEIGHT);
        static_assert(get<2>(typename DEPTH_SPEC::SHAPE{}) == SPEC::CAM_WIDTH);
        constexpr typename SPEC::TI expected = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        cudaMemcpy(data(out_depth), owlBufferGetPointer((OWLBuffer)renderer.backend.owl_depth_buffer, 0), expected * sizeof(float), cudaMemcpyDeviceToHost);
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
                   owlBufferGetPointer((OWLBuffer)renderer.backend.owl_frame_buffer, 0),
                   fb_count * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        const uint32_t* fb = fb_host.data();

        constexpr int grid_width = SPEC::GRID_COLS * SPEC::CAM_WIDTH;
        constexpr int grid_height = SPEC::GRID_ROWS * SPEC::CAM_HEIGHT;
        std::vector<uint32_t> grid_image(grid_width * grid_height, 0);

        for(int i = 0; i < (int)SPEC::NUM_CAMERAS; i++){
            int col = i % SPEC::GRID_COLS;
            int row = i / SPEC::GRID_COLS;
            int offset_x = col * SPEC::CAM_WIDTH;
            int offset_y = row * SPEC::CAM_HEIGHT;

            for(int y = 0; y < (int)SPEC::CAM_HEIGHT; y++){
                memcpy(&grid_image[(offset_y + y) * grid_width + offset_x],
                       &fb[i * cam_pixels + y * SPEC::CAM_WIDTH],
                       SPEC::CAM_WIDTH * sizeof(uint32_t));
            }
        }

        stbi_write_png(filename, grid_width, grid_height, 4,
                       grid_image.data(), grid_width * sizeof(uint32_t));
        RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("Written grid image (" << SPEC::GRID_COLS << "x" << SPEC::GRID_ROWS
               << " cameras, " << grid_width << "x" << grid_height << " px) to " << filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth_image requires a depth-capable renderer specification");
        using TI = typename SPEC::TI;
        constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        const size_t depth_count = (size_t)SPEC::NUM_CAMERAS * cam_pixels;

        std::vector<float> depth_host(depth_count);
        cudaMemcpy(depth_host.data(),
                   owlBufferGetPointer((OWLBuffer)renderer.backend.owl_depth_buffer, 0),
                   depth_count * sizeof(float),
                   cudaMemcpyDeviceToHost);

        constexpr int grid_width = SPEC::GRID_COLS * SPEC::CAM_WIDTH;
        constexpr int grid_height = SPEC::GRID_ROWS * SPEC::CAM_HEIGHT;
        std::vector<uint32_t> grid_image(grid_width * grid_height, 0);
        const float max_depth = renderer.camera_radius > 0 ? renderer.camera_radius * 2.0f : 1e30f;
        const float valid_max_depth = max_depth * 0.999f;
        float min_valid_depth = std::numeric_limits<float>::max();
        float max_valid_depth = std::numeric_limits<float>::lowest();
        for(const float depth : depth_host){
            if(std::isfinite(depth) && depth > 0.f && depth < valid_max_depth){
                min_valid_depth = std::min(min_valid_depth, depth);
                max_valid_depth = std::max(max_valid_depth, depth);
            }
        }
        const bool has_valid_depth = min_valid_depth <= max_valid_depth;
        const float valid_depth_range = has_valid_depth ? max_valid_depth - min_valid_depth : 0.f;

        for(int i = 0; i < (int)SPEC::NUM_CAMERAS; i++){
            int col = i % SPEC::GRID_COLS;
            int row = i / SPEC::GRID_COLS;
            int offset_x = col * SPEC::CAM_WIDTH;
            int offset_y = row * SPEC::CAM_HEIGHT;

            for(int y = 0; y < (int)SPEC::CAM_HEIGHT; y++){
                for(int x = 0; x < (int)SPEC::CAM_WIDTH; x++){
                    const float depth = depth_host[i * cam_pixels + y * SPEC::CAM_WIDTH + x];
                    uint8_t value = 0;
                    if(has_valid_depth && std::isfinite(depth) && depth > 0.f && depth < valid_max_depth){
                        const float normalized = fminf(fmaxf((depth - min_valid_depth) / (valid_depth_range + 1e-6f), 0.f), 1.f);
                        value = static_cast<uint8_t>((1.f - normalized) * 255.f);
                    }
                    grid_image[(offset_y + y) * grid_width + offset_x + x] =
                        (0xFFu << 24) | (uint32_t(value) << 16) | (uint32_t(value) << 8) | uint32_t(value);
                }
            }
        }

        stbi_write_png(filename, grid_width, grid_height, 4,
                       grid_image.data(), grid_width * sizeof(uint32_t));
        RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("Written depth image (" << SPEC::GRID_COLS << "x" << SPEC::GRID_ROWS
               << " cameras, " << grid_width << "x" << grid_height << " px) to " << filename);
    }

    template <typename DEVICE, typename SPEC>
    void save_depth(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        static_assert(SPEC::HAS_DEPTH, "save_depth requires a depth-capable renderer specification");
        constexpr size_t depth_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        std::vector<float> depth_host(depth_count);
        cudaMemcpy(depth_host.data(),
                   owlBufferGetPointer((OWLBuffer)renderer.backend.owl_depth_buffer, 0),
                   depth_count * sizeof(float),
                   cudaMemcpyDeviceToHost);
        FILE* f = fopen(filename, "wb");
        if(f){
            int nc = SPEC::NUM_CAMERAS;
            int h = SPEC::CAM_HEIGHT;
            int w = SPEC::CAM_WIDTH;
            fwrite(&nc, sizeof(int), 1, f);
            fwrite(&h, sizeof(int), 1, f);
            fwrite(&w, sizeof(int), 1, f);
            fwrite(depth_host.data(), sizeof(float), depth_count, f);
            fclose(f);
            RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("Written depth data (" << depth_count << " values) to " << filename);
        }
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
            (const CollisionResult*)owlBufferGetPointer((OWLBuffer)renderer.backend.owl_collision_results_buffer, 0);

        int total_hits = 0;
        float min_hit_dist = 1e30f, max_hit_dist = 0.f;
        double sum_hit_dist = 0.0;
        for(int i = 0; i < (int)(SPEC::NUM_CAMERAS * SPEC::NUM_PROBES); i++){
            if(probe_results[i].hit){
                total_hits++;
                sum_hit_dist += probe_results[i].distance;
                if(probe_results[i].distance < min_hit_dist)
                    min_hit_dist = probe_results[i].distance;
                if(probe_results[i].distance > max_hit_dist)
                    max_hit_dist = probe_results[i].distance;
            }
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("=== COLLISION PROBE RESULTS ===");
        RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Total probes:  " << SPEC::NUM_CAMERAS * SPEC::NUM_PROBES);
        RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Hits:          " << total_hits
               << " (" << (100.0 * total_hits / (SPEC::NUM_CAMERAS * SPEC::NUM_PROBES)) << "%)");
        RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Misses:        " << (SPEC::NUM_CAMERAS * SPEC::NUM_PROBES - total_hits));
        if(total_hits > 0){
            RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Min hit dist:  " << min_hit_dist);
            RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Max hit dist:  " << max_hit_dist);
            RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("  Avg hit dist:  " << (sum_hit_dist / total_hits));
        }
        RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("===============================");

        {
            FILE* f = fopen(filename, "wb");
            if(f){
                int nc = SPEC::NUM_CAMERAS, np = SPEC::NUM_PROBES;
                fwrite(&nc, sizeof(int), 1, f);
                fwrite(&np, sizeof(int), 1, f);
                fwrite(probe_results, sizeof(CollisionResult),
                       (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES, f);
                fclose(f);
                RL_TOOLS_RENDERING_RAYTRACING_LOG_OK("Written probe data (" << SPEC::NUM_CAMERAS * SPEC::NUM_PROBES
                       << " results) to " << filename);
            }
        }
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
        if(renderer.backend.owl_collision_results_buffer != nullptr){
            memcpy(data(out),
                   owlBufferGetPointer((OWLBuffer)renderer.backend.owl_collision_results_buffer, 0),
                   SPEC::NUM_CAMERAS * SPEC::NUM_PROBES * sizeof(rendering::raytracing::CollisionResult));
        }
#endif
    }

    template <typename DEVICE, typename SPEC>
    const rendering::raytracing::CollisionResult* read_collision_results_raw(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
#if RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
        return nullptr;
#else
        if(renderer.backend.owl_collision_results_buffer == nullptr){
            return nullptr;
        }
        return (const rendering::raytracing::CollisionResult*)owlBufferGetPointer((OWLBuffer)renderer.backend.owl_collision_results_buffer, 0);
#endif
    }

    template <typename DEVICE, typename SPEC>
    uint32_t* get_framebuffer_device_ptr(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_RGB, "get_framebuffer_device_ptr requires an RGB-capable renderer specification");
        return (uint32_t*)owlBufferGetPointer((OWLBuffer)renderer.backend.owl_frame_buffer, 0);
    }

    template <typename DEVICE, typename SPEC>
    float* get_depthbuffer_device_ptr(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        static_assert(SPEC::HAS_DEPTH, "get_depthbuffer_device_ptr requires a depth-capable renderer specification");
        return (float*)owlBufferGetPointer((OWLBuffer)renderer.backend.owl_depth_buffer, 0);
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
