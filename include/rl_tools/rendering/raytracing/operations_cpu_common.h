#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_COMMON_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_COMMON_H

#include "renderer.h"
#include "transforms_generic.h"

// STATIC gives the stb implementations internal linkage so multiple TUs of one binary may include
// this header without duplicate-symbol link errors; RL_TOOLS_STB_PROVIDED arbitrates with other
// stb-providing headers (e.g. the test golden_io.h) so the implementation lands exactly once per
// TU regardless of include order (stb's implementation section has no include guard).
#ifndef RL_TOOLS_STB_PROVIDED
#define RL_TOOLS_STB_PROVIDED
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/GltfMaterial.h>
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
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>

#define RL_TOOLS_RENDERING_RAYTRACING_LOG(message) do { std::cout << "\033[0;34m" << "#rl_tools::rendering::raytracing: " << message << "\033[0m" << std::endl; } while(false)
#define RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR(message) do { std::cerr << "\033[0;31m" << "#rl_tools::rendering::raytracing: " << message << "\033[0m" << std::endl; } while(false)

#ifndef RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS
#define RL_TOOLS_RENDERING_RAYTRACING_DISABLE_PROBE_RAYS 0
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    // =========================================================================
    // Default cube geometry
    // =========================================================================
    namespace rendering::raytracing::constants{
        const int NUM_VERTICES = 8;
        const float default_vertices[8][3] = {
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
        const int default_indices[12][3] = {
            { 0,1,3 }, { 2,3,0 },
            { 5,7,6 }, { 5,6,4 },
            { 0,4,5 }, { 0,5,1 },
            { 2,3,7 }, { 2,7,6 },
            { 1,5,7 }, { 1,7,3 },
            { 4,0,2 }, { 4,2,6 }
        };
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

    // =========================================================================
    // Decode an embedded texture from assimp into RGBA8 pixels
    // =========================================================================
    namespace rendering::raytracing{
        struct RepresentativeTextureColor {
            float color[3] = {1.0f, 1.0f, 1.0f};
        };

        static inline float srgb_to_linear(unsigned char value) {
            const float x = static_cast<float>(value) / 255.0f;
            return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
        }

        static inline bool representative_texture_color(const std::vector<uint8_t>& pixels, int w, int h, RepresentativeTextureColor& out) {
            if(w <= 0 || h <= 0 || pixels.size() < static_cast<size_t>(w) * static_cast<size_t>(h) * 4) {
                return false;
            }
            double sum[3] = {0.0, 0.0, 0.0};
            double weight_sum = 0.0;
            for(int i = 0; i < w * h; i++) {
                const double alpha = static_cast<double>(pixels[i * 4 + 3]) / 255.0;
                sum[0] += static_cast<double>(srgb_to_linear(pixels[i * 4 + 0])) * alpha;
                sum[1] += static_cast<double>(srgb_to_linear(pixels[i * 4 + 1])) * alpha;
                sum[2] += static_cast<double>(srgb_to_linear(pixels[i * 4 + 2])) * alpha;
                weight_sum += alpha;
            }
            if(weight_sum <= 0.0) {
                for(int i = 0; i < w * h; i++) {
                    sum[0] += static_cast<double>(srgb_to_linear(pixels[i * 4 + 0]));
                    sum[1] += static_cast<double>(srgb_to_linear(pixels[i * 4 + 1]));
                    sum[2] += static_cast<double>(srgb_to_linear(pixels[i * 4 + 2]));
                }
                weight_sum = static_cast<double>(w) * static_cast<double>(h);
            }
            out.color[0] = static_cast<float>(sum[0] / weight_sum);
            out.color[1] = static_cast<float>(sum[1] / weight_sum);
            out.color[2] = static_cast<float>(sum[2] / weight_sum);
            return true;
        }

        static inline bool decode_embedded_texture(const aiTexture* tex,
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

        static inline bool decode_image_bytes(const std::vector<uint8_t>& bytes,
                                              std::vector<uint8_t>& pixels,
                                              int& w, int& h){
            int channels;
            unsigned char* data = stbi_load_from_memory(bytes.data(), (int)bytes.size(), &w, &h, &channels, 4);
            if(!data) return false;
            pixels.assign(data, data + (size_t)w * h * 4);
            stbi_image_free(data);
            return true;
        }

        static inline bool load_representative_texture_color(const aiScene* scene,
                                                      const aiMaterial* mat,
                                                      aiTextureType texture_type,
                                                      std::map<std::string, RepresentativeTextureColor>& cache,
                                                      RepresentativeTextureColor& out,
                                                      size_t& decoded_count) {
            if(mat == nullptr || mat->GetTextureCount(texture_type) == 0) {
                return false;
            }
            aiString tex_path;
            if(mat->GetTexture(texture_type, 0, &tex_path) != AI_SUCCESS) {
                return false;
            }
            const std::string path_str(tex_path.C_Str());
            if(path_str.empty()) {
                return false;
            }
            const std::string cache_key = std::to_string(static_cast<int>(texture_type)) + ":" + path_str;
            auto cached = cache.find(cache_key);
            if(cached != cache.end()) {
                out = cached->second;
                return true;
            }

            int w = 0;
            int h = 0;
            std::vector<uint8_t> pixels;
            const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
            if(emb_tex) {
                if(!decode_embedded_texture(emb_tex, pixels, w, h)) {
                    return false;
                }
            }
            else {
                int channels = 0;
                unsigned char* data = stbi_load(path_str.c_str(), &w, &h, &channels, 4);
                if(!data) {
                    return false;
                }
                pixels.assign(data, data + static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
                stbi_image_free(data);
            }

            if(!representative_texture_color(pixels, w, h, out)) {
                return false;
            }
            cache[cache_key] = out;
            decoded_count++;
            return true;
        }
    }

    // Lights are parsed from the raw GLB JSON instead of using Assimp's aiScene::mLights: Assimp's
    // KHR_lights_punctual import associates lights to nodes by name and mangles per-light
    // intensity/attenuation (see commit 8278f9a9 "removing assimp loading because it fumbled the lights").
    namespace rendering::raytracing::glb{
        struct Material {
            float metallic;
            float roughness;
            int metallic_roughness_image = -1; // glTF image index, -1 when the material has no MR texture
        };
        struct ParsedMetadata {
            std::vector<rendering::raytracing::SceneLight> lights; // world frame (welded scene loads)
            std::map<int, std::vector<rendering::raytracing::SceneLight>> root_lights; // by scene-root ordinal, in the root node's frame (assembly loads)
            std::map<std::string, Material> materials; // by material name; glTF spec defaults when absent
            std::map<int, std::vector<uint8_t>> images; // encoded bytes of referenced MR images, by glTF image index
        };

        // accumulates local transforms from the ancestors down to node_idx; stop_ancestor (when
        // >= 0) is excluded, yielding the transform relative to that ancestor's frame
        static inline void node_world_transform(const nlohmann::json& nodes, int node_idx, const std::vector<int>& parent_map, float out[16], int stop_ancestor = -1) {
            float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
            std::memcpy(out, identity, sizeof(identity));

            std::vector<int> chain;
            for (int cur = node_idx; cur >= 0 && cur != stop_ancestor; cur = parent_map[cur]) chain.push_back(cur);

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

        static inline void swizzle_gltf_direction_to_renderer(float dx, float dy, float dz, float out[3]) {
            float len = sqrtf(dx*dx + dy*dy + dz*dz);
            if (len <= 1e-6f) {
                out[0] = 0.f;
                out[1] = 0.f;
                out[2] = 1.f;
                return;
            }
            dx /= len;
            dy /= len;
            dz /= len;
            out[0] = dx;
            out[1] = -dz;
            out[2] = dy;
        }

        static inline ParsedMetadata parse(const std::string& filename) {
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

            // BIN chunk (glTF buffer 0) directly follows the JSON chunk; chunk lengths include padding
            long bin_data_start = -1;
            long bin_data_length = 0;
            if (fread(chunk_header, 4, 2, f) == 2 && chunk_header[1] == 0x004E4942u) {
                bin_data_start = 12 + 8 + (long)json_len + 8;
                bin_data_length = (long)chunk_header[0];
            }

            nlohmann::json gltf = nlohmann::json::parse(json_str, nullptr, false);
            if (gltf.is_discarded()) { fclose(f); return result; }

            // Assimp's defaults for absent glTF pbrMetallicRoughness factors vary across versions;
            // the GLB JSON is authoritative here so scene import is identical on every machine.
            // Absent factors resolve to the glTF spec default 1: exporters omit values equal to
            // the default, which is why intentionally non-metallic materials carry an explicit
            // metallicFactor of 0 while chrome/steel/gold-style materials omit it entirely.
            if (gltf.contains("materials")) {
                for (auto& material : gltf["materials"]) {
                    if (!material.contains("name")) continue;
                    Material material_import{1.0f, 1.0f};
                    if (material.contains("pbrMetallicRoughness")) {
                        auto& pbr = material["pbrMetallicRoughness"];
                        if (pbr.contains("metallicFactor")) {
                            material_import.metallic = pbr["metallicFactor"].get<float>();
                        }
                        if (pbr.contains("roughnessFactor")) {
                            material_import.roughness = pbr["roughnessFactor"].get<float>();
                        }
                        if (pbr.contains("metallicRoughnessTexture") && pbr["metallicRoughnessTexture"].contains("index")) {
                            const int texture_index = pbr["metallicRoughnessTexture"]["index"].get<int>();
                            if (gltf.contains("textures") && texture_index >= 0 && texture_index < (int)gltf["textures"].size()
                                && gltf["textures"][texture_index].contains("source")) {
                                material_import.metallic_roughness_image = gltf["textures"][texture_index]["source"].get<int>();
                            }
                        }
                    }
                    result.materials[material["name"].get<std::string>()] = material_import;
                }

                if (bin_data_start >= 0 && gltf.contains("images") && gltf.contains("bufferViews")) {
                    for (auto& [material_name, material_import] : result.materials) {
                        const int image_index = material_import.metallic_roughness_image;
                        if (image_index < 0 || result.images.count(image_index) > 0) continue;
                        if (image_index >= (int)gltf["images"].size()) continue;
                        const auto& image = gltf["images"][image_index];
                        if (!image.contains("bufferView")) continue;
                        const int buffer_view_index = image["bufferView"].get<int>();
                        if (buffer_view_index < 0 || buffer_view_index >= (int)gltf["bufferViews"].size()) continue;
                        const auto& buffer_view = gltf["bufferViews"][buffer_view_index];
                        if (buffer_view.value("buffer", 0) != 0) continue;
                        const long byte_offset = (long)buffer_view.value("byteOffset", 0LL);
                        const long byte_length = (long)buffer_view.value("byteLength", 0LL);
                        if (byte_length <= 0 || byte_offset < 0 || byte_offset + byte_length > bin_data_length) continue;
                        if (fseek(f, bin_data_start + byte_offset, SEEK_SET) != 0) continue;
                        std::vector<uint8_t> bytes((size_t)byte_length);
                        if (fread(bytes.data(), 1, (size_t)byte_length, f) != (size_t)byte_length) continue;
                        result.images[image_index] = std::move(bytes);
                    }
                }
            }
            fclose(f);

            auto& nodes = gltf["nodes"];
            std::vector<int> parent_map(nodes.size(), -1);
            for (int i = 0; i < (int)nodes.size(); i++) {
                if (nodes[i].contains("children")) {
                    for (auto& child : nodes[i]["children"]) parent_map[child.get<int>()] = i;
                }
            }

            std::map<int, int> root_ordinal;
            if (gltf.contains("scenes")) {
                const int scene_index = gltf.value("scene", 0);
                if (scene_index >= 0 && scene_index < (int)gltf["scenes"].size() && gltf["scenes"][scene_index].contains("nodes")) {
                    int ordinal = 0;
                    for (auto& root : gltf["scenes"][scene_index]["nodes"]) root_ordinal[root.get<int>()] = ordinal++;
                }
            }

            if (gltf.contains("extensions") && gltf["extensions"].contains("KHR_lights_punctual")) {
                auto& light_defs = gltf["extensions"]["KHR_lights_punctual"]["lights"];
                const auto make_light = [](const nlohmann::json& ldef, const std::string& type_str, float intensity, float color_r, float color_g, float color_b, const float transform[16]) {
                    rendering::raytracing::SceneLight sl{};
                    if (type_str == "directional") sl.type = 0;
                    else if (type_str == "spot") sl.type = 2;
                    else sl.type = 1;

                    sl.position[0] = transform[12]; sl.position[1] = -transform[14]; sl.position[2] = transform[13];
                    sl.color[0] = color_r * intensity; sl.color[1] = color_g * intensity; sl.color[2] = color_b * intensity;
                    sl.attenuation_constant = 0.f; sl.attenuation_linear = 0.f; sl.attenuation_quadratic = 1.f;

                    if (type_str == "directional") {
                        swizzle_gltf_direction_to_renderer(transform[8], transform[9], transform[10], sl.direction);
                    } else if (type_str == "spot") {
                        float inner = 0.0f;
                        float outer = 0.7854f;
                        if (ldef.contains("spot")) {
                            inner = ldef["spot"].value("innerConeAngle", inner);
                            outer = ldef["spot"].value("outerConeAngle", outer);
                        }
                        sl.cos_inner_cone = cosf(inner);
                        sl.cos_outer_cone = cosf(outer);
                        swizzle_gltf_direction_to_renderer(-transform[8], -transform[9], -transform[10], sl.direction);
                    }
                    return sl;
                };

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
                    result.lights.push_back(make_light(ldef, type_str, intensity, color_r, color_g, color_b, world));

                    int owning_root = ni;
                    while (parent_map[owning_root] >= 0) owning_root = parent_map[owning_root];
                    const auto ordinal = root_ordinal.find(owning_root);
                    if (ordinal != root_ordinal.end()) {
                        float relative[16];
                        node_world_transform(nodes, ni, parent_map, relative, owning_root);
                        result.root_lights[ordinal->second].push_back(make_light(ldef, type_str, intensity, color_r, color_g, color_b, relative));
                    }
                }
            }

            return result;
        }
    }

    // =========================================================================
    // load: Assimp scene/object loading
    // =========================================================================
    namespace rendering::raytracing::detail{
    struct DecodedTexture{
        std::vector<uint8_t> pixels;
        int width;
        int height;
    };
    struct MeshConversionState{
        std::map<std::string, size_t> tex_cache;
        std::vector<DecodedTexture> decoded_textures;
        std::map<std::string, rendering::raytracing::RepresentativeTextureColor> representative_texture_color_cache;
        size_t representative_texture_color_meshes = 0;
        size_t representative_texture_color_decoded = 0;
    };

    template <typename SHADING, bool HAS_RGB>
    rendering::raytracing::Mesh convert_mesh(const aiScene* scene, const aiMesh* mesh, const aiMatrix4x4& global_transform, const rendering::raytracing::glb::ParsedMetadata& glb_metadata, MeshConversionState& state){
        rendering::raytracing::Mesh md;
        [[maybe_unused]] const aiMaterial* mat = nullptr;
        if constexpr (HAS_RGB) {
            if(mesh->mMaterialIndex < scene->mNumMaterials){
                mat = scene->mMaterials[mesh->mMaterialIndex];
            }
        }

        // vertices: apply node transform, then GLB (Y-up) → FLU (Z-up)
        for(unsigned int v = 0; v < mesh->mNumVertices; v++){
            aiVector3D pos = mesh->mVertices[v];
            pos = global_transform * pos;
            md.vertices.push_back(pos.x);
            md.vertices.push_back(-pos.z);
            md.vertices.push_back(pos.y);
        }

        if constexpr (HAS_RGB && SHADING::PBR_SHADING) {
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

        if constexpr (HAS_RGB && SHADING::LOAD_TEXTURES) {
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
                    if constexpr (SHADING::PBR_SHADING) {
                        md.tex_coords.push_back(1.0f - tc.y);
                    } else {
                        md.tex_coords.push_back(tc.y);
                    }
                }
            }
        }

        // material / texture
        if constexpr (SHADING::PBR_SHADING) {
            md.color[0] = 1.0f; md.color[1] = 1.0f; md.color[2] = 1.0f;
        } else {
            md.color[0] = 0.8f; md.color[1] = 0.8f; md.color[2] = 0.8f;
        }
        if constexpr (HAS_RGB) {
        if(mat != nullptr){

            if constexpr (SHADING::PBR_SHADING) {
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
                else if constexpr (!SHADING::LOAD_TEXTURES) {
                    aiColor4D base_color;
                    if (aiGetMaterialColor(mat, AI_MATKEY_BASE_COLOR, &base_color) == AI_SUCCESS) {
                        md.color[0] = base_color.r; md.color[1] = base_color.g; md.color[2] = base_color.b;
                    }
                }
            }

            if constexpr (!SHADING::LOAD_TEXTURES) {
                rendering::raytracing::RepresentativeTextureColor representative_color;
                bool has_representative_color = rendering::raytracing::load_representative_texture_color(
                    scene, mat, aiTextureType_DIFFUSE, state.representative_texture_color_cache,
                    representative_color, state.representative_texture_color_decoded
                );
                if(!has_representative_color) {
                    has_representative_color = rendering::raytracing::load_representative_texture_color(
                        scene, mat, aiTextureType_BASE_COLOR, state.representative_texture_color_cache,
                        representative_color, state.representative_texture_color_decoded
                    );
                }
                if(has_representative_color) {
                    md.color[0] *= representative_color.color[0];
                    md.color[1] *= representative_color.color[1];
                    md.color[2] *= representative_color.color[2];
                    state.representative_texture_color_meshes++;
                }
            }

            if constexpr (SHADING::LOAD_TEXTURES) {
            if(mat->GetTextureCount(aiTextureType_DIFFUSE) > 0){
                aiString tex_path;
                if(mat->GetTexture(aiTextureType_DIFFUSE, 0, &tex_path) == AI_SUCCESS){
                    std::string path_str(tex_path.C_Str());

                    auto it = state.tex_cache.find(path_str);
                    if(it != state.tex_cache.end()){
                        auto& cached = state.decoded_textures[it->second];
                        md.texture.pixels = cached.pixels;
                        md.texture.width = cached.width;
                        md.texture.height = cached.height;
                    } else {
                        const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                        if(emb_tex){
                            int w, h;
                            std::vector<uint8_t> pixels;
                            if(rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)){
                                md.texture.pixels = pixels;
                                md.texture.width = w;
                                md.texture.height = h;
                                state.tex_cache[path_str] = state.decoded_textures.size();
                                state.decoded_textures.push_back({std::move(pixels), w, h});
                            }
                        } else if(!path_str.empty()){
                            int w, h, channels;
                            unsigned char* data = stbi_load(path_str.c_str(), &w, &h, &channels, 4);
                            if(data){
                                md.texture.pixels.assign(data, data + w * h * 4);
                                md.texture.width = w;
                                md.texture.height = h;
                                stbi_image_free(data);
                                state.tex_cache[path_str] = state.decoded_textures.size();
                                state.decoded_textures.push_back({md.texture.pixels, w, h});
                            }
                        }
                    }
                }
            }
            }

            if constexpr (SHADING::METALLIC_REFLECTIONS || SHADING::PBR_SHADING) {
            float metallic_factor = 0.0f;
            mat->Get(AI_MATKEY_METALLIC_FACTOR, metallic_factor);
            md.metallic = metallic_factor;
            }

            if constexpr (SHADING::PBR_SHADING) {
                float metallic_factor_pbr = 1.0f;
                mat->Get(AI_MATKEY_METALLIC_FACTOR, metallic_factor_pbr);
                md.metallic = metallic_factor_pbr;

                float roughness_factor = 1.0f;
                mat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness_factor);
                md.roughness = roughness_factor;

                if constexpr (SHADING::LOAD_TEXTURES) {
                if (mat->GetTextureCount(aiTextureType_NORMALS) > 0) {
                    aiString tex_path;
                    if (mat->GetTexture(aiTextureType_NORMALS, 0, &tex_path) == AI_SUCCESS) {
                        const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                        if (emb_tex) {
                            int w, h;
                            std::vector<uint8_t> pixels;
                            if (rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)) {
                                md.normal_map.pixels = std::move(pixels);
                                md.normal_map.width = w;
                                md.normal_map.height = h;
                            }
                        }
                    }
                }

                // the metallicRoughness texture is resolved from the GLB JSON, not through
                // Assimp's texture-type taxonomy (which relabels it across versions and orders
                // embedded textures differently from the glTF image array)
                {
                    const auto glb_material = glb_metadata.materials.find(mat->GetName().C_Str());
                    if (glb_material != glb_metadata.materials.end() && glb_material->second.metallic_roughness_image >= 0) {
                        const auto image = glb_metadata.images.find(glb_material->second.metallic_roughness_image);
                        if (image != glb_metadata.images.end()) {
                            int w, h;
                            std::vector<uint8_t> pixels;
                            if (rendering::raytracing::decode_image_bytes(image->second, pixels, w, h)) {
                                md.metallic_roughness_map.pixels = std::move(pixels);
                                md.metallic_roughness_map.width = w;
                                md.metallic_roughness_map.height = h;
                            }
                        }
                    }
                }
                }

                aiColor3D emissive_color(0.f, 0.f, 0.f);
                mat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive_color);
                md.emissive[0] = emissive_color.r; md.emissive[1] = emissive_color.g; md.emissive[2] = emissive_color.b;

                if constexpr (SHADING::LOAD_TEXTURES) {
                if (mat->GetTextureCount(aiTextureType_EMISSIVE) > 0) {
                    aiString tex_path;
                    if (mat->GetTexture(aiTextureType_EMISSIVE, 0, &tex_path) == AI_SUCCESS) {
                        const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                        if (emb_tex) {
                            int w, h;
                            std::vector<uint8_t> pixels;
                            if (rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)) {
                                md.emissive_map.pixels = std::move(pixels);
                                md.emissive_map.width = w;
                                md.emissive_map.height = h;
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
                                md.occlusion_map.pixels = std::move(pixels);
                                md.occlusion_map.width = w;
                                md.occlusion_map.height = h;
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
                                md.occlusion_map.pixels = std::move(pixels);
                                md.occlusion_map.width = w;
                                md.occlusion_map.height = h;
                            }
                        }
                    }
                }
                }
                float opacity_val = 1.0f;
                mat->Get(AI_MATKEY_OPACITY, opacity_val);
                aiColor4D opacity_base_color(1.0f, 1.0f, 1.0f, 1.0f);
                if (aiGetMaterialColor(mat, AI_MATKEY_BASE_COLOR, &opacity_base_color) == AI_SUCCESS) {
                    opacity_val = fminf(opacity_val, opacity_base_color.a);
                }
                float transmission_factor = 0.0f;
                mat->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmission_factor);
                if (transmission_factor > 0.0f) {
                    opacity_val = fminf(opacity_val, 1.0f - transmission_factor);
                }
                md.opacity = opacity_val;

                aiString alpha_mode;
                if (mat->Get(AI_MATKEY_GLTF_ALPHAMODE, alpha_mode) == AI_SUCCESS) {
                    std::string alpha_mode_str = alpha_mode.C_Str();
                    if (alpha_mode_str == "MASK") {
                        md.alpha_mode = 1;
                    } else if (alpha_mode_str == "BLEND") {
                        md.alpha_mode = 2;
                    } else {
                        md.alpha_mode = 0;
                    }
                }
                float alpha_cutoff = 0.5f;
                if (mat->Get(AI_MATKEY_GLTF_ALPHACUTOFF, alpha_cutoff) == AI_SUCCESS) {
                    md.alpha_cutoff = alpha_cutoff;
                }
            }

            if constexpr (SHADING::LOAD_TEXTURES) {
            if(!md.texture.present() && mat->GetTextureCount(aiTextureType_BASE_COLOR) > 0){
                aiString tex_path;
                if(mat->GetTexture(aiTextureType_BASE_COLOR, 0, &tex_path) == AI_SUCCESS){
                    const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                    if(emb_tex){
                        int w, h;
                        std::vector<uint8_t> pixels;
                        if(rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)){
                            md.texture.pixels = pixels;
                            md.texture.width = w;
                            md.texture.height = h;
                        }
                    }
                }
            }
            }
        }
        }

        if constexpr (HAS_RGB && (SHADING::METALLIC_REFLECTIONS || SHADING::PBR_SHADING)) {
            if(mat != nullptr){
                const auto glb_material = glb_metadata.materials.find(mat->GetName().C_Str());
                if(glb_material != glb_metadata.materials.end()){
                    md.metallic = glb_material->second.metallic;
                    if constexpr (SHADING::PBR_SHADING) {
                        md.roughness = glb_material->second.roughness;
                    }
                }
            }
        }

        return md;
    }

    template <typename SHADING, bool HAS_RGB>
    bool load_scene_data(std::vector<rendering::raytracing::Mesh>& out_meshes, std::vector<rendering::raytracing::SceneLight>& out_lights, const std::string& filename){
        Assimp::Importer importer;
        unsigned int import_flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality;
        if constexpr (HAS_RGB && SHADING::PBR_SHADING) {
            import_flags |= aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace;
        } else if constexpr (HAS_RGB && SHADING::NORMAL_SHADING) {
            import_flags |= aiProcess_GenNormals;
        }
        const aiScene* scene = importer.ReadFile(filename, import_flags);

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode){
            RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Assimp error: " << importer.GetErrorString());
            return false;
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Loaded model with " << scene->mNumMeshes << " mesh(es)");

        [[maybe_unused]] MeshConversionState conversion_state;

        size_t total_verts = 0, total_tris = 0;

        const size_t first_new_mesh = out_meshes.size();
        const auto glb_metadata = rendering::raytracing::glb::parse(filename);

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
            rendering::raytracing::Mesh md = convert_mesh<SHADING, HAS_RGB>(scene, mesh, global_transform, glb_metadata, conversion_state);
            total_verts += md.vertices.size() / 3;
            total_tris += md.indices.size() / 3;
            out_meshes.push_back(std::move(md));
            } // end for global_transform
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Total vertices: " << total_verts << ", triangles: " << total_tris);

        const size_t num_new_meshes = out_meshes.size() - first_new_mesh;
        int textured_count = 0;
        int metallic_count = 0;
        for(size_t m = first_new_mesh; m < out_meshes.size(); m++){
            if(out_meshes[m].texture.present()) textured_count++;
            if(out_meshes[m].metallic > 0.f) metallic_count++;
        }
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Meshes with textures: " << textured_count << "/" << num_new_meshes
              << ", metallic: " << metallic_count << "/" << num_new_meshes);
        if constexpr (HAS_RGB && !SHADING::LOAD_TEXTURES) {
            RL_TOOLS_RENDERING_RAYTRACING_LOG("Representative texture colors: " << conversion_state.representative_texture_color_meshes
                  << "/" << num_new_meshes << " meshes, decoded "
                  << conversion_state.representative_texture_color_decoded << " texture(s)");
        }

        for (auto& sl : glb_metadata.lights) {
            out_lights.push_back(sl);
        }

        return true;
    }

    inline bool transform_is_identity(const float transform[12]){
        const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        for(int element = 0; element < 12; element++){
            if(transform[element] != identity[element]){
                return false;
            }
        }
        return true;
    }

    inline void transform_point(const float transform[12], const float point[3], float out[3]){
        for(int row = 0; row < 3; row++){
            out[row] = transform[row * 4 + 0] * point[0] + transform[row * 4 + 1] * point[1] + transform[row * 4 + 2] * point[2] + transform[row * 4 + 3];
        }
    }

    inline void transform_vector(const float transform[12], const float vector[3], float out[3]){
        for(int row = 0; row < 3; row++){
            out[row] = transform[row * 4 + 0] * vector[0] + transform[row * 4 + 1] * vector[1] + transform[row * 4 + 2] * vector[2];
        }
    }

    // conjugates an Assimp (glTF Y-up) node transform into the FLU frame: T_flu = S * T * S^-1,
    // with S the same Y-up -> FLU swizzle that is applied to vertices
    inline void flu_from_assimp(const aiMatrix4x4& transform, float out[12]){
        out[0] = transform.a1;  out[1] = -transform.a3; out[2]  = transform.a2;  out[3]  = transform.a4;
        out[4] = -transform.c1; out[5] = transform.c3;  out[6]  = -transform.c2; out[7]  = -transform.c4;
        out[8] = transform.b1;  out[9] = -transform.b3; out[10] = transform.b2;  out[11] = transform.b4;
    }

    template <typename SHADING, bool HAS_RGB>
    bool load_assembly_data(rendering::raytracing::ObjectAssembly& assembly, const std::string& filename){
        Assimp::Importer importer;
        unsigned int import_flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality;
        if constexpr (HAS_RGB && SHADING::PBR_SHADING) {
            import_flags |= aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace;
        } else if constexpr (HAS_RGB && SHADING::NORMAL_SHADING) {
            import_flags |= aiProcess_GenNormals;
        }
        const aiScene* scene = importer.ReadFile(filename, import_flags);
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode){
            RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Assimp error: " << importer.GetErrorString());
            return false;
        }
        const auto glb_metadata = rendering::raytracing::glb::parse(filename);
        MeshConversionState conversion_state; // shared across parts: textures used by several parts decode once

        const aiNode* file_root = scene->mRootNode;
        const size_t first_new_part = assembly.parts.size();
        for(unsigned int root_i = 0; root_i < file_root->mNumChildren; root_i++){
            const aiNode* root = file_root->mChildren[root_i];
            rendering::raytracing::Object object;
            object.name = root->mName.C_Str();

            // geometry lands in the root node's frame: the recursion starts below the root's own
            // transform, which becomes the part placement instead
            std::function<void(const aiNode*, const aiMatrix4x4&)> collect = [&](const aiNode* node, const aiMatrix4x4& parent_transform){
                const aiMatrix4x4 relative_transform = parent_transform * node->mTransformation;
                for(unsigned int mesh_i = 0; mesh_i < node->mNumMeshes; mesh_i++){
                    object.meshes.push_back(convert_mesh<SHADING, HAS_RGB>(scene, scene->mMeshes[node->mMeshes[mesh_i]], relative_transform, glb_metadata, conversion_state));
                }
                for(unsigned int child_i = 0; child_i < node->mNumChildren; child_i++){
                    collect(node->mChildren[child_i], relative_transform);
                }
            };
            for(unsigned int mesh_i = 0; mesh_i < root->mNumMeshes; mesh_i++){
                object.meshes.push_back(convert_mesh<SHADING, HAS_RGB>(scene, scene->mMeshes[root->mMeshes[mesh_i]], aiMatrix4x4(), glb_metadata, conversion_state));
            }
            for(unsigned int child_i = 0; child_i < root->mNumChildren; child_i++){
                collect(root->mChildren[child_i], aiMatrix4x4());
            }

            const auto root_lights = glb_metadata.root_lights.find((int)root_i);
            if(root_lights != glb_metadata.root_lights.end()){
                object.lights = root_lights->second;
            }

            rendering::raytracing::ObjectAssembly::Part part;
            part.object = assembly.objects.size();
            flu_from_assimp(root->mTransformation, part.transform);
            assembly.objects.push_back(std::move(object));
            assembly.parts.push_back(part);
        }
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Loaded assembly with " << (assembly.parts.size() - first_new_part) << " part(s)");
        return true;
    }

    // compose_transforms and slerp_transform live in transforms_generic.h (shared with device
    // producers)

    inline constexpr float IDENTITY_TRANSFORM[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

    // stages host-verb writes (the per-slot transform_entry mirrors) into the transforms tensor;
    // backends whose tensor is host-resident call this at the top of update(). Device producers
    // write the tensor directly and are not staged — a dirty overlay row is owned by the host.
    template <typename SPEC, typename BACKEND>
    void flush_overlay_transforms(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        using TI = typename SPEC::TI;
        float* transforms = data(renderer.transforms);
        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            auto& overlay_state = renderer.overlays[overlay];
            if(!overlay_state.dirty) continue;
            for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
                std::memcpy(transforms + ((size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES + slot) * 12, overlay_state.slots[slot].transform_entry, 12 * sizeof(float));
            }
            overlay_state.dirty = false;
        }
    }

    // world = pose ∘ part_local ∘ articulation: the root slot's tensor entry carries the
    // placement pose, non-root entries articulate their part in the part frame. The
    // transforms_base overload lets the dynamic-motion-blur loop compose from a per-sample
    // slab of transforms_motion, which shares the transforms tensor layout.
    template <typename SPEC, typename BACKEND>
    void compose_overlay_slot_transform(const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const float* transforms_base, typename SPEC::TI overlay, typename SPEC::TI slot_index, float out[12]){
        const auto& slot = renderer.overlays[overlay].slots[slot_index];
        const float* row = transforms_base + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES * 12;
        float composed[12];
        compose_transforms(row + (size_t)slot.pose_slot * 12, slot.part_local, composed);
        if(slot_index == slot.pose_slot){
            std::memcpy(out, composed, sizeof(composed));
        }
        else{
            compose_transforms(composed, row + (size_t)slot_index * 12, out);
        }
    }

    template <typename SPEC, typename BACKEND>
    void compose_overlay_slot_transform(const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, typename SPEC::TI overlay, typename SPEC::TI slot_index, float out[12]){
        compose_overlay_slot_transform(renderer, data(renderer.transforms), overlay, slot_index, out);
    }

    inline void invert_transform(const float transform[12], float out[12]){
        const float a = transform[0], b = transform[1], c = transform[2];
        const float d = transform[4], e = transform[5], f = transform[6];
        const float g = transform[8], h = transform[9], i = transform[10];
        const float cofactor_a = e*i - f*h;
        const float cofactor_b = f*g - d*i;
        const float cofactor_c = d*h - e*g;
        const float inv_det = 1.0f / (a*cofactor_a + b*cofactor_b + c*cofactor_c);
        out[0] = cofactor_a * inv_det; out[1] = (c*h - b*i) * inv_det; out[2]  = (b*f - c*e) * inv_det;
        out[4] = cofactor_b * inv_det; out[5] = (a*i - c*g) * inv_det; out[6]  = (c*d - a*f) * inv_det;
        out[8] = cofactor_c * inv_det; out[9] = (b*g - a*h) * inv_det; out[10] = (a*e - b*d) * inv_det;
        out[3]  = -(out[0]*transform[3] + out[1]*transform[7] + out[2] *transform[11]);
        out[7]  = -(out[4]*transform[3] + out[5]*transform[7] + out[6] *transform[11]);
        out[11] = -(out[8]*transform[3] + out[9]*transform[7] + out[10]*transform[11]);
    }

    // writes one entry into every motion sample of the staging mirror (constant across the
    // shutter = sharp); the single-pose verbs route through this so a dynamic-motion-blur spec
    // driven only by them renders identically to camera-only blur
    template <typename SPEC, typename BACKEND>
    void stage_motion_entry(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, typename SPEC::TI overlay, typename SPEC::TI slot, const float entry[12]){
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
            for(typename SPEC::TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                std::memcpy(renderer.transforms_motion_staging.data() + sample * SLAB + ((size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES + slot) * 12, entry, 12 * sizeof(float));
            }
            renderer.transforms_motion_dirty[overlay] = true;
        }
    }

    // CPU expansion of the transforms_pair tensor for backends whose tensors are host-resident
    // (generic/Vulkan); OptiX runs the same math on-device (overlay_accel_expand_motion).
    // Producer-style: writes the tensors directly with no dirty flags, so the host-verb flush
    // never clobbers it — per overlay, use either the set_transform* verbs or the pair path.
    template <typename SPEC, typename BACKEND>
    void expand_motion_transforms_host(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        using TI = typename SPEC::TI;
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        const float* pairs = data(renderer.transforms_pair);
        float* transforms_motion = data(renderer.transforms_motion);
        float* transforms = data(renderer.transforms);
        for(size_t slot = 0; slot < SLOTS; slot++){
            const float* open = pairs + slot * 12;
            const float* close = pairs + (SLOTS + slot) * 12;
            for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                const float shutter_t = ((float)sample + 0.5f) / (float)SPEC::MOTION_BLUR_SAMPLES;
                slerp_transform(open, close, shutter_t, transforms_motion + ((size_t)sample * SLOTS + slot) * 12);
            }
            std::memcpy(transforms + slot * 12, close, 12 * sizeof(float));
        }
    }

    // stages host-verb writes into the transforms_motion tensor, mirroring
    // flush_overlay_transforms; backends with a host-resident tensor call it in update()
    template <typename SPEC, typename BACKEND>
    void flush_overlay_motion_transforms(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        using TI = typename SPEC::TI;
        constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
        float* transforms_motion = data(renderer.transforms_motion);
        for(TI overlay = 0; overlay < SPEC::NUM_OVERLAYS; overlay++){
            if(!renderer.transforms_motion_dirty[overlay]) continue;
            for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                const size_t offset = sample * SLAB + (size_t)overlay * SPEC::MAX_OVERLAY_INSTANCES * 12;
                std::memcpy(transforms_motion + offset, renderer.transforms_motion_staging.data() + offset, (size_t)SPEC::MAX_OVERLAY_INSTANCES * 12 * sizeof(float));
            }
            renderer.transforms_motion_dirty[overlay] = false;
        }
    }

    // flattens the asset pool for overlay spawns: appends pool objects to the combined object
    // list and records per-part global object indices + local transforms on the renderer
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void register_pool_assets(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const rendering::raytracing::AssetPool& pool, std::vector<const rendering::raytracing::Object*>& all_objects){
        using TI = typename SPEC::TI;
        renderer.assets.clear();
        renderer.asset_part_objects.clear();
        renderer.asset_part_transforms.clear();
        for(const auto& assembly : pool.assemblies){
            typename rendering::raytracing::Renderer<SPEC, BACKEND>::AssetRecord record;
            record.first_part = (TI)renderer.asset_part_objects.size();
            record.num_parts = (TI)assembly.parts.size();
            utils::assert_exit(device, record.num_parts <= SPEC::MAX_OVERLAY_INSTANCES, "asset has more parts than the overlay capacity");
            const TI object_base = (TI)all_objects.size();
            for(const auto& object : assembly.objects){
                all_objects.push_back(&object);
            }
            for(const auto& part : assembly.parts){
                renderer.asset_part_objects.push_back(object_base + (TI)part.object);
                renderer.asset_part_transforms.insert(renderer.asset_part_transforms.end(), part.transform, part.transform + 12);
            }
            renderer.assets.push_back(record);
        }
    }

    // Deterministic first-fit is a contract, not an implementation detail: the chosen slot defines
    // the global instance id (segmentation output), which must be reproducible across runs and
    // identical across backends. Do not replace with a free-list or best-fit strategy.
    template <typename SPEC, typename BACKEND>
    typename SPEC::TI first_fit_slot(const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, size_t overlay, typename SPEC::TI num_parts){
        using TI = typename SPEC::TI;
        const auto& state = renderer.overlays[overlay];
        TI run = 0;
        for(TI slot = 0; slot < SPEC::MAX_OVERLAY_INSTANCES; slot++){
            run = state.slots[slot].active ? 0 : run + 1;
            if(run == num_parts){
                return slot + 1 - num_parts;
            }
        }
        return SPEC::MAX_OVERLAY_INSTANCES;
    }

    template <typename SPEC, typename BACKEND>
    void reset_overlay_state(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        using TI = typename SPEC::TI;
        for(auto& overlay_state : renderer.overlays){
            for(auto& slot : overlay_state.slots){
                slot.active = false;
            }
            overlay_state.dirty = true;
        }
        for(TI attachment_i = 0; attachment_i < SPEC::NUM_CAMERAS * SPEC::MAX_OVERLAYS_PER_CAMERA; attachment_i++){
            renderer.attachments[attachment_i] = rendering::raytracing::Renderer<SPEC, BACKEND>::INVALID_OVERLAY;
        }
        renderer.attachments_dirty = true;
    }

    template <typename SPEC, typename BACKEND>
    void compute_scene_bounds(rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const rendering::raytracing::Scene& scene){
        // resolved-configuration echo: makes a silently-defaulted (e.g. misspelled) fringe
        // config member visible on the first run
        RL_TOOLS_RENDERING_RAYTRACING_LOG("config: " << SPEC::NUM_CAMERAS << " camera(s) " << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT
            << " outputs[rgb=" << SPEC::HAS_RGB << " depth=" << SPEC::HAS_DEPTH << " segmentation=" << SPEC::HAS_SEGMENTATION << (SPEC::SEMANTIC_SEGMENTATION ? " (semantic)" : "") << "]"
            << " probes=" << SPEC::NUM_PROBES
            << " motion_blur_samples=" << (SPEC::ENABLE_MOTION_BLUR ? SPEC::MOTION_BLUR_SAMPLES : 0)
            << " anti_aliasing_grid=" << (SPEC::ENABLE_ANTI_ALIASING ? SPEC::ANTI_ALIASING_GRID_SIZE : 0)
            << " overlays=" << SPEC::NUM_OVERLAYS << "x" << SPEC::MAX_OVERLAY_INSTANCES << " overlays_per_camera=" << SPEC::MAX_OVERLAYS_PER_CAMERA);
        float bbox_min[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        float bbox_max[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
        for(const auto& instance : scene.instances){
            for(const auto& mesh : scene.objects[instance.object].meshes){
                for(size_t vertex_i = 0; vertex_i + 2 < mesh.vertices.size(); vertex_i += 3){
                    float world[3] = {mesh.vertices[vertex_i], mesh.vertices[vertex_i + 1], mesh.vertices[vertex_i + 2]};
                    if(!instance.identity){
                        const float local[3] = {world[0], world[1], world[2]};
                        transform_point(instance.transform, local, world);
                    }
                    for(int d = 0; d < 3; d++){
                        bbox_min[d] = std::min(bbox_min[d], world[d]);
                        bbox_max[d] = std::max(bbox_max[d], world[d]);
                    }
                }
            }
        }
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Bounding box: [" << bbox_min[0] << "," << bbox_min[1] << "," << bbox_min[2] << "] - ["
              << bbox_max[0] << "," << bbox_max[1] << "," << bbox_max[2] << "]");

        float center[3], size[3];
        for(int d = 0; d < 3; d++){
            center[d] = 0.5f * (bbox_min[d] + bbox_max[d]);
            size[d] = bbox_max[d] - bbox_min[d];
        }
        float max_dim = std::max({size[0], size[1], size[2]});
        float look_from[3] = {center[0] + max_dim * 1.5f, center[1] + max_dim * 1.5f, center[2] + max_dim * 0.8f};
        renderer.scene_center[0] = center[0];
        renderer.scene_center[1] = center[1];
        renderer.scene_center[2] = center[2];
        renderer.scene_half_extent[0] = size[0] * 0.5f;
        renderer.scene_half_extent[1] = size[1] * 0.5f;
        renderer.scene_half_extent[2] = size[2] * 0.5f;
        float look_offset[3];
        rendering::raytracing::vec3::sub(look_from, center, look_offset);
        renderer.camera_radius = rendering::raytracing::vec3::length(look_offset);
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Camera positioned at [" << look_from[0] << "," << look_from[1] << "," << look_from[2] << "]");
    }

    // Lights as uploaded to the device: only the PBR tiers consume punctual lights. Object lights
    // are authored in the object's local frame and follow the instance placing it.
    template <bool APPLY>
    std::vector<rendering::raytracing::SceneLight> effective_scene_lights(const rendering::raytracing::Scene& scene){
        std::vector<rendering::raytracing::SceneLight> lights;
        if constexpr (APPLY) {
            lights = scene.lights;
            for(const auto& instance : scene.instances){
                for(const auto& object_light : scene.objects[instance.object].lights){
                    rendering::raytracing::SceneLight light = object_light;
                    if(!instance.identity){
                        transform_point(instance.transform, object_light.position, light.position);
                        transform_vector(instance.transform, object_light.direction, light.direction);
                        const float length = sqrtf(light.direction[0]*light.direction[0] + light.direction[1]*light.direction[1] + light.direction[2]*light.direction[2]);
                        if(length > 0){
                            light.direction[0] /= length;
                            light.direction[1] /= length;
                            light.direction[2] /= length;
                        }
                    }
                    lights.push_back(light);
                }
            }
            RL_TOOLS_RENDERING_RAYTRACING_LOG("Scene lights: " << lights.size());
            for (size_t li = 0; li < lights.size(); li++) {
                auto& sl = lights[li];
                RL_TOOLS_RENDERING_RAYTRACING_LOG("  light " << li << ": pos=(" << sl.position[0] << "," << sl.position[1] << "," << sl.position[2]
                    << ") dir=(" << sl.direction[0] << "," << sl.direction[1] << "," << sl.direction[2]
                    << ") color=(" << sl.color[0] << "," << sl.color[1] << "," << sl.color[2] << ")");
            }
        }
        return lights;
    }
    } // namespace rendering::raytracing::detail

    template <typename SHADING = rendering::raytracing::VeryHigh, bool HAS_RGB = true, typename DEVICE>
    bool load(DEVICE& device, rendering::raytracing::Object& object, const std::string& filename){
        return rendering::raytracing::detail::load_scene_data<SHADING, HAS_RGB>(object.meshes, object.lights, filename);
    }

    namespace rendering::raytracing::detail{
        // FNV-1a over the file bytes: deterministic scene-content identity for library dedup
        inline bool hash_scene_file(const char* path, uint64_t& hash_out){
            std::ifstream file(path, std::ios::binary);
            if(!file){
                return false;
            }
            uint64_t hash = 1469598103934665603ull;
            char buffer[1 << 16];
            while(file.read(buffer, sizeof(buffer)) || file.gcount() > 0){
                const std::streamsize count = file.gcount();
                for(std::streamsize byte_i = 0; byte_i < count; byte_i++){
                    hash = (hash ^ (uint64_t)(unsigned char)buffer[byte_i]) * 1099511628211ull;
                }
                if(count < (std::streamsize)sizeof(buffer)){
                    break;
                }
            }
            hash_out = hash;
            return true;
        }
    }

    // Appends the scene file as one welded static object placed at identity: explicit
    // composition, the append twin of load below.
    template <typename SHADING = rendering::raytracing::VeryHigh, bool HAS_RGB = true, typename DEVICE>
    bool add(DEVICE& device, rendering::raytracing::Scene& scene, const std::string& filename){
        rendering::raytracing::Object object;
        if(!rendering::raytracing::detail::load_scene_data<SHADING, HAS_RGB>(object.meshes, scene.lights, filename)){
            return false;
        }
        scene.objects.push_back(std::move(object));
        scene.instances.push_back({scene.objects.size() - 1, {1,0,0,0, 0,1,0,0, 0,0,1,0}, true});
        return true;
    }

    // Scene-level load: the scene file becomes one welded static object placed at identity. The
    // scene must be empty — reusing a scene across loads silently accumulates geometry (every
    // renderer re-uploads every previously loaded scene) and no rendering test can see it, so it
    // is a hard failure; composition is the explicit add(device, scene, filename). A scene file
    // without any punctual lights gets a neutral 3-directional fill so PBR-shaded content is not
    // lit by ambient only. Object/asset loads deliberately do not: fill lighting is a scene
    // decision, not an asset property.
    template <typename SHADING = rendering::raytracing::VeryHigh, bool HAS_RGB = true, typename DEVICE>
    bool load(DEVICE& device, rendering::raytracing::Scene& scene, const std::string& filename){
        utils::assert_exit(device, scene.objects.empty() && scene.instances.empty() && scene.lights.empty(), "load: scene is not empty — use add(device, scene, filename) to compose");
        if(!add<SHADING, HAS_RGB>(device, scene, filename)){
            return false;
        }
        if(scene.lights.empty()){
            float inv_sqrt2 = 0.70710678f;
            scene.lights.push_back({0, {0,0,0}, {-inv_sqrt2, 0.f, inv_sqrt2}, {0.4f, 0.4f, 0.4f}, 0,0,0, 0,0});
            scene.lights.push_back({0, {0,0,0}, {0.f, -inv_sqrt2, inv_sqrt2}, {0.3f, 0.3f, 0.3f}, 0,0,0, 0,0});
            scene.lights.push_back({0, {0,0,0}, {0.f, inv_sqrt2, inv_sqrt2}, {0.2f, 0.2f, 0.2f}, 0,0,0, 0,0});
            RL_TOOLS_RENDERING_RAYTRACING_LOG("Scene file has no lights: adding 3 directional fill lights");
        }
        return true;
    }

    namespace rendering::raytracing::detail{
        // content-hash lookup into the library's owned scenes; loads the file into a fresh
        // library-owned scene on a miss. Callers key their per-scene data off the returned index.
        template <typename DEVICE, typename SPEC, typename BACKEND>
        typename SPEC::TI library_lookup_or_load(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, BACKEND>& library, const char* scene_path, bool& is_new){
            using TI = typename SPEC::TI;
            uint64_t hash = 0;
            utils::assert_exit(device, hash_scene_file(scene_path, hash), "library: failed to read scene file");
            for(TI scene_i = 0; scene_i < (TI)library.hashes.size(); scene_i++){
                if(library.hashes[scene_i] == hash){
                    is_new = false;
                    RL_TOOLS_RENDERING_RAYTRACING_LOG("library: scene " << scene_path << " shares build " << scene_i << " (content hash match)");
                    return scene_i;
                }
            }
            is_new = true;
            library.hashes.push_back(hash);
            library.scenes.emplace_back();
            library.assets.push_back(nullptr);
            const bool loaded = load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, library.scenes.back(), std::string(scene_path));
            utils::assert_exit(device, loaded, "library: failed to load scene");
            return (TI)(library.scenes.size() - 1);
        }
    }

    template <typename DEVICE>
    size_t add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::Object& object, const float transform[12]){
        scene.objects.push_back(object);
        scene.instances.push_back({scene.objects.size() - 1, {}, rendering::raytracing::detail::transform_is_identity(transform)});
        std::memcpy(scene.instances.back().transform, transform, 12 * sizeof(float));
        return scene.instances.size() - 1;
    }

    template <typename DEVICE>
    size_t add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::Object& object){
        const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        return add(device, scene, object, identity);
    }

    template <typename DEVICE>
    size_t add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::Mesh& mesh){
        rendering::raytracing::Object object;
        object.meshes.push_back(mesh);
        return add(device, scene, object);
    }

    template <typename SHADING = rendering::raytracing::VeryHigh, bool HAS_RGB = true, typename DEVICE>
    bool load(DEVICE& device, rendering::raytracing::ObjectAssembly& assembly, const std::string& filename){
        return rendering::raytracing::detail::load_assembly_data<SHADING, HAS_RGB>(assembly, filename);
    }

    template <typename DEVICE>
    rendering::raytracing::Placement add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::ObjectAssembly& assembly, const float transform[12]){
        rendering::raytracing::Placement placement{scene.instances.size(), assembly.parts.size()};
        const size_t object_base = scene.objects.size();
        scene.objects.insert(scene.objects.end(), assembly.objects.begin(), assembly.objects.end());
        for(const auto& part : assembly.parts){
            float composed[12];
            rendering::raytracing::detail::compose_transforms(transform, part.transform, composed);
            scene.instances.push_back({object_base + part.object, {}, rendering::raytracing::detail::transform_is_identity(composed)});
            std::memcpy(scene.instances.back().transform, composed, sizeof(composed));
        }
        return placement;
    }

    template <typename DEVICE>
    rendering::raytracing::Placement add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::ObjectAssembly& assembly){
        const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        return add(device, scene, assembly, identity);
    }

    template <typename DEVICE>
    rendering::raytracing::AssetHandle add(DEVICE& device, rendering::raytracing::AssetPool& pool, const rendering::raytracing::ObjectAssembly& assembly){
        pool.assemblies.push_back(assembly);
        return {pool.assemblies.size() - 1};
    }

    template <typename DEVICE>
    rendering::raytracing::AssetHandle add(DEVICE& device, rendering::raytracing::AssetPool& pool, const rendering::raytracing::Object& object){
        rendering::raytracing::ObjectAssembly assembly;
        assembly.objects.push_back(object);
        assembly.parts.push_back({0, {1,0,0,0, 0,1,0,0, 0,0,1,0}});
        return add(device, pool, assembly);
    }

    template <typename DEVICE>
    rendering::raytracing::AssetHandle add(DEVICE& device, rendering::raytracing::AssetPool& pool, const rendering::raytracing::Mesh& mesh){
        rendering::raytracing::Object object;
        object.meshes.push_back(mesh);
        return add(device, pool, object);
    }

    // Validation predicates for boundaries (language bindings, C interface) that must not trip
    // the fail-fast asserts inside the verbs: check first, then call.
    template <typename DEVICE, typename SPEC, typename BACKEND>
    bool can_attach(DEVICE& device, const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, typename SPEC::TI camera, rendering::raytracing::OverlayIndex overlay){
        static_assert(SPEC::ENABLE_OVERLAYS, "can_attach requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        if(camera >= SPEC::NUM_CAMERAS || overlay.index >= SPEC::NUM_OVERLAYS){
            return false;
        }
        const TI* row = &renderer.attachments[camera * SPEC::MAX_OVERLAYS_PER_CAMERA];
        for(TI slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++){
            if(row[slot] == (TI)overlay.index || row[slot] == rendering::raytracing::Renderer<SPEC, BACKEND>::INVALID_OVERLAY){
                return true;
            }
        }
        return false;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    bool can_spawn(DEVICE& device, const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, rendering::raytracing::AssetHandle asset){
        static_assert(SPEC::ENABLE_OVERLAYS, "can_spawn requires an overlay-enabled renderer specification");
        if(overlay.index >= SPEC::NUM_OVERLAYS || asset.index >= renderer.assets.size()){
            return false;
        }
        return rendering::raytracing::detail::first_fit_slot<SPEC>(renderer, overlay.index, renderer.assets[asset.index].num_parts) < SPEC::MAX_OVERLAY_INSTANCES;
    }

    // Overlay verbs mutate host-side truth on the renderer and mark it dirty; update(device,
    // renderer) is the single point where the backend consumes it. All bookkeeping is
    // deterministic: slot allocation is a first-fit scan, so identical call sequences yield
    // identical slots (and therefore identical global instance ids).
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void attach(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, typename SPEC::TI camera, rendering::raytracing::OverlayIndex overlay){
        static_assert(SPEC::ENABLE_OVERLAYS, "attach requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        utils::assert_exit(device, overlay.index < SPEC::NUM_OVERLAYS, "attach: overlay index out of range");
        constexpr TI INVALID = rendering::raytracing::Renderer<SPEC, BACKEND>::INVALID_OVERLAY;
        TI* row = &renderer.attachments[camera * SPEC::MAX_OVERLAYS_PER_CAMERA];
        TI free_slot = SPEC::MAX_OVERLAYS_PER_CAMERA;
        for(TI slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++){
            if(row[slot] == (TI)overlay.index){
                return; // a camera's attachments form a set: attach is idempotent
            }
            if(row[slot] == INVALID && free_slot == SPEC::MAX_OVERLAYS_PER_CAMERA){
                free_slot = slot;
            }
        }
        utils::assert_exit(device, free_slot < SPEC::MAX_OVERLAYS_PER_CAMERA, "attach: camera already holds MAX_OVERLAYS_PER_CAMERA overlays");
        row[free_slot] = (TI)overlay.index;
        renderer.attachments_dirty = true;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void detach(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, typename SPEC::TI camera, rendering::raytracing::OverlayIndex overlay){
        static_assert(SPEC::ENABLE_OVERLAYS, "detach requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        TI* row = &renderer.attachments[camera * SPEC::MAX_OVERLAYS_PER_CAMERA];
        for(TI slot = 0; slot < SPEC::MAX_OVERLAYS_PER_CAMERA; slot++){
            if(row[slot] == (TI)overlay.index){
                row[slot] = rendering::raytracing::Renderer<SPEC, BACKEND>::INVALID_OVERLAY;
                renderer.attachments_dirty = true;
                return;
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    rendering::raytracing::OverlayPlacement spawn(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, rendering::raytracing::AssetHandle asset, const float transform[12]){
        static_assert(SPEC::ENABLE_OVERLAYS, "spawn requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        if(overlay.index >= SPEC::NUM_OVERLAYS){
            utils::assert_exit(device, false, "spawn: overlay index out of range");
            return {0, 0, 0};
        }
        if(asset.index >= renderer.assets.size()){
            utils::assert_exit(device, false, "spawn: asset handle out of range");
            return {0, 0, 0};
        }
        auto& state = renderer.overlays[overlay.index];
        const auto& record = renderer.assets[asset.index];

        const TI first_slot = rendering::raytracing::detail::first_fit_slot<SPEC>(renderer, overlay.index, record.num_parts);
        if(first_slot >= SPEC::MAX_OVERLAY_INSTANCES){
            utils::assert_exit(device, false, "spawn: overlay capacity exceeded");
            return {0, 0, 0};
        }

        for(TI part = 0; part < record.num_parts && first_slot + part < SPEC::MAX_OVERLAY_INSTANCES; part++){
            auto& slot = state.slots[first_slot + part];
            slot.object = renderer.asset_part_objects[record.first_part + part];
            slot.pose_slot = first_slot;
            std::memcpy(slot.part_local, &renderer.asset_part_transforms[(record.first_part + part) * 12], sizeof(slot.part_local));
            std::memcpy(slot.transform_entry, part == 0 ? transform : rendering::raytracing::detail::IDENTITY_TRANSFORM, sizeof(slot.transform_entry));
            slot.active = true;
            rendering::raytracing::detail::stage_motion_entry(renderer, (TI)overlay.index, first_slot + part, slot.transform_entry);
        }
        state.dirty = true;
        return {(size_t)first_slot, (size_t)record.num_parts, (size_t)record.first_part};
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void despawn(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement){
        static_assert(SPEC::ENABLE_OVERLAYS, "despawn requires an overlay-enabled renderer specification");
        auto& state = renderer.overlays[overlay.index];
        for(size_t part = 0; part < placement.num_parts; part++){
            state.slots[placement.first_slot + part].active = false;
        }
        state.dirty = true;
    }

    // per-part: part 0 sets the placement pose, other parts articulate in their part frame
    // (world = pose ∘ part_local ∘ articulation)
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transform(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, typename SPEC::TI part, const float transform[12]){
        static_assert(SPEC::ENABLE_OVERLAYS, "set_transform requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        auto& state = renderer.overlays[overlay.index];
        std::memcpy(state.slots[placement.first_slot + part].transform_entry, transform, 12 * sizeof(float));
        state.dirty = true;
        rendering::raytracing::detail::stage_motion_entry(renderer, (TI)overlay.index, (TI)(placement.first_slot + part), transform);
    }

    // rigid move: sets the placement pose and resets per-part articulation state
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transform(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const float transform[12]){
        static_assert(SPEC::ENABLE_OVERLAYS, "set_transform requires an overlay-enabled renderer specification");
        using TI = typename SPEC::TI;
        auto& state = renderer.overlays[overlay.index];
        std::memcpy(state.slots[placement.first_slot].transform_entry, transform, 12 * sizeof(float));
        rendering::raytracing::detail::stage_motion_entry(renderer, (TI)overlay.index, (TI)placement.first_slot, transform);
        for(size_t part = 1; part < placement.num_parts; part++){
            std::memcpy(state.slots[placement.first_slot + part].transform_entry, rendering::raytracing::detail::IDENTITY_TRANSFORM, 12 * sizeof(float));
            rendering::raytracing::detail::stage_motion_entry(renderer, (TI)overlay.index, (TI)(placement.first_slot + part), rendering::raytracing::detail::IDENTITY_TRANSFORM);
        }
        state.dirty = true;
    }

    // dynamic motion blur: per-part shutter-open/close entries, slerped into every motion sample
    // at the same midpoint shutter times the camera lerp uses; the close entry also becomes the
    // steady-state transform (segmentation, probes, and the next frame render at shutter close)
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transform_pair(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, typename SPEC::TI part, const float open[12], const float close[12]){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "set_transform_pair requires a dynamic-motion-blur renderer specification");
        using TI = typename SPEC::TI;
        auto& state = renderer.overlays[overlay.index];
        std::memcpy(state.slots[placement.first_slot + part].transform_entry, close, 12 * sizeof(float));
        state.dirty = true;
        constexpr size_t SLAB = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES * 12;
        for(TI sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
            const float shutter_t = ((float)sample + 0.5f) / (float)SPEC::MOTION_BLUR_SAMPLES;
            float entry[12];
            rendering::raytracing::detail::slerp_transform(open, close, shutter_t, entry);
            std::memcpy(renderer.transforms_motion_staging.data() + sample * SLAB + ((size_t)overlay.index * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot + part) * 12, entry, 12 * sizeof(float));
        }
        renderer.transforms_motion_dirty[overlay.index] = true;
    }

    // rigid move with shutter-open/close poses: resets articulation in every sample
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void set_transform_pair(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const float open[12], const float close[12]){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "set_transform_pair requires a dynamic-motion-blur renderer specification");
        using TI = typename SPEC::TI;
        set_transform_pair(device, renderer, overlay, placement, (TI)0, open, close);
        auto& state = renderer.overlays[overlay.index];
        for(size_t part = 1; part < placement.num_parts; part++){
            std::memcpy(state.slots[placement.first_slot + part].transform_entry, rendering::raytracing::detail::IDENTITY_TRANSFORM, 12 * sizeof(float));
            rendering::raytracing::detail::stage_motion_entry(renderer, (TI)overlay.index, (TI)(placement.first_slot + part), rendering::raytracing::detail::IDENTITY_TRANSFORM);
        }
        state.dirty = true;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& transforms(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::ENABLE_OVERLAYS, "transforms requires an overlay-enabled renderer specification");
        return renderer.transforms;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& transforms_motion(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "transforms_motion requires a dynamic-motion-blur renderer specification");
        return renderer.transforms_motion;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& transforms_pair(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "transforms_pair requires a dynamic-motion-blur renderer specification");
        return renderer.transforms_pair;
    }

    // camera input tensors, backend-native residency like transforms: device memory on OptiX,
    // host on generic, shared/mapped on Metal/Vulkan. Producers write them via rlt::copy or
    // kernels; the launch verbs consume them directly. Under motion blur the pair is
    // cameras_open (shutter open) and cameras_close (shutter close, aliasing cameras) — both
    // must be written each step (identical values for a blur-free frame).
    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        return renderer.cameras;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras_open(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::ENABLE_MOTION_BLUR, "cameras_open requires a motion-blur renderer specification");
        return renderer.cameras_open;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& cameras_close(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::ENABLE_MOTION_BLUR, "cameras_close requires a motion-blur renderer specification");
        return renderer.cameras;
    }

    // output tensors, same backend-native residency as the inputs: consumers on the device read
    // them in place (zero-copy); host readers stage through an explicit copy at readback
    // boundaries (after a _sync)
    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& frame_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_RGB, "frame_buffer requires an RGB-capable renderer specification");
        return renderer.frame_buffer;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& depth_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_DEPTH, "depth_buffer requires a depth-capable renderer specification");
        return renderer.depth_buffer;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& segmentation_buffer(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_SEGMENTATION, "segmentation_buffer requires a segmentation-capable renderer specification");
        return renderer.segmentation_buffer;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& collision_results(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        return renderer.collision_results;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    auto& observation(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        static_assert(SPEC::HAS_OBSERVATION, "observation requires OUTPUT_OBSERVATION in the renderer specification");
        return renderer.observation;
    }

    // decodes a rendered segmentation id back to the object it references. Single owner of the
    // global id layout: scene instances occupy [0, S); overlay o's slot s sits at S + o*CAP + s;
    // object indices count scene objects first, then each pool assembly's objects in
    // registration order. Returns nullptr for the miss sentinel and out-of-range ids.
    // This layout is cross-backend API surface: the generic flat instance array, Metal's user-ID
    // descriptors, and OptiX's user instance ids all realize it identically, and segmentation
    // consumers depend on that equivalence — treat any change to it as breaking.
    template <typename DEVICE, typename SPEC, typename BACKEND>
    const rendering::raytracing::Object* segmentation_object(DEVICE& device, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool, const rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, uint32_t id){
        if(id == 0xFFFFFFFFu){
            return nullptr;
        }
        if(id < scene.instances.size()){
            return &scene.objects[scene.instances[id].object];
        }
        if constexpr (SPEC::ENABLE_OVERLAYS){
            const size_t relative = id - scene.instances.size();
            const size_t overlay = relative / SPEC::MAX_OVERLAY_INSTANCES;
            const size_t slot = relative % SPEC::MAX_OVERLAY_INSTANCES;
            if(overlay >= SPEC::NUM_OVERLAYS){
                return nullptr;
            }
            size_t object = renderer.overlays[overlay].slots[slot].object;
            if(object < scene.objects.size()){
                return &scene.objects[object];
            }
            object -= scene.objects.size();
            for(const auto& assembly : pool.assemblies){
                if(object < assembly.objects.size()){
                    return &assembly.objects[object];
                }
                object -= assembly.objects.size();
            }
        }
        return nullptr;
    }

    inline void make_transform(const float position[3], const float orientation_wxyz[4], float out[12]){
        const float w = orientation_wxyz[0], x = orientation_wxyz[1], y = orientation_wxyz[2], z = orientation_wxyz[3];
        out[0] = 1 - 2*(y*y + z*z); out[1] = 2*(x*y - w*z);     out[2] = 2*(x*z + w*y);     out[3] = position[0];
        out[4] = 2*(x*y + w*z);     out[5] = 1 - 2*(x*x + z*z); out[6] = 2*(y*z - w*x);     out[7] = position[1];
        out[8] = 2*(x*z - w*y);     out[9] = 2*(y*z + w*x);     out[10] = 1 - 2*(x*x + y*y); out[11] = position[2];
    }

    inline void compose_transforms(const float a[12], const float b[12], float out[12]){
        rendering::raytracing::detail::compose_transforms(a, b, out);
    }

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT rendering::raytracing::Camera<T> make_camera_data(const T position[3], const T look_at[3], const T up[3], T fov, T aspect){
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
        rendering::raytracing::Camera<T> cam;
        v3::add(tmp, half_dv, cam.dir_00);
        cam.pos[0] = position[0]; cam.pos[1] = position[1]; cam.pos[2] = position[2];
        cam.dir_du[0] = du[0]; cam.dir_du[1] = du[1]; cam.dir_du[2] = du[2];
        v3::scale(dv, T{-1}, cam.dir_dv);
        return cam;
    }

    namespace rendering::raytracing::detail{
        // fills a host staging buffer; the backend-specific generate_cameras writes it into the
        // backend-native camera tensors
        template <typename SPEC, typename DEVICE>
        void generate_camera_poses(DEVICE& device, rendering::raytracing::Camera<typename SPEC::T>* cameras_out,
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

            cameras_out[i] = make_camera_data(cam_pos, center, up, fov, aspect);
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Generated " << SPEC::NUM_CAMERAS << " camera positions");
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Per-camera resolution: " << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT);
        }

        template <typename SPEC>
        std::vector<float> generate_probe_direction_vectors(){
            std::vector<float> dirs;
            dirs.reserve(SPEC::NUM_PROBES * 3);

            const float golden_ratio = (1.0f + sqrtf(5.0f)) / 2.0f;

            for(int i = 0; i < (int)SPEC::NUM_PROBES; i++){
                float theta = 2.0f * (float)M_PI * i / golden_ratio;
                float cos_inc = 1.0f - 2.0f * (i + 0.5f) / SPEC::NUM_PROBES;
                float sin_inc = sqrtf(1.0f - cos_inc * cos_inc);

                const float dir[3] = {sin_inc * cosf(theta), sin_inc * sinf(theta), cos_inc};
                const float inv_len = 1.0f / sqrtf(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
                dirs.push_back(dir[0] * inv_len);
                dirs.push_back(dir[1] * inv_len);
                dirs.push_back(dir[2] * inv_len);
            }

            RL_TOOLS_RENDERING_RAYTRACING_LOG("Generated " << SPEC::NUM_PROBES << " probe directions per camera");
            return dirs;
        }

        template <typename SPEC>
        void write_grid_png(const uint32_t* fb, const char* filename){
            using TI = typename SPEC::TI;
            constexpr TI cam_pixels = SPEC::CAM_PIXELS;
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
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Written grid image (" << SPEC::GRID_COLS << "x" << SPEC::GRID_ROWS
               << " cameras, " << grid_width << "x" << grid_height << " px) to " << filename);
        }

        // one distinct color per instance id (golden-ratio hue hash); miss sentinel renders black
        inline uint32_t segmentation_id_to_rgba(uint32_t id){
            if(id == 0xFFFFFFFFu){
                return 0xFF000000u;
            }
            const float hue = std::fmod((float)id * 0.61803398875f, 1.0f) * 6.0f;
            const float descending = 1.0f - std::fabs(std::fmod(hue, 2.0f) - 1.0f);
            float r = 0, g = 0, b = 0;
            switch((int)hue){
                case 0: r = 1; g = descending; break;
                case 1: r = descending; g = 1; break;
                case 2: g = 1; b = descending; break;
                case 3: g = descending; b = 1; break;
                case 4: r = descending; b = 1; break;
                default: r = 1; b = descending; break;
            }
            return 0xFF000000u | ((uint32_t)(b * 255) << 16) | ((uint32_t)(g * 255) << 8) | (uint32_t)(r * 255);
        }
        template <typename SPEC>
        void write_segmentation_grid_png(const uint32_t* segmentation, const char* filename){
            constexpr typename SPEC::TI num_pixels = SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
            std::vector<uint32_t> colored(num_pixels);
            for(size_t pixel_i = 0; pixel_i < (size_t)num_pixels; pixel_i++){
                colored[pixel_i] = segmentation_id_to_rgba(segmentation[pixel_i]);
            }
            write_grid_png<SPEC>(colored.data(), filename);
        }

        template <typename SPEC>
        void write_depth_grid_png(const float* depth_host, float camera_radius, const char* filename){
            using TI = typename SPEC::TI;
            constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        constexpr int grid_width = SPEC::GRID_COLS * SPEC::CAM_WIDTH;
        constexpr int grid_height = SPEC::GRID_ROWS * SPEC::CAM_HEIGHT;
        std::vector<uint32_t> grid_image(grid_width * grid_height, 0);
        constexpr size_t depth_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        const float max_depth = camera_radius > 0 ? camera_radius * 2.0f : 1e30f;
        const float valid_max_depth = max_depth * 0.999f;
        float min_valid_depth = std::numeric_limits<float>::max();
        float max_valid_depth = std::numeric_limits<float>::lowest();
        for(size_t depth_i = 0; depth_i < depth_count; depth_i++){
            const float depth = depth_host[depth_i];
            // no std::isfinite: unreliable under -ffast-math; the range check excludes inf/NaN
            if(depth > 0.f && depth < valid_max_depth){
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
                    if(has_valid_depth && depth > 0.f && depth < valid_max_depth){
                        const float normalized = fminf(fmaxf((depth - min_valid_depth) / (valid_depth_range + 1e-6f), 0.f), 1.f);
                        value = static_cast<uint8_t>(normalized * 255.f);
                    }
                    grid_image[(offset_y + y) * grid_width + offset_x + x] =
                        (0xFFu << 24) | (uint32_t(value) << 16) | (uint32_t(value) << 8) | uint32_t(value);
                }
            }
        }

        stbi_write_png(filename, grid_width, grid_height, 4,
                       grid_image.data(), grid_width * sizeof(uint32_t));
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Written depth image (" << SPEC::GRID_COLS << "x" << SPEC::GRID_ROWS
               << " cameras, " << grid_width << "x" << grid_height << " px) to " << filename);
        }

        template <typename SPEC>
        void write_depth_bin(const float* depth_host, const char* filename){
            constexpr size_t depth_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        FILE* f = fopen(filename, "wb");
        if(f){
            int nc = SPEC::NUM_CAMERAS;
            int h = SPEC::CAM_HEIGHT;
            int w = SPEC::CAM_WIDTH;
            fwrite(&nc, sizeof(int), 1, f);
            fwrite(&h, sizeof(int), 1, f);
            fwrite(&w, sizeof(int), 1, f);
            fwrite(depth_host, sizeof(float), depth_count, f);
            fclose(f);
            RL_TOOLS_RENDERING_RAYTRACING_LOG("Written depth data (" << depth_count << " values) to " << filename);
        }
        }

        template <typename SPEC>
        void write_probes_bin_and_log(const CollisionResult* probe_results, const char* filename){
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

        RL_TOOLS_RENDERING_RAYTRACING_LOG("=== COLLISION PROBE RESULTS ===");
        RL_TOOLS_RENDERING_RAYTRACING_LOG("  Total probes:  " << SPEC::NUM_CAMERAS * SPEC::NUM_PROBES);
        RL_TOOLS_RENDERING_RAYTRACING_LOG("  Hits:          " << total_hits
               << " (" << (100.0 * total_hits / (SPEC::NUM_CAMERAS * SPEC::NUM_PROBES)) << "%)");
        RL_TOOLS_RENDERING_RAYTRACING_LOG("  Misses:        " << (SPEC::NUM_CAMERAS * SPEC::NUM_PROBES - total_hits));
        if(total_hits > 0){
            RL_TOOLS_RENDERING_RAYTRACING_LOG("  Min hit dist:  " << min_hit_dist);
            RL_TOOLS_RENDERING_RAYTRACING_LOG("  Max hit dist:  " << max_hit_dist);
            RL_TOOLS_RENDERING_RAYTRACING_LOG("  Avg hit dist:  " << (sum_hit_dist / total_hits));
        }
        RL_TOOLS_RENDERING_RAYTRACING_LOG("===============================");

        {
            FILE* f = fopen(filename, "wb");
            if(f){
                int nc = SPEC::NUM_CAMERAS, np = SPEC::NUM_PROBES;
                fwrite(&nc, sizeof(int), 1, f);
                fwrite(&np, sizeof(int), 1, f);
                fwrite(probe_results, sizeof(CollisionResult),
                       (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES, f);
                fclose(f);
                RL_TOOLS_RENDERING_RAYTRACING_LOG("Written probe data (" << SPEC::NUM_CAMERAS * SPEC::NUM_PROBES
                       << " results) to " << filename);
            }
        }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
