#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_GLB_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_GLB_OPERATIONS_CPU_H

#include "../operations_cpu.h"

// STATIC gives the stb implementation internal linkage so multiple TUs of one binary may include
// this header without duplicate-symbol link errors; RL_TOOLS_STB_IMAGE_PROVIDED arbitrates with
// other stb-providing headers (e.g. the test golden_io.h) so the implementation lands exactly
// once per TU regardless of include order (stb's implementation section has no include guard).
#ifndef RL_TOOLS_STB_IMAGE_PROVIDED
#define RL_TOOLS_STB_IMAGE_PROVIDED
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
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    // =========================================================================
    // Decode an embedded texture from assimp into RGBA8 pixels
    // =========================================================================
    namespace rendering::datasets::glb{
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
    namespace rendering::datasets::glb{
        struct Material {
            float metallic;
            float roughness;
            int metallic_roughness_image = -1; // glTF image index, -1 when the material has no MR texture
        };
        struct ParsedMetadata {
            std::vector<rendering::SceneLight> lights; // world frame (welded scene loads)
            std::map<int, std::vector<rendering::SceneLight>> root_lights; // by scene-root ordinal, in the root node's frame (assembly loads)
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
                    rendering::SceneLight sl{};
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

    namespace rendering::datasets::glb{
    struct DecodedTexture{
        std::vector<uint8_t> pixels;
        int width;
        int height;
    };
    struct MeshConversionState{
        std::map<std::string, size_t> tex_cache;
        std::vector<DecodedTexture> decoded_textures;
        std::map<std::string, RepresentativeTextureColor> representative_texture_color_cache;
        size_t representative_texture_color_meshes = 0;
        size_t representative_texture_color_decoded = 0;
    };

    template <typename SHADING, bool HAS_RGB>
    rendering::Mesh convert_mesh(const aiScene* scene, const aiMesh* mesh, const aiMatrix4x4& global_transform, const ParsedMetadata& glb_metadata, MeshConversionState& state){
        rendering::Mesh md;
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
                RepresentativeTextureColor representative_color;
                bool has_representative_color = load_representative_texture_color(
                    scene, mat, aiTextureType_DIFFUSE, state.representative_texture_color_cache,
                    representative_color, state.representative_texture_color_decoded
                );
                if(!has_representative_color) {
                    has_representative_color = load_representative_texture_color(
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
                            if(decode_embedded_texture(emb_tex, pixels, w, h)){
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
                            if (decode_embedded_texture(emb_tex, pixels, w, h)) {
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
                            if (decode_image_bytes(image->second, pixels, w, h)) {
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
                            if (decode_embedded_texture(emb_tex, pixels, w, h)) {
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
                            if (decode_embedded_texture(emb_tex, pixels, w, h)) {
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
                            if (decode_embedded_texture(emb_tex, pixels, w, h)) {
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
                        if(decode_embedded_texture(emb_tex, pixels, w, h)){
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
    bool load_scene_data(std::vector<rendering::Mesh>& out_meshes, std::vector<rendering::SceneLight>& out_lights, const std::string& filename){
        Assimp::Importer importer;
        unsigned int import_flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality;
        if constexpr (HAS_RGB && SHADING::PBR_SHADING) {
            import_flags |= aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace;
        } else if constexpr (HAS_RGB && SHADING::NORMAL_SHADING) {
            import_flags |= aiProcess_GenNormals;
        }
        const aiScene* scene = importer.ReadFile(filename, import_flags);

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode){
            RL_TOOLS_RENDERING_DATASETS_LOG_ERR("Assimp error: " << importer.GetErrorString());
            return false;
        }

        RL_TOOLS_RENDERING_DATASETS_LOG("Loaded model with " << scene->mNumMeshes << " mesh(es)");

        [[maybe_unused]] MeshConversionState conversion_state;

        size_t total_verts = 0, total_tris = 0;

        const size_t first_new_mesh = out_meshes.size();
        const auto glb_metadata = parse(filename);

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
            rendering::Mesh md = convert_mesh<SHADING, HAS_RGB>(scene, mesh, global_transform, glb_metadata, conversion_state);
            total_verts += md.vertices.size() / 3;
            total_tris += md.indices.size() / 3;
            out_meshes.push_back(std::move(md));
            } // end for global_transform
        }

        RL_TOOLS_RENDERING_DATASETS_LOG("Total vertices: " << total_verts << ", triangles: " << total_tris);

        const size_t num_new_meshes = out_meshes.size() - first_new_mesh;
        int textured_count = 0;
        int metallic_count = 0;
        for(size_t m = first_new_mesh; m < out_meshes.size(); m++){
            if(out_meshes[m].texture.present()) textured_count++;
            if(out_meshes[m].metallic > 0.f) metallic_count++;
        }
        RL_TOOLS_RENDERING_DATASETS_LOG("Meshes with textures: " << textured_count << "/" << num_new_meshes
              << ", metallic: " << metallic_count << "/" << num_new_meshes);
        if constexpr (HAS_RGB && !SHADING::LOAD_TEXTURES) {
            RL_TOOLS_RENDERING_DATASETS_LOG("Representative texture colors: " << conversion_state.representative_texture_color_meshes
                  << "/" << num_new_meshes << " meshes, decoded "
                  << conversion_state.representative_texture_color_decoded << " texture(s)");
        }

        for (auto& sl : glb_metadata.lights) {
            out_lights.push_back(sl);
        }

        return true;
    }

    // conjugates an Assimp (glTF Y-up) node transform into the FLU frame: T_flu = S * T * S^-1,
    // with S the same Y-up -> FLU swizzle that is applied to vertices
    inline void flu_from_assimp(const aiMatrix4x4& transform, float out[12]){
        out[0] = transform.a1;  out[1] = -transform.a3; out[2]  = transform.a2;  out[3]  = transform.a4;
        out[4] = -transform.c1; out[5] = transform.c3;  out[6]  = -transform.c2; out[7]  = -transform.c4;
        out[8] = transform.b1;  out[9] = -transform.b3; out[10] = transform.b2;  out[11] = transform.b4;
    }

    template <typename SHADING, bool HAS_RGB>
    bool load_assembly_data(rendering::ObjectAssembly& assembly, const std::string& filename){
        Assimp::Importer importer;
        unsigned int import_flags = aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality;
        if constexpr (HAS_RGB && SHADING::PBR_SHADING) {
            import_flags |= aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace;
        } else if constexpr (HAS_RGB && SHADING::NORMAL_SHADING) {
            import_flags |= aiProcess_GenNormals;
        }
        const aiScene* scene = importer.ReadFile(filename, import_flags);
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode){
            RL_TOOLS_RENDERING_DATASETS_LOG_ERR("Assimp error: " << importer.GetErrorString());
            return false;
        }
        const auto glb_metadata = parse(filename);
        MeshConversionState conversion_state; // shared across parts: textures used by several parts decode once

        const aiNode* file_root = scene->mRootNode;
        const size_t first_new_part = assembly.parts.size();
        for(unsigned int root_i = 0; root_i < file_root->mNumChildren; root_i++){
            const aiNode* root = file_root->mChildren[root_i];
            rendering::Object object;
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

            rendering::ObjectAssembly::Part part;
            part.object = assembly.objects.size();
            flu_from_assimp(root->mTransformation, part.transform);
            assembly.objects.push_back(std::move(object));
            assembly.parts.push_back(part);
        }
        RL_TOOLS_RENDERING_DATASETS_LOG("Loaded assembly with " << (assembly.parts.size() - first_new_part) << " part(s)");
        return true;
    }
    }

    template <typename SHADING = rendering::VeryHigh, bool HAS_RGB = true, typename DEVICE>
    bool load(DEVICE& device, rendering::Object& object, const std::string& reference){
        std::string path;
        if(!rendering::datasets::resolve_reference(device, reference, path)){
            return false;
        }
        return rendering::datasets::glb::load_scene_data<SHADING, HAS_RGB>(object.meshes, object.lights, path);
    }

    template <typename SHADING = rendering::VeryHigh, bool HAS_RGB = true, typename DEVICE>
    bool load(DEVICE& device, rendering::ObjectAssembly& assembly, const std::string& reference){
        std::string path;
        if(!rendering::datasets::resolve_reference(device, reference, path)){
            return false;
        }
        return rendering::datasets::glb::load_assembly_data<SHADING, HAS_RGB>(assembly, path);
    }

    // Appends the scene file as one welded static object placed at identity: explicit
    // composition, the append twin of load below. Composition does not touch the metadata —
    // finish a composed bundle with datasets::compute_bounds and datasets::content_hash.
    template <typename SHADING = rendering::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T>
    bool add(DEVICE& device, rendering::Bundle<T>& bundle, const std::string& reference){
        std::string path;
        if(!rendering::datasets::resolve_reference(device, reference, path)){
            return false;
        }
        rendering::Object object;
        if(!rendering::datasets::glb::load_scene_data<SHADING, HAS_RGB>(object.meshes, bundle.scene.lights, path)){
            return false;
        }
        bundle.scene.objects.push_back(std::move(object));
        bundle.scene.instances.push_back({bundle.scene.objects.size() - 1, {1,0,0,0, 0,1,0,0, 0,0,1,0}, true});
        return true;
    }

    // Bundle-level load: the scene file becomes one welded static object placed at identity and
    // the metadata (bounds, max ray length, content hash) is loader-filled. The bundle must be
    // empty — reusing one across loads silently accumulates geometry and no rendering test can
    // see it, so it is a hard failure; composition is the explicit add(device, bundle, reference).
    // A scene file without any punctual lights gets a neutral 3-directional fill so PBR-shaded
    // content is not lit by ambient only. Object/asset loads deliberately do not: fill lighting
    // is a scene decision, not an asset property.
    template <typename SHADING = rendering::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T>
    bool load(DEVICE& device, rendering::Bundle<T>& bundle, const std::string& reference){
        utils::assert_exit(device, bundle.scene.objects.empty() && bundle.scene.instances.empty() && bundle.scene.lights.empty(), "load: bundle is not empty — use add(device, bundle, reference) to compose");
        if(!add<SHADING, HAS_RGB>(device, bundle, reference)){
            return false;
        }
        if(bundle.scene.lights.empty()){
            float inv_sqrt2 = 0.70710678f;
            bundle.scene.lights.push_back({0, {0,0,0}, {-inv_sqrt2, 0.f, inv_sqrt2}, {0.4f, 0.4f, 0.4f}, 0,0,0, 0,0});
            bundle.scene.lights.push_back({0, {0,0,0}, {0.f, -inv_sqrt2, inv_sqrt2}, {0.3f, 0.3f, 0.3f}, 0,0,0, 0,0});
            bundle.scene.lights.push_back({0, {0,0,0}, {0.f, inv_sqrt2, inv_sqrt2}, {0.2f, 0.2f, 0.2f}, 0,0,0, 0,0});
            RL_TOOLS_RENDERING_DATASETS_LOG("Scene file has no lights: adding 3 directional fill lights");
        }
        rendering::datasets::compute_bounds(device, bundle);
        std::string path;
        rendering::datasets::resolve_reference(device, reference, path);
        utils::assert_exit(device, rendering::datasets::content_hash(device, path, bundle.metadata.content_hash), "load: failed to hash scene file");
        return true;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
