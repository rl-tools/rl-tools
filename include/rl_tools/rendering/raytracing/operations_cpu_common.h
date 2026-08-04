#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_COMMON_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_COMMON_H

#include "renderer.h"

// STATIC gives the stb implementations internal linkage so multiple TUs of one binary may include
// this header without duplicate-symbol link errors.
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
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
        struct MaterialFactors {
            float metallic;
            float roughness;
        };
        struct ParsedMetadata {
            std::vector<rendering::raytracing::SceneLight> lights;
            std::map<std::string, MaterialFactors> material_factors; // by material name; glTF spec defaults when absent
        };

        static inline void node_world_transform(const nlohmann::json& nodes, int node_idx, const std::vector<int>& parent_map, float out[16]) {
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
            fclose(f);

            nlohmann::json gltf = nlohmann::json::parse(json_str, nullptr, false);
            if (gltf.is_discarded()) return result;

            // Assimp's defaults for absent glTF pbrMetallicRoughness factors vary across versions;
            // the GLB JSON is authoritative here so scene import is identical on every machine.
            // Absent metallicFactor resolves to 1 when a metallicRoughnessTexture is present (the
            // factor is the texture's multiplier, the texture decides per texel) and to 0 otherwise
            // (untextured materials without an explicit factor are authored as non-metallic in this
            // pipeline; the glTF spec default of 1 would render them as mirrors). Absent
            // roughnessFactor resolves to the spec default 1.
            if (gltf.contains("materials")) {
                for (auto& material : gltf["materials"]) {
                    if (!material.contains("name") || !material.contains("pbrMetallicRoughness")) continue;
                    auto& pbr = material["pbrMetallicRoughness"];
                    MaterialFactors factors{1.0f, 1.0f};
                    if (pbr.contains("metallicFactor")) {
                        factors.metallic = pbr["metallicFactor"].get<float>();
                    } else {
                        factors.metallic = pbr.contains("metallicRoughnessTexture") ? 1.0f : 0.0f;
                    }
                    if (pbr.contains("roughnessFactor")) {
                        factors.roughness = pbr["roughnessFactor"].get<float>();
                    }
                    result.material_factors[material["name"].get<std::string>()] = factors;
                }
            }

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

                    if (type_str == "directional") {
                        swizzle_gltf_direction_to_renderer(world[8], world[9], world[10], sl.direction);
                    } else if (type_str == "spot") {
                        float inner = 0.0f;
                        float outer = 0.7854f;
                        if (ldef.contains("spot")) {
                            inner = ldef["spot"].value("innerConeAngle", inner);
                            outer = ldef["spot"].value("outerConeAngle", outer);
                        }
                        sl.cos_inner_cone = cosf(inner);
                        sl.cos_outer_cone = cosf(outer);
                        swizzle_gltf_direction_to_renderer(-world[8], -world[9], -world[10], sl.direction);
                    }

                    result.lights.push_back(sl);
                }
            }

            return result;
        }
    }

    // =========================================================================
    // load: Assimp scene/object loading
    // =========================================================================
    namespace rendering::raytracing::detail{
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

        [[maybe_unused]] std::map<std::string, size_t> tex_cache;
        struct DecodedTex { std::vector<uint8_t> pixels; int w, h; };
        [[maybe_unused]] std::vector<DecodedTex> decoded_textures;
        [[maybe_unused]] std::map<std::string, rendering::raytracing::RepresentativeTextureColor> representative_texture_color_cache;
        [[maybe_unused]] size_t representative_texture_color_meshes = 0;
        [[maybe_unused]] size_t representative_texture_color_decoded = 0;

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
                        scene, mat, aiTextureType_DIFFUSE, representative_texture_color_cache,
                        representative_color, representative_texture_color_decoded
                    );
                    if(!has_representative_color) {
                        has_representative_color = rendering::raytracing::load_representative_texture_color(
                            scene, mat, aiTextureType_BASE_COLOR, representative_texture_color_cache,
                            representative_color, representative_texture_color_decoded
                        );
                    }
                    if(has_representative_color) {
                        md.color[0] *= representative_color.color[0];
                        md.color[1] *= representative_color.color[1];
                        md.color[2] *= representative_color.color[2];
                        representative_texture_color_meshes++;
                    }
                }

                if constexpr (SHADING::LOAD_TEXTURES) {
                if(mat->GetTextureCount(aiTextureType_DIFFUSE) > 0){
                    aiString tex_path;
                    if(mat->GetTexture(aiTextureType_DIFFUSE, 0, &tex_path) == AI_SUCCESS){
                        std::string path_str(tex_path.C_Str());

                        auto it = tex_cache.find(path_str);
                        if(it != tex_cache.end()){
                            auto& cached = decoded_textures[it->second];
                            md.texture.pixels = cached.pixels;
                            md.texture.width = cached.w;
                            md.texture.height = cached.h;
                        } else {
                            const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                            if(emb_tex){
                                int w, h;
                                std::vector<uint8_t> pixels;
                                if(rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)){
                                    md.texture.pixels = pixels;
                                    md.texture.width = w;
                                    md.texture.height = h;
                                    tex_cache[path_str] = decoded_textures.size();
                                    decoded_textures.push_back({std::move(pixels), w, h});
                                }
                            } else if(!path_str.empty()){
                                int w, h, channels;
                                unsigned char* data = stbi_load(path_str.c_str(), &w, &h, &channels, 4);
                                if(data){
                                    md.texture.pixels.assign(data, data + w * h * 4);
                                    md.texture.width = w;
                                    md.texture.height = h;
                                    stbi_image_free(data);
                                    tex_cache[path_str] = decoded_textures.size();
                                    decoded_textures.push_back({md.texture.pixels, w, h});
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

                    if (mat->GetTextureCount(aiTextureType_UNKNOWN) > 0) {
                        aiString tex_path;
                        if (mat->GetTexture(aiTextureType_UNKNOWN, 0, &tex_path) == AI_SUCCESS) {
                            const aiTexture* emb_tex = scene->GetEmbeddedTexture(tex_path.C_Str());
                            if (emb_tex) {
                                int w, h;
                                std::vector<uint8_t> pixels;
                                if (rendering::raytracing::decode_embedded_texture(emb_tex, pixels, w, h)) {
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
                    const auto glb_factors = glb_metadata.material_factors.find(mat->GetName().C_Str());
                    if(glb_factors != glb_metadata.material_factors.end()){
                        md.metallic = glb_factors->second.metallic;
                        if constexpr (SHADING::PBR_SHADING) {
                            md.roughness = glb_factors->second.roughness;
                        }
                    }
                }
            }

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
            RL_TOOLS_RENDERING_RAYTRACING_LOG("Representative texture colors: " << representative_texture_color_meshes
                  << "/" << num_new_meshes << " meshes, decoded "
                  << representative_texture_color_decoded << " texture(s)");
        }

        for (auto& sl : glb_metadata.lights) {
            out_lights.push_back(sl);
        }

        return true;
    }

    template <typename SPEC>
    void compute_scene_bounds(rendering::raytracing::Renderer<SPEC>& renderer, const rendering::raytracing::Scene& scene){
        float bbox_min[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        float bbox_max[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
        for(const auto& mesh : scene.meshes){
            for(size_t vertex_i = 0; vertex_i + 2 < mesh.vertices.size(); vertex_i += 3){
                for(int d = 0; d < 3; d++){
                    bbox_min[d] = std::min(bbox_min[d], mesh.vertices[vertex_i + d]);
                    bbox_max[d] = std::max(bbox_max[d], mesh.vertices[vertex_i + d]);
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

    // Lights as uploaded to the device: only the PBR tiers consume punctual lights
    template <bool APPLY>
    std::vector<rendering::raytracing::SceneLight> effective_scene_lights(const rendering::raytracing::Scene& scene){
        std::vector<rendering::raytracing::SceneLight> lights;
        if constexpr (APPLY) {
            lights = scene.lights;
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

    // Scene-level load: a scene file without any punctual lights gets a neutral 3-directional
    // fill so PBR-shaded content is not lit by ambient only. Object/asset loads deliberately do
    // not: fill lighting is a scene decision, not an asset property.
    template <typename SHADING = rendering::raytracing::VeryHigh, bool HAS_RGB = true, typename DEVICE>
    bool load(DEVICE& device, rendering::raytracing::Scene& scene, const std::string& filename){
        if(!rendering::raytracing::detail::load_scene_data<SHADING, HAS_RGB>(scene.meshes, scene.lights, filename)){
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

    template <typename DEVICE>
    void add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::Object& object){
        scene.meshes.insert(scene.meshes.end(), object.meshes.begin(), object.meshes.end());
        scene.lights.insert(scene.lights.end(), object.lights.begin(), object.lights.end());
    }

    template <typename DEVICE>
    void add(DEVICE& device, rendering::raytracing::Scene& scene, const rendering::raytracing::Mesh& mesh){
        scene.meshes.push_back(mesh);
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
        template <typename DEVICE, typename SPEC>
        void generate_camera_poses(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer,
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
