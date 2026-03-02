#pragma once

#include "renderer.h"
#include "device.h"

#include "owl/owl.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <vector>
#include <limits>
#include <algorithm>
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

extern "C" char device_ptx[];

namespace rl_tools{

    // =========================================================================
    // Default cube geometry
    // =========================================================================
    namespace rendering::raytracing::constants{
        const int NUM_VERTICES = 8;
        const vec3f default_vertices[8] = {
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
        const vec3i default_indices[12] = {
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

    // =========================================================================
    // malloc: create OWL contexts and allocate buffers
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        using TI = typename SPEC::TI;

        // RGB rendering context
        OWLContext context = owlContextCreate(nullptr, 1);
        OWLModule module = owlModuleCreate(context, device_ptx);

        constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        OWLBuffer frame_buffer = owlDeviceBufferCreate(context, OWL_INT,
                                                        (size_t)SPEC::NUM_CAMERAS * cam_pixels, nullptr);

        // Miss program
        OWLVarDecl miss_prog_vars[] = {
            { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color0)},
            { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color1)},
            { /* sentinel */ }
        };
        OWLMissProg miss_prog = owlMissProgCreate(context, module, "miss",
                                                    sizeof(MissProgData), miss_prog_vars, -1);
        owlMissProgSet3f(miss_prog, "color0", owl3f{.8f, 0.f, 0.f});
        owlMissProgSet3f(miss_prog, "color1", owl3f{.8f, .8f, .8f});

        // Ray gen
        OWLVarDecl ray_gen_vars[] = {
            { "fbPtr",       OWL_BUFPTR, OWL_OFFSETOF(RayGenData, fbPtr)},
            { "fbSize",      OWL_INT2,   OWL_OFFSETOF(RayGenData, fbSize)},
            { "camSize",     OWL_INT2,   OWL_OFFSETOF(RayGenData, camSize)},
            { "gridCols",    OWL_INT,    OWL_OFFSETOF(RayGenData, gridCols)},
            { "numCameras",  OWL_INT,    OWL_OFFSETOF(RayGenData, numCameras)},
            { "world",       OWL_GROUP,  OWL_OFFSETOF(RayGenData, world)},
            { "cameras",     OWL_BUFPTR, OWL_OFFSETOF(RayGenData, cameras)},
            { /* sentinel */ }
        };
        OWLRayGen ray_gen = owlRayGenCreate(context, module, "simpleRayGen",
                                             sizeof(RayGenData), ray_gen_vars, -1);

        const owl2i fb_size  = {(int)SPEC::FB_WIDTH, (int)SPEC::FB_HEIGHT};
        const owl2i cam_size = {(int)SPEC::CAM_WIDTH, (int)SPEC::CAM_HEIGHT};

        owlRayGenSetBuffer(ray_gen, "fbPtr", frame_buffer);
        owlRayGenSet2i    (ray_gen, "fbSize", fb_size);
        owlRayGenSet2i    (ray_gen, "camSize", cam_size);
        owlRayGenSet1i    (ray_gen, "gridCols", SPEC::GRID_COLS);
        owlRayGenSet1i    (ray_gen, "numCameras", SPEC::NUM_CAMERAS);

        renderer.context = context;
        renderer.module = module;
        renderer.ray_gen = ray_gen;
        renderer.frame_buffer = frame_buffer;

        // Collision context
        OWLContext coll_context = owlContextCreate(nullptr, 1);
        OWLModule coll_module = owlModuleCreate(coll_context, device_ptx);

        OWLVarDecl collision_ray_gen_vars[] = {
            { "results",         OWL_BUFPTR, OWL_OFFSETOF(CollisionRayGenData, results)},
            { "probeDirections", OWL_BUFPTR, OWL_OFFSETOF(CollisionRayGenData, probeDirections)},
            { "cameras",         OWL_BUFPTR, OWL_OFFSETOF(CollisionRayGenData, cameras)},
            { "world",           OWL_GROUP,  OWL_OFFSETOF(CollisionRayGenData, world)},
            { "numProbes",       OWL_INT,    OWL_OFFSETOF(CollisionRayGenData, numProbes)},
            { "numCameras",      OWL_INT,    OWL_OFFSETOF(CollisionRayGenData, numCameras)},
            { "maxDist",         OWL_FLOAT,  OWL_OFFSETOF(CollisionRayGenData, maxDist)},
            { /* sentinel */ }
        };
        OWLRayGen collision_ray_gen = owlRayGenCreate(coll_context, coll_module, "collisionRayGen",
                                                       sizeof(CollisionRayGenData),
                                                       collision_ray_gen_vars, -1);

        OWLBuffer collision_results_buffer = owlHostPinnedBufferCreate(coll_context, OWL_USER_TYPE(CollisionResult),
                                                                        (size_t)SPEC::NUM_CAMERAS * SPEC::NUM_PROBES);

        renderer.coll_context = coll_context;
        renderer.coll_module = coll_module;
        renderer.collision_ray_gen = collision_ray_gen;
        renderer.collision_results_buffer = collision_results_buffer;
    }

    // =========================================================================
    // load_model: Assimp model loading
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    bool load_model(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const std::string& filename){
        using T = typename SPEC::T;

        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(filename,
            aiProcess_Triangulate |
            aiProcess_GenNormals |
            aiProcess_JoinIdenticalVertices |
            aiProcess_PreTransformVertices |
            aiProcess_ImproveCacheLocality);

        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode){
            RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Assimp error: " << importer.GetErrorString());
            return false;
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Loaded model with " << scene->mNumMeshes << " mesh(es)");

        vec3f bbox_min(std::numeric_limits<float>::max());
        vec3f bbox_max(std::numeric_limits<float>::lowest());

        std::map<std::string, size_t> tex_cache;
        struct DecodedTex { std::vector<uint8_t> pixels; int w, h; };
        std::vector<DecodedTex> decoded_textures;

        size_t total_verts = 0, total_tris = 0;

        renderer.meshes.clear();

        for(unsigned int m = 0; m < scene->mNumMeshes; m++){
            const aiMesh* mesh = scene->mMeshes[m];
            rendering::raytracing::MeshData<SPEC> md;

            // vertices
            for(unsigned int v = 0; v < mesh->mNumVertices; v++){
                const aiVector3D& pos = mesh->mVertices[v];
                vec3f vertex(pos.x, pos.y, pos.z);
                md.vertices.push_back(pos.x);
                md.vertices.push_back(pos.y);
                md.vertices.push_back(pos.z);
                bbox_min = min(bbox_min, vertex);
                bbox_max = max(bbox_max, vertex);
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

            // texture coordinates
            if(mesh->mTextureCoords[0]){
                for(unsigned int v = 0; v < mesh->mNumVertices; v++){
                    const aiVector3D& tc = mesh->mTextureCoords[0][v];
                    md.tex_coords.push_back(tc.x);
                    md.tex_coords.push_back(tc.y);
                }
            }

            // material / texture
            md.color[0] = 0.8f; md.color[1] = 0.8f; md.color[2] = 0.8f;
            if(mesh->mMaterialIndex < scene->mNumMaterials){
                const aiMaterial* mat = scene->mMaterials[mesh->mMaterialIndex];

                aiColor4D diffuse;
                if(aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diffuse) == AI_SUCCESS){
                    md.color[0] = diffuse.r; md.color[1] = diffuse.g; md.color[2] = diffuse.b;
                }

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

            total_verts += md.vertices.size() / 3;
            total_tris += md.indices.size() / 3;
            renderer.meshes.push_back(std::move(md));
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Total vertices: " << total_verts << ", triangles: " << total_tris);
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Bounding box: [" << bbox_min.x << "," << bbox_min.y << "," << bbox_min.z << "] - ["
              << bbox_max.x << "," << bbox_max.y << "," << bbox_max.z << "]");

        int textured_count = 0;
        for(auto& m : renderer.meshes) if(m.has_texture) textured_count++;
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Meshes with textures: " << textured_count << "/" << renderer.meshes.size());

        // Adjust camera based on bounding box
        vec3f center = 0.5f * (bbox_min + bbox_max);
        vec3f size = bbox_max - bbox_min;
        float max_dim = std::max({size.x, size.y, size.z});
        vec3f look_from = center + vec3f(max_dim * 1.5f, max_dim * 0.8f, max_dim * 1.5f);
        renderer.scene_center[0] = center.x;
        renderer.scene_center[1] = center.y;
        renderer.scene_center[2] = center.z;
        renderer.camera_radius = length(look_from - center);
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Camera positioned at [" << look_from.x << "," << look_from.y << "," << look_from.z << "]");

        return true;
    }

    // =========================================================================
    // load_default_cube: fallback cube geometry
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void load_default_cube(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        renderer.meshes.clear();
        rendering::raytracing::MeshData<SPEC> md;
        for(int i = 0; i < rendering::raytracing::constants::NUM_VERTICES; i++){
            md.vertices.push_back(rendering::raytracing::constants::default_vertices[i].x);
            md.vertices.push_back(rendering::raytracing::constants::default_vertices[i].y);
            md.vertices.push_back(rendering::raytracing::constants::default_vertices[i].z);
        }
        for(int i = 0; i < rendering::raytracing::constants::NUM_INDICES; i++){
            md.indices.push_back(rendering::raytracing::constants::default_indices[i].x);
            md.indices.push_back(rendering::raytracing::constants::default_indices[i].y);
            md.indices.push_back(rendering::raytracing::constants::default_indices[i].z);
        }
        md.color[0] = 0.f; md.color[1] = 1.f; md.color[2] = 0.f;
        md.has_texture = false;
        renderer.meshes.push_back(std::move(md));

        // Default camera distance (matches original: lookFrom(-4,-3,-2) lookAt(0,0,0))
        renderer.camera_radius = length(vec3f(-4.f, -3.f, -2.f));
    }

    // =========================================================================
    // upload_geometry: upload meshes to both RGB and collision OWL contexts + build BVH
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void upload_geometry(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        OWLContext context = (OWLContext)renderer.context;
        OWLModule module = (OWLModule)renderer.module;
        OWLContext coll_context = (OWLContext)renderer.coll_context;
        OWLModule coll_module = (OWLModule)renderer.coll_module;

        // --- RGB context geometry ---
        OWLVarDecl triangles_geom_vars[] = {
            { "index",      OWL_BUFPTR,  OWL_OFFSETOF(TrianglesGeomData, index)},
            { "vertex",     OWL_BUFPTR,  OWL_OFFSETOF(TrianglesGeomData, vertex)},
            { "texCoord",   OWL_BUFPTR,  OWL_OFFSETOF(TrianglesGeomData, texCoord)},
            { "color",      OWL_FLOAT3,  OWL_OFFSETOF(TrianglesGeomData, color)},
            { "texture",    OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData, texture)},
            { "hasTexture", OWL_INT,     OWL_OFFSETOF(TrianglesGeomData, hasTexture)},
            { /* sentinel */ }
        };
        OWLGeomType triangles_geom_type = owlGeomTypeCreate(context, OWL_TRIANGLES,
                                                             sizeof(TrianglesGeomData),
                                                             triangles_geom_vars, -1);
        owlGeomTypeSetClosestHit(triangles_geom_type, 0, module, "TriangleMesh");

        RL_TOOLS_RENDERING_RAYTRACING_LOG("building " << renderer.meshes.size() << " geometries (render context) ...");

        std::vector<OWLGeom> geoms;
        for(size_t m = 0; m < renderer.meshes.size(); m++){
            auto& md = renderer.meshes[m];
            size_t num_vertices = md.vertices.size() / 3;
            size_t num_indices = md.indices.size() / 3;

            OWLBuffer vb = owlDeviceBufferCreate(context, OWL_FLOAT3, num_vertices, md.vertices.data());
            OWLBuffer ib = owlDeviceBufferCreate(context, OWL_INT3, num_indices, md.indices.data());

            OWLGeom geom = owlGeomCreate(context, triangles_geom_type);
            owlTrianglesSetVertices(geom, vb, num_vertices, sizeof(vec3f), 0);
            owlTrianglesSetIndices(geom, ib, num_indices, sizeof(vec3i), 0);
            owlGeomSetBuffer(geom, "vertex", vb);
            owlGeomSetBuffer(geom, "index", ib);
            owlGeomSet3f(geom, "color", owl3f{md.color[0], md.color[1], md.color[2]});

            if(!md.tex_coords.empty()){
                size_t num_tc = md.tex_coords.size() / 2;
                OWLBuffer tcb = owlDeviceBufferCreate(context, OWL_FLOAT2, num_tc, md.tex_coords.data());
                owlGeomSetBuffer(geom, "texCoord", tcb);
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
                owlGeomSet1i(geom, "hasTexture", 1);
            } else {
                owlGeomSet1i(geom, "hasTexture", 0);
            }

            geoms.push_back(geom);
        }

        OWLGroup triangles_group = owlTrianglesGeomGroupCreate(context, geoms.size(), geoms.data());
        owlGroupBuildAccel(triangles_group);
        OWLGroup world = owlInstanceGroupCreate(context, 1, &triangles_group);
        owlGroupBuildAccel(world);

        owlRayGenSetGroup((OWLRayGen)renderer.ray_gen, "world", world);
        renderer.world = world;

        // --- Collision context geometry ---
        OWLVarDecl collision_geom_vars[] = {
            { "dummy", OWL_INT, OWL_OFFSETOF(CollisionGeomData, dummy)},
            { /* sentinel */ }
        };
        OWLGeomType collision_geom_type = owlGeomTypeCreate(coll_context, OWL_TRIANGLES,
                                                             sizeof(CollisionGeomData),
                                                             collision_geom_vars, -1);
        owlGeomTypeSetClosestHit(collision_geom_type, 0, coll_module, "collisionHit");

        OWLVarDecl collision_miss_vars[] = {
            { "dummy", OWL_INT, OWL_OFFSETOF(CollisionMissData, dummy)},
            { /* sentinel */ }
        };
        OWLMissProg collision_miss_prog = owlMissProgCreate(coll_context, coll_module, "collisionMiss",
                                                             sizeof(CollisionMissData), collision_miss_vars, -1);
        (void)collision_miss_prog;

        RL_TOOLS_RENDERING_RAYTRACING_LOG("building " << renderer.meshes.size() << " geometries (collision context) ...");

        std::vector<OWLGeom> coll_geoms;
        for(size_t m = 0; m < renderer.meshes.size(); m++){
            auto& md = renderer.meshes[m];
            size_t num_vertices = md.vertices.size() / 3;
            size_t num_indices = md.indices.size() / 3;

            OWLBuffer vb = owlDeviceBufferCreate(coll_context, OWL_FLOAT3, num_vertices, md.vertices.data());
            OWLBuffer ib = owlDeviceBufferCreate(coll_context, OWL_INT3, num_indices, md.indices.data());

            OWLGeom geom = owlGeomCreate(coll_context, collision_geom_type);
            owlTrianglesSetVertices(geom, vb, num_vertices, sizeof(vec3f), 0);
            owlTrianglesSetIndices(geom, ib, num_indices, sizeof(vec3i), 0);
            coll_geoms.push_back(geom);
        }

        OWLGroup coll_tri_group = owlTrianglesGeomGroupCreate(coll_context, coll_geoms.size(), coll_geoms.data());
        owlGroupBuildAccel(coll_tri_group);
        OWLGroup coll_world = owlInstanceGroupCreate(coll_context, 1, &coll_tri_group);
        owlGroupBuildAccel(coll_world);

        owlRayGenSetGroup((OWLRayGen)renderer.collision_ray_gen, "world", coll_world);
        renderer.coll_world = coll_world;
    }

    // =========================================================================
    // generate_cameras: Fibonacci sphere camera generation + upload
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void generate_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer,
                          vec3f center, float radius, vec3f up, float cos_fov){
        using TI = typename SPEC::TI;

        std::vector<CameraData> cameras;
        cameras.reserve(SPEC::NUM_CAMERAS);

        const float golden_ratio = (1.0f + sqrtf(5.0f)) / 2.0f;
        const vec2i cam_size(SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        const float aspect = cam_size.x / float(cam_size.y);

        for(int i = 0; i < (int)SPEC::NUM_CAMERAS; i++){
            float theta = 2.0f * (float)M_PI * i / golden_ratio;
            float cos_inc = 1.0f - 2.0f * (i + 0.5f) / SPEC::NUM_CAMERAS;
            cos_inc = cos_inc * 0.85f;
            float sin_inc = sqrtf(1.0f - cos_inc * cos_inc);

            vec3f cam_pos;
            cam_pos.x = center.x + radius * sin_inc * cosf(theta);
            cam_pos.y = center.y + radius * cos_inc;
            cam_pos.z = center.z + radius * sin_inc * sinf(theta);

            if(cam_pos.y < center.y - radius * 0.1f)
                cam_pos.y = center.y + radius * 0.3f;

            vec3f dir = normalize(center - cam_pos);
            vec3f du = cos_fov * aspect * normalize(cross(dir, up));
            vec3f dv = cos_fov * normalize(cross(du, dir));

            vec3f dir_00 = dir - 0.5f * du + 0.5f * dv;
            dv = -dv;

            cameras.push_back({cam_pos, dir_00, du, dv});
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Generated " << cameras.size() << " camera positions");
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Per-camera resolution: " << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT);

        // Upload to RGB context
        OWLContext context = (OWLContext)renderer.context;
        OWLBuffer cameras_buffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(CameraData),
                                                          cameras.size(), cameras.data());
        owlRayGenSetBuffer((OWLRayGen)renderer.ray_gen, "cameras", cameras_buffer);
        renderer.cameras_buffer = cameras_buffer;

        // Upload to collision context
        OWLContext coll_context = (OWLContext)renderer.coll_context;
        OWLBuffer coll_cameras_buffer = owlDeviceBufferCreate(coll_context, OWL_USER_TYPE(CameraData),
                                                               cameras.size(), cameras.data());
        owlRayGenSetBuffer((OWLRayGen)renderer.collision_ray_gen, "cameras", coll_cameras_buffer);
        renderer.coll_cameras_buffer = coll_cameras_buffer;
    }

    // =========================================================================
    // generate_probe_directions: Fibonacci probe directions + upload
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void generate_probe_directions(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        using TI = typename SPEC::TI;

        std::vector<vec3f> dirs;
        dirs.reserve(SPEC::NUM_PROBES);

        const float golden_ratio = (1.0f + sqrtf(5.0f)) / 2.0f;

        for(int i = 0; i < (int)SPEC::NUM_PROBES; i++){
            float theta = 2.0f * (float)M_PI * i / golden_ratio;
            float cos_inc = 1.0f - 2.0f * (i + 0.5f) / SPEC::NUM_PROBES;
            float sin_inc = sqrtf(1.0f - cos_inc * cos_inc);

            dirs.push_back(normalize(vec3f(sin_inc * cosf(theta),
                                           cos_inc,
                                           sin_inc * sinf(theta))));
        }

        RL_TOOLS_RENDERING_RAYTRACING_LOG("Generated " << dirs.size() << " probe directions per camera");

        OWLContext coll_context = (OWLContext)renderer.coll_context;
        OWLBuffer probe_dirs_buffer = owlDeviceBufferCreate(coll_context, OWL_USER_TYPE(vec3f),
                                                              dirs.size(), dirs.data());
        owlRayGenSetBuffer((OWLRayGen)renderer.collision_ray_gen, "probeDirections", probe_dirs_buffer);
        renderer.probe_dirs_buffer = probe_dirs_buffer;

        owlRayGenSetBuffer((OWLRayGen)renderer.collision_ray_gen, "results", (OWLBuffer)renderer.collision_results_buffer);
        owlRayGenSet1i    ((OWLRayGen)renderer.collision_ray_gen, "numProbes", SPEC::NUM_PROBES);
        owlRayGenSet1i    ((OWLRayGen)renderer.collision_ray_gen, "numCameras", SPEC::NUM_CAMERAS);
        owlRayGenSet1f    ((OWLRayGen)renderer.collision_ray_gen, "maxDist", renderer.camera_radius * 2.0f);
    }

    // =========================================================================
    // build_pipeline: build programs, pipeline, and SBT for both contexts
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void build_pipeline(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        OWLContext context = (OWLContext)renderer.context;
        OWLContext coll_context = (OWLContext)renderer.coll_context;

        owlBuildPrograms(context);
        owlBuildPipeline(context);
        owlBuildSBT(context);

        owlBuildPrograms(coll_context);
        owlBuildPipeline(coll_context);
        owlBuildSBT(coll_context);

        // Create async launch params
        OWLParams rgb_lp = owlParamsCreate(context, 0, nullptr, 0);
        OWLParams coll_lp = owlParamsCreate(coll_context, 0, nullptr, 0);
        renderer.rgb_launch_params = rgb_lp;
        renderer.coll_launch_params = coll_lp;
    }

    // =========================================================================
    // render: async launch RGB + collision, then sync
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void render(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        OWLRayGen ray_gen = (OWLRayGen)renderer.ray_gen;
        OWLRayGen collision_ray_gen = (OWLRayGen)renderer.collision_ray_gen;
        OWLParams rgb_lp = (OWLParams)renderer.rgb_launch_params;
        OWLParams coll_lp = (OWLParams)renderer.coll_launch_params;

        owlAsyncLaunch2D(ray_gen, SPEC::FB_WIDTH, SPEC::FB_HEIGHT, rgb_lp);
        owlAsyncLaunch2D(collision_ray_gen, SPEC::NUM_CAMERAS, SPEC::NUM_PROBES, coll_lp);
        owlLaunchSync(rgb_lp);
        owlLaunchSync(coll_lp);
    }

    // =========================================================================
    // save_image: readback framebuffer, rearrange to grid, write PNG
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void save_image(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        using TI = typename SPEC::TI;
        constexpr TI cam_pixels = SPEC::CAM_PIXELS;
        const size_t fb_count = (size_t)SPEC::NUM_CAMERAS * cam_pixels;

        std::vector<uint32_t> fb_host(fb_count);
        cudaMemcpy(fb_host.data(),
                   owlBufferGetPointer((OWLBuffer)renderer.frame_buffer, 0),
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

    // =========================================================================
    // save_probes: readback collision results, write binary
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void save_probes(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer, const char* filename){
        const CollisionResult* probe_results =
            (const CollisionResult*)owlBufferGetPointer((OWLBuffer)renderer.collision_results_buffer, 0);

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
    }

    // =========================================================================
    // free: destroy OWL contexts
    // =========================================================================
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC>& renderer){
        RL_TOOLS_RENDERING_RAYTRACING_LOG("destroying devicegroups ...");
        if(renderer.coll_context) owlContextDestroy((OWLContext)renderer.coll_context);
        if(renderer.context) owlContextDestroy((OWLContext)renderer.context);
        renderer.context = nullptr;
        renderer.coll_context = nullptr;
    }

}
