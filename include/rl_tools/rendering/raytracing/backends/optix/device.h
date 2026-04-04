#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_DEVICE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_DEVICE_H

#include "../../types.h"

#include <owl/owl.h>
#include <owl/common/math/vec.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    /* variables for the triangle mesh geometry */
    struct TrianglesGeomData
    {
        /*! base color we use when there is no texture */
        owl::vec3f color;
        /*! array/buffer of vertex indices */
        owl::vec3i *index;
        /*! array/buffer of vertex positions */
        owl::vec3f *vertex;
        /*! array/buffer of texture coordinates (may be null) */
        owl::vec2f *tex_coord;
        /*! diffuse texture (0 if none) */
        cudaTextureObject_t texture;
        /*! whether this geometry has a valid texture */
        int has_texture;
        float metallic;
        OptixTraversableHandle world;
        owl::vec3f *normal;
        float roughness;
        cudaTextureObject_t normal_map;
        int has_normal_map;
        cudaTextureObject_t metallic_roughness_map;
        int has_metallic_roughness_map;
        owl::vec3f light_dir_0;
        owl::vec3f light_color_0;
        owl::vec3f light_dir_1;
        owl::vec3f light_color_1;
        owl::vec3f light_dir_2;
        owl::vec3f light_color_2;
        owl::vec3f ambient_color;
    };

    struct OptixCameraData
    {
        owl::vec3f pos;
        owl::vec3f dir_00;
        owl::vec3f dir_du;
        owl::vec3f dir_dv;
    };
    static_assert(sizeof(OptixCameraData) == sizeof(rendering::raytracing::CameraData<float>), "OptixCameraData and CameraData<float> must have identical layout");

    /* variables for the ray generation program */
    struct RayGenData
    {
        uint32_t *fb_ptr;
        owl::vec2i  fb_size;      // total framebuffer size (full grid)
        owl::vec2i  cam_size;     // per-camera resolution
        int    grid_cols;    // number of columns in the grid
        int    num_cameras;  // total number of cameras
        OptixTraversableHandle world;
        OptixCameraData *cameras; // device array of all cameras
    };

    /* variables for the miss program */
    struct MissProgData
    {
        owl::vec3f  color_0;
        owl::vec3f  color_1;
    };

    // ---- Collision probing ----

    using CollisionResult = rendering::raytracing::CollisionResult;

    struct CollisionGeomData
    {
        int dummy;
    };

    struct CollisionMissData
    {
        int dummy;
    };

    struct CollisionRayGenData
    {
        CollisionResult *results;    // [num_cameras * num_probes] output
        owl::vec3f           *probe_directions; // [num_probes] unit directions (index 0 = placeholder, overridden by camera forward)
        OptixCameraData *cameras;        // reused camera array
        OptixTraversableHandle world;
        int   num_probes;
        int   num_cameras;
        float max_dist;
    };

}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif