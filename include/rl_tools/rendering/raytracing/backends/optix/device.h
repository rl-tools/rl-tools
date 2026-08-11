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
        owl::vec3f emissive;
        cudaTextureObject_t emissive_map;
        int has_emissive_map;
        cudaTextureObject_t occlusion_map;
        int has_occlusion_map;
        float opacity;
        int alpha_mode;
        float alpha_cutoff;
        rendering::raytracing::SceneLight *scene_lights;
        int num_scene_lights;
        owl::vec3f ambient_color;
    };

    struct OptixCameraData
    {
        owl::vec3f pos;
        owl::vec3f dir_00;
        owl::vec3f dir_du;
        owl::vec3f dir_dv;
    };
    static_assert(sizeof(OptixCameraData) == sizeof(rendering::raytracing::Camera<float>), "OptixCameraData and Camera<float> must have identical layout");

    /* variables for the ray generation program */
    struct RayGenData
    {
        uint32_t *fb_ptr;
        float *obs_ptr;           // observation output (3 floats per pixel), null when disabled
        owl::vec2i  fb_size;      // total framebuffer size (full grid)
        owl::vec2i  cam_size;     // per-camera resolution
        int    grid_cols;    // number of columns in the grid
        int    num_cameras;  // total number of cameras
        OptixTraversableHandle world;
        OptixCameraData *cameras; // device array of all cameras
    };

    struct MotionBlurRayGenData
    {
        uint32_t *fb_ptr;
        float *obs_ptr;
        owl::vec2i  fb_size;
        owl::vec2i  cam_size;
        int    grid_cols;
        int    num_cameras;
        OptixTraversableHandle world;
        OptixCameraData *cameras_open;
        OptixCameraData *cameras_close;
    };

    // dynamic motion blur: one launch per motion sample against per-sample overlay TLAS state;
    // the shutter time is a device word published on-stream by the overlay fill kernel and the
    // pass result is added into a linear float accumulator, resolved after the last pass
    struct AccumulateRayGenData
    {
        float *accum_ptr;         // 3 floats per pixel, linear radiance summed across passes
        const float *shutter_ptr; // per-pass shutter time in [0, 1]
        owl::vec2i  fb_size;
        owl::vec2i  cam_size;
        int    grid_cols;
        int    num_cameras;
        OptixTraversableHandle world;
        OptixCameraData *cameras_open;
        OptixCameraData *cameras_close;
    };

    struct AccumulateDepthRayGenData
    {
        float *depth_accum_ptr;
        const float *shutter_ptr;
        owl::vec2i  fb_size;
        owl::vec2i  cam_size;
        int    grid_cols;
        int    num_cameras;
        OptixTraversableHandle world;
        OptixCameraData *cameras_open;
        OptixCameraData *cameras_close;
        float max_depth;
    };

    struct DepthRayGenData
    {
        float *depth_ptr;
        owl::vec2i  fb_size;
        owl::vec2i  cam_size;
        int    grid_cols;
        int    num_cameras;
        OptixTraversableHandle world;
        OptixCameraData *cameras;
        float max_depth;
    };

    struct MotionBlurDepthRayGenData
    {
        float *depth_ptr;
        owl::vec2i  fb_size;
        owl::vec2i  cam_size;
        int    grid_cols;
        int    num_cameras;
        OptixTraversableHandle world;
        OptixCameraData *cameras_open;
        OptixCameraData *cameras_close;
        float max_depth;
    };

    // launch params shared by every raygen/hit program: overlay traversables + per-camera
    // attachment table for min-t composition; overlay_count == 0 keeps every loop empty
    struct OverlayLaunchParams
    {
        unsigned long long *overlays;   // OptixTraversableHandle per overlay
        uint32_t *attachments;          // [num_cameras * overlay_count]
        int overlay_count;
        int cam_width;                  // framebuffer tile math so closest-hit programs can
        int cam_height;                 // recover the camera index for composed secondaries
        int grid_cols;
        unsigned int *instance_classes; // per global instance id (semantic segmentation)
        int semantic_segmentation;
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

    struct SegmentationRayGenData
    {
        uint32_t *seg_ptr;
        owl::vec2i  fb_size;
        owl::vec2i  cam_size;
        int    grid_cols;
        int    num_cameras;
        OptixTraversableHandle world;
        OptixCameraData *cameras;
    };

    struct NormalsRayGenData
    {
        float *normals_ptr;
        owl::vec2i  fb_size;
        owl::vec2i  cam_size;
        int    grid_cols;
        int    num_cameras;
        OptixTraversableHandle world;
        OptixCameraData *cameras;
    };

    struct FlowRayGenData
    {
        float *flow_ptr;
        owl::vec2i  fb_size;
        owl::vec2i  cam_size;
        int    grid_cols;
        int    num_cameras;
        OptixTraversableHandle world;
        OptixCameraData *cameras_open;
        OptixCameraData *cameras_close;
        const float *flow_deltas;            // 12 per overlay slot (world_open ∘ world_close⁻¹), null without overlays
        unsigned int first_overlay_instance; // global instance ids >= this index the delta table
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
