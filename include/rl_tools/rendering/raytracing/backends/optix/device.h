#pragma once

#include <owl/owl.h>
#include <owl/common/math/vec.h>

using namespace owl;

/* variables for the triangle mesh geometry */
struct TrianglesGeomData
{
  /*! base color we use when there is no texture */
  vec3f color;
  /*! array/buffer of vertex indices */
  vec3i *index;
  /*! array/buffer of vertex positions */
  vec3f *vertex;
  /*! array/buffer of texture coordinates (may be null) */
  vec2f *tex_coord;
  /*! diffuse texture (0 if none) */
  cudaTextureObject_t texture;
  /*! whether this geometry has a valid texture */
  int has_texture;
};

/* per-camera parameters (shared between host and device) */
struct CameraData
{
  vec3f pos;
  vec3f dir_00;
  vec3f dir_du;
  vec3f dir_dv;
};

/* variables for the ray generation program */
struct RayGenData
{
  uint32_t *fb_ptr;
  vec2i  fb_size;      // total framebuffer size (full grid)
  vec2i  cam_size;     // per-camera resolution
  int    grid_cols;    // number of columns in the grid
  int    num_cameras;  // total number of cameras
  OptixTraversableHandle world;
  CameraData *cameras; // device array of all cameras
};

/* variables for the miss program */
struct MissProgData
{
  vec3f  color_0;
  vec3f  color_1;
};

// ---- Collision probing ----

struct CollisionResult
{
  float distance; // hit distance, or max_dist on miss
  int   hit;      // 1 = geometry, 0 = miss (skybox)
};

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
  vec3f           *probe_directions; // [num_probes] unit directions (index 0 = placeholder, overridden by camera forward)
  CameraData      *cameras;        // reused camera array
  OptixTraversableHandle world;
  int   num_probes;
  int   num_cameras;
  float max_dist;
};
