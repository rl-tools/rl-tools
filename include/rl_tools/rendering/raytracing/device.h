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
  vec2f *texCoord;
  /*! diffuse texture (0 if none) */
  cudaTextureObject_t texture;
  /*! whether this geometry has a valid texture */
  int hasTexture;
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
  uint32_t *fbPtr;
  vec2i  fbSize;      // total framebuffer size (full grid)
  vec2i  camSize;     // per-camera resolution
  int    gridCols;    // number of columns in the grid
  int    numCameras;  // total number of cameras
  OptixTraversableHandle world;
  CameraData *cameras; // device array of all cameras
};

/* variables for the miss program */
struct MissProgData
{
  vec3f  color0;
  vec3f  color1;
};

// ---- Collision probing ----

struct CollisionResult
{
  float distance; // hit distance, or maxDist on miss
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
  CollisionResult *results;    // [numCameras * numProbes] output
  vec3f           *probeDirections; // [numProbes] unit directions (index 0 = placeholder, overridden by camera forward)
  CameraData      *cameras;        // reused camera array
  OptixTraversableHandle world;
  int   numProbes;
  int   numCameras;
  float maxDist;
};
