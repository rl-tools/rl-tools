#include "device.h"
#include <optix_device.h>

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();

  // Determine which camera tile this pixel belongs to
  const int tileCol = pixelID.x / self.camSize.x;
  const int tileRow = pixelID.y / self.camSize.y;
  const int camIdx  = tileRow * self.gridCols + tileCol;

  // Pixels in padding tiles (beyond numCameras) are discarded
  if (camIdx >= self.numCameras)
    return;

  // Local pixel coordinates within this camera's tile
  const int localX = pixelID.x - tileCol * self.camSize.x;
  const int localY = pixelID.y - tileRow * self.camSize.y;
  const vec2f screen = (vec2f(localX, localY) + vec2f(.5f)) / vec2f(self.camSize);

  const CameraData &cam = self.cameras[camIdx];
  owl::Ray ray;
  ray.origin    = cam.pos;
  ray.direction = normalize(cam.dir_00
                            + screen.u * cam.dir_du
                            + screen.v * cam.dir_dv);

  vec3f color;
  owl::traceRay(self.world, ray, color);

  // Flat per-camera layout: camera i occupies [i*W*H .. (i+1)*W*H)
  const int fbOfs = camIdx * self.camSize.x * self.camSize.y
                  + localY * self.camSize.x + localX;
  self.fbPtr[fbOfs] = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();

  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  const vec3i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];
  const vec3f Ng     = normalize(cross(B-A,C-A));

  // determine base color: sample texture if available, otherwise use flat color
  vec3f baseColor;
  if (self.hasTexture && self.texCoord) {
    const vec2f bary = optixGetTriangleBarycentrics();
    const vec2f tc
      = (1.f - bary.x - bary.y) * self.texCoord[index.x]
        +      bary.x           * self.texCoord[index.y]
        +             bary.y    * self.texCoord[index.z];
    vec4f texColor = tex2D<float4>(self.texture, tc.x, tc.y);
    baseColor = vec3f(texColor.x, texColor.y, texColor.z);
  } else {
    baseColor = self.color;
  }

  const vec3f rayDir = optixGetWorldRayDirection();
  prd = (.2f + .8f*fabs(dot(rayDir,Ng))) * baseColor;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();

  vec3f &prd = owl::getPRD<vec3f>();
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  prd = (pattern&1) ? self.color1 : self.color0;
}

// =====================================================================
// Collision probing programs (ray type 1)
// =====================================================================

OPTIX_RAYGEN_PROGRAM(collisionRayGen)()
{
  const CollisionRayGenData &self
    = owl::getProgramData<CollisionRayGenData>();
  const vec2i idx = owl::getLaunchIndex();
  const int camIdx   = idx.x;
  const int probeIdx = idx.y;

  if (camIdx >= self.numCameras || probeIdx >= self.numProbes)
    return;

  const CameraData &cam = self.cameras[camIdx];

  vec3f dir;
  if (probeIdx == 0) {
    dir = normalize(cam.dir_00 + 0.5f * cam.dir_du + 0.5f * cam.dir_dv);
  } else {
    dir = self.probeDirections[probeIdx];
  }

  // Step B: raw optixTrace + CollisionResult PRD, no FIRST_HIT
  CollisionResult result;
  result.distance = self.maxDist;
  result.hit = 0;

  unsigned int u0, u1;
  u0 = __float_as_uint(result.distance);
  u1 = result.hit;

  optixTrace(
      self.world,
      (const float3&)cam.pos,
      (const float3&)dir,
      1e-3f,               // tmin
      self.maxDist,         // tmax
      0.0f,                 // rayTime
      OptixVisibilityMask(255),
      OPTIX_RAY_FLAG_NONE,
      0, 1, 0,             // SBT offset, stride, miss
      u0, u1);

  result.distance = __uint_as_float(u0);
  result.hit = u1;
  self.results[camIdx * self.numProbes + probeIdx] = result;
}

OPTIX_CLOSEST_HIT_PROGRAM(collisionHit)()
{
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(1);
}

OPTIX_MISS_PROGRAM(collisionMiss)()
{
  // Payloads stay as initialized by caller: distance = maxDist, hit = 0
}
