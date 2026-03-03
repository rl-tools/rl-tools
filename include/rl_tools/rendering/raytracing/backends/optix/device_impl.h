#include "device.h"
#include <optix_device.h>

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixel_id = owl::getLaunchIndex();

  // Determine which camera tile this pixel belongs to
  const int tile_col = pixel_id.x / self.cam_size.x;
  const int tile_row = pixel_id.y / self.cam_size.y;
  const int cam_idx  = tile_row * self.grid_cols + tile_col;

  // Pixels in padding tiles (beyond num_cameras) are discarded
  if (cam_idx >= self.num_cameras)
    return;

  // Local pixel coordinates within this camera's tile
  const int local_x = pixel_id.x - tile_col * self.cam_size.x;
  const int local_y = pixel_id.y - tile_row * self.cam_size.y;
  const vec2f screen = (vec2f(local_x, local_y) + vec2f(.5f)) / vec2f(self.cam_size);

  const CameraData &cam = self.cameras[cam_idx];
  owl::Ray ray;
  ray.origin    = cam.pos;
  ray.direction = normalize(cam.dir_00
                            + screen.u * cam.dir_du
                            + screen.v * cam.dir_dv);

  vec3f color;
  owl::traceRay(self.world, ray, color);

  // Flat per-camera layout: camera i occupies [i*W*H .. (i+1)*W*H)
  const int fb_offset = cam_idx * self.cam_size.x * self.cam_size.y
                  + local_y * self.cam_size.x + local_x;
  self.fb_ptr[fb_offset] = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();

  // compute normal:
  const int   prim_id = optixGetPrimitiveIndex();
  const vec3i index  = self.index[prim_id];
  const vec3f &vertex_a = self.vertex[index.x];
  const vec3f &vertex_b = self.vertex[index.y];
  const vec3f &vertex_c = self.vertex[index.z];
  const vec3f normal_geometric = normalize(cross(vertex_b - vertex_a, vertex_c - vertex_a));

  // determine base color: sample texture if available, otherwise use flat color
  vec3f base_color;
  if (self.has_texture && self.tex_coord) {
    const vec2f bary = optixGetTriangleBarycentrics();
    const vec2f tc
      = (1.f - bary.x - bary.y) * self.tex_coord[index.x]
        +      bary.x           * self.tex_coord[index.y]
        +             bary.y    * self.tex_coord[index.z];
    vec4f tex_color = tex2D<float4>(self.texture, tc.x, tc.y);
    base_color = vec3f(tex_color.x, tex_color.y, tex_color.z);
  } else {
    base_color = self.color;
  }

  const vec3f ray_dir = optixGetWorldRayDirection();
  prd = (.2f + .8f * fabs(dot(ray_dir, normal_geometric))) * base_color;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixel_id = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();

  vec3f &prd = owl::getPRD<vec3f>();
  int checker_pattern = (pixel_id.x / 8) ^ (pixel_id.y/8);
  prd = (checker_pattern&1) ? self.color_1 : self.color_0;
}

// =====================================================================
// Collision probing programs (ray type 1)
// =====================================================================

OPTIX_RAYGEN_PROGRAM(collisionRayGen)()
{
  const CollisionRayGenData &self
    = owl::getProgramData<CollisionRayGenData>();
  const vec2i idx = owl::getLaunchIndex();
  const int cam_idx   = idx.x;
  const int probe_idx = idx.y;

  if (cam_idx >= self.num_cameras || probe_idx >= self.num_probes)
    return;

  const CameraData &cam = self.cameras[cam_idx];

  vec3f dir;
  if (probe_idx == 0) {
    dir = normalize(cam.dir_00 + 0.5f * cam.dir_du + 0.5f * cam.dir_dv);
  } else {
    dir = self.probe_directions[probe_idx];
  }

  // Step B: raw optixTrace + CollisionResult PRD, no FIRST_HIT
  CollisionResult result;
  result.distance = self.max_dist;
  result.hit = 0;

  unsigned int u0, u1;
  u0 = __float_as_uint(result.distance);
  u1 = result.hit;

  optixTrace(
      self.world,
      (const float3&)cam.pos,
      (const float3&)dir,
      1e-3f,               // tmin
      self.max_dist,         // tmax
      0.0f,                 // rayTime
      OptixVisibilityMask(255),
      OPTIX_RAY_FLAG_NONE,
      0, 1, 0,             // SBT offset, stride, miss
      u0, u1);

  result.distance = __uint_as_float(u0);
  result.hit = u1;
  self.results[cam_idx * self.num_probes + probe_idx] = result;
}

OPTIX_CLOSEST_HIT_PROGRAM(collisionHit)()
{
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(1);
}

OPTIX_MISS_PROGRAM(collisionMiss)()
{
  // Payloads stay as initialized by caller: distance = max_dist, hit = 0
}
