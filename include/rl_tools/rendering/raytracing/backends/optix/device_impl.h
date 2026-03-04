#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_DEVICE_IMPL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_DEVICE_IMPL_H

#include "device.h"
#include <optix_device.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools
{
  static constexpr int NUM_RAY_TYPES = 2;

  OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
  {
    const RayGenData &self = owl::getProgramData<RayGenData>();
    const owl::vec2i pixel_id = owl::getLaunchIndex();

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
    const owl::vec2f screen = (owl::vec2f(local_x, local_y) + owl::vec2f(.5f)) / owl::vec2f(self.cam_size);

    const CameraData &cam = self.cameras[cam_idx];
    owl::Ray ray;
    ray.origin    = cam.pos;
    ray.direction = normalize(cam.dir_00
                              + screen.u * cam.dir_du
                              + screen.v * cam.dir_dv);

    owl::vec3f color;
    unsigned int p0 = 0, p1 = 0;
    owl::packPointer(&color, p0, p1);
    unsigned int p2 = 0;
    optixTrace(self.world,
               (const float3&)ray.origin,
               (const float3&)ray.direction,
               ray.tmin,
               ray.tmax,
               0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0, NUM_RAY_TYPES, 0,
               p0, p1, p2);

    // Flat per-camera layout: camera i occupies [i*W*H .. (i+1)*W*H)
    const int fb_offset = cam_idx * self.cam_size.x * self.cam_size.y
                    + local_y * self.cam_size.x + local_x;
    self.fb_ptr[fb_offset] = owl::make_rgba(color);
  }

  OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
  {
    owl::vec3f &prd = owl::getPRD<owl::vec3f>();

    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();

    // compute normal:
    const int   prim_id = optixGetPrimitiveIndex();
    const owl::vec3i index  = self.index[prim_id];
    const owl::vec3f &vertex_a = self.vertex[index.x];
    const owl::vec3f &vertex_b = self.vertex[index.y];
    const owl::vec3f &vertex_c = self.vertex[index.z];
    const owl::vec3f normal_geometric = normalize(cross(vertex_b - vertex_a, vertex_c - vertex_a));

    // determine base color: sample texture if available, otherwise use flat color
    owl::vec3f base_color;
    if (self.has_texture && self.tex_coord) {
      const owl::vec2f bary = optixGetTriangleBarycentrics();
      const owl::vec2f tc
        = (1.f - bary.x - bary.y) * self.tex_coord[index.x]
          +      bary.x           * self.tex_coord[index.y]
          +             bary.y    * self.tex_coord[index.z];
      owl::vec4f tex_color = tex2D<float4>(self.texture, tc.x, tc.y);
      base_color = owl::vec3f(tex_color.x, tex_color.y, tex_color.z);
    } else {
      base_color = self.color;
    }

    const owl::vec3f ray_dir = optixGetWorldRayDirection();
    owl::vec3f direct = (.2f + .8f * fabs(dot(ray_dir, normal_geometric))) * base_color;

    unsigned int depth = optixGetPayload_2();
    if (depth < 1 && self.metallic > 0.f) {
      owl::vec3f hit_point = ray_dir * optixGetRayTmax();
      hit_point.x += optixGetWorldRayOrigin().x;
      hit_point.y += optixGetWorldRayOrigin().y;
      hit_point.z += optixGetWorldRayOrigin().z;
      owl::vec3f n = dot(ray_dir, normal_geometric) > 0.f ? -normal_geometric : normal_geometric;
      owl::vec3f reflect_dir = ray_dir - 2.f * dot(ray_dir, n) * n;

      owl::vec3f reflected_color;
      unsigned int rp0 = 0, rp1 = 0;
      owl::packPointer(&reflected_color, rp0, rp1);
      unsigned int rp2 = depth + 1;
      optixTrace(self.world,
                 (const float3&)hit_point,
                 (const float3&)reflect_dir,
                 1e-3f,
                 1e20f,
                 0.0f,
                 OptixVisibilityMask(255),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                 0, NUM_RAY_TYPES, 0,
                 rp0, rp1, rp2);

      float cos_theta = fabs(dot(ray_dir, n));
      float fresnel = self.metallic * (0.04f + 0.96f * powf(1.f - cos_theta, 5.f));
      prd = direct * ((1.f - fresnel) + fresnel * reflected_color);
    } else {
      prd = direct;
    }
  }

  OPTIX_MISS_PROGRAM(miss)()
  {
    const owl::vec2i pixel_id = owl::getLaunchIndex();

    const MissProgData &self = owl::getProgramData<MissProgData>();

    owl::vec3f &prd = owl::getPRD<owl::vec3f>();
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
    const owl::vec2i idx = owl::getLaunchIndex();
    const int cam_idx   = idx.x;
    const int probe_idx = idx.y;

    if (cam_idx >= self.num_cameras || probe_idx >= self.num_probes)
      return;

    const CameraData &cam = self.cameras[cam_idx];

    owl::vec3f dir;
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
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        1, NUM_RAY_TYPES, 1, // SBT offset, stride, miss (ray type 1)
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
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif