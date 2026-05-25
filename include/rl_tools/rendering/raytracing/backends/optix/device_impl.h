#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_DEVICE_IMPL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_DEVICE_IMPL_H

#include "device.h"
#include <optix_device.h>

#ifndef RL_TOOLS_RENDERING_RAYTRACING_ENABLE_DEPTH_PROGRAMS
#define RL_TOOLS_RENDERING_RAYTRACING_ENABLE_DEPTH_PROGRAMS 0
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools
{
  static constexpr int NUM_RAY_TYPES = 2;

  inline __device__ float linear_to_srgb(float x) {
    if (x <= 0.0031308f) return 12.92f * x;
    return 1.055f * powf(x, 1.f / 2.4f) - 0.055f;
  }

  inline __device__ uint32_t make_srgb_rgba_from_linear(owl::vec3f color) {
    color.x = linear_to_srgb(fminf(fmaxf(color.x, 0.f), 1.f));
    color.y = linear_to_srgb(fminf(fmaxf(color.y, 0.f), 1.f));
    color.z = linear_to_srgb(fminf(fmaxf(color.z, 0.f), 1.f));
    return owl::make_rgba(color);
  }

  inline __device__ uint32_t make_linear_rgba_from_linear(owl::vec3f color) {
    color.x = fminf(fmaxf(color.x, 0.f), 1.f);
    color.y = fminf(fmaxf(color.y, 0.f), 1.f);
    color.z = fminf(fmaxf(color.z, 0.f), 1.f);
    return owl::make_rgba(color);
  }

  inline __device__ owl::vec3f lerp_camera_vec(const owl::vec3f &a, const owl::vec3f &b, float t)
  {
    return (1.f - t) * a + t * b;
  }

  struct PixelLaunchContext
  {
    int cam_idx;
    int local_x;
    int local_y;
    int fb_offset;
    bool valid;
  };

  template <typename RAYGEN_DATA>
  inline __device__ PixelLaunchContext pixel_launch_context(const RAYGEN_DATA &self)
  {
    const owl::vec2i pixel_id = owl::getLaunchIndex();
    const int tile_col = pixel_id.x / self.cam_size.x;
    const int tile_row = pixel_id.y / self.cam_size.y;
    const int cam_idx  = tile_row * self.grid_cols + tile_col;
    const int local_x = pixel_id.x - tile_col * self.cam_size.x;
    const int local_y = pixel_id.y - tile_row * self.cam_size.y;
    const int fb_offset = cam_idx * self.cam_size.x * self.cam_size.y + local_y * self.cam_size.x + local_x;
    return {cam_idx, local_x, local_y, fb_offset, cam_idx < self.num_cameras};
  }

  inline __device__ owl::vec3f trace_rgb_color(OptixTraversableHandle world, const owl::vec3f &pos, const owl::vec3f &direction)
  {
    owl::vec3f color;
    unsigned int p0 = 0, p1 = 0;
    owl::packPointer(&color, p0, p1);
    unsigned int p2 = 0;
    optixTrace(world,
               (const float3&)pos,
               (const float3&)direction,
               0.f,
               1e30f,
               0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0, NUM_RAY_TYPES, 0,
               p0, p1, p2);
    return color;
  }

#if RL_TOOLS_RENDERING_RAYTRACING_ENABLE_DEPTH_PROGRAMS
  inline __device__ float trace_depth_distance(OptixTraversableHandle world, const owl::vec3f &pos, const owl::vec3f &direction, float max_depth)
  {
    unsigned int u0 = __float_as_uint(max_depth);
    unsigned int u1 = 0;
    optixTrace(world,
               (const float3&)pos,
               (const float3&)direction,
               0.f,
               max_depth,
               0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               1, NUM_RAY_TYPES, 1,
               u0, u1);
    return __uint_as_float(u0);
  }
#endif

  template <bool T_SRGB_OUTPUT>
  struct RgbOutput
  {
    using Accumulator = owl::vec3f;
    inline __device__ static Accumulator zero() { return owl::vec3f(0.f); }
    template <typename RAYGEN_DATA>
    inline __device__ static void accumulate(const RAYGEN_DATA &self, Accumulator &acc, const owl::vec3f &pos, const owl::vec3f &direction)
    {
      acc = acc + trace_rgb_color(self.world, pos, direction);
    }
    template <typename RAYGEN_DATA>
    inline __device__ static void store(const RAYGEN_DATA &self, const PixelLaunchContext &ctx, Accumulator acc, int samples)
    {
      const owl::vec3f color = acc * (1.f / float(samples));
      if constexpr (T_SRGB_OUTPUT) {
        self.fb_ptr[ctx.fb_offset] = make_srgb_rgba_from_linear(color);
      }
      else {
        self.fb_ptr[ctx.fb_offset] = make_linear_rgba_from_linear(color);
      }
    }
  };

#if RL_TOOLS_RENDERING_RAYTRACING_ENABLE_DEPTH_PROGRAMS
  struct DepthOutput
  {
    using Accumulator = float;
    inline __device__ static Accumulator zero() { return 0.f; }
    template <typename RAYGEN_DATA>
    inline __device__ static void accumulate(const RAYGEN_DATA &self, Accumulator &acc, const owl::vec3f &pos, const owl::vec3f &direction)
    {
      acc += trace_depth_distance(self.world, pos, direction, self.max_depth);
    }
    template <typename RAYGEN_DATA>
    inline __device__ static void store(const RAYGEN_DATA &self, const PixelLaunchContext &ctx, Accumulator acc, int samples)
    {
      self.depth_ptr[ctx.fb_offset] = acc * (1.f / float(samples));
    }
  };
#endif

  template <typename OUTPUT, bool MOTION_BLUR, int MOTION_SAMPLES, int AA_GRID, typename RAYGEN_DATA>
  inline __device__ void rayGenImpl()
  {
    const RAYGEN_DATA &self = owl::getProgramData<RAYGEN_DATA>();
    const PixelLaunchContext ctx = pixel_launch_context(self);
    if (!ctx.valid)
      return;

    typename OUTPUT::Accumulator accumulated = OUTPUT::zero();
    const float inv_aa_grid = 1.f / float(AA_GRID);
    for (int motion_i = 0; motion_i < MOTION_SAMPLES; motion_i++) {
      owl::vec3f pos;
      owl::vec3f dir_00;
      owl::vec3f dir_du;
      owl::vec3f dir_dv;
      if constexpr (MOTION_BLUR) {
        const OptixCameraData &cam_open = self.cameras_open[ctx.cam_idx];
        const OptixCameraData &cam_close = self.cameras_close[ctx.cam_idx];
        const float shutter_t = (float(motion_i) + .5f) * (1.f / float(MOTION_SAMPLES));
        pos = lerp_camera_vec(cam_open.pos, cam_close.pos, shutter_t);
        dir_00 = lerp_camera_vec(cam_open.dir_00, cam_close.dir_00, shutter_t);
        dir_du = lerp_camera_vec(cam_open.dir_du, cam_close.dir_du, shutter_t);
        dir_dv = lerp_camera_vec(cam_open.dir_dv, cam_close.dir_dv, shutter_t);
      }
      else {
        const OptixCameraData &cam = self.cameras[ctx.cam_idx];
        pos = cam.pos;
        dir_00 = cam.dir_00;
        dir_du = cam.dir_du;
        dir_dv = cam.dir_dv;
      }
      for (int aa_y = 0; aa_y < AA_GRID; aa_y++) {
        for (int aa_x = 0; aa_x < AA_GRID; aa_x++) {
          const owl::vec2f screen = (owl::vec2f(ctx.local_x, ctx.local_y) + owl::vec2f((float(aa_x) + .5f) * inv_aa_grid, (float(aa_y) + .5f) * inv_aa_grid)) / owl::vec2f(self.cam_size);
          const owl::vec3f direction = normalize(dir_00 + screen.u * dir_du + screen.v * dir_dv);
          OUTPUT::accumulate(self, accumulated, pos, direction);
        }
      }
    }
    OUTPUT::store(self, ctx, accumulated, MOTION_SAMPLES * AA_GRID * AA_GRID);
  }

#define RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(SRGB_NAME, LINEAR_NAME, MOTION_BLUR, MOTION_SAMPLES, AA_GRID, DATA) \
  OPTIX_RAYGEN_PROGRAM(SRGB_NAME)() { rayGenImpl<RgbOutput<true>, MOTION_BLUR, MOTION_SAMPLES, AA_GRID, DATA>(); } \
  OPTIX_RAYGEN_PROGRAM(LINEAR_NAME)() { rayGenImpl<RgbOutput<false>, MOTION_BLUR, MOTION_SAMPLES, AA_GRID, DATA>(); }

  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGen, linearRayGen, false, 1, 1, RayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur2, linearRayGenMotionBlur2, true, 2, 1, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur4, linearRayGenMotionBlur4, true, 4, 1, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur8, linearRayGenMotionBlur8, true, 8, 1, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur16, linearRayGenMotionBlur16, true, 16, 1, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur32, linearRayGenMotionBlur32, true, 32, 1, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenAA2, linearRayGenAA2, false, 1, 2, RayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenAA3, linearRayGenAA3, false, 1, 3, RayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenAA4, linearRayGenAA4, false, 1, 4, RayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur2AA2, linearRayGenMotionBlur2AA2, true, 2, 2, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur2AA3, linearRayGenMotionBlur2AA3, true, 2, 3, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur2AA4, linearRayGenMotionBlur2AA4, true, 2, 4, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur4AA2, linearRayGenMotionBlur4AA2, true, 4, 2, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur4AA3, linearRayGenMotionBlur4AA3, true, 4, 3, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur4AA4, linearRayGenMotionBlur4AA4, true, 4, 4, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur8AA2, linearRayGenMotionBlur8AA2, true, 8, 2, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur8AA3, linearRayGenMotionBlur8AA3, true, 8, 3, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur8AA4, linearRayGenMotionBlur8AA4, true, 8, 4, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur16AA2, linearRayGenMotionBlur16AA2, true, 16, 2, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur16AA3, linearRayGenMotionBlur16AA3, true, 16, 3, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur16AA4, linearRayGenMotionBlur16AA4, true, 16, 4, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur32AA2, linearRayGenMotionBlur32AA2, true, 32, 2, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur32AA3, linearRayGenMotionBlur32AA3, true, 32, 3, MotionBlurRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN(simpleRayGenMotionBlur32AA4, linearRayGenMotionBlur32AA4, true, 32, 4, MotionBlurRayGenData)

#undef RL_TOOLS_RENDERING_RAYTRACING_RGB_RAYGEN

#if RL_TOOLS_RENDERING_RAYTRACING_ENABLE_DEPTH_PROGRAMS
#define RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(NAME, MOTION_BLUR, MOTION_SAMPLES, AA_GRID, DATA) \
  OPTIX_RAYGEN_PROGRAM(NAME)() { rayGenImpl<DepthOutput, MOTION_BLUR, MOTION_SAMPLES, AA_GRID, DATA>(); }

  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGen, false, 1, 1, DepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur2, true, 2, 1, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur4, true, 4, 1, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur8, true, 8, 1, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur16, true, 16, 1, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur32, true, 32, 1, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenAA2, false, 1, 2, DepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenAA3, false, 1, 3, DepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenAA4, false, 1, 4, DepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur2AA2, true, 2, 2, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur2AA3, true, 2, 3, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur2AA4, true, 2, 4, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur4AA2, true, 4, 2, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur4AA3, true, 4, 3, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur4AA4, true, 4, 4, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur8AA2, true, 8, 2, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur8AA3, true, 8, 3, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur8AA4, true, 8, 4, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur16AA2, true, 16, 2, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur16AA3, true, 16, 3, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur16AA4, true, 16, 4, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur32AA2, true, 32, 2, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur32AA3, true, 32, 3, MotionBlurDepthRayGenData)
  RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN(depthRayGenMotionBlur32AA4, true, 32, 4, MotionBlurDepthRayGenData)

#undef RL_TOOLS_RENDERING_RAYTRACING_DEPTH_RAYGEN
#endif

  template <bool LOAD_TEXTURES, bool NORMAL_SHADING, bool METALLIC_REFLECTIONS>
  inline __device__ void triangleMeshBasicImpl()
  {
    owl::vec3f &prd = owl::getPRD<owl::vec3f>();

    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();

    owl::vec3f base_color = self.color;
    owl::vec3f normal_geometric(0.f, 0.f, 1.f);
    owl::vec3f ray_dir(0.f);

    if constexpr (LOAD_TEXTURES || NORMAL_SHADING || METALLIC_REFLECTIONS) {
      const int prim_id = optixGetPrimitiveIndex();
      const owl::vec3i index = self.index[prim_id];

      if constexpr (NORMAL_SHADING || METALLIC_REFLECTIONS) {
        const owl::vec3f &vertex_a = self.vertex[index.x];
        const owl::vec3f &vertex_b = self.vertex[index.y];
        const owl::vec3f &vertex_c = self.vertex[index.z];
        normal_geometric = normalize(cross(vertex_b - vertex_a, vertex_c - vertex_a));
        ray_dir = optixGetWorldRayDirection();
      }

      if constexpr (LOAD_TEXTURES) {
        if (self.has_texture && self.tex_coord) {
          const owl::vec2f bary = optixGetTriangleBarycentrics();
          const owl::vec2f tc
            = (1.f - bary.x - bary.y) * self.tex_coord[index.x]
              +      bary.x           * self.tex_coord[index.y]
              +             bary.y    * self.tex_coord[index.z];
          owl::vec4f tex_color = tex2D<float4>(self.texture, tc.x, tc.y);
          base_color = owl::vec3f(tex_color.x, tex_color.y, tex_color.z) * self.color;
        }
      }
    }

    owl::vec3f direct = base_color;
    if constexpr (NORMAL_SHADING) {
      direct = (.2f + .8f * fabs(dot(ray_dir, normal_geometric))) * base_color;
    }

    if constexpr (METALLIC_REFLECTIONS) {
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
    else {
      prd = direct;
    }
  }

#define RL_TOOLS_RENDERING_RAYTRACING_BASIC_CLOSEST_HIT(NAME, LOAD_TEXTURES, NORMAL_SHADING, METALLIC_REFLECTIONS) \
  OPTIX_CLOSEST_HIT_PROGRAM(NAME)() { triangleMeshBasicImpl<LOAD_TEXTURES, NORMAL_SHADING, METALLIC_REFLECTIONS>(); }

  RL_TOOLS_RENDERING_RAYTRACING_BASIC_CLOSEST_HIT(TriangleMeshBasicTTT, true, true, true)
  RL_TOOLS_RENDERING_RAYTRACING_BASIC_CLOSEST_HIT(TriangleMeshBasicTTF, true, true, false)
  RL_TOOLS_RENDERING_RAYTRACING_BASIC_CLOSEST_HIT(TriangleMeshBasicTFT, true, false, true)
  RL_TOOLS_RENDERING_RAYTRACING_BASIC_CLOSEST_HIT(TriangleMeshBasicTFF, true, false, false)
  RL_TOOLS_RENDERING_RAYTRACING_BASIC_CLOSEST_HIT(TriangleMeshBasicFTT, false, true, true)
  RL_TOOLS_RENDERING_RAYTRACING_BASIC_CLOSEST_HIT(TriangleMeshBasicFTF, false, true, false)
  RL_TOOLS_RENDERING_RAYTRACING_BASIC_CLOSEST_HIT(TriangleMeshBasicFFT, false, false, true)
  RL_TOOLS_RENDERING_RAYTRACING_BASIC_CLOSEST_HIT(TriangleMeshBasicFFF, false, false, false)

#undef RL_TOOLS_RENDERING_RAYTRACING_BASIC_CLOSEST_HIT

  OPTIX_CLOSEST_HIT_PROGRAM(TriangleMeshPBR)()
  {
    owl::vec3f &prd = owl::getPRD<owl::vec3f>();
    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();

    const int prim_id = optixGetPrimitiveIndex();
    const owl::vec3i index = self.index[prim_id];
    const owl::vec3f &vertex_a = self.vertex[index.x];
    const owl::vec3f &vertex_b = self.vertex[index.y];
    const owl::vec3f &vertex_c = self.vertex[index.z];
    const owl::vec2f bary = optixGetTriangleBarycentrics();
    const float w0 = 1.f - bary.x - bary.y;

    const owl::vec3f edge1 = vertex_b - vertex_a;
    const owl::vec3f edge2 = vertex_c - vertex_a;
    const owl::vec3f normal_geometric = normalize(cross(edge1, edge2));

    owl::vec3f N;
    if (self.normal) {
      N = normalize(w0 * self.normal[index.x] + bary.x * self.normal[index.y] + bary.y * self.normal[index.z]);
    } else {
      N = normal_geometric;
    }

    const owl::vec3f ray_dir = optixGetWorldRayDirection();
    if (dot(ray_dir, N) > 0.f) N = -N;

    const owl::vec2f tc = (self.tex_coord)
      ? w0 * self.tex_coord[index.x] + bary.x * self.tex_coord[index.y] + bary.y * self.tex_coord[index.z]
      : owl::vec2f(0.f);

    owl::vec3f base_color = self.color;
    float alpha = self.opacity;
    if (self.has_texture && self.tex_coord) {
      owl::vec4f tex_color = tex2D<float4>(self.texture, tc.x, tc.y);
      base_color = owl::vec3f(tex_color.x, tex_color.y, tex_color.z) * self.color;
      alpha *= tex_color.w;
    }

    float metallic = self.metallic;
    float roughness = self.roughness;
    if (self.has_metallic_roughness_map && self.tex_coord) {
      owl::vec4f mr_sample = tex2D<float4>(self.metallic_roughness_map, tc.x, tc.y);
      roughness = mr_sample.y * self.roughness;
      metallic = mr_sample.z * self.metallic;
    }

    if (self.has_normal_map && self.tex_coord) {
      const owl::vec2f tc0 = self.tex_coord[index.x];
      const owl::vec2f tc1 = self.tex_coord[index.y];
      const owl::vec2f tc2 = self.tex_coord[index.z];
      const owl::vec2f duv1 = tc1 - tc0;
      const owl::vec2f duv2 = tc2 - tc0;
      float det = duv1.x * duv2.y - duv2.x * duv1.y;
      if (fabsf(det) > 1e-8f) {
        float inv_det = 1.f / det;
        owl::vec3f T = normalize(inv_det * (duv2.y * edge1 - duv1.y * edge2));
        T = normalize(T - dot(T, N) * N);
        owl::vec3f B = cross(N, T);
        owl::vec4f nm_sample = tex2D<float4>(self.normal_map, w0 * tc0.x + bary.x * tc1.x + bary.y * tc2.x, w0 * tc0.y + bary.x * tc1.y + bary.y * tc2.y);
        owl::vec3f n_tangent = owl::vec3f(nm_sample.x * 2.f - 1.f, -(nm_sample.y * 2.f - 1.f), nm_sample.z * 2.f - 1.f);
        N = normalize(T * n_tangent.x + B * n_tangent.y + N * n_tangent.z);
      }
    }

    roughness = fmaxf(roughness, 0.04f);
    float roughness_alpha = roughness * roughness;
    float alpha2 = roughness_alpha * roughness_alpha;
    float k = (roughness + 1.f) * (roughness + 1.f) / 8.f;

    owl::vec3f V = -ray_dir;
    float NdotV = fmaxf(dot(N, V), 1e-4f);
    owl::vec3f F0 = owl::vec3f(0.04f) * (1.f - metallic) + base_color * metallic;

    owl::vec3f hit_point = ray_dir * optixGetRayTmax();
    hit_point.x += optixGetWorldRayOrigin().x;
    hit_point.y += optixGetWorldRayOrigin().y;
    hit_point.z += optixGetWorldRayOrigin().z;

    owl::vec3f Lo(0.f);
    for (int li = 0; li < self.num_scene_lights; li++) {
      const rendering::raytracing::SceneLight& light = self.scene_lights[li];
      owl::vec3f Lc(light.color[0], light.color[1], light.color[2]);
      owl::vec3f L;
      float attenuation = 1.f;

      if (light.type == 0) {
        L = owl::vec3f(light.direction[0], light.direction[1], light.direction[2]);
      } else {
        owl::vec3f to_light = owl::vec3f(light.position[0], light.position[1], light.position[2]) - hit_point;
        float dist = length(to_light);
        L = to_light * (1.f / fmaxf(dist, 1e-6f));
        attenuation = 1.f / (light.attenuation_constant + light.attenuation_linear * dist + light.attenuation_quadratic * dist * dist);
        if (light.type == 2) {
          owl::vec3f spot_dir(light.direction[0], light.direction[1], light.direction[2]);
          float cos_angle = dot(-L, spot_dir);
          float denom = light.cos_inner_cone - light.cos_outer_cone;
          float spot_t = (cos_angle - light.cos_outer_cone) / (fabsf(denom) > 1e-6f ? denom : 1e-6f);
          attenuation *= fmaxf(fminf(spot_t, 1.f), 0.f);
        }
      }

      float NdotL = fmaxf(dot(N, L), 0.f);
      if (NdotL <= 0.f) continue;

      owl::vec3f H = normalize(V + L);
      float NdotH = fmaxf(dot(N, H), 0.f);
      float VdotH = fmaxf(dot(V, H), 0.f);

      float denom_D = NdotH * NdotH * (alpha2 - 1.f) + 1.f;
      float D = alpha2 / (3.14159265f * denom_D * denom_D);

      float G1_V = NdotV / (NdotV * (1.f - k) + k);
      float G1_L = NdotL / (NdotL * (1.f - k) + k);
      float G = G1_V * G1_L;

      float pow5 = powf(1.f - VdotH, 5.f);
      owl::vec3f F = F0 + (owl::vec3f(1.f) - F0) * pow5;

      owl::vec3f specular = D * G * F * (1.f / (4.f * NdotV * NdotL + 1e-4f));
      owl::vec3f kd = (owl::vec3f(1.f) - F) * (1.f - metallic);
      owl::vec3f diffuse = kd * base_color * (1.f / 3.14159265f);

      Lo = Lo + (diffuse + specular) * Lc * (attenuation * NdotL);
    }

    float occlusion = 1.f;
    if (self.has_occlusion_map && self.tex_coord) {
      owl::vec4f ao_sample = tex2D<float4>(self.occlusion_map, tc.x, tc.y);
      occlusion = ao_sample.x;
    }

    owl::vec3f emissive_color(0.f);
    if (self.has_emissive_map && self.tex_coord) {
      owl::vec4f em_sample = tex2D<float4>(self.emissive_map, tc.x, tc.y);
      emissive_color = owl::vec3f(em_sample.x, em_sample.y, em_sample.z) * self.emissive;
    } else {
      emissive_color = self.emissive;
    }

    owl::vec3f ambient = self.ambient_color * base_color * occlusion;
    owl::vec3f color = ambient + Lo + emissive_color;

    unsigned int depth = optixGetPayload_2();
    if (depth < 1 && metallic > 0.1f) {
      owl::vec3f reflect_dir = ray_dir - 2.f * dot(ray_dir, N) * N;

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

      float fresnel_refl = F0.x + (1.f - F0.x) * powf(1.f - fmaxf(dot(V, N), 0.f), 5.f);
      float reflection_weight = fresnel_refl * (1.f - roughness);
      color = color * (1.f - reflection_weight) + reflected_color * reflection_weight;
    }

    bool transparent = (self.alpha_mode == 2 && alpha < 0.99f) || (self.alpha_mode == 1 && alpha < self.alpha_cutoff);
    if (transparent && depth < 1) {
      owl::vec3f behind_color;
      unsigned int tp0 = 0, tp1 = 0;
      owl::packPointer(&behind_color, tp0, tp1);
      unsigned int tp2 = depth + 1;
      optixTrace(self.world,
                 (const float3&)hit_point,
                 (const float3&)ray_dir,
                 1e-3f,
                 1e20f,
                 0.0f,
                 OptixVisibilityMask(255),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                 0, NUM_RAY_TYPES, 0,
                 tp0, tp1, tp2);

      if (self.alpha_mode == 1) {
        color = behind_color;
      } else {
        color = color * alpha + behind_color * (1.f - alpha);
      }
    }

    prd = color;
  }

  OPTIX_MISS_PROGRAM(miss)()
  {
    const owl::vec2i pixel_id = owl::getLaunchIndex();

    const MissProgData &self = owl::getProgramData<MissProgData>();

    owl::vec3f &prd = owl::getPRD<owl::vec3f>();
    int checker_pattern = (pixel_id.x / 8) ^ (pixel_id.y/8);
    prd = (checker_pattern&1) ? self.color_1 : self.color_0;
  }

  OPTIX_MISS_PROGRAM(missConstant)()
  {
    const MissProgData &self = owl::getProgramData<MissProgData>();
    owl::vec3f &prd = owl::getPRD<owl::vec3f>();
    prd = self.color_0;
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

    const OptixCameraData &cam = self.cameras[cam_idx];

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
