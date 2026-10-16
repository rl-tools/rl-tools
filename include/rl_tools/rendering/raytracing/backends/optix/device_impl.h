#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_DEVICE_IMPL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_DEVICE_IMPL_H

#include "device.h"
#include "overlay_accel.h"
#include "../../outputs/slots.h"
#include <optix_device.h>

#ifndef RL_TOOLS_RENDERING_RAYTRACING_ENABLE_NORMALS_PROGRAMS
#define RL_TOOLS_RENDERING_RAYTRACING_ENABLE_NORMALS_PROGRAMS 0
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools
{
  // ray type 2 (normals) only exists in normals-enabled PTX; the host mirrors this via
  // owlContextSetRayTypeCount keyed on SPEC::HAS_NORMALS, keeping the SBT stride in sync
  static constexpr int NUM_RAY_TYPES = RL_TOOLS_RENDERING_RAYTRACING_ENABLE_NORMALS_PROGRAMS ? 3 : 2;

  extern "C" __constant__ OverlayLaunchParams optixLaunchParams;

  inline __device__ int camera_from_fb_tile()
  {
    const owl::vec2i pixel_id = owl::getLaunchIndex();
    const int tile_col = pixel_id.x / optixLaunchParams.cam_width;
    const int tile_row = pixel_id.y / optixLaunchParams.cam_height;
    return tile_row * optixLaunchParams.grid_cols + tile_col;
  }

  inline __device__ owl::vec3f normalize_shading(const owl::vec3f& v) {
    return v * rsqrtf(dot(v, v));
  }

  inline __device__ float power_five(float value)
  {
    const float squared = value * value;
    return squared * squared * value;
  }

  inline __device__ float linear_to_srgb(float x) {
    if (x <= 0.0031308f) {
      return 12.92f * x;
    }
    return 1.055f * __powf(x, 1.f / 2.4f) - 0.055f;
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

  inline __device__ uint32_t packed_auxiliary_color(unsigned int r, unsigned int g, unsigned int b){
    return 0xFF000000u | r | (g << 8u) | (b << 16u);
  }

  inline __device__ owl::vec3f auxiliary_hue(float hue){
    const float x = 1.f - fabsf((hue - 2.f * floorf(hue * .5f)) - 1.f);
    switch((unsigned int)hue % 6u){
      case 0u: { return owl::vec3f(1.f, x, 0.f); }
      case 1u: { return owl::vec3f(x, 1.f, 0.f); }
      case 2u: { return owl::vec3f(0.f, 1.f, x); }
      case 3u: { return owl::vec3f(0.f, x, 1.f); }
      case 4u: { return owl::vec3f(x, 0.f, 1.f); }
      default: { return owl::vec3f(1.f, 0.f, x); }
    }
  }

  inline __device__ uint32_t segmentation_color(unsigned int id){
    if(id == 0xFFFFFFFFu){
      return 0xFF000000u;
    }
    const float scaled = float(id) * .61803398875f;
    const owl::vec3f color = auxiliary_hue((scaled - floorf(scaled)) * 6.f) * 255.f;
    return packed_auxiliary_color((unsigned int)color.x, (unsigned int)color.y, (unsigned int)color.z);
  }

  inline __device__ unsigned int normal_channel(float normal){
    const float unit = __fadd_rn(__fmul_rn(fminf(fmaxf(normal, -1.f), 1.f), .5f), .5f);
    return (unsigned int)((double)__fmul_rn(unit, 255.f) + .5);
  }

  inline __device__ uint32_t normal_color(float x, float y, float z){
    return packed_auxiliary_color(normal_channel(x), normal_channel(y), normal_channel(z));
  }

  inline __device__ uint32_t status_color(unsigned int status){
    switch(status){
      case 0u: { return 0xFFFFFFFFu; }
      case 1u: { return 0xFF000000u; }
      case 2u: { return 0xFFFF00FFu; }
      case 3u: { return 0xFFFF4020u; }
      case 4u: { return 0xFF0090FFu; }
      default: { return 0xFF0000FFu; }
    }
  }

  inline __device__ uint32_t flow_color(float u, float v, float scale){
    if((u == 0.f && v == 0.f) || scale <= 0.f){
      return 0xFFFFFFFFu;
    }
    const float saturation = fminf(sqrtf(u * u + v * v) / scale, 1.f);
    const float hue = (atan2f(v, u) / 6.28318530717958647692f + .5f) * 6.f;
    const owl::vec3f color = (owl::vec3f(1.f) - saturation * (owl::vec3f(1.f) - auxiliary_hue(hue))) * 255.f;
    return packed_auxiliary_color((unsigned int)color.x, (unsigned int)color.y, (unsigned int)color.z);
  }

  inline __device__ owl::vec3f lerp_camera_vec(const owl::vec3f &a, const owl::vec3f &b, float t)
  {
    return (1.f - t) * a + t * b;
  }

  using OutputSlot = rendering::raytracing::outputs::Slot;

  struct PixelLaunchContext
  {
    int cam_idx;
    int local_x;
    int local_y;
    int fb_offset;        // frame-buffer order, for the internal motion-blur accumulator
    unsigned int pixel;   // within the camera
    unsigned int outputs; // the camera's output-slot mask; zero outside the grid
  };

  template <typename RAYGEN_DATA>
  inline __device__ PixelLaunchContext pixel_launch_context(const RAYGEN_DATA &self)
  {
    const owl::vec2i pixel_id = owl::getLaunchIndex();
    const int tile_col = pixel_id.x / optixLaunchParams.cam_width;
    const int tile_row = pixel_id.y / optixLaunchParams.cam_height;
    const int cam_idx  = tile_row * optixLaunchParams.grid_cols + tile_col;
    const int local_x = pixel_id.x - tile_col * optixLaunchParams.cam_width;
    const int local_y = pixel_id.y - tile_row * optixLaunchParams.cam_height;
    const int fb_offset = cam_idx * optixLaunchParams.cam_width * optixLaunchParams.cam_height + local_y * optixLaunchParams.cam_width + local_x;
    const unsigned int camera_outputs = cam_idx < self.num_cameras ? (optixLaunchParams.all_cameras ? optixLaunchParams.output_mask : (optixLaunchParams.camera_table[cam_idx * rendering::raytracing::outputs::CAMERA_TABLE_STRIDE] & optixLaunchParams.output_mask)) : 0u;
    return {cam_idx, local_x, local_y, fb_offset, (unsigned int)(local_y * optixLaunchParams.cam_width + local_x), camera_outputs};
  }

  inline __device__ bool selected(const PixelLaunchContext &ctx, OutputSlot slot)
  {
    return ((ctx.outputs >> (unsigned int)slot) & 1u) != 0u;
  }

  // element index of the pixel in an output: the camera's row, then the pixel within it
  inline __device__ size_t output_element(const PixelLaunchContext &ctx, OutputSlot slot)
  {
    if (optixLaunchParams.all_cameras) {
      return (size_t)ctx.fb_offset;
    }
    const unsigned int row = optixLaunchParams.camera_table[ctx.cam_idx * rendering::raytracing::outputs::CAMERA_TABLE_STRIDE + 1 + (unsigned int)slot];
    return (size_t)row * optixLaunchParams.cam_width * optixLaunchParams.cam_height + ctx.pixel;
  }

  inline __device__ owl::vec3f trace_rgb_color_composed(OptixTraversableHandle world, int camera, unsigned int depth, const owl::vec3f &pos, const owl::vec3f &direction, float tmin, float tmax)
  {
    owl::vec3f color;
    unsigned int p0 = 0, p1 = 0;
    owl::packPointer(&color, p0, p1);
    unsigned int p2 = depth;
    unsigned int p3 = __float_as_uint(1e30f);
    optixTrace(world,
               (const float3&)pos,
               (const float3&)direction,
               tmin,
               tmax,
               0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0, NUM_RAY_TYPES, 0,
               p0, p1, p2, p3);
    const int overlay_count = optixLaunchParams.overlay_count;
    float best_t = __uint_as_float(p3);
    for (int k = 0; k < overlay_count; k++) {
      const uint32_t overlay = optixLaunchParams.attachments[camera * overlay_count + k];
      if (overlay == 0xFFFFFFFFu) continue;
      unsigned int q2 = depth;
      unsigned int q3 = __float_as_uint(best_t);
      optixTrace((OptixTraversableHandle)optixLaunchParams.overlays[overlay],
                 (const float3&)pos,
                 (const float3&)direction,
                 tmin,
                 fminf(best_t, tmax),
                 0.0f,
                 OptixVisibilityMask(1u << (overlay % rendering::raytracing::backends::optix::OVERLAYS_PER_ACCEL)),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                 0, NUM_RAY_TYPES, 1, // no-op miss: an overlay miss keeps the current color
                 p0, p1, q2, q3);
      best_t = fminf(best_t, __uint_as_float(q3));
    }
    return color;
  }

  inline __device__ bool trace_shadow_occluded(OptixTraversableHandle world, const owl::vec3f &pos, const owl::vec3f &direction, float max_dist, unsigned int visibility = 255u)
  {
    unsigned int u0 = __float_as_uint(max_dist);
    unsigned int u1 = 0;
    unsigned int u2 = 0; // collisionHit writes payload 2 (instance id); must be declared here
    optixTrace(world,
               (const float3&)pos,
               (const float3&)direction,
               1e-3f,
               max_dist,
               0.0f,
               OptixVisibilityMask(visibility),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               1, NUM_RAY_TYPES, 1,
               u0, u1, u2);
    return u1 != 0;
  }

  inline __device__ bool trace_shadow_occluded_composed(OptixTraversableHandle world, int camera, const owl::vec3f &pos, const owl::vec3f &direction, float max_dist)
  {
    if (trace_shadow_occluded(world, pos, direction, max_dist)) return true;
    const int overlay_count = optixLaunchParams.overlay_count;
    for (int k = 0; k < overlay_count; k++) {
      const uint32_t overlay = optixLaunchParams.attachments[camera * overlay_count + k];
      if (overlay == 0xFFFFFFFFu) continue;
      if (trace_shadow_occluded((OptixTraversableHandle)optixLaunchParams.overlays[overlay], pos, direction, max_dist, 1u << (overlay % rendering::raytracing::backends::optix::OVERLAYS_PER_ACCEL))) {
        return true;
      }
    }
    return false;
  }



  template <bool T_SRGB_OUTPUT>
  struct RgbOutput
  {
    using Accumulator = owl::vec3f;
    inline __device__ static Accumulator zero() { return owl::vec3f(0.f); }
    template <typename RAYGEN_DATA>
    inline __device__ static void accumulate(const RAYGEN_DATA &self, Accumulator &acc, int camera, const owl::vec3f &pos, const owl::vec3f &direction)
    {
      acc = acc + trace_rgb_color_composed(self.world, camera, 0, pos, direction, 0.f, 1e30f);
    }
    template <typename RAYGEN_DATA>
    inline __device__ static void store(const RAYGEN_DATA &self, const PixelLaunchContext &ctx, Accumulator acc, int samples)
    {
      const owl::vec3f color = acc * (1.f / float(samples));
      if (selected(ctx, OutputSlot::RGB_FLOAT32)) {
        // the observation is the frame-buffer color before 8-bit quantization — same transfer
        // curve, full float precision
        float *obs = self.obs_ptr + output_element(ctx, OutputSlot::RGB_FLOAT32) * 3;
        if constexpr (T_SRGB_OUTPUT) {
          obs[0] = linear_to_srgb(fminf(fmaxf(color.x, 0.f), 1.f));
          obs[1] = linear_to_srgb(fminf(fmaxf(color.y, 0.f), 1.f));
          obs[2] = linear_to_srgb(fminf(fmaxf(color.z, 0.f), 1.f));
        }
        else {
          obs[0] = fminf(fmaxf(color.x, 0.f), 1.f);
          obs[1] = fminf(fmaxf(color.y, 0.f), 1.f);
          obs[2] = fminf(fmaxf(color.z, 0.f), 1.f);
        }
      }
      if(selected(ctx, OutputSlot::RGB_RGBA8)){
        if constexpr (T_SRGB_OUTPUT) {
          self.fb_ptr[output_element(ctx, OutputSlot::RGB_RGBA8)] = make_srgb_rgba_from_linear(color);
        }
        else {
          self.fb_ptr[output_element(ctx, OutputSlot::RGB_RGBA8)] = make_linear_rgba_from_linear(color);
        }
      }
    }
  };



  template <typename OUTPUT, bool MOTION_BLUR, int MOTION_SAMPLES, int AA_GRID, typename RAYGEN_DATA>
  inline __device__ void rayGenImpl()
  {
    const RAYGEN_DATA &self = owl::getProgramData<RAYGEN_DATA>();
    const PixelLaunchContext ctx = pixel_launch_context(self);
    if ((ctx.outputs & rendering::raytracing::outputs::RGB_SLOTS) == 0u) {
      return;
    }

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
          const owl::vec2f screen = (owl::vec2f(ctx.local_x, ctx.local_y) + owl::vec2f((float(aa_x) + .5f) * inv_aa_grid, (float(aa_y) + .5f) * inv_aa_grid)) / owl::vec2f(optixLaunchParams.cam_width, optixLaunchParams.cam_height);
          const owl::vec3f direction = normalize(dir_00 + screen.u * dir_du + screen.v * dir_dv);
          OUTPUT::accumulate(self, accumulated, ctx.cam_idx, pos, direction);
        }
      }
    }
    OUTPUT::store(self, ctx, accumulated, MOTION_SAMPLES * AA_GRID * AA_GRID);
  }

  // dynamic-motion-blur pass outputs: accumulate the pass's linear mean into a float buffer
  // (transfer curve + quantization happen once, in the resolve kernel), so there is no
  // srgb/linear raygen split. Each thread owns its pixel — no atomics.
  struct RgbAccumulateOutput
  {
    using Accumulator = owl::vec3f;
    inline __device__ static Accumulator zero() { return owl::vec3f(0.f); }
    template <typename RAYGEN_DATA>
    inline __device__ static void accumulate(const RAYGEN_DATA &self, Accumulator &acc, int camera, const owl::vec3f &pos, const owl::vec3f &direction)
    {
      acc = acc + trace_rgb_color_composed(self.world, camera, 0, pos, direction, 0.f, 1e30f);
    }
    template <typename RAYGEN_DATA>
    inline __device__ static void store(const RAYGEN_DATA &self, const PixelLaunchContext &ctx, Accumulator acc, int samples)
    {
      const owl::vec3f mean = acc * (1.f / float(samples));
      float *accumulation = self.accum_ptr + (size_t)ctx.fb_offset * 3;
      accumulation[0] += mean.x;
      accumulation[1] += mean.y;
      accumulation[2] += mean.z;
    }
  };



  template <typename OUTPUT, int AA_GRID, typename RAYGEN_DATA>
  inline __device__ void rayGenAccumulateImpl()
  {
    const RAYGEN_DATA &self = owl::getProgramData<RAYGEN_DATA>();
    const PixelLaunchContext ctx = pixel_launch_context(self);
    if ((ctx.outputs & rendering::raytracing::outputs::RGB_SLOTS) == 0u) {
      return;
    }

    const float shutter_t = *self.shutter_ptr;
    const OptixCameraData &cam_open = self.cameras_open[ctx.cam_idx];
    const OptixCameraData &cam_close = self.cameras_close[ctx.cam_idx];
    const owl::vec3f pos = lerp_camera_vec(cam_open.pos, cam_close.pos, shutter_t);
    const owl::vec3f dir_00 = lerp_camera_vec(cam_open.dir_00, cam_close.dir_00, shutter_t);
    const owl::vec3f dir_du = lerp_camera_vec(cam_open.dir_du, cam_close.dir_du, shutter_t);
    const owl::vec3f dir_dv = lerp_camera_vec(cam_open.dir_dv, cam_close.dir_dv, shutter_t);

    typename OUTPUT::Accumulator accumulated = OUTPUT::zero();
    const float inv_aa_grid = 1.f / float(AA_GRID);
    for (int aa_y = 0; aa_y < AA_GRID; aa_y++) {
      for (int aa_x = 0; aa_x < AA_GRID; aa_x++) {
        const owl::vec2f screen = (owl::vec2f(ctx.local_x, ctx.local_y) + owl::vec2f((float(aa_x) + .5f) * inv_aa_grid, (float(aa_y) + .5f) * inv_aa_grid)) / owl::vec2f(optixLaunchParams.cam_width, optixLaunchParams.cam_height);
        const owl::vec3f direction = normalize(dir_00 + screen.u * dir_du + screen.v * dir_dv);
        OUTPUT::accumulate(self, accumulated, ctx.cam_idx, pos, direction);
      }
    }
    OUTPUT::store(self, ctx, accumulated, AA_GRID * AA_GRID);
  }

  OPTIX_RAYGEN_PROGRAM(accumRayGen)() { rayGenAccumulateImpl<RgbAccumulateOutput, 1, AccumulateRayGenData>(); }
  OPTIX_RAYGEN_PROGRAM(accumRayGenAA2)() { rayGenAccumulateImpl<RgbAccumulateOutput, 2, AccumulateRayGenData>(); }
  OPTIX_RAYGEN_PROGRAM(accumRayGenAA3)() { rayGenAccumulateImpl<RgbAccumulateOutput, 3, AccumulateRayGenData>(); }
  OPTIX_RAYGEN_PROGRAM(accumRayGenAA4)() { rayGenAccumulateImpl<RgbAccumulateOutput, 4, AccumulateRayGenData>(); }



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



  // geometry lives in object space under the instance (top-level) group; the intrinsics apply
  // the current instance transform (exact for identity instances)
  inline __device__ owl::vec3f transform_point_to_world(const owl::vec3f& point){
    const float3 world = optixTransformPointFromObjectToWorldSpace(make_float3(point.x, point.y, point.z));
    return owl::vec3f(world.x, world.y, world.z);
  }

  inline __device__ owl::vec3f transform_normal_to_world(const owl::vec3f& normal){
    const float3 world = optixTransformNormalFromObjectToWorldSpace(make_float3(normal.x, normal.y, normal.z));
    return owl::vec3f(world.x, world.y, world.z);
  }

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
        const owl::vec3f vertex_a = transform_point_to_world(self.vertex[index.x]);
        const owl::vec3f vertex_b = transform_point_to_world(self.vertex[index.y]);
        const owl::vec3f vertex_c = transform_point_to_world(self.vertex[index.z]);
        normal_geometric = normalize_shading(cross(vertex_b - vertex_a, vertex_c - vertex_a));
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

    if(self.unlit != 0){
      prd = base_color;
      optixSetPayload_3(__float_as_uint(optixGetRayTmax()));
      return;
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

        const owl::vec3f reflected_color = trace_rgb_color_composed(self.world, camera_from_fb_tile(), depth + 1, hit_point, reflect_dir, 1e-3f, 1e20f);

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
    optixSetPayload_3(__float_as_uint(optixGetRayTmax()));
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

  template <bool ENABLE_PUNCTUAL_LIGHT_SHADOWS>
  inline __device__ void triangleMeshPBRImpl()
  {
    owl::vec3f &prd = owl::getPRD<owl::vec3f>();
    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();

    const int prim_id = optixGetPrimitiveIndex();
    const owl::vec3i index = self.index[prim_id];
    const owl::vec3f vertex_a = transform_point_to_world(self.vertex[index.x]);
    const owl::vec3f vertex_b = transform_point_to_world(self.vertex[index.y]);
    const owl::vec3f vertex_c = transform_point_to_world(self.vertex[index.z]);
    const owl::vec2f bary = optixGetTriangleBarycentrics();
    const float w0 = 1.f - bary.x - bary.y;

    const owl::vec3f edge1 = vertex_b - vertex_a;
    const owl::vec3f edge2 = vertex_c - vertex_a;
    const owl::vec3f normal_geometric = normalize_shading(cross(edge1, edge2));

    owl::vec3f N;
    if (self.normal) {
      N = normalize_shading(transform_normal_to_world(w0 * self.normal[index.x] + bary.x * self.normal[index.y] + bary.y * self.normal[index.z]));
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

    owl::vec3f hit_point = ray_dir * optixGetRayTmax();
    hit_point.x += optixGetWorldRayOrigin().x;
    hit_point.y += optixGetWorldRayOrigin().y;
    hit_point.z += optixGetWorldRayOrigin().z;

    const int camera = camera_from_fb_tile();

    owl::vec3f color = base_color;
    unsigned int depth = optixGetPayload_2();
    if(self.unlit == 0){
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
          owl::vec3f T = normalize_shading(inv_det * (duv2.y * edge1 - duv1.y * edge2));
          T = normalize_shading(T - dot(T, N) * N);
          owl::vec3f B = cross(N, T);
          owl::vec4f nm_sample = tex2D<float4>(self.normal_map, w0 * tc0.x + bary.x * tc1.x + bary.y * tc2.x, w0 * tc0.y + bary.x * tc1.y + bary.y * tc2.y);
          owl::vec3f n_tangent = owl::vec3f(nm_sample.x * 2.f - 1.f, -(nm_sample.y * 2.f - 1.f), nm_sample.z * 2.f - 1.f);
          N = normalize_shading(T * n_tangent.x + B * n_tangent.y + N * n_tangent.z);
        }
      }

      roughness = fmaxf(roughness, 0.04f);
      float roughness_alpha = roughness * roughness;
      float alpha2 = roughness_alpha * roughness_alpha;
      float k = (roughness + 1.f) * (roughness + 1.f) / 8.f;

      owl::vec3f V = -ray_dir;
      float NdotV = fmaxf(dot(N, V), 1e-4f);
      owl::vec3f F0 = owl::vec3f(0.04f) * (1.f - metallic) + base_color * metallic;

      owl::vec3f Lo(0.f);
      for (int li = 0; li < self.num_scene_lights; li++) {
        const rendering::raytracing::SceneLight& light = self.scene_lights[li];
        owl::vec3f Lc(light.color[0], light.color[1], light.color[2]);
        owl::vec3f L;
        float attenuation = 1.f;
        float light_distance = 1e20f;

        if (light.type == 0) {
          L = owl::vec3f(light.direction[0], light.direction[1], light.direction[2]);
        } else {
          owl::vec3f to_light = owl::vec3f(light.position[0], light.position[1], light.position[2]) - hit_point;
          float dist = length(to_light);
          light_distance = fmaxf(dist - 1e-3f, 0.f);
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
        if constexpr (ENABLE_PUNCTUAL_LIGHT_SHADOWS) {
          if (light.type != 0 && trace_shadow_occluded_composed(self.world, camera, hit_point + N * 2e-3f, L, light_distance)) continue;
        }

        owl::vec3f H = normalize_shading(V + L);
        float NdotH = fmaxf(dot(N, H), 0.f);
        float VdotH = fmaxf(dot(V, H), 0.f);

        float denom_D = NdotH * NdotH * (alpha2 - 1.f) + 1.f;
        float D = __fdividef(alpha2, 3.14159265f * denom_D * denom_D);

        float G1_V = __fdividef(NdotV, NdotV * (1.f - k) + k);
        float G1_L = __fdividef(NdotL, NdotL * (1.f - k) + k);
        float G = G1_V * G1_L;

        float pow5 = power_five(1.f - VdotH);
        owl::vec3f F = F0 + (owl::vec3f(1.f) - F0) * pow5;

        owl::vec3f specular = D * G * F * (__fdividef(1.f, 4.f * NdotV * NdotL + 1e-4f));
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

      owl::vec3f ambient = self.ambient_color * base_color * ((1.f - metallic) * occlusion);
      color = ambient + Lo + emissive_color;

      if (depth < 1 && metallic > 0.1f) {
        owl::vec3f reflect_dir = ray_dir - 2.f * dot(ray_dir, N) * N;

        const owl::vec3f reflected_color = trace_rgb_color_composed(self.world, camera, depth + 1, hit_point, reflect_dir, 1e-3f, 1e20f);

        float fresnel_refl = F0.x + (1.f - F0.x) * power_five(1.f - fmaxf(dot(V, N), 0.f));
        float reflection_weight = fresnel_refl * (1.f - roughness);
        color = color * (1.f - reflection_weight) + reflected_color * reflection_weight;
      }

    }

    bool transparent = (self.alpha_mode == 2 && alpha < 0.99f) || (self.alpha_mode == 1 && alpha < self.alpha_cutoff);
    if (transparent && depth < 1) {
      const owl::vec3f behind_color = trace_rgb_color_composed(self.world, camera, depth + 1, hit_point, ray_dir, 1e-5f, 1e20f);

      if (self.alpha_mode == 1) {
        color = behind_color;
      } else {
        color = color * alpha + behind_color * (1.f - alpha);
      }
    }

    prd = color;
    optixSetPayload_3(__float_as_uint(optixGetRayTmax()));
  }

  OPTIX_CLOSEST_HIT_PROGRAM(TriangleMeshPBR)()
  {
    triangleMeshPBRImpl<false>();
  }

  OPTIX_CLOSEST_HIT_PROGRAM(TriangleMeshPBRShadows)()
  {
    triangleMeshPBRImpl<true>();
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
    const unsigned int row = optixLaunchParams.all_cameras ? (unsigned int)cam_idx : optixLaunchParams.camera_table[cam_idx * rendering::raytracing::outputs::CAMERA_TABLE_STRIDE + 1 + (unsigned int)OutputSlot::PROBES];
    if (row == rendering::raytracing::outputs::NO_ROW) {
      return;
    }

    const OptixCameraData &cam = self.cameras[cam_idx];

    const owl::vec3f forward = normalize(cam.dir_00 + 0.5f * cam.dir_du + 0.5f * cam.dir_dv);
    owl::vec3f dir;
    if (probe_idx == 0) {
      dir = forward;
    } else {
      const owl::vec3f local = self.probe_directions[probe_idx];
      const owl::vec3f left = -normalize(cam.dir_du);
      const owl::vec3f up = -normalize(cam.dir_dv);
      dir = normalize(local.x * forward + local.y * left + local.z * up);
    }

    // Step B: raw optixTrace + CollisionResult PRD, no FIRST_HIT
    CollisionResult result;
    result.distance = self.max_dist;
    result.hit = 0;

    unsigned int u0, u1, u2;
    u0 = __float_as_uint(result.distance);
    u1 = result.hit;
    u2 = 0xFFFFFFFFu;

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
        u0, u1, u2);
    const int overlay_count = optixLaunchParams.overlay_count;
    for (int k = 0; k < overlay_count; k++) {
      const uint32_t overlay = optixLaunchParams.attachments[cam_idx * overlay_count + k];
      if (overlay == 0xFFFFFFFFu) continue;
      optixTrace(
          (OptixTraversableHandle)optixLaunchParams.overlays[overlay],
          (const float3&)cam.pos,
          (const float3&)dir,
          1e-3f,
          __uint_as_float(u0),
          0.0f,
          OptixVisibilityMask(1u << (overlay % rendering::raytracing::backends::optix::OVERLAYS_PER_ACCEL)),
          OPTIX_RAY_FLAG_DISABLE_ANYHIT,
          1, NUM_RAY_TYPES, 1,
          u0, u1, u2);
    }

    if(u1 != 0){
      result.distance = __uint_as_float(u0);
    }
    else{
      store_miss(result.distance);
    }
    result.hit = u1;
    self.results[row * self.num_probes + probe_idx] = result;
  }

  OPTIX_CLOSEST_HIT_PROGRAM(collisionHit)()
  {
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(1);
    optixSetPayload_2(optixGetInstanceId());
  }

  OPTIX_MISS_PROGRAM(collisionMiss)()
  {
    // Payloads stay as initialized by caller: distance = max_dist, hit = 0, instance = sentinel
  }

  // world point → screen fraction through the linear camera model: solves
  // dir_00 + screen_x·dir_du + screen_y·dir_dv = lambda·(point − pos) by Cramer's rule;
  // returns false for a degenerate basis or a point at/behind the camera (lambda <= 0)
  inline __device__ bool project_camera(const OptixCameraData &cam, const owl::vec3f &point, float &screen_x, float &screen_y)
  {
    const owl::vec3f direction = point - cam.pos;
    const owl::vec3f negative_direction = -direction;
    const owl::vec3f cross_dv_negative_direction = cross(cam.dir_dv, negative_direction);
    const float det = dot(cam.dir_du, cross_dv_negative_direction);
    if (det > -1e-12f && det < 1e-12f)
      return false;
    const float inv_det = 1.f / det;
    const owl::vec3f b = -cam.dir_00;
    screen_x = dot(b, cross_dv_negative_direction) * inv_det;
    screen_y = dot(cam.dir_du, cross(b, negative_direction)) * inv_det;
    const float lambda = dot(cam.dir_du, cross(cam.dir_dv, b)) * inv_det;
    return lambda > 0.f;
  }

  inline __device__ owl::vec3f transform_point_rows(const float *m, const owl::vec3f &p)
  {
    return owl::vec3f(
        m[0]*p.x + m[1]*p.y + m[2]*p.z + m[3],
        m[4]*p.x + m[5]*p.y + m[6]*p.z + m[7],
        m[8]*p.x + m[9]*p.y + m[10]*p.z + m[11]);
  }

  // any-hit visibility against the world plus one instant's overlays attached to the camera:
  // the collision ray type with terminate-on-first-hit, like the punctual-light shadows
  inline __device__ bool trace_flow_visibility_occluded(OptixTraversableHandle world, const unsigned long long *overlays, int camera, const owl::vec3f &pos, const owl::vec3f &direction, float max_dist)
  {
    if (trace_shadow_occluded(world, pos, direction, max_dist)) return true;
    const int overlay_count = optixLaunchParams.overlay_count;
    for (int k = 0; k < overlay_count; k++) {
      const uint32_t overlay = optixLaunchParams.attachments[camera * overlay_count + k];
      if (overlay == 0xFFFFFFFFu) continue;
      if (trace_shadow_occluded((OptixTraversableHandle)overlays[overlay], pos, direction, max_dist, 1u << (overlay % rendering::raytracing::backends::optix::OVERLAYS_PER_ACCEL))) {
        return true;
      }
    }
    return false;
  }

  inline __device__ void write_flow_result(const FlowRayGenData &self, const PixelLaunchContext &ctx, const OptixCameraData &cam, const owl::vec3f &direction, unsigned int u0, unsigned int u1, unsigned int u2)
  {
    using FlowStatus = rendering::raytracing::FlowStatus;
    const int cam_idx = ctx.cam_idx;
    const int local_x = ctx.local_x;
    const int local_y = ctx.local_y;
    const unsigned long long *overlays_target = self.source_previous != 0 ? optixLaunchParams.overlays : optixLaunchParams.overlays_previous;
    float flow_u = 0.f, flow_v = 0.f;
    unsigned int status = FlowStatus::MISS;
    if (u1 != 0u) {
      const float t = __uint_as_float(u0);
      owl::vec3f point = owl::vec3f(cam.pos) + direction * t;
      if (self.flow_deltas != nullptr && u2 >= self.source_first_overlay_instance) {
        point = transform_point_rows(self.flow_deltas + (size_t)(u2 - self.source_first_overlay_instance) * 12, point);
      }
      const OptixCameraData &cam_target = self.cameras_target[cam_idx];
      float target_x, target_y;
      if (!project_camera(cam_target, point, target_x, target_y)) {
        status = FlowStatus::BEHIND;
      } else {
        flow_u = target_x * (float)optixLaunchParams.cam_width - ((float)local_x + 0.5f);
        flow_v = target_y * (float)optixLaunchParams.cam_height - ((float)local_y + 0.5f);
        if (target_x < 0.f || target_x >= 1.f || target_y < 0.f || target_y >= 1.f) {
          status = FlowStatus::OUT_OF_FRAME;
        } else if((ctx.outputs & (self.source_previous != 0 ? rendering::raytracing::outputs::FLOW_FORWARD_VISIBILITY_SLOTS : rendering::raytracing::outputs::FLOW_BACKWARD_VISIBILITY_SLOTS)) != 0u){
          const owl::vec3f to_point = point - owl::vec3f(cam_target.pos);
          const float length = sqrtf(dot(to_point, to_point));
          const owl::vec3f visibility_direction = to_point * (1.f / length);
          const bool occluded = trace_flow_visibility_occluded(self.world, overlays_target, cam_idx, owl::vec3f(cam_target.pos), visibility_direction, length * (1.f - 1e-4f)); // FLOW_VISIBILITY_EPSILON
          status = occluded ? FlowStatus::OCCLUDED : FlowStatus::VALID;
        }
      }
    }
    const OutputSlot vector_slot = self.source_previous != 0 ? OutputSlot::FLOW_FORWARD_FLOAT32 : OutputSlot::FLOW_FLOAT32;
    const OutputSlot color_slot = self.source_previous != 0 ? OutputSlot::FLOW_FORWARD_RGBA8 : OutputSlot::FLOW_RGBA8;
    const OutputSlot status_slot = self.source_previous != 0 ? OutputSlot::FLOW_FORWARD_STATUS_UINT32 : OutputSlot::FLOW_STATUS_UINT32;
    const OutputSlot status_color_slot = self.source_previous != 0 ? OutputSlot::FLOW_FORWARD_STATUS_RGBA8 : OutputSlot::FLOW_STATUS_RGBA8;
    if(selected(ctx, vector_slot)){
      const size_t element = output_element(ctx, vector_slot);
      self.flow_ptr[element * 2 + 0] = flow_u;
      self.flow_ptr[element * 2 + 1] = flow_v;
    }
    if(selected(ctx, status_slot)){
      self.status_ptr[output_element(ctx, status_slot)] = status;
    }
    if(selected(ctx, color_slot)){
      self.color_ptr[output_element(ctx, color_slot)] = status == FlowStatus::VALID ? flow_color(flow_u, flow_v, self.color_scale) : status_color(status);
    }
    if(selected(ctx, status_color_slot)){
      self.status_color_ptr[output_element(ctx, status_color_slot)] = status_color(status);
    }
  }

  inline __device__ void trace_flow_pixel(const FlowRayGenData &self)
  {
    const PixelLaunchContext ctx = pixel_launch_context(self);
    if ((ctx.outputs & (self.source_previous != 0 ? rendering::raytracing::outputs::FLOW_FORWARD_SLOTS : rendering::raytracing::outputs::FLOW_BACKWARD_SLOTS)) == 0u) {
      return;
    }
    const int cam_idx = ctx.cam_idx;
    const int local_x = ctx.local_x;
    const int local_y = ctx.local_y;
    const unsigned long long *overlays_source = self.source_previous != 0 ? optixLaunchParams.overlays_previous : optixLaunchParams.overlays;

    const OptixCameraData &cam = self.cameras_source[cam_idx];
    const float screen_x = ((float)local_x + 0.5f) / (float)optixLaunchParams.cam_width;
    const float screen_y = ((float)local_y + 0.5f) / (float)optixLaunchParams.cam_height;
    const owl::vec3f direction = normalize(cam.dir_00 + screen_x * cam.dir_du + screen_y * cam.dir_dv);

    unsigned int u0 = __float_as_uint(1e30f);
    unsigned int u1 = 0;
    unsigned int u2 = 0xFFFFFFFFu;
    optixTrace(
        self.world,
        (const float3&)cam.pos,
        (const float3&)direction,
        0.f,
        1e30f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        1, NUM_RAY_TYPES, 1, // collision ray type
        u0, u1, u2);
    const int overlay_count = optixLaunchParams.overlay_count;
    for (int k = 0; k < overlay_count; k++) {
      const uint32_t overlay = optixLaunchParams.attachments[cam_idx * overlay_count + k];
      if (overlay == 0xFFFFFFFFu) continue;
      optixTrace(
          (OptixTraversableHandle)overlays_source[overlay],
          (const float3&)cam.pos,
          (const float3&)direction,
          0.f,
          __uint_as_float(u0),
          0.0f,
          OptixVisibilityMask(1u << (overlay % rendering::raytracing::backends::optix::OVERLAYS_PER_ACCEL)),
          OPTIX_RAY_FLAG_DISABLE_ANYHIT,
          1, NUM_RAY_TYPES, 1,
          u0, u1, u2);
    }
    write_flow_result(self, ctx, cam, direction, u0, u1, u2);
  }

  OPTIX_RAYGEN_PROGRAM(auxiliaryRayGen)()
  {
    const AuxiliaryRayGenData &self = owl::getProgramData<AuxiliaryRayGenData>();
    const auto &flow = self.flow;
    const PixelLaunchContext ctx = pixel_launch_context(flow);
    namespace outputs = rendering::raytracing::outputs;
    if((ctx.outputs & (outputs::SURFACE_SLOTS | outputs::FLOW_BACKWARD_SLOTS)) == 0u){
      return;
    }
    const OptixCameraData &cam = flow.cameras_source[ctx.cam_idx];
    const float screen_x = ((float)ctx.local_x + .5f) / (float)optixLaunchParams.cam_width;
    const float screen_y = ((float)ctx.local_y + .5f) / (float)optixLaunchParams.cam_height;
    const owl::vec3f direction = normalize(cam.dir_00 + screen_x * cam.dir_du + screen_y * cam.dir_dv);
    unsigned int distance = __float_as_uint(1e30f), instance = 0xffffffffu;
    unsigned int nx = __float_as_uint(0.f), ny = __float_as_uint(0.f), nz = __float_as_uint(0.f);
#if RL_TOOLS_RENDERING_RAYTRACING_ENABLE_NORMALS_PROGRAMS
    if((ctx.outputs & outputs::NORMALS_SLOTS) != 0u){
      optixTrace(flow.world, (const float3&)cam.pos, (const float3&)direction, 0.f, 1e30f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 2, NUM_RAY_TYPES, 1, distance, instance, nx, ny, nz);
      const int overlay_count = optixLaunchParams.overlay_count;
      for(int slot = 0; slot < overlay_count; slot++){
        const uint32_t overlay = optixLaunchParams.attachments[ctx.cam_idx * overlay_count + slot];
        if(overlay != 0xffffffffu){
          optixTrace((OptixTraversableHandle)optixLaunchParams.overlays[overlay], (const float3&)cam.pos, (const float3&)direction, 0.f, __uint_as_float(distance), 0.f, OptixVisibilityMask(1u << (overlay % rendering::raytracing::backends::optix::OVERLAYS_PER_ACCEL)), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 2, NUM_RAY_TYPES, 1, distance, instance, nx, ny, nz);
        }
      }
    }
    else
#endif
    {
      unsigned int hit = 0;
      optixTrace(flow.world, (const float3&)cam.pos, (const float3&)direction, 0.f, 1e30f, 0.f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 1, NUM_RAY_TYPES, 1, distance, hit, instance);
      const int overlay_count = optixLaunchParams.overlay_count;
      for(int slot = 0; slot < overlay_count; slot++){
        const uint32_t overlay = optixLaunchParams.attachments[ctx.cam_idx * overlay_count + slot];
        if(overlay != 0xffffffffu){
          optixTrace((OptixTraversableHandle)optixLaunchParams.overlays[overlay], (const float3&)cam.pos, (const float3&)direction, 0.f, __uint_as_float(distance), 0.f, OptixVisibilityMask(1u << (overlay % rendering::raytracing::backends::optix::OVERLAYS_PER_ACCEL)), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 1, NUM_RAY_TYPES, 1, distance, hit, instance);
        }
      }
    }
    if((ctx.outputs & outputs::DEPTH_SLOTS) != 0u){
      const owl::vec3f normal = normalize(cross(cam.dir_du, cam.dir_dv));
      const float depth = instance != 0xffffffffu ? __uint_as_float(distance) * fabsf(dot(direction, normal)) : __uint_as_float(0x7f800000u);
      if(selected(ctx, OutputSlot::DEPTH_FLOAT32)){
        self.depth_ptr[output_element(ctx, OutputSlot::DEPTH_FLOAT32)] = depth;
      }
      if(selected(ctx, OutputSlot::DEPTH_RGBA8)){
        const unsigned int shade = (unsigned int)(fminf(fmaxf(depth / self.depth_scale, 0.f), 1.f) * 255.f);
        self.depth_color_ptr[output_element(ctx, OutputSlot::DEPTH_RGBA8)] = packed_auxiliary_color(shade, shade, shade);
      }
    }
    if(selected(ctx, OutputSlot::SEGMENTATION_UINT32)){
      self.seg_ptr[output_element(ctx, OutputSlot::SEGMENTATION_UINT32)] = instance;
    }
    if(selected(ctx, OutputSlot::SEGMENTATION_RGBA8)){
      self.seg_color_ptr[output_element(ctx, OutputSlot::SEGMENTATION_RGBA8)] = segmentation_color(instance);
    }
    if(selected(ctx, OutputSlot::SEMANTIC_UINT32) || selected(ctx, OutputSlot::SEMANTIC_RGBA8)){
      const unsigned int semantic = instance == 0xffffffffu ? instance : optixLaunchParams.instance_classes[instance];
      if(selected(ctx, OutputSlot::SEMANTIC_UINT32)){
        self.semantic_ptr[output_element(ctx, OutputSlot::SEMANTIC_UINT32)] = semantic;
      }
      if(selected(ctx, OutputSlot::SEMANTIC_RGBA8)){
        self.semantic_color_ptr[output_element(ctx, OutputSlot::SEMANTIC_RGBA8)] = segmentation_color(semantic);
      }
    }
    if(selected(ctx, OutputSlot::NORMALS_FLOAT32)){
      const size_t element = output_element(ctx, OutputSlot::NORMALS_FLOAT32);
      self.normals_ptr[element * 3] = __uint_as_float(nx);
      self.normals_ptr[element * 3 + 1] = __uint_as_float(ny);
      self.normals_ptr[element * 3 + 2] = __uint_as_float(nz);
    }
    if(selected(ctx, OutputSlot::NORMALS_RGBA8)){
      self.normals_color_ptr[output_element(ctx, OutputSlot::NORMALS_RGBA8)] = normal_color(__uint_as_float(nx), __uint_as_float(ny), __uint_as_float(nz));
    }
    if((ctx.outputs & outputs::FLOW_BACKWARD_SLOTS) != 0u){
      write_flow_result(flow, ctx, cam, direction, distance, instance != 0xffffffffu, instance);
    }
  }

  OPTIX_RAYGEN_PROGRAM(flowForwardRayGen)()
  {
    trace_flow_pixel(owl::getProgramData<FlowRayGenData>());
  }

#if RL_TOOLS_RENDERING_RAYTRACING_ENABLE_NORMALS_PROGRAMS
  OPTIX_CLOSEST_HIT_PROGRAM(auxiliaryHit)()
  {
    const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
    const owl::vec3i index = self.index[optixGetPrimitiveIndex()];
    const owl::vec3f vertex_a = transform_point_to_world(self.vertex[index.x]);
    const owl::vec3f vertex_b = transform_point_to_world(self.vertex[index.y]);
    const owl::vec3f vertex_c = transform_point_to_world(self.vertex[index.z]);
    owl::vec3f normal = normalize(cross(vertex_b - vertex_a, vertex_c - vertex_a));
    const owl::vec3f ray_dir = optixGetWorldRayDirection();
    if (dot(ray_dir, normal) > 0.f) {
      normal = -normal;
    }
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(optixGetInstanceId());
    optixSetPayload_2(__float_as_uint(normal.x));
    optixSetPayload_3(__float_as_uint(normal.y));
    optixSetPayload_4(__float_as_uint(normal.z));
  }

#endif


}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
