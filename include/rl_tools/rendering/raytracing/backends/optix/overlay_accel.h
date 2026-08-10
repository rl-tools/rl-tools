#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_OVERLAY_ACCEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_OPTIX_OVERLAY_ACCEL_H

#include "../../../../rl_tools.h"

#include <cuda_runtime.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
// device-side overlay TLAS path: raw optixAccelBuild over device-resident instance descriptors,
// bypassing OWL's host-staged instance groups so update_launch never synchronizes the host.
// Implemented in src/rendering/raytracing/backends/optix/overlay_accel.cu (linked via the
// rendering_raytracing_backend* targets); this header stays compilable by non-CUDA TUs.
namespace rl_tools::rendering::raytracing::backends::optix{
    struct OverlaySlotStructure{
        unsigned int active;
        unsigned int pose_slot;
        unsigned int object;
        unsigned int padding;
        float part_local[12];
    };
    static_assert(sizeof(OverlaySlotStructure) == 64, "OverlaySlotStructure layout must match the fill kernel");
    struct OverlayObjectEntry{
        unsigned long long traversable;
        unsigned int sbt_offset;
        unsigned int segmentation_class;
    };
    struct OverlayAccelState;

    OverlayAccelState* overlay_accel_create(void* optix_device_context, unsigned int num_overlays, unsigned int max_instances, unsigned int num_scene_instances, unsigned long long filler_traversable);
    void overlay_accel_destroy(OverlayAccelState* state);
    // device pointer to the [num_overlays * max_instances] slot structure table (staging target)
    OverlaySlotStructure* overlay_accel_structure(OverlayAccelState* state);
    // per-object BLAS traversables + SBT offsets + classes; upload once, after owlBuildSBT
    void overlay_accel_upload_objects(OverlayAccelState* state, const OverlayObjectEntry* objects, unsigned int num_objects);
    // per-overlay TLAS handles; stable across rebuilds (fixed output buffers), valid after create
    const unsigned long long* overlay_accel_traversables(const OverlayAccelState* state);
    // enqueues the instance fill kernel and one optixAccelBuild per overlay on stream; no host sync.
    // transforms: [num_overlays * max_instances * 12] device; instance_classes: per global
    // instance id, device — the kernel maintains the overlay range [num_scene_instances, ...).
    // shutter_out (optional device float): the fill kernel stores shutter_t there on-stream, so
    // per-pass shutter times reach the accumulate ray gens without per-launch param uploads
    // (which would race: OWL stages launch params through a single pinned host buffer)
    void overlay_accel_build(OverlayAccelState* state, const float* transforms, unsigned int* instance_classes, float shutter_t, float* shutter_out, cudaStream_t stream);
    // divides the linear accumulators by num_samples and writes the packed frame buffer (+
    // observation) / depth buffer with the exact quantization of the single-launch path;
    // null accumulator pointers skip the corresponding output
    void overlay_accel_resolve(const float* rgb_accumulation, unsigned int* frame_buffer, float* observation, int srgb_output, const float* depth_accumulation, float* depth_buffer, unsigned int num_pixels, unsigned int num_samples, cudaStream_t stream);
    // expands per-slot shutter pairs (pairs: [2][num_slots][12], open then close) into the
    // per-sample transforms_motion slabs (slerp at (s+0.5)/num_samples) and writes the close
    // entries into the transforms tensor — the on-device counterpart of set_transform_pair
    void overlay_accel_expand_motion(const float* pairs, float* transforms_motion, float* transforms, unsigned int num_slots, unsigned int num_samples, cudaStream_t stream);
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
