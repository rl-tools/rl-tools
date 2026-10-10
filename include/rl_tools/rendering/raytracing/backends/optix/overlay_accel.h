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
        unsigned int object;
    };
    static_assert(sizeof(OverlaySlotStructure) == 8, "OverlaySlotStructure layout must match the fill kernel");
    struct OverlayObjectEntry{
        unsigned long long traversable;
        unsigned int sbt_offset;
        unsigned int segmentation_class;
    };
    struct OverlayAccelState;

    // num_instants: 1 for the steady state alone, 2 when flow keeps the previous instant's TLASes
    // resident next to the current ones — instant i's instances carry the global ids
    // [num_scene_instances + i * slots, num_scene_instances + (i + 1) * slots)
    OverlayAccelState* overlay_accel_create(void* optix_device_context, unsigned int num_overlays, unsigned int max_instances, unsigned int num_scene_instances, unsigned long long filler_traversable, unsigned int num_instants);
    void overlay_accel_destroy(OverlayAccelState* state);
    // device pointer to the [num_overlays * max_instances] slot structure table (staging target)
    OverlaySlotStructure* overlay_accel_structure(OverlayAccelState* state);
    // per-object BLAS traversables + SBT offsets + classes; upload once, after owlBuildSBT
    void overlay_accel_upload_objects(OverlayAccelState* state, const OverlayObjectEntry* objects, unsigned int num_objects);
    // per-overlay TLAS handles of one instant; stable across rebuilds (fixed output buffers), valid after create
    const unsigned long long* overlay_accel_traversables(const OverlayAccelState* state, unsigned int instant);
    // enqueues the instance fill kernel and one optixAccelBuild per overlay of the given instant
    // on stream; no host sync. transforms: [num_overlays * max_instances * 12] device
    // geometry-to-world rows; instance_classes: per global instance id, device — the kernel
    // maintains the overlay range of instant 0 (null skips the classes: the previous instant
    // never feeds segmentation). shutter_out (optional device float): the fill kernel stores
    // shutter_t there on-stream, so per-pass shutter times reach the accumulate ray gens without
    // per-launch param uploads (which would race: OWL stages launch params through a single
    // pinned host buffer)
    void overlay_accel_build(OverlayAccelState* state, const float* transforms, unsigned int* instance_classes, float shutter_t, float* shutter_out, cudaStream_t stream, unsigned int instant);
    // divides the linear accumulators by num_samples and writes the packed frame buffer (+
    // observation) / depth buffer with the exact quantization of the single-launch path;
    // null accumulator pointers skip the corresponding output
    void overlay_accel_resolve(const float* rgb_accumulation, unsigned int* frame_buffer, float* observation, int srgb_output, unsigned int num_pixels, unsigned int num_samples, cudaStream_t stream);
    // the device half of the host publish: for each of num_slabs slabs slab_stride rows apart,
    // writes rows[(slab * count + row) * 12] into target[(slab * slab_stride + indices[row]) * 12]
    void overlay_accel_scatter_rows(const unsigned int* indices, unsigned int count, const float* rows, float* target, unsigned int num_slabs, unsigned int slab_stride, cudaStream_t stream);

    // flow delta table [2][slots][12] from the current and previous transforms slabs and the
    // slot structure table: slab 0 carries a current-instant point to the previous instant
    // (world_previous ∘ world_current⁻¹), slab 1 the inverse; identity for inactive slots — the
    // on-device counterpart of detail::compose_flow_deltas
    void overlay_accel_compose_flow_deltas(const OverlaySlotStructure* structure, const float* transforms, const float* transforms_previous, float* deltas, unsigned int num_overlays, unsigned int max_instances, cudaStream_t stream);
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
