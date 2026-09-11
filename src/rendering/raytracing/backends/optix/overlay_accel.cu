#include "../../../../../include/rl_tools/rendering/raytracing/backends/optix/overlay_accel.h"
#define RL_TOOLS_FUNCTION_PLACEMENT __device__ __host__
#include "../../../../../include/rl_tools/rendering/raytracing/transforms_generic.h"

#include <optix.h>
#include <optix_stubs.h>
// owl keeps its own function table private to libowl.so, so this library carries one of its own
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::backends::optix{
    namespace overlay_accel_detail{
        inline void check_cuda(cudaError_t result, const char* what){
            if(result != cudaSuccess){
                std::fprintf(stderr, "overlay_accel: %s failed: %s\n", what, cudaGetErrorString(result));
                std::abort();
            }
        }
        inline void check_optix(OptixResult result, const char* what){
            if(result != OPTIX_SUCCESS){
                std::fprintf(stderr, "overlay_accel: %s failed: %d\n", what, (int)result);
                std::abort();
            }
        }
        constexpr size_t align_up(size_t value, size_t alignment){
            return (value + alignment - 1) & ~(alignment - 1);
        }

        using rl_tools::rendering::raytracing::detail::compose_transforms;

        __global__ void fill_instances(const OverlaySlotStructure* structure, const float* transforms, const OverlayObjectEntry* objects, OptixInstance* instances, unsigned int* instance_classes, unsigned long long filler_traversable, unsigned int num_overlays, unsigned int max_instances, unsigned int instance_id_base, float shutter_t, float* shutter_out){
            const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
            if(index == 0 && shutter_out != nullptr){
                *shutter_out = shutter_t; // stream-ordered publication for the accumulate ray gens
            }
            if(index >= num_overlays * max_instances){
                return;
            }
            const unsigned int overlay = index / max_instances;
            const unsigned int slot = index % max_instances;
            const OverlaySlotStructure row = structure[index];
            OptixInstance instance{};
            instance.instanceId = instance_id_base + index; // global id layout (contract: segmentation_object, operations_cpu_common.h); the previous instant's range follows the current one
            instance.visibilityMask = 255;
            instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            if(row.active != 0){
                const float* overlay_transforms = transforms + (size_t)overlay * max_instances * 12;
                float composed[12];
                compose_transforms(overlay_transforms + (size_t)row.pose_slot * 12, row.part_local, composed);
                if(slot == row.pose_slot){
                    for(int element = 0; element < 12; element++){
                        instance.transform[element] = composed[element];
                    }
                }
                else{
                    compose_transforms(composed, overlay_transforms + (size_t)slot * 12, instance.transform);
                }
                const OverlayObjectEntry object = objects[row.object];
                instance.sbtOffset = object.sbt_offset;
                instance.traversableHandle = object.traversable;
                if(instance_classes != nullptr){
                    instance_classes[instance_id_base + index] = object.segmentation_class;
                }
            }
            else{
                instance.transform[0] = 1.0f; instance.transform[5] = 1.0f; instance.transform[10] = 1.0f;
                instance.sbtOffset = 0; // degenerate filler triangle never hits, so any valid record works
                instance.traversableHandle = filler_traversable;
                if(instance_classes != nullptr){
                    instance_classes[instance_id_base + index] = 0;
                }
            }
            instances[index] = instance;
        }
    }

    struct OverlayAccelState{
        OptixDeviceContext context;
        unsigned int num_overlays;
        unsigned int max_instances;
        unsigned int num_scene_instances;
        unsigned int num_instants;
        unsigned long long filler_traversable;
        OverlaySlotStructure* structure = nullptr;
        OverlayObjectEntry* objects = nullptr;
        OptixInstance* instances = nullptr; // [num_instants][num_overlays * max_instances]
        CUdeviceptr tlas_slab = 0;           // [num_instants][num_overlays] fixed output buffers
        size_t tlas_stride = 0;
        CUdeviceptr scratch_slab = 0;        // per overlay, shared by the instants (builds are stream-ordered)
        size_t scratch_stride = 0;
        std::vector<unsigned long long> traversables; // [num_instants][num_overlays]
    };

    namespace overlay_accel_detail{
        inline size_t num_slots(const OverlayAccelState& state){
            return (size_t)state.num_overlays * state.max_instances;
        }
        inline OptixBuildInput make_build_input(const OverlayAccelState& state, unsigned int instant, unsigned int overlay){
            OptixBuildInput build_input{};
            build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            build_input.instanceArray.instances = (CUdeviceptr)(state.instances + (size_t)instant * num_slots(state) + (size_t)overlay * state.max_instances);
            build_input.instanceArray.numInstances = state.max_instances;
            return build_input;
        }
        inline OptixAccelBuildOptions build_options(){
            OptixAccelBuildOptions options{};
            options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            options.operation = OPTIX_BUILD_OPERATION_BUILD;
            return options;
        }
        inline void enqueue_builds(OverlayAccelState& state, cudaStream_t stream, unsigned int instant){
            const OptixAccelBuildOptions options = build_options();
            for(unsigned int overlay = 0; overlay < state.num_overlays; overlay++){
                const OptixBuildInput build_input = make_build_input(state, instant, overlay);
                const size_t tlas_index = (size_t)instant * state.num_overlays + overlay;
                OptixTraversableHandle handle = 0;
                check_optix(optixAccelBuild(state.context, stream, &options, &build_input, 1,
                                            state.scratch_slab + (CUdeviceptr)overlay * state.scratch_stride, state.scratch_stride,
                                            state.tlas_slab + (CUdeviceptr)tlas_index * state.tlas_stride, state.tlas_stride,
                                            &handle, nullptr, 0), "optixAccelBuild");
                state.traversables[tlas_index] = handle; // stable: fixed output buffer per overlay and instant
            }
        }
        inline void enqueue_fill(OverlayAccelState& state, const float* transforms, unsigned int* instance_classes, float shutter_t, float* shutter_out, cudaStream_t stream, unsigned int instant){
            const size_t slots = num_slots(state);
            const unsigned int block_size = 128;
            const unsigned int grid_size = (unsigned int)((slots + block_size - 1) / block_size);
            fill_instances<<<grid_size, block_size, 0, stream>>>(state.structure, transforms, state.objects, state.instances + (size_t)instant * slots, instance_classes, state.filler_traversable, state.num_overlays, state.max_instances, state.num_scene_instances + (unsigned int)(instant * slots), shutter_t, shutter_out);
            check_cuda(cudaGetLastError(), "fill_instances launch");
        }
    }

    OverlayAccelState* overlay_accel_create(void* optix_device_context, unsigned int num_overlays, unsigned int max_instances, unsigned int num_scene_instances, unsigned long long filler_traversable, unsigned int num_instants){
        namespace detail = overlay_accel_detail;
        static const OptixResult optix_init_result = optixInit();
        detail::check_optix(optix_init_result, "optixInit");
        auto* state = new OverlayAccelState{};
        state->context = (OptixDeviceContext)optix_device_context;
        state->num_overlays = num_overlays;
        state->max_instances = max_instances;
        state->num_scene_instances = num_scene_instances;
        state->num_instants = num_instants;
        state->filler_traversable = filler_traversable;
        const size_t num_slots = (size_t)num_overlays * max_instances;
        detail::check_cuda(cudaMalloc(&state->structure, num_slots * sizeof(OverlaySlotStructure)), "structure cudaMalloc");
        detail::check_cuda(cudaMemset(state->structure, 0, num_slots * sizeof(OverlaySlotStructure)), "structure cudaMemset");
        detail::check_cuda(cudaMalloc(&state->instances, (size_t)num_instants * num_slots * sizeof(OptixInstance)), "instances cudaMalloc");

        const OptixAccelBuildOptions options = detail::build_options();
        const OptixBuildInput size_input = detail::make_build_input(*state, 0, 0);
        OptixAccelBufferSizes sizes{};
        detail::check_optix(optixAccelComputeMemoryUsage(state->context, &options, &size_input, 1, &sizes), "optixAccelComputeMemoryUsage");
        state->tlas_stride = detail::align_up(sizes.outputSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
        state->scratch_stride = detail::align_up(sizes.tempSizeInBytes, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
        detail::check_cuda(cudaMalloc((void**)&state->tlas_slab, (size_t)num_instants * num_overlays * state->tlas_stride), "tlas cudaMalloc");
        detail::check_cuda(cudaMalloc((void**)&state->scratch_slab, (size_t)num_overlays * state->scratch_stride), "scratch cudaMalloc");
        state->traversables.assign((size_t)num_instants * num_overlays, 0);

        // initial build: the zeroed structure emits filler-only instances (transforms/objects/classes
        // unread), so the stable per-overlay handles of every instant exist before the first update
        for(unsigned int instant = 0; instant < num_instants; instant++){
            detail::enqueue_fill(*state, nullptr, nullptr, 0.0f, nullptr, nullptr, instant);
            detail::enqueue_builds(*state, nullptr, instant);
        }
        detail::check_cuda(cudaStreamSynchronize(nullptr), "initial build sync");
        return state;
    }

    void overlay_accel_destroy(OverlayAccelState* state){
        if(state == nullptr){
            return;
        }
        cudaFree(state->structure);
        cudaFree(state->objects);
        cudaFree(state->instances);
        cudaFree((void*)state->tlas_slab);
        cudaFree((void*)state->scratch_slab);
        delete state;
    }

    OverlaySlotStructure* overlay_accel_structure(OverlayAccelState* state){
        return state->structure;
    }

    void overlay_accel_upload_objects(OverlayAccelState* state, const OverlayObjectEntry* objects, unsigned int num_objects){
        namespace detail = overlay_accel_detail;
        cudaFree(state->objects);
        detail::check_cuda(cudaMalloc(&state->objects, (size_t)num_objects * sizeof(OverlayObjectEntry)), "objects cudaMalloc");
        detail::check_cuda(cudaMemcpy(state->objects, objects, (size_t)num_objects * sizeof(OverlayObjectEntry), cudaMemcpyHostToDevice), "objects upload");
    }

    const unsigned long long* overlay_accel_traversables(const OverlayAccelState* state, unsigned int instant){
        return state->traversables.data() + (size_t)instant * state->num_overlays;
    }

    void overlay_accel_build(OverlayAccelState* state, const float* transforms, unsigned int* instance_classes, float shutter_t, float* shutter_out, cudaStream_t stream, unsigned int instant){
        namespace detail = overlay_accel_detail;
        detail::enqueue_fill(*state, transforms, instance_classes, shutter_t, shutter_out, stream, instant);
        detail::enqueue_builds(*state, stream, instant);
    }

    namespace overlay_accel_detail{
        __device__ inline unsigned int resolve_make_8bit(float f){
            const int value = (int)(f * 256.0f);
            return (unsigned int)(value < 0 ? 0 : (value > 255 ? 255 : value));
        }
        __device__ inline float resolve_linear_to_srgb(float x){
            if(x <= 0.0031308f) return 12.92f * x;
            return 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
        }
        __device__ inline float resolve_clamp01(float x){
            return fminf(fmaxf(x, 0.0f), 1.0f);
        }
        __global__ void resolve_accumulators(const float* rgb_accumulation, unsigned int* frame_buffer, float* observation, int srgb_output, const float* depth_accumulation, float* depth_buffer, unsigned int num_pixels, float inv_samples){
            const unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
            if(pixel >= num_pixels){
                return;
            }
            if(rgb_accumulation != nullptr){
                float color[3];
                for(int channel = 0; channel < 3; channel++){
                    color[channel] = rgb_accumulation[(size_t)pixel * 3 + channel] * inv_samples;
                    if(srgb_output != 0){
                        color[channel] = resolve_linear_to_srgb(resolve_clamp01(color[channel]));
                    }
                    else{
                        color[channel] = resolve_clamp01(color[channel]);
                    }
                }
                if(observation != nullptr){
                    observation[(size_t)pixel * 3 + 0] = color[0];
                    observation[(size_t)pixel * 3 + 1] = color[1];
                    observation[(size_t)pixel * 3 + 2] = color[2];
                }
                frame_buffer[pixel] = (resolve_make_8bit(color[0]) << 0) | (resolve_make_8bit(color[1]) << 8) | (resolve_make_8bit(color[2]) << 16) | (0xffu << 24);
            }
            if(depth_accumulation != nullptr){
                depth_buffer[pixel] = depth_accumulation[pixel] * inv_samples;
            }
        }
    }

    void overlay_accel_resolve(const float* rgb_accumulation, unsigned int* frame_buffer, float* observation, int srgb_output, const float* depth_accumulation, float* depth_buffer, unsigned int num_pixels, unsigned int num_samples, cudaStream_t stream){
        namespace detail = overlay_accel_detail;
        const unsigned int block_size = 128;
        const unsigned int grid_size = (num_pixels + block_size - 1) / block_size;
        detail::resolve_accumulators<<<grid_size, block_size, 0, stream>>>(rgb_accumulation, frame_buffer, observation, srgb_output, depth_accumulation, depth_buffer, num_pixels, 1.0f / (float)num_samples);
        detail::check_cuda(cudaGetLastError(), "resolve_accumulators launch");
    }

    namespace overlay_accel_detail{
        __global__ void expand_motion(const float* pairs, float* transforms_motion, float* transforms, unsigned int num_slots, unsigned int num_samples){
            const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
            if(index >= num_slots * num_samples){
                return;
            }
            const unsigned int slot = index / num_samples;
            const unsigned int sample = index % num_samples;
            const float* open = pairs + (size_t)slot * 12;
            const float* close = pairs + ((size_t)num_slots + slot) * 12;
            const float shutter_t = ((float)sample + 0.5f) / (float)num_samples;
            rl_tools::rendering::raytracing::detail::slerp_transform(open, close, shutter_t, transforms_motion + ((size_t)sample * num_slots + slot) * 12);
            if(sample == 0){
                for(int element = 0; element < 12; element++){
                    transforms[(size_t)slot * 12 + element] = close[element];
                }
            }
        }
    }

    void overlay_accel_expand_motion(const float* pairs, float* transforms_motion, float* transforms, unsigned int num_slots, unsigned int num_samples, cudaStream_t stream){
        namespace detail = overlay_accel_detail;
        const unsigned int block_size = 128;
        const unsigned int grid_size = (num_slots * num_samples + block_size - 1) / block_size;
        detail::expand_motion<<<grid_size, block_size, 0, stream>>>(pairs, transforms_motion, transforms, num_slots, num_samples);
        detail::check_cuda(cudaGetLastError(), "expand_motion launch");
    }

    namespace overlay_accel_detail{
        // world composition mirrors fill_instances; the deltas carry a hit point between the
        // instants for the flow ray gens
        __device__ void compose_world(const float* overlay_transforms, const OverlaySlotStructure& row, unsigned int slot, float out[12]){
            float composed[12];
            compose_transforms(overlay_transforms + (size_t)row.pose_slot * 12, row.part_local, composed);
            if(slot == row.pose_slot){
                for(int element = 0; element < 12; element++){
                    out[element] = composed[element];
                }
            }
            else{
                compose_transforms(composed, overlay_transforms + (size_t)slot * 12, out);
            }
        }

        __global__ void compose_flow_deltas(const OverlaySlotStructure* structure, const float* transforms, const float* transforms_previous, float* deltas, unsigned int num_overlays, unsigned int max_instances){
            const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int num_slots = num_overlays * max_instances;
            if(index >= num_slots){
                return;
            }
            const unsigned int overlay = index / max_instances;
            const unsigned int slot = index % max_instances;
            float* backward = deltas + (size_t)index * 12;
            float* forward = deltas + ((size_t)num_slots + index) * 12;
            const OverlaySlotStructure row = structure[index];
            if(row.active == 0){
                for(int element = 0; element < 12; element++){
                    backward[element] = (element % 5 == 0) ? 1.0f : 0.0f; // identity [R|t] rows
                    forward[element] = backward[element];
                }
                return;
            }
            const float* current_row = transforms + (size_t)overlay * max_instances * 12;
            const float* previous_row = transforms_previous + (size_t)overlay * max_instances * 12;
            float world_current[12], world_previous[12], world_current_inverse[12], world_previous_inverse[12];
            compose_world(current_row, row, slot, world_current);
            compose_world(previous_row, row, slot, world_previous);
            rl_tools::rendering::raytracing::detail::invert_transform(world_current, world_current_inverse);
            rl_tools::rendering::raytracing::detail::invert_transform(world_previous, world_previous_inverse);
            compose_transforms(world_previous, world_current_inverse, backward);
            compose_transforms(world_current, world_previous_inverse, forward);
        }
    }

    void overlay_accel_compose_flow_deltas(const OverlaySlotStructure* structure, const float* transforms, const float* transforms_previous, float* deltas, unsigned int num_overlays, unsigned int max_instances, cudaStream_t stream){
        namespace detail = overlay_accel_detail;
        const unsigned int block_size = 128;
        const unsigned int grid_size = (num_overlays * max_instances + block_size - 1) / block_size;
        detail::compose_flow_deltas<<<grid_size, block_size, 0, stream>>>(structure, transforms, transforms_previous, deltas, num_overlays, max_instances);
        detail::check_cuda(cudaGetLastError(), "compose_flow_deltas launch");
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
