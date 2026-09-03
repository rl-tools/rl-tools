#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_OPERATIONS_CUDA_H

#include "operations_cpu.h"
#include "../../operations_cuda.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::hyperdrone::tasks::target_frame::cuda{
        template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void sample_initial_parameters_kernel(DEVICE device, typename World<TASK_SPEC>::NEXT_WORLD::DYNAMICS_ENV dynamics, typename World<TASK_SPEC>::NEXT_WORLD::Parameters defaults, Tensor<PARAMETER_SPEC> parameters, const Tensor<RESET_SPEC> reset_mask, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES && get(device, reset_mask, instance_i)){
                auto& rng_state = get(rng.states, 0, instance_i);
                // qualified: ADL through the base-typed defaults argument would otherwise also
                // consider the base helper and hard-instantiate the base World with TASK_SPEC
                tasks::target_frame::_sample_initial_parameters<DEVICE, TASK_SPEC>(device, dynamics, defaults, get_ref(device, parameters, instance_i), rng_state);
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC>
        __global__ void target_camera_kernel(DEVICE device, Tensor<PARAMETER_SPEC> parameters, rendering::raytracing::Camera<typename TASK_SPEC::T>* cameras, rendering::raytracing::Camera<typename TASK_SPEC::T>* cameras_open, typename TASK_SPEC::T aspect){
            using T = typename TASK_SPEC::T;
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                auto camera = make_target_camera<DEVICE, T>(device, instance_parameters.camera_mount, instance_parameters.fov, aspect, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin, instance_parameters.target_roll, instance_parameters.target_pitch);
                cameras[instance_i] = camera;
                if(cameras_open != cameras){
                    cameras_open[instance_i] = camera;
                }
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename RESET_SPEC>
        __global__ void target_scatter_kernel(DEVICE device, const float* __restrict__ observation, float* __restrict__ target_frames, Tensor<PARAMETER_SPEC> parameters, const Tensor<RESET_SPEC> reset_mask){
            using TI = typename DEVICE::index_t;
            using WORLD = World<TASK_SPEC>;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            using BASE_SPEC = typename NEXT_WORLD::SPEC;
            constexpr TI INSTANCES = WORLD::INSTANCES;
            constexpr TI CAM_PIXELS = BASE_SPEC::CAM_WIDTH * BASE_SPEC::CAM_HEIGHT;
            TI thread_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(thread_i < INSTANCES * CAM_PIXELS){
                TI instance_i = thread_i / CAM_PIXELS;
                if(!get(device, reset_mask, instance_i)){
                    return;
                }
                TI pixel_i = thread_i % CAM_PIXELS;
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                const float scale = (float)(instance_parameters.brightness_scale * instance_parameters.brightness_mismatch);
                for(TI channel_i = 0; channel_i < 3; channel_i++){
                    float value = observation[(instance_i * CAM_PIXELS + pixel_i) * 3 + channel_i] * scale;
                    value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
                    target_frames[instance_i * NEXT_WORLD::FRAME_DIM + pixel_i * NEXT_WORLD::IMAGE_CHANNELS + channel_i] = value;
                }
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename OBSERVATION_SPEC>
        __global__ void observe_kernel(DEVICE device, const float* __restrict__ history, const typename TASK_SPEC::TI* __restrict__ episode_start, const float* __restrict__ target_frames, typename TASK_SPEC::TI latest, Tensor<OBSERVATION_SPEC> observations){
            using T = typename TASK_SPEC::T;
            using TI = typename DEVICE::index_t;
            using WORLD = World<TASK_SPEC>;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            using BASE_SPEC = typename NEXT_WORLD::SPEC;
            constexpr TI INSTANCES = WORLD::INSTANCES;
            constexpr TI CAM_PIXELS = BASE_SPEC::CAM_WIDTH * BASE_SPEC::CAM_HEIGHT;
            constexpr TI IMAGE_CHANNELS = NEXT_WORLD::IMAGE_CHANNELS;
            constexpr TI TOTAL_CHANNELS = WORLD::OBSERVATION_CHANNELS;
            constexpr TI STACK_N = TASK_SPEC::IMAGE_STACK_N;
            constexpr TI STACK_STRIDE = TASK_SPEC::IMAGE_STACK_STRIDE;
            TI thread_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(thread_i < INSTANCES * WORLD::OBSERVATION_DIM){
                TI instance_i = thread_i / WORLD::OBSERVATION_DIM;
                TI offset = thread_i % WORLD::OBSERVATION_DIM;
                TI pixel_i = offset / TOTAL_CHANNELS;
                TI channel = offset % TOTAL_CHANNELS;
                float value;
                if(channel < STACK_N * IMAGE_CHANNELS){
                    TI frame_i = channel / IMAGE_CHANNELS;
                    TI channel_i = channel % IMAGE_CHANNELS;
                    TI back = frame_i * STACK_STRIDE;
                    TI start = episode_start[instance_i];
                    TI desired = latest >= back ? latest - back : start;
                    if(desired < start){
                        desired = start;
                    }
                    TI history_slot = desired % BASE_SPEC::HISTORY_LENGTH;
                    value = history[(history_slot * INSTANCES * NEXT_WORLD::N_VIEWS + instance_i) * NEXT_WORLD::FRAME_DIM + pixel_i * IMAGE_CHANNELS + channel_i];
                }
                else if(channel < (STACK_N + 1) * IMAGE_CHANNELS){
                    TI channel_i = channel - STACK_N * IMAGE_CHANNELS;
                    value = target_frames[instance_i * NEXT_WORLD::FRAME_DIM + pixel_i * IMAGE_CHANNELS + channel_i];
                }
                else{
                    value = 0.0f;
                }
                set(device, observations, (T)value, instance_i, offset);
            }
        }
    }

    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI INSTANCES = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>::INSTANCES;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::tasks::target_frame::cuda::sample_initial_parameters_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, world.parameters, parameters, reset_mask, rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename TASK_SPEC::T;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        using BASE_SPEC = typename NEXT_WORLD::SPEC;
        constexpr TI INSTANCES = NEXT_WORLD::INSTANCES;
        constexpr TI CAM_PIXELS = BASE_SPEC::CAM_WIDTH * BASE_SPEC::CAM_HEIGHT;
        // the cache-refresh decision needs the reset mask on the host: a pinned mirror and a
        // short join on the producing stream (the payoff is skipping a whole render pass on
        // non-reset steps)
        if(world.cuda_reset_staging == nullptr){
            cudaMallocHost(&world.cuda_reset_staging, INSTANCES * sizeof(bool));
        }
        bool* reset_staging = (bool*)world.cuda_reset_staging;
        cudaMemcpyAsync(reset_staging, data(reset_mask), INSTANCES * sizeof(bool), cudaMemcpyDeviceToHost, device.stream);
        cudaStreamSynchronize(device.stream);
        bool any_reset = false;
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            any_reset = any_reset || reset_staging[instance_i];
        }
        if(any_reset){
            cudaStream_t render_stream = stream(device, world.renderer);
            rl::environments::hyperdrone::cuda::stream_barrier(world, device.stream, render_stream);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            const T aspect = static_cast<T>(BASE_SPEC::CAM_WIDTH) / static_cast<T>(BASE_SPEC::CAM_HEIGHT);
            {
                constexpr TI BLOCKSIZE = 32;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
                auto* cameras_data = data(cameras(device, world.renderer));
                auto* cameras_open_data = cameras_data;
                if constexpr(BASE_SPEC::ENABLE_MOTION_BLUR){
                    cameras_open_data = data(cameras_open(device, world.renderer));
                }
                rl::environments::hyperdrone::tasks::target_frame::cuda::target_camera_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, render_stream>>>(tag_device, parameters, cameras_data, cameras_open_data, aspect);
            }
            render_launch(device, world.renderer);
            {
                constexpr TI BLOCKSIZE = 256;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES * CAM_PIXELS, BLOCKSIZE);
                rl::environments::hyperdrone::tasks::target_frame::cuda::target_scatter_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, render_stream>>>(tag_device, data(world.renderer.observation), data(world.target_frames), parameters, reset_mask);
            }
            check_status(device);
        }
        render(device, static_cast<NEXT_WORLD&>(world), parameters, states, reset_mask);
    }
    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>::Observation, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        static_assert(get<0>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::INSTANCES);
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::OBSERVATION_DIM);
        if(world.render_pending){
            render(device, world, parameters, states, rl::environments::hyperdrone::render_reset(device, world));
        }
        utils::assert_exit(device, world.history_step > 0, "hyperdrone::tasks::target_frame::observe: no frame available");
        cudaStream_t render_stream = stream(device, world.renderer);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        constexpr TI BLOCKSIZE = 256;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(WORLD::INSTANCES * WORLD::OBSERVATION_DIM, BLOCKSIZE);
        rl::environments::hyperdrone::tasks::target_frame::cuda::observe_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, render_stream>>>(tag_device, data(world.history), data(world.episode_start), data(world.target_frames), world.history_step - 1, observations);
        rl::environments::hyperdrone::cuda::stream_barrier(world, render_stream, device.stream);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
