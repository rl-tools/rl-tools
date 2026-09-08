#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_OPERATIONS_CUDA_H

#include "operations_cpu.h"

#include <cuda_runtime.h>
#include <stdexcept>

// device-resident per-step path: pose/sample/step kernels are enqueued on the caller's stream or
// the renderer's stream, joined by events — no host synchronization inside the verbs

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::hyperdrone::cuda{
        template <typename DEVICE, typename WORLD, typename RNG>
        RNG instance_rng(DEVICE&, const WORLD& world, const RNG& rng){
            static_assert(RNG::NUM_RNGS >= WORLD::INSTANCES, "Please increase the number of CUDA RNGs");
            if(world.rng_offset > RNG::NUM_RNGS - WORLD::INSTANCES){
                throw std::out_of_range("hyperdrone: World RNG range exceeds the CUDA RNG allocation");
            }
            auto result = rng;
            result.states._data += world.rng_offset;
            return result;
        }

        template <typename SPEC>
        void stream_barrier(World<SPEC>& world, cudaStream_t from, cudaStream_t to){
            if(from == to){
                return;
            }
            if(world.cuda_sync_event == nullptr){
                cudaEvent_t event;
                cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
                world.cuda_sync_event = (void*)event;
            }
            cudaEvent_t event = (cudaEvent_t)world.cuda_sync_event;
            cudaEventRecord(event, from);
            cudaStreamWaitEvent(to, event, 0);
        }

        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void sample_initial_parameters_kernel(DEVICE device, typename World<SPEC>::DYNAMICS_ENV dynamics, Parameters<SPEC> defaults, Tensor<PARAMETER_SPEC> parameters, const Tensor<RESET_SPEC> reset_mask, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<SPEC>::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES && get(device, reset_mask, instance_i)){
                auto& rng_state = get(rng.states, 0, instance_i);
                _sample_initial_parameters<DEVICE, SPEC>(device, dynamics, defaults, get_ref(device, parameters, instance_i), rng_state);
            }
        }
        template <typename DEVICE, typename SPEC, typename SCENE_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void sample_initial_state_kernel(DEVICE device, typename World<SPEC>::DYNAMICS_ENV dynamics, Tensor<SCENE_SPEC> active_annotations, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<RESET_SPEC> reset_mask, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<SPEC>::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES && get(device, reset_mask, instance_i)){
                auto& rng_state = get(rng.states, 0, instance_i);
                _sample_initial_state<DEVICE, SPEC>(device, dynamics, get_ref(device, active_annotations, 0), get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng_state);
            }
        }
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
        __global__ void pose_kernel(DEVICE device, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<RESET_SPEC> reset_mask, rendering::raytracing::Camera<typename SPEC::T>* cameras, rendering::raytracing::Camera<typename SPEC::T>* cameras_open, rendering::raytracing::Camera<typename SPEC::T>* prev_cameras, typename SPEC::TI history_step, typename SPEC::T aspect){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using WORLD = World<SPEC>;
            constexpr TI NUM_CAMERAS = WORLD::INSTANCES * WORLD::N_VIEWS;
            TI camera_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(camera_i < NUM_CAMERAS){
                const TI instance_i = camera_i / WORLD::N_VIEWS;
                const TI agent_i = camera_i % WORLD::N_VIEWS;
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                const auto& agent_state = _agent_state<SPEC>(get_ref(device, states, instance_i), agent_i);
                auto close = make_camera<DEVICE, T>(device, instance_parameters.camera_mount, instance_parameters.fov, agent_state.orientation, agent_state.position, aspect, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin);
                cameras[camera_i] = close;
                if constexpr(SPEC::ENABLE_MOTION_BLUR){
                    bool reset = history_step == 0 || get(device, reset_mask, instance_i);
                    cameras_open[camera_i] = reset ? close : interpolate_camera(close, prev_cameras[camera_i], instance_parameters.shutter_fraction);
                    prev_cameras[camera_i] = close;
                }
            }
        }
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC>
        __global__ void drone_pose_kernel(DEVICE device, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, float* staging){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using WORLD = World<SPEC>;
            constexpr TI ENTRIES = WORLD::INSTANCES * WORLD::N_AGENTS;
            TI entry_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(entry_i < ENTRIES){
                const TI instance_i = entry_i / WORLD::N_AGENTS;
                const TI agent_i = entry_i % WORLD::N_AGENTS;
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                const auto& agent_state = _agent_state<SPEC>(get_ref(device, states, instance_i), agent_i);
                float* entry = staging + entry_i * 20;
                rig::make_body_transform(device, agent_state.orientation, agent_state.position, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin, entry);
                for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                    entry[12 + rotor_i] = 0;
                    entry[16 + rotor_i] = 0;
                }
                if constexpr (world::HasRotorPhase<typename WORLD::DYNAMICS_ENV::State>::VALUE){
                    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                        float close = (float)agent_state.rotor_phase[rotor_i];
                        float open = close;
                        if constexpr (WORLD::RENDERER_CONFIG::ENABLE_DYNAMIC_MOTION_BLUR){
                            open = close - (float)(agent_state.rpm[rotor_i] * (T)2 * math::PI<T> / (T)60 * instance_parameters.dynamics.integration.dt * instance_parameters.shutter_fraction);
                        }
                        entry[12 + rotor_i] = open;
                        entry[16 + rotor_i] = close;
                    }
                }
            }
        }
        // the per-instance episode-start bookkeeping rides in the scatter grid (one designated
        // thread per instance) to save a launch
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename RESET_SPEC>
        __global__ void scatter_kernel(DEVICE device, const float* __restrict__ observation, float* __restrict__ history_row, Tensor<PARAMETER_SPEC> parameters, typename SPEC::TI* episode_start, const Tensor<RESET_SPEC> reset_mask, typename SPEC::TI history_step){
            using TI = typename DEVICE::index_t;
            using WORLD = World<SPEC>;
            constexpr TI NUM_CAMERAS = WORLD::INSTANCES * WORLD::N_VIEWS;
            constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
            TI thread_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(thread_i < NUM_CAMERAS * CAM_PIXELS){
                TI camera_i = thread_i / CAM_PIXELS;
                TI instance_i = camera_i / WORLD::N_VIEWS;
                TI pixel_i = thread_i % CAM_PIXELS;
                if(pixel_i == 0 && camera_i % WORLD::N_VIEWS == 0 && get(device, reset_mask, instance_i)){
                    episode_start[instance_i] = history_step;
                }
                const float scale = (float)get_ref(device, parameters, instance_i).brightness_scale;
                for(TI channel_i = 0; channel_i < 3; channel_i++){
                    float value = observation[(camera_i * CAM_PIXELS + pixel_i) * 3 + channel_i] * scale;
                    value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
                    history_row[camera_i * WORLD::FRAME_DIM + pixel_i * WORLD::IMAGE_CHANNELS + channel_i] = value;
                }
            }
        }
        template <typename DEVICE, typename SPEC, typename OBSERVATION_SPEC>
        __global__ void observe_kernel(DEVICE device, const float* __restrict__ history_row, Tensor<OBSERVATION_SPEC> observations){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            using WORLD = World<SPEC>;
            TI thread_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(thread_i < WORLD::INSTANCES * WORLD::OBSERVATION_DIM){
                TI instance_i = thread_i / WORLD::OBSERVATION_DIM;
                TI dim_i = thread_i % WORLD::OBSERVATION_DIM;
                set(device, observations, (T)history_row[thread_i], instance_i, dim_i);
            }
        }
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG>
        __global__ void observe_dynamics_kernel(DEVICE device, typename World<SPEC>::DYNAMICS_ENV dynamics, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC> observations, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<SPEC>::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                auto& rng_state = get(rng.states, 0, instance_i);
                auto observation_slice = view(device, observations, instance_i);
                auto observation_matrix = matrix_view(device, observation_slice);
                if constexpr (utils::typing::is_same_v<OBSERVATION_TYPE, typename World<SPEC>::ObservationPrivileged>){
                    _observe_dynamics<DEVICE, SPEC>(device, dynamics, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), observation_matrix, rng_state);
                } else {
                    static_assert(World<SPEC>::N_AGENTS == 1, "multi-agent Worlds expose only the privileged observation on the dynamics side");
                    observe(device, dynamics, get_ref(device, parameters, instance_i).dynamics, get_ref(device, states, instance_i), observation_type, observation_matrix, rng_state);
                }
            }
        }
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
        __global__ void step_kernel(DEVICE device, typename World<SPEC>::DYNAMICS_ENV dynamics, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<ACTION_SPEC> actions, Tensor<NEXT_STATE_SPEC> next_states, RNG rng){
            using TI = typename DEVICE::index_t;
            using WORLD = World<SPEC>;
            constexpr TI INSTANCES = WORLD::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                auto& rng_state = get(rng.states, 0, instance_i);
                Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, WORLD::ACTION_DIM, false>> action_matrix;
                for(TI action_i = 0; action_i < WORLD::ACTION_DIM; action_i++){
                    set(action_matrix, 0, action_i, get(device, actions, instance_i, action_i));
                }
                _step<DEVICE, SPEC>(device, dynamics, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), action_matrix, get_ref(device, next_states, instance_i), rng_state);
            }
        }
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
        __global__ void reward_kernel(DEVICE device, typename World<SPEC>::DYNAMICS_ENV dynamics, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<ACTION_SPEC> actions, Tensor<NEXT_STATE_SPEC> next_states, Tensor<REWARD_SPEC> rewards, RNG rng){
            using TI = typename DEVICE::index_t;
            using WORLD = World<SPEC>;
            constexpr TI INSTANCES = WORLD::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                auto& rng_state = get(rng.states, 0, instance_i);
                Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, WORLD::ACTION_DIM, false>> action_matrix;
                for(TI action_i = 0; action_i < WORLD::ACTION_DIM; action_i++){
                    set(action_matrix, 0, action_i, get(device, actions, instance_i, action_i));
                }
                set(device, rewards, _reward<DEVICE, SPEC>(device, dynamics, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), action_matrix, get_ref(device, next_states, instance_i), rng_state), instance_i);
            }
        }
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
        __global__ void terminated_kernel(DEVICE device, typename World<SPEC>::DYNAMICS_ENV dynamics, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, Tensor<TERMINATED_SPEC> terminated_flags, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<SPEC>::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                auto& rng_state = get(rng.states, 0, instance_i);
                set(device, terminated_flags, _terminated<DEVICE, SPEC>(device, dynamics, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng_state), instance_i);
            }
        }
    }

    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI INSTANCES = rl::environments::hyperdrone::World<SPEC>::INSTANCES;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::cuda::sample_initial_parameters_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, world.parameters, parameters, reset_mask, world_rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI INSTANCES = rl::environments::hyperdrone::World<SPEC>::INSTANCES;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::cuda::sample_initial_state_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, world.active_annotations, parameters, states, reset_mask, world_rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        static_assert(SPEC::OUTPUT_RGB, "hyperdrone::World::render requires the RGB observation output");
        constexpr TI INSTANCES = WORLD::INSTANCES;
        constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
        if constexpr (SPEC::SELF_VISIBLE){
            devices::cuda::TAG<DEVICE, true> pose_tag_device{};
            {
                constexpr TI BLOCKSIZE = 32;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES * WORLD::N_AGENTS, BLOCKSIZE);
                rl::environments::hyperdrone::cuda::drone_pose_kernel<decltype(pose_tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(pose_tag_device, parameters, states, data(world.drone_pose_staging));
            }
            check_status(device);
            if(world.cuda_drone_pose_staging == nullptr){
                cudaMallocHost(&world.cuda_drone_pose_staging, INSTANCES * WORLD::N_AGENTS * 20 * sizeof(float));
            }
            float* staging_host = (float*)world.cuda_drone_pose_staging;
            cudaMemcpyAsync(staging_host, data(world.drone_pose_staging), INSTANCES * WORLD::N_AGENTS * 20 * sizeof(float), cudaMemcpyDeviceToHost, device.stream);
            cudaStreamSynchronize(device.stream);
            auto& slot = world.slots[world.active_slot];
            const TI kinds = (TI)world.entity_kinds.size();
            for(TI entry_i = 0; entry_i < INSTANCES * WORLD::N_AGENTS; entry_i++){
                const TI instance_i = entry_i / WORLD::N_AGENTS;
                const TI agent_i = entry_i % WORLD::N_AGENTS;
                const float* entry = staging_host + entry_i * 20;
                const auto& placement = slot.entity_placements[instance_i * kinds + world.drone_entity_kind_index + agent_i];
                if constexpr (WORLD::RENDERER_CONFIG::ENABLE_DYNAMIC_MOTION_BLUR){
                    T phase_open[4], phase_close[4];
                    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                        phase_open[rotor_i] = (T)entry[12 + rotor_i];
                        phase_close[rotor_i] = (T)entry[16 + rotor_i];
                    }
                    set_transform_pair(device, world.renderer, rendering::raytracing::OverlayIndex{instance_i}, placement, world.drone_rig, entry, entry, phase_open, phase_close);
                } else {
                    set_transform(device, world.renderer, rendering::raytracing::OverlayIndex{instance_i}, placement, entry);
                    for(TI prop_i = 0; prop_i < world.drone_rig.num_props; prop_i++){
                        float spin[12];
                        rl::environments::hyperdrone::rig::prop_spin_transform(device, world.drone_rig.prop_pivots[prop_i][0], world.drone_rig.prop_pivots[prop_i][1], world.drone_rig.prop_directions[prop_i] * (T)entry[16 + prop_i], spin);
                        set_transform(device, world.renderer, rendering::raytracing::OverlayIndex{instance_i}, placement, world.drone_rig.prop_parts[prop_i], spin);
                    }
                }
            }
        }
        cudaStream_t render_stream = stream(device, world.renderer);
        rl::environments::hyperdrone::cuda::stream_barrier(world, device.stream, render_stream);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        {
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES * WORLD::N_VIEWS, BLOCKSIZE);
            auto* cameras_data = data(cameras(device, world.renderer));
            auto* cameras_open_data = cameras_data;
            if constexpr(SPEC::ENABLE_MOTION_BLUR){
                cameras_open_data = data(cameras_open(device, world.renderer));
            }
            rl::environments::hyperdrone::cuda::pose_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, render_stream>>>(tag_device, parameters, states, reset_mask, cameras_data, cameras_open_data, data(world.prev_cameras), world.history_step, aspect);
        }
        if constexpr (WORLD::RENDERER_CONFIG::NUM_OVERLAYS > 0) {
            // wrapper render overloads staged their entity poses pre-forward; publish them
            update(device, world.renderer);
        }
        render_launch(device, world.renderer);
        const TI history_slot = world.history_step % SPEC::HISTORY_LENGTH;
        float* history_row = data(world.history) + history_slot * INSTANCES * WORLD::N_VIEWS * WORLD::FRAME_DIM;
        {
            constexpr TI BLOCKSIZE = 256;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES * WORLD::N_VIEWS * CAM_PIXELS, BLOCKSIZE);
            rl::environments::hyperdrone::cuda::scatter_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, render_stream>>>(tag_device, data(world.renderer.observation), history_row, parameters, data(world.episode_start), reset_mask, world.history_step);
        }
        check_status(device);
        world.history_step++;
    }
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::World<SPEC>::Observation, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        static_assert(get<0>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::INSTANCES);
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::OBSERVATION_DIM);
        utils::assert_exit(device, world.history_step > 0, "hyperdrone::World::observe: render must be called before observe");
        cudaStream_t render_stream = stream(device, world.renderer);
        const TI history_slot = (world.history_step - 1) % SPEC::HISTORY_LENGTH;
        const float* history_row = data(world.history) + history_slot * WORLD::INSTANCES * WORLD::N_VIEWS * WORLD::FRAME_DIM;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        constexpr TI BLOCKSIZE = 256;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(WORLD::INSTANCES * WORLD::OBSERVATION_DIM, BLOCKSIZE);
        rl::environments::hyperdrone::cuda::observe_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, render_stream>>>(tag_device, history_row, observations);
        rl::environments::hyperdrone::cuda::stream_barrier(world, render_stream, device.stream);
        check_status(device);
    }
    // dynamics-side observations (e.g. the privileged observation for asymmetric critics); the
    // visual Image observation has its own overload above
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG, typename utils::typing::enable_if<!utils::typing::is_same_v<OBSERVATION_TYPE, typename rl::environments::hyperdrone::World<SPEC>::Observation>, bool>::type = true>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI INSTANCES = rl::environments::hyperdrone::World<SPEC>::INSTANCES;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::cuda::observe_dynamics_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, parameters, states, observation_type, observations, world_rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI INSTANCES = rl::environments::hyperdrone::World<SPEC>::INSTANCES;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::cuda::step_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, parameters, states, actions, next_states, world_rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI INSTANCES = rl::environments::hyperdrone::World<SPEC>::INSTANCES;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::cuda::reward_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, parameters, states, actions, next_states, rewards, world_rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void terminated(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI INSTANCES = rl::environments::hyperdrone::World<SPEC>::INSTANCES;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::cuda::terminated_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, parameters, states, terminated_flags, world_rng);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
