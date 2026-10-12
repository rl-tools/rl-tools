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
        template <typename DEVICE, typename TASK_SPEC, typename SCENE_SPEC, typename DRONE_SPEC, typename SETUPS_SPEC, typename RNG>
        __global__ void sample_setups_kernel(DEVICE device, Tensor<SCENE_SPEC> active_annotations, typename World<TASK_SPEC>::NEXT_WORLD::Parameters defaults, const Tensor<DRONE_SPEC> drones, typename TASK_SPEC::TI num_drones, Tensor<SETUPS_SPEC> setups, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI setup_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(setup_i < INSTANCES){
                auto& rng_state = get(rng.states, 0, setup_i);
                tasks::target_frame::_sample_setup<DEVICE, TASK_SPEC>(device, get_ref(device, active_annotations, 0), defaults, drones, num_drones, get_ref(device, setups, setup_i), rng_state);
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename SETUPS_SPEC>
        __global__ void bank_camera_kernel(DEVICE device, Tensor<SETUPS_SPEC> setups, rendering::raytracing::Camera<typename TASK_SPEC::T>* cameras, rendering::raytracing::Camera<typename TASK_SPEC::T>* cameras_open, typename TASK_SPEC::T aspect){
            using T = typename TASK_SPEC::T;
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            TI setup_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(setup_i < INSTANCES){
                const auto& setup = get_ref(device, setups, setup_i);
                auto camera = make_target_camera<DEVICE, T>(device, setup.agents[0].cameras[0], aspect, setup.scene_translation, setup.scene_yaw_cos, setup.scene_yaw_sin, setup.target_roll, setup.target_pitch);
                cameras[setup_i] = camera;
                if(cameras_open != cameras){
                    cameras_open[setup_i] = camera;
                }
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename SETUPS_SPEC, typename BODIES_SPEC, typename STRUCTURE>
        __global__ void bank_drone_rows_kernel(DEVICE device, Tensor<SETUPS_SPEC> setups, const Tensor<BODIES_SPEC> bodies, typename TASK_SPEC::TI slot_offset, typename TASK_SPEC::TI parts, float* transforms, float* transforms_motion, hyperdrone::cuda::DroneMeshes<typename TASK_SPEC::NEXT_WORLD::SPEC, STRUCTURE> meshes){
            using T = typename TASK_SPEC::T;
            using TI = typename DEVICE::index_t;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            TI setup_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(setup_i < INSTANCES){
                const auto& setup = get_ref(device, setups, setup_i);
                T orientation[4], position[3];
                make_target_body_pose(device, setup.agents[0].cameras[0], setup.target_roll, setup.target_pitch, orientation, position);
                float base[12];
                rig::make_body_transform(device, orientation, position, setup.scene_translation, setup.scene_yaw_cos, setup.scene_yaw_sin, base);
                const float coordinates[NEXT_WORLD::DYNAMICS_ENV::ACTION_DIM] = {};
                const TI row = setup_i * NEXT_WORLD::RENDERER_CONFIG::MAX_OVERLAY_INSTANCES + slot_offset;
                const auto& body = get_ref(device, bodies, setup.agents[0].drone);
                const float placement_scale = rotor_scale(device, body.rig, setup.agents[0].dynamics.dynamics.rotor_positions, (TI)NEXT_WORLD::DYNAMICS_ENV::ACTION_DIM);
                hyperdrone::cuda::drone_entity_rows<DEVICE, typename NEXT_WORLD::SPEC>(device, body, base, base, coordinates, coordinates, transforms, transforms_motion, meshes, parts, row, placement_scale);
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename SETUPS_SPEC>
        __global__ void targets_scatter_kernel(DEVICE device, const float* __restrict__ observation, Tensor<SETUPS_SPEC> setups, const bool* __restrict__ mask, float* __restrict__ frames){
            using TI = typename DEVICE::index_t;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            using BASE_SPEC = typename NEXT_WORLD::SPEC;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            constexpr TI CAM_PIXELS = BASE_SPEC::CAM_WIDTH * BASE_SPEC::CAM_HEIGHT;
            TI thread_i = threadIdx.x + blockIdx.x * blockDim.x;
            const TI setup_i = thread_i / CAM_PIXELS;
            if(thread_i < INSTANCES * CAM_PIXELS && (mask == nullptr || mask[setup_i])){
                const TI pixel_i = thread_i % CAM_PIXELS;
                const auto& setup = get_ref(device, setups, setup_i);
                const float scale = (float)(setup.brightness_scale * setup.brightness_mismatch);
                for(TI channel_i = 0; channel_i < 3; channel_i++){
                    frames[setup_i * NEXT_WORLD::FRAME_DIM + pixel_i * NEXT_WORLD::IMAGE_CHANNELS + channel_i] = tasks::target_frame::_target_pixel(observation[(setup_i * CAM_PIXELS + pixel_i) * 3 + channel_i], scale);
                }
            }
        }
        // one thread hands the resetting instances the next setups in order: deterministic and
        // free of atomics
        template <typename DEVICE, typename TASK_SPEC, typename INDEX_SPEC, typename CURSOR_SPEC, typename RESET_SPEC>
        __global__ void assign_setups_kernel(DEVICE device, Tensor<INDEX_SPEC> setup_index, Tensor<CURSOR_SPEC> bank_cursor, const Tensor<RESET_SPEC> reset_mask){
            using TI = typename TASK_SPEC::TI;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            TI cursor = get(device, bank_cursor, (TI)0);
            for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
                if(get(device, reset_mask, instance_i)){
                    set(device, setup_index, cursor % INSTANCES, instance_i);
                    cursor++;
                }
            }
            set(device, bank_cursor, cursor, (TI)0);
        }
        template <typename DEVICE, typename TASK_SPEC, typename DRONE_SPEC, typename SETUPS_SPEC, typename PARAMETER_SPEC, typename INDEX_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void sample_initial_parameters_kernel(DEVICE device, typename World<TASK_SPEC>::NEXT_WORLD::DYNAMICS_ENV dynamics, const Tensor<DRONE_SPEC> drones, Tensor<SETUPS_SPEC> setups, Tensor<PARAMETER_SPEC> parameters, Tensor<INDEX_SPEC> setup_index, const Tensor<RESET_SPEC> reset_mask, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES && get(device, reset_mask, instance_i)){
                auto& rng_state = get(rng.states, 0, instance_i);
                tasks::target_frame::_adopt_setup<DEVICE, TASK_SPEC>(device, dynamics, drones, get_ref(device, setups, get(device, setup_index, instance_i)), get_ref(device, parameters, instance_i), rng_state);
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename INDEX_SPEC, typename RESET_SPEC>
        __global__ void copy_target_frames_kernel(DEVICE device, const float* __restrict__ bank_frames, Tensor<INDEX_SPEC> setup_index, const Tensor<RESET_SPEC> reset_mask, float* __restrict__ target_frames){
            using TI = typename DEVICE::index_t;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            TI thread_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(thread_i < INSTANCES * NEXT_WORLD::FRAME_DIM){
                const TI instance_i = thread_i / NEXT_WORLD::FRAME_DIM;
                const TI element_i = thread_i % NEXT_WORLD::FRAME_DIM;
                if(get(device, reset_mask, instance_i)){
                    target_frames[thread_i] = bank_frames[get(device, setup_index, instance_i) * NEXT_WORLD::FRAME_DIM + element_i];
                }
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void sample_initial_state_kernel(DEVICE device, typename World<TASK_SPEC>::NEXT_WORLD::DYNAMICS_ENV dynamics, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<RESET_SPEC> reset_mask, RNG rng){
            using TI = typename DEVICE::index_t;
            using BASE_SPEC = typename TASK_SPEC::NEXT_WORLD::SPEC;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES && get(device, reset_mask, instance_i)){
                auto& rng_state = get(rng.states, 0, instance_i);
                hyperdrone::_sample_dynamics_state<DEVICE, BASE_SPEC>(device, dynamics, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng_state);
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

    template <typename DEV_SPEC, typename TASK_SPEC, typename RNG>
    void sample_target_setups(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(WORLD::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::tasks::target_frame::cuda::sample_setups_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.active_annotations, world.parameters, world.drone_parameters, world.num_drones, world.setups, world_rng);
        check_status(device);
    }
    namespace rl::environments::hyperdrone::tasks::target_frame {
        template <typename DEV_SPEC, typename TASK_SPEC, typename SETUPS_SPEC, typename FRAMES_SPEC>
        void _render_targets(devices::CUDA<DEV_SPEC>& device, World<TASK_SPEC>& world, Tensor<SETUPS_SPEC>& setups, Tensor<FRAMES_SPEC>& frames, const bool* mask){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using T = typename TASK_SPEC::T;
            using TI = typename DEVICE::index_t;
            using WORLD = World<TASK_SPEC>;
            using BASE_SPEC = typename TASK_SPEC::NEXT_WORLD::SPEC;
            static_assert(utils::typing::is_same_v<typename SETUPS_SPEC::T, typename WORLD::Parameters>);
            constexpr TI CAM_PIXELS = BASE_SPEC::CAM_WIDTH * BASE_SPEC::CAM_HEIGHT;
            const T aspect = static_cast<T>(BASE_SPEC::CAM_WIDTH) / static_cast<T>(BASE_SPEC::CAM_HEIGHT);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            auto& renderer = get_renderer(device, world);
            if constexpr (BASE_SPEC::SELF_VISIBLE){
                // the setup's drone at rest at the target pose, overwritten by the next frame's rows
                float* transforms_motion_rows = nullptr;
                if constexpr (TASK_SPEC::NEXT_WORLD::RENDERER_SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                    transforms_motion_rows = data(transforms_motion(device, renderer));
                }
                constexpr TI BLOCKSIZE = 32;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(WORLD::INSTANCES, BLOCKSIZE);
                cuda::bank_drone_rows_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, setups, world.drone_bodies, world.drone_slot_offset, world.drone_parts, data(transforms(device, renderer)), transforms_motion_rows, hyperdrone::cuda::drone_meshes(device, world, renderer));
                update_launch(device, renderer);
            }
            {
                constexpr TI BLOCKSIZE = 32;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(WORLD::INSTANCES, BLOCKSIZE);
                auto* cameras_data = data(cameras(device, renderer));
                auto* cameras_open_data = cameras_data;
                if constexpr(BASE_SPEC::ENABLE_MOTION_BLUR){
                    cameras_open_data = data(cameras_open(device, renderer));
                }
                cuda::bank_camera_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, setups, cameras_data, cameras_open_data, aspect);
            }
            if constexpr (TASK_SPEC::NEXT_WORLD::RENDERER_SPEC::HAS_FLOW){
                if(world.flow_scene_version != world.scene_version){
                    advance_flow_launch(device, renderer);
                    if constexpr (TASK_SPEC::TASK_SPEC::NEXT_WORLD::RENDERER_SPEC::HAS_FLOW_OVERLAYS){
                        update_launch(device, renderer);
                    }
                }
            }
            render_launch(device, renderer);
            {
                constexpr TI BLOCKSIZE = 256;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(WORLD::INSTANCES * CAM_PIXELS, BLOCKSIZE);
                cuda::targets_scatter_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, data(output(device, renderer, rendering::raytracing::outputs::RGB<rendering::raytracing::outputs::format::Float32>{})), setups, mask, data(frames));
            }
            check_status(device);
        }
    }
    template <typename DEV_SPEC, typename TASK_SPEC>
    void render_target_bank(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world){
        rl::environments::hyperdrone::tasks::target_frame::_render_targets(device, world, world.setups, world.bank_frames, nullptr);
        set_all(device, world.bank_cursor, (typename TASK_SPEC::TI)0);
        world.bank_step = world.history_step;
        world.bank_scene_version = world.scene_version;
    }
    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename RESET_SPEC>
    void render_target_frames(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask){
        rl::environments::hyperdrone::tasks::target_frame::_render_targets(device, world, parameters, world.target_frames, data(reset_mask));
    }
    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        constexpr TI INSTANCES = WORLD::INSTANCES;
        if(rl::environments::hyperdrone::tasks::target_frame::_bank_due(world)){
            sample_target_setups(device, world, rng);
            render_target_bank(device, world);
        }
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::tasks::target_frame::cuda::assign_setups_kernel<decltype(tag_device), TASK_SPEC><<<dim3(1), dim3(1), 0, device.stream>>>(tag_device, world.setup_index, world.bank_cursor, reset_mask);
        {
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
            rl::environments::hyperdrone::tasks::target_frame::cuda::sample_initial_parameters_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, world.drone_parameters, world.setups, parameters, world.setup_index, reset_mask, world_rng);
        }
        {
            constexpr TI BLOCKSIZE = 256;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES * NEXT_WORLD::FRAME_DIM, BLOCKSIZE);
            rl::environments::hyperdrone::tasks::target_frame::cuda::copy_target_frames_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, data(world.bank_frames), world.setup_index, reset_mask, data(world.target_frames));
        }
        check_status(device);
    }
    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI INSTANCES = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>::INSTANCES;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::tasks::target_frame::cuda::sample_initial_state_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, parameters, states, reset_mask, world_rng);
        check_status(device);
        request_render(device, world, reset_mask);
    }
    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename TASK_SPEC::NEXT_WORLD::Observation observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        if(world.render_pending){
            render(device, world, parameters, states, world.render_reset);
        }
        observe(device, static_cast<NEXT_WORLD&>(world), parameters, states, observation_type, observations, rng);
    }
    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>::Observation, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
        static_assert(get<0>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::INSTANCES);
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::OBSERVATION_DIM);
        if(world.render_pending){
            render(device, world, parameters, states, world.render_reset);
        }
        utils::assert_exit(device, world.history_step > 0, "hyperdrone::tasks::target_frame::observe: no frame available");
        devices::cuda::TAG<DEVICE, true> tag_device{};
        constexpr TI BLOCKSIZE = 256;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(WORLD::INSTANCES * WORLD::OBSERVATION_DIM, BLOCKSIZE);
        rl::environments::hyperdrone::tasks::target_frame::cuda::observe_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, data(world.history), data(world.episode_start), data(world.target_frames), world.history_step - 1, observations);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
