#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_OPERATIONS_CUDA_H

#include "operations_cpu.h"

#include <cuda_runtime.h>
#include <stdexcept>

// device-resident per-step path, stream-ordered on device.stream with no host synchronization: the
// World's kernels are enqueued on it and the renderer's launch verbs fork from and join back into it;
// a MultiEnvironment renders and observes its members concurrently (fan_out)

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

        // the device counterpart of the rig verbs: one entity's part rows (and motion samples)
        // written straight into the renderer's resident tensors at row (overlay-major slot index)
        template <typename DEVICE, typename RENDERER_SPEC, typename RIG_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void entity_rows(DEVICE& device, const rendering::rig::Model<RIG_SPEC>& rig, const float base_open[12], const float base_close[12], const float* coordinates_open, const float* coordinates_close, float* transforms, float* transforms_motion, typename RIG_SPEC::TI row, float placement_scale = 1){
            using TI = typename RIG_SPEC::TI;
            rendering::rig::Buffer<RIG_SPEC> buffer;
            if constexpr (RENDERER_SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                constexpr TI SAMPLE_STRIDE = (TI)RENDERER_SPEC::NUM_OVERLAYS * (TI)RENDERER_SPEC::MAX_OVERLAY_INSTANCES * 12;
                sample(device, rig, base_open, base_close, coordinates_open, coordinates_close, (TI)RENDERER_SPEC::MOTION_BLUR_SAMPLES, buffer, transforms_motion + row * 12, SAMPLE_STRIDE, transforms + row * 12, placement_scale);
            }
            else{
                evaluate(device, rig, base_close, coordinates_close, buffer, transforms + row * 12, placement_scale);
            }
        }

        template <typename DEVICE, typename SPEC, typename DRONE_SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void sample_initial_parameters_kernel(DEVICE device, typename World<SPEC>::DYNAMICS_ENV dynamics, GroupParameters<SPEC> defaults, const Tensor<DRONE_SPEC> drones, typename SPEC::TI num_drones, Tensor<PARAMETER_SPEC> parameters, const Tensor<RESET_SPEC> reset_mask, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<SPEC>::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES && get(device, reset_mask, instance_i)){
                auto& rng_state = get(rng.states, 0, instance_i);
                _sample_initial_parameters<DEVICE, SPEC>(device, dynamics, defaults, drones, num_drones, get_ref(device, parameters, instance_i), rng_state);
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
                const TI view_i = camera_i % WORLD::N_VIEWS;
                const TI agent_i = view_i / WORLD::N_CAMERAS_PER_AGENT;
                const TI mount_i = view_i % WORLD::N_CAMERAS_PER_AGENT;
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                const auto& agent_state = _agent_state<SPEC>(get_ref(device, states, instance_i), agent_i);
                auto close = make_camera<DEVICE, T>(device, instance_parameters.agents[agent_i].cameras[mount_i], agent_state.orientation, agent_state.position, aspect, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin);
                cameras[camera_i] = close;
                if constexpr(rl::environments::hyperdrone::World<SPEC>::RENDERER_SPEC::ENABLE_MOTION_BLUR){
                    bool reset = history_step <= 1 || get(device, reset_mask, instance_i);
                    cameras_open[camera_i] = reset ? close : interpolate_camera(close, prev_cameras[camera_i], instance_parameters.shutter_fraction);
                }
            }
        }
        template <typename DEVICE, typename RESET_SPEC, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void reset_episode_start(DEVICE& device, const Tensor<RESET_SPEC> reset_mask, TI* episode_start, TI history_step, TI instance){
            if(get(device, reset_mask, instance)){
                episode_start[instance] = history_step - 1;
            }
        }
        template <typename DEVICE, typename SPEC, typename RESET_SPEC>
        __global__ void reset_render_state_kernel(DEVICE device, const Tensor<RESET_SPEC> reset_mask, bool first, const rendering::raytracing::Camera<typename SPEC::T>* current, rendering::raytracing::Camera<typename SPEC::T>* previous, const float* transforms_current, float* transforms_previous, typename SPEC::TI* episode_start, typename SPEC::TI history_step){
            using TI = typename SPEC::TI;
            using WORLD = World<SPEC>;
            const TI instance = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance < WORLD::INSTANCES){
                const bool reset = get(device, reset_mask, instance);
                reset_episode_start(device, reset_mask, episode_start, history_step, instance);
                if constexpr(rl::environments::hyperdrone::World<SPEC>::RENDERER_SPEC::HAS_FLOW){
                    if(first || reset){
                        for(TI view = 0; view < WORLD::N_VIEWS; view++){
                            const TI camera = instance * WORLD::N_VIEWS + view;
                            previous[camera] = current[camera];
                        }
                        if constexpr(WORLD::RENDERER_SPEC::HAS_FLOW_OVERLAYS){
                            constexpr TI ELEMENTS = SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE * 12;
                            for(TI element = 0; element < ELEMENTS; element++){
                                const TI offset = instance * ELEMENTS + element;
                                transforms_previous[offset] = transforms_current[offset];
                            }
                        }
                    }
                }
            }
        }
        // the drone meshes in the active renderer, by value: the object ids of every mesh's parts
        // (the pool's object base differs per hot slot) and the device structure rows their
        // object column goes to, null with a single mesh (the init mesh stays)
        template <typename SPEC, typename STRUCTURE>
        struct DroneMeshes {
            using TI = typename SPEC::TI;
            static constexpr TI MAX_PARTS = decltype(DroneBody<SPEC>::rig)::SPEC::MAX_PARTS;
            STRUCTURE* structure;
            unsigned int objects[SPEC::MAX_DRONES][MAX_PARTS];
        };
        // one agent's drone region at row: the rows from the entry's rig and, with several meshes,
        // the object column (host truth keeps the init mesh, see overlay_structure)
        template <typename DEVICE, typename SPEC, typename STRUCTURE>
        RL_TOOLS_FUNCTION_PLACEMENT void drone_entity_rows(DEVICE& device, const DroneBody<SPEC>& body, const float base_open[12], const float base_close[12], const float* coordinates_open, const float* coordinates_close, float* transforms, float* transforms_motion, const DroneMeshes<SPEC, STRUCTURE>& meshes, typename SPEC::TI parts, typename SPEC::TI row, float placement_scale){
            using TI = typename SPEC::TI;
            entity_rows<DEVICE, typename World<SPEC>::RENDERER_SPEC>(device, body.rig, base_open, base_close, coordinates_open, coordinates_close, transforms, transforms_motion, row, placement_scale);
            if(meshes.structure != nullptr){
                for(TI part_i = 0; part_i < parts; part_i++){
                    meshes.structure[row + part_i].object = meshes.objects[body.mesh][part_i];
                }
            }
        }
        // every agent's drone region follows its own drone entry; the body is posed at
        // shutter close and only the props blur
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename BODIES_SPEC, typename STRUCTURE>
        __global__ void drone_rows_kernel(DEVICE device, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<BODIES_SPEC> bodies, typename SPEC::TI slot_offset, typename SPEC::TI parts, float* transforms, float* transforms_motion, DroneMeshes<SPEC, STRUCTURE> meshes){
            using TI = typename DEVICE::index_t;
            using WORLD = World<SPEC>;
            constexpr TI ENTRIES = WORLD::INSTANCES * WORLD::N_AGENTS;
            TI entry_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(entry_i < ENTRIES){
                const TI instance_i = entry_i / WORLD::N_AGENTS;
                const TI agent_i = entry_i % WORLD::N_AGENTS;
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                const auto& agent_state = _agent_state<SPEC>(get_ref(device, states, instance_i), agent_i);
                float base[12];
                rig::make_body_transform(device, agent_state.orientation, agent_state.position, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin, base);
                float coordinates_open[WORLD::DYNAMICS_ENV::ACTION_DIM], coordinates_close[WORLD::DYNAMICS_ENV::ACTION_DIM];
                _drone_coordinates<DEVICE, SPEC>(device, instance_parameters, agent_i, agent_state, coordinates_open, coordinates_close);
                const TI row = instance_i * WORLD::RENDERER_CONFIG::MAX_OVERLAY_INSTANCES + slot_offset + agent_i * parts;
                const auto& body = get_ref(device, bodies, instance_parameters.agents[agent_i].drone);
                const float placement_scale = rotor_scale(device, body.rig, instance_parameters.agents[agent_i].dynamics.dynamics.rotor_positions, WORLD::DYNAMICS_ENV::ACTION_DIM);
                drone_entity_rows<DEVICE, SPEC>(device, body, base, base, coordinates_open, coordinates_close, transforms, transforms_motion, meshes, parts, row, placement_scale);
            }
        }
        // Fuse no-flow episode bookkeeping into scatter or step to avoid a separate kernel launch.
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename RESET_SPEC>
        __global__ void scatter_kernel(DEVICE device, const float* __restrict__ observation, float* __restrict__ history_row, Tensor<PARAMETER_SPEC> parameters, typename SPEC::TI* episode_start, const Tensor<RESET_SPEC> reset_mask, typename SPEC::TI history_step){
            using TI = typename DEVICE::index_t;
            using WORLD = World<SPEC>;
            constexpr TI NUM_CAMERAS = WORLD::INSTANCES * WORLD::N_OBSERVATION_VIEWS;
            constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
            TI thread_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(thread_i < NUM_CAMERAS * CAM_PIXELS){
                TI camera_i = thread_i / CAM_PIXELS;
                TI instance_i = camera_i / WORLD::N_OBSERVATION_VIEWS;
                const TI rgb_row = WORLD::observation_rgb_row(camera_i);
                TI pixel_i = thread_i % CAM_PIXELS;
                if constexpr(!rl::environments::hyperdrone::World<SPEC>::RENDERER_SPEC::HAS_FLOW){
                    if(pixel_i == 0 && camera_i % WORLD::N_OBSERVATION_VIEWS == 0){
                        reset_episode_start(device, reset_mask, episode_start, history_step, static_cast<typename SPEC::TI>(instance_i));
                    }
                }
                const float scale = (float)get_ref(device, parameters, instance_i).brightness_scale;
                for(TI channel_i = 0; channel_i < 3; channel_i++){
                    float value = observation[(rgb_row * CAM_PIXELS + pixel_i) * 3 + channel_i] * scale;
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
                    observe(device, dynamics, get_ref(device, parameters, instance_i).agents[0].dynamics, get_ref(device, states, instance_i), observation_type, observation_matrix, rng_state);
                }
            }
        }
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void step_kernel(DEVICE device, typename World<SPEC>::DYNAMICS_ENV dynamics, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<ACTION_SPEC> actions, Tensor<NEXT_STATE_SPEC> next_states, RNG rng, rendering::raytracing::Camera<typename SPEC::T>* prev_cameras, const Tensor<RESET_SPEC> reset_mask, typename SPEC::TI* episode_start, typename SPEC::TI history_step){
            using TI = typename DEVICE::index_t;
            using WORLD = World<SPEC>;
            constexpr TI INSTANCES = WORLD::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                if constexpr((SPEC::HISTORY_LENGTH == 1 || !WORLD::HAS_IMAGE_OBSERVATION) && !WORLD::RENDERER_SPEC::HAS_FLOW){
                    reset_episode_start(device, reset_mask, episode_start, history_step, static_cast<typename SPEC::TI>(instance_i));
                }
                if constexpr (rl::environments::hyperdrone::World<SPEC>::RENDERER_SPEC::ENABLE_MOTION_BLUR){
                    const auto& instance_parameters = get_ref(device, parameters, instance_i);
                    const auto& state = get_ref(device, states, instance_i);
                    const typename SPEC::T aspect = static_cast<typename SPEC::T>(SPEC::CAM_WIDTH) / static_cast<typename SPEC::T>(SPEC::CAM_HEIGHT);
                    for(TI agent_i = 0; agent_i < WORLD::N_CAMERA_AGENTS; agent_i++){
                        const auto& agent = _agent_state<SPEC>(state, agent_i);
                        for(TI camera_i = 0; camera_i < WORLD::N_CAMERAS_PER_AGENT; camera_i++){
                            const TI view_i = agent_i * WORLD::N_CAMERAS_PER_AGENT + camera_i;
                            prev_cameras[instance_i * WORLD::N_VIEWS + view_i] = make_camera<DEVICE, typename SPEC::T>(device, instance_parameters.agents[agent_i].cameras[camera_i], agent.orientation, agent.position, aspect, instance_parameters.scene_translation, instance_parameters.scene_yaw_cos, instance_parameters.scene_yaw_sin);
                        }
                    }
                }
                auto& rng_state = get(rng.states, 0, instance_i);
                Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, WORLD::DYNAMICS_ACTION_DIM, false>> action_matrix;
                for(TI action_i = 0; action_i < WORLD::DYNAMICS_ACTION_DIM; action_i++){
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

    namespace rl::environments::hyperdrone::cuda{
        template <typename DEV_SPEC, typename SPEC, typename RENDERER>
        auto drone_meshes(devices::CUDA<DEV_SPEC>& device, const World<SPEC>& world, RENDERER& renderer){
            using TI = typename SPEC::TI;
            using STRUCTURE = typename utils::typing::remove_pointer<decltype(overlay_structure(device, renderer))>::type;
            for(TI overlay_i = 0; overlay_i < World<SPEC>::INSTANCES; overlay_i++){
                utils::assert_exit(device, !renderer.overlays[overlay_i].dirty, "hyperdrone::World: a CUDA World spawns only at init, an update would overwrite the device-written drone meshes");
            }
            DroneMeshes<SPEC, STRUCTURE> meshes{};
            if(world.drone_meshes.size() > 1){
                meshes.structure = overlay_structure(device, renderer);
                for(TI mesh_i = 0; mesh_i < (TI)world.drone_meshes.size(); mesh_i++){
                    const auto& record = renderer.assets[world.drone_meshes[mesh_i].index];
                    for(TI part_i = 0; part_i < world.drone_parts; part_i++){
                        meshes.objects[mesh_i][part_i] = (unsigned int)renderer.asset_part_objects[record.first_part + part_i];
                    }
                }
            }
            return meshes;
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
        rl::environments::hyperdrone::cuda::sample_initial_parameters_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, world.parameters, world.drone_parameters, world.num_drones, parameters, reset_mask, world_rng);
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
        complete_reset(device, world, reset_mask, rng);
    }
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void prepare_render_inputs(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask_input){
        if(!world.render_pending){
            return;
        }
        request_render(device, world, reset_mask_input);
        auto& reset_mask = world.render_reset;
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;

        constexpr TI INSTANCES = WORLD::INSTANCES;
        const T aspect = static_cast<T>(SPEC::CAM_WIDTH) / static_cast<T>(SPEC::CAM_HEIGHT);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto& renderer = get_renderer(device, world);
        if constexpr (SPEC::SELF_VISIBLE){
            float* transforms_motion_rows = nullptr;
            if constexpr (WORLD::RENDERER_SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                transforms_motion_rows = data(transforms_motion(device, renderer));
            }
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES * WORLD::N_AGENTS, BLOCKSIZE);
            rl::environments::hyperdrone::cuda::drone_rows_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, parameters, states, world.drone_bodies, world.drone_slot_offset, world.drone_parts, data(transforms(device, renderer)), transforms_motion_rows, rl::environments::hyperdrone::cuda::drone_meshes(device, world, renderer));
            check_status(device);
        }
        {
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES * WORLD::N_VIEWS, BLOCKSIZE);
            auto* cameras_data = data(cameras(device, renderer));
            auto* cameras_open_data = cameras_data;
            if constexpr(rl::environments::hyperdrone::World<SPEC>::RENDERER_SPEC::ENABLE_MOTION_BLUR){
                cameras_open_data = data(cameras_open(device, renderer));
            }
            rl::environments::hyperdrone::cuda::pose_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, parameters, states, reset_mask, cameras_data, cameras_open_data, data(world.prev_cameras), world.history_step, aspect);
        }
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename RESET_SPEC>
    void reset_render_state(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, const Tensor<RESET_SPEC>& reset_mask){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        auto& renderer = get_renderer(device, world);
        rendering::raytracing::Camera<typename SPEC::T>* previous_cameras = nullptr;
        if constexpr(rl::environments::hyperdrone::World<SPEC>::RENDERER_SPEC::HAS_FLOW){
            previous_cameras = data(cameras_previous(device, renderer));
        }
        const float* current_transforms = nullptr;
        float* previous_transforms = nullptr;
        if constexpr(WORLD::RENDERER_SPEC::HAS_FLOW_OVERLAYS){
            current_transforms = data(transforms(device, renderer));
            previous_transforms = data(transforms_previous(device, renderer));
        }
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = (WORLD::INSTANCES + BLOCKSIZE - 1) / BLOCKSIZE;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::cuda::reset_render_state_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, reset_mask, world.flow_scene_version != world.scene_version, data(cameras(device, renderer)), previous_cameras, current_transforms, previous_transforms, data(world.episode_start), world.history_step);
        check_status(device);
        if constexpr(rl::environments::hyperdrone::World<SPEC>::RENDERER_SPEC::HAS_FLOW){
            world.flow_scene_version = world.scene_version;
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask_input){
        if(!world.render_pending){
            return;
        }
        prepare_render_inputs(device, world, parameters, states, reset_mask_input);
        if constexpr(rl::environments::hyperdrone::World<SPEC>::RENDERER_SPEC::HAS_FLOW || !rl::environments::hyperdrone::World<SPEC>::HAS_IMAGE_OBSERVATION){
            reset_render_state(device, world, world.render_reset);
        }
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        constexpr TI INSTANCES = WORLD::INSTANCES;
        constexpr TI CAM_PIXELS = SPEC::CAM_WIDTH * SPEC::CAM_HEIGHT;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto& renderer = get_renderer(device, world);
        if constexpr (WORLD::RENDERER_CONFIG::NUM_OVERLAYS > 0) {
            update_launch(device, renderer);
        }
        render_launch(device, renderer);
        if constexpr (WORLD::HAS_IMAGE_OBSERVATION){
            const TI history_slot = (world.history_step - 1) % SPEC::HISTORY_LENGTH;
            float* history_row = data(world.history) + history_slot * INSTANCES * WORLD::N_OBSERVATION_VIEWS * WORLD::FRAME_DIM;
            {
                constexpr TI BLOCKSIZE = 256;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES * WORLD::N_OBSERVATION_VIEWS * CAM_PIXELS, BLOCKSIZE);
                rl::environments::hyperdrone::cuda::scatter_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, data(output(device, renderer, rendering::raytracing::outputs::RGB<rendering::raytracing::outputs::format::Float32>{})), history_row, parameters, data(world.episode_start), world.render_reset, world.history_step);
            }
        }
        check_status(device);
        world.render_pending = false;
    }
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::World<SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::World<SPEC>::Observation, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::World<SPEC>;
        static_assert(get<0>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::INSTANCES);
        static_assert(WORLD::HAS_IMAGE_OBSERVATION, "image observations require RGB<Float32> in OUTPUTS");
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::OBSERVATION_DIM);
        if(world.render_pending){
            render(device, world, parameters, states, world.render_reset);
        }
        utils::assert_exit(device, world.history_step > 0, "hyperdrone::World::observe: no frame available");
        const TI history_slot = (world.history_step - 1) % SPEC::HISTORY_LENGTH;
        const float* history_row = data(world.history) + history_slot * WORLD::INSTANCES * WORLD::N_OBSERVATION_VIEWS * WORLD::FRAME_DIM;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        constexpr TI BLOCKSIZE = 256;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(WORLD::INSTANCES * WORLD::OBSERVATION_DIM, BLOCKSIZE);
        rl::environments::hyperdrone::cuda::observe_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, history_row, observations);
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
        auto dynamics_actions = control(device, world.controller, world.controller_state, world.controller_buffer, parameters, states, actions, rng);
        if(world.render_pending){
            if constexpr(rl::environments::hyperdrone::World<SPEC>::HAS_IMAGE_OBSERVATION && SPEC::HISTORY_LENGTH > 1){
                render(device, world, parameters, states, world.render_reset);
            }
            else{
                if constexpr(rl::environments::hyperdrone::World<SPEC>::RENDERER_SPEC::HAS_FLOW){
                    prepare_render_inputs(device, world, parameters, states, world.render_reset);
                    reset_render_state(device, world, world.render_reset);
                }
            }
        }
        if constexpr (rl::environments::hyperdrone::World<SPEC>::RENDERER_SPEC::HAS_FLOW){
            advance_flow_launch(device, get_renderer(device, world));
        }
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::cuda::step_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.dynamics, parameters, states, dynamics_actions, next_states, world_rng, data(world.prev_cameras), world.render_reset, data(world.episode_start), world.history_step);
        check_status(device);
        world.history_step++;
        set_all(device, world.render_reset, false);
        request_render(device, world);
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

    namespace rl::environments::hyperdrone::cuda{
        // runs the verb for every member concurrently: all members' renderer streams wait for one
        // record of the caller's stream, taken before any member work (on the legacy default
        // stream a later record would also wait for the members enqueued so far), then
        // device.stream points at each renderer stream in turn, where the launch verbs need no
        // fork/join of their own; the guard restores the caller's stream and joins every member
        // back into it, also when a verb throws. Members write disjoint memory and draw from
        // disjoint RNG partitions
        template <typename DEV_SPEC, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename VERB>
        void fan_out(devices::CUDA<DEV_SPEC>& device, MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, VERB&& verb){
            using TI = typename MEMBER::TI;
            struct Join{
                devices::CUDA<DEV_SPEC>& device;
                MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env;
                const cudaStream_t caller_stream;
                ~Join(){
                    device.stream = caller_stream;
                    for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
                        devices::cuda::order_after(device.ordering_event, stream(device, get_renderer(device, env.environments[environment_i])), caller_stream);
                    }
                }
            } join{device, env, device.stream};
            cudaEventRecord(device.ordering_event, join.caller_stream);
            for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
                cudaStreamWaitEvent(stream(device, get_renderer(device, env.environments[environment_i])), device.ordering_event, 0);
            }
            for (TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++) {
                device.stream = stream(device, get_renderer(device, env.environments[environment_i]));
                verb(environment_i);
            }
        }
    }
    template <typename DEV_SPEC, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask){
        using MULTI_ENVIRONMENT = rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MEMBER::TI;
        constexpr TI M = rl::environments::hyperdrone::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        rl::environments::hyperdrone::cuda::fan_out(device, env, [&](TI environment_i){
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto reset_block = view_range(device, reset_mask, environment_i * M, tensor::ViewSpec<0, M>{});
            render(device, env.environments[environment_i], parameters_block, states_block, reset_block);
        });
    }
    template <typename DEV_SPEC, typename MEMBER, typename MEMBER::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using MULTI_ENVIRONMENT = rl::environments::hyperdrone::MultiEnvironment<MEMBER, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MEMBER::TI;
        constexpr TI M = rl::environments::hyperdrone::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        rl::environments::hyperdrone::cuda::fan_out(device, env, [&](TI environment_i){
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto observations_block = view_range(device, observations, environment_i * M, tensor::ViewSpec<0, M>{});
            observe(device, env.environments[environment_i], parameters_block, states_block, observation_type, observations_block, rng);
        });
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
