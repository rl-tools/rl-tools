#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_MOVING_GATE_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_MOVING_GATE_OPERATIONS_CUDA_H

#include "operations_cpu.h"
#include "../../operations_cuda.h"

#include <cuda_runtime.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::hyperdrone::tasks::moving_gate::cuda{
        template <typename DEVICE, typename TASK_SPEC, typename SCENE_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void reset_state_kernel(DEVICE device, Tensor<SCENE_SPEC> active_annotations, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<RESET_SPEC> reset_mask, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES && get(device, reset_mask, instance_i)){
                auto& rng_state = get(rng.states, 0, instance_i);
                const auto& annotations = get_ref(device, active_annotations, 0);
                auto& instance_parameters = get_ref(device, parameters, instance_i);
                auto& state = get_ref(device, states, instance_i);
                tasks::moving_gate::_sample_gate<DEVICE, TASK_SPEC>(device, annotations, instance_parameters, state, rng_state);
            }
        }
        // the gate is a rigid entity moved along the constant-twist screw between the shutter-open
        // and shutter-close gate poses
        template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC>
        __global__ void gate_rows_kernel(DEVICE device, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, hyperlux::PoseSink<typename TASK_SPEC::NEXT_WORLD::RENDERER_SPEC> sink, hyperlux::OverlayPlacement placement){
            using T = typename TASK_SPEC::T;
            using TI = typename DEVICE::index_t;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                const auto& state = get_ref(device, states, instance_i);
                float pose_close[12];
                gate_pose(device, instance_parameters, state.gate_phase, pose_close);
                if constexpr(NEXT_WORLD::RENDERER_SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
                    constexpr T TWO_PI = (T)2 * math::PI<T>;
                    const T dt = instance_parameters.agents[0].dynamics.integration.dt;
                    const T phase_open = state.gate_phase - TWO_PI * instance_parameters.gate_frequency * dt * instance_parameters.shutter_fraction;
                    float pose_open[12];
                    gate_pose(device, instance_parameters, phase_open, pose_open);
                    set_pose(device, sink, hyperlux::OverlayIndex{instance_i}, placement, make_pose(pose_open), make_pose(pose_close));
                }
                else{
                    set_pose(device, sink, hyperlux::OverlayIndex{instance_i}, placement, make_pose(pose_close));
                }
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename NEXT_STATE_SPEC>
        __global__ void gate_step_kernel(DEVICE device, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, Tensor<NEXT_STATE_SPEC> next_states){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                _gate_step<DEVICE, TASK_SPEC>(device, instance_parameters, get_ref(device, states, instance_i), get_ref(device, next_states, instance_i), instance_parameters.agents[0].dynamics.integration.dt);
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename STATE_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC>
        __global__ void gate_pass_reward_kernel(DEVICE device, Tensor<STATE_SPEC> states, Tensor<NEXT_STATE_SPEC> next_states, Tensor<REWARD_SPEC> rewards){
            using T = typename TASK_SPEC::T;
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                const auto& state = get_ref(device, states, instance_i);
                const auto& next_state = get_ref(device, next_states, instance_i);
                if(next_state.gate_passed && !state.gate_passed){
                    set(device, rewards, get(device, rewards, instance_i) + (typename REWARD_SPEC::T)TASK_SPEC::GATE_PASS_REWARD, instance_i);
                }
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC>
        __global__ void gate_crash_terminated_kernel(DEVICE device, Tensor<STATE_SPEC> states, Tensor<TERMINATED_SPEC> terminated_flags){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = World<TASK_SPEC>::INSTANCES;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                if(get_ref(device, states, instance_i).gate_crashed){
                    set(device, terminated_flags, true, instance_i);
                }
            }
        }
        template <typename DEVICE, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC>
        __global__ void gate_observe_kernel(DEVICE device, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, Tensor<OBSERVATION_SPEC> observations){
            using T = typename TASK_SPEC::T;
            using TI = typename DEVICE::index_t;
            using WORLD = World<TASK_SPEC>;
            using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
            constexpr TI INSTANCES = WORLD::INSTANCES;
            constexpr TI BASE_DIM = NEXT_WORLD::OBSERVATION_DIM_PRIVILEGED;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                const auto& instance_parameters = get_ref(device, parameters, instance_i);
                const auto& state = get_ref(device, states, instance_i);
                T position[3];
                drone_scene_position<DEVICE, TASK_SPEC>(device, instance_parameters, state, position);
                T offset = instance_parameters.gate_amplitude * math::sin(device.math, state.gate_phase);
                T gate_velocity_magnitude = instance_parameters.gate_amplitude * math::cos(device.math, state.gate_phase) * (T)2 * math::PI<T> * instance_parameters.gate_frequency;
                for(TI dim = 0; dim < 3; dim++){
                    T gate_position = instance_parameters.gate_center[dim] + offset * instance_parameters.gate_axis[dim];
                    set(device, observations, gate_position - position[dim], instance_i, BASE_DIM + dim);
                    set(device, observations, gate_velocity_magnitude * instance_parameters.gate_axis[dim], instance_i, BASE_DIM + 3 + dim);
                }
                set(device, observations, math::sin(device.math, state.gate_phase), instance_i, BASE_DIM + 6);
                set(device, observations, math::cos(device.math, state.gate_phase), instance_i, BASE_DIM + 7);
            }
        }
    }

    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void reset_state(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        constexpr TI INSTANCES = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>::INSTANCES;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        reset_state(device, static_cast<NEXT_WORLD&>(world), parameters, states, reset_mask, rng);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto world_rng = rl::environments::hyperdrone::cuda::instance_rng(device, world, rng);
        rl::environments::hyperdrone::tasks::moving_gate::cuda::reset_state_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, world.active_annotations, parameters, states, reset_mask, world_rng);
        check_status(device);
    }

    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC>
    void prepare_entity_poses(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        constexpr TI INSTANCES = WORLD::INSTANCES;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        const auto& kind = world.entity_kinds[world.entity_kind_index];
        auto& renderer = get_renderer(device, world);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        rl::environments::hyperdrone::tasks::moving_gate::cuda::gate_rows_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, parameters, states, pose_sink(device, renderer), rl::environments::hyperdrone::_entity_placement(world, kind));
        check_status(device);
    }

    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask_input){
        if(!world.render_pending){
            return;
        }
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        request_render(device, world, reset_mask_input);
        prepare_entity_poses(device, world, parameters, states);
        render(device, static_cast<NEXT_WORLD&>(world), parameters, states, world.render_reset);
    }

    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        constexpr TI INSTANCES = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>::INSTANCES;
        if constexpr (NEXT_WORLD::SPEC::HISTORY_LENGTH > 1 || NEXT_WORLD::RENDERER_SPEC::HAS_FLOW){
            if(world.render_pending){
                prepare_entity_poses(device, world, parameters, states);
            }
        }
        step(device, static_cast<NEXT_WORLD&>(world), parameters, states, actions, next_states, rng);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::tasks::moving_gate::cuda::gate_step_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, parameters, states, next_states);
        check_status(device);
    }

    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        constexpr TI INSTANCES = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>::INSTANCES;
        reward(device, static_cast<NEXT_WORLD&>(world), parameters, states, actions, next_states, rewards, rng);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::tasks::moving_gate::cuda::gate_pass_reward_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, states, next_states, rewards);
        check_status(device);
    }

    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void terminated(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        constexpr TI INSTANCES = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>::INSTANCES;
        terminated(device, static_cast<NEXT_WORLD&>(world), parameters, states, terminated_flags, rng);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::tasks::moving_gate::cuda::gate_crash_terminated_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, states, terminated_flags);
        check_status(device);
    }

    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename TASK_SPEC::NEXT_WORLD::Observation observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        if(world.render_pending){
            render(device, world, parameters, states, world.render_reset);
        }
        observe(device, static_cast<NEXT_WORLD&>(world), parameters, states, observation_type, observations, rng);
    }

    template <typename DEV_SPEC, typename TASK_SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, typename rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>::ObservationPrivileged, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        using WORLD = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == WORLD::OBSERVATION_DIM_PRIVILEGED);
        constexpr TI BASE_DIM = NEXT_WORLD::OBSERVATION_DIM_PRIVILEGED;
        auto base_observations = view_range(device, observations, 0, tensor::ViewSpec<1, BASE_DIM>{});
        observe(device, static_cast<NEXT_WORLD&>(world), parameters, states, typename NEXT_WORLD::ObservationPrivileged{}, base_observations, rng);
        constexpr TI INSTANCES = WORLD::INSTANCES;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::tasks::moving_gate::cuda::gate_observe_kernel<decltype(tag_device), TASK_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, parameters, states, observations);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
