#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_RAPTOR_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_RAPTOR_OPERATIONS_CUDA_H

#include "raptor.h"
#include "../../../../../nn/layers/dense/operations_cuda.h"
#include "../../../../../nn/layers/gru/helper_operations_cuda.h"
#include "operations_cpu.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::environments::hyperdrone::controllers::raptor::cuda {
        template <typename DEVICE, typename ACTION_SPEC, typename MASK_SPEC>
        __global__ void reset_action_history_kernel(DEVICE device, Tensor<ACTION_SPEC> actions, const Tensor<MASK_SPEC> mask) {
            using TI = typename DEVICE::index_t;
            TI row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < get<0>(typename ACTION_SPEC::SHAPE{}) && get(device, mask, row)) {
                for (TI motor = 0; motor < get<1>(typename ACTION_SPEC::SHAPE{}); motor++) {
                    set(device, actions, 0, row, motor);
                }
            }
        }
        template <typename DEVICE, typename MASK_SPEC, typename OUTPUT_SPEC>
        __global__ void expand_mask_kernel(DEVICE device, const Tensor<MASK_SPEC> mask, Tensor<OUTPUT_SPEC> output) {
            using TI = typename DEVICE::index_t;
            constexpr TI ROWS = get<0>(typename OUTPUT_SPEC::SHAPE{});
            constexpr TI AGENTS = ROWS / get<0>(typename MASK_SPEC::SHAPE{});
            TI row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < ROWS) {
                set(device, output, get(device, mask, row / AGENTS), row);
            }
        }
        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename PHYSICAL_SPEC, typename COMMAND_SPEC>
        __global__ void prepare_kernel(DEVICE device, Controller<SPEC> controller, State<SPEC> state, Buffer<SPEC> buffer, const Tensor<PARAMETER_SPEC> parameters, const Tensor<PHYSICAL_SPEC> physical_states, const Tensor<COMMAND_SPEC> commands) {
            using TI = typename SPEC::TI;
            constexpr TI AGENTS = get<1>(typename COMMAND_SPEC::SHAPE{}) / 4;
            TI row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < SPEC::INSTANCES) {
                TI instance = row / AGENTS;
                auto command = view(device, commands, instance);
                observe(device, controller, state, buffer, get_ref(device, parameters, instance), get_ref(device, physical_states, instance), command, row, row % AGENTS);
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC>
    void reset_action_history(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, const Tensor<MASK_SPEC>& mask) {
        devices::cuda::TAG<devices::CUDA<DEV_SPEC>, true> tag_device{};
        rl::environments::hyperdrone::controllers::raptor::cuda::reset_action_history_kernel<<<(SPEC::INSTANCES + 31) / 32, 32, 0, device.stream>>>(tag_device, state.previous_actions, mask);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC, typename RNG>
    void reset(devices::CUDA<DEV_SPEC>& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, const Tensor<MASK_SPEC>& mask, RNG& rng) {
        static_assert(length(typename MASK_SPEC::SHAPE{}) == 1 && get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::INSTANCES);
        auto mask_matrix = matrix_view(device, mask);
        Mode<mode::sequential::ResetMask<mode::Default<>, mode::sequential::ResetMaskSpecification<decltype(mask_matrix)>>> mode;
        mode.mask = mask_matrix;
        reset(device, controller.model, state.recurrent, rng, mode);
        reset_action_history(device, state, mask);
    }
    template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC, typename RNG>
    void reset(devices::CUDA<DEV_SPEC>& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, rl::environments::hyperdrone::controllers::raptor::Buffer<SPEC>& buffer, const Tensor<MASK_SPEC>& mask, RNG& rng) {
        constexpr auto INSTANCES = get<0>(typename MASK_SPEC::SHAPE{});
        static_assert(length(typename MASK_SPEC::SHAPE{}) == 1 && SPEC::INSTANCES % INSTANCES == 0);
        if constexpr (INSTANCES == SPEC::INSTANCES) {
            reset(device, controller, state, mask, rng);
        }
        else {
            devices::cuda::TAG<devices::CUDA<DEV_SPEC>, true> tag_device{};
            rl::environments::hyperdrone::controllers::raptor::cuda::expand_mask_kernel<<<(SPEC::INSTANCES + 31) / 32, 32, 0, device.stream>>>(tag_device, mask, buffer.expanded_mask);
            check_status(device);
            reset(device, controller, state, buffer.expanded_mask, rng);
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename PHYSICAL_SPEC, typename COMMAND_SPEC, typename RNG>
    auto control(devices::CUDA<DEV_SPEC>& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, rl::environments::hyperdrone::controllers::raptor::Buffer<SPEC>& buffer, const Tensor<PARAMETER_SPEC>& parameters, const Tensor<PHYSICAL_SPEC>& physical_states, const Tensor<COMMAND_SPEC>& commands, RNG& rng) {
        using TI = typename SPEC::TI;
        constexpr TI INSTANCES = get<0>(typename COMMAND_SPEC::SHAPE{});
        constexpr TI AGENTS = get<1>(typename COMMAND_SPEC::SHAPE{}) / 4;
        static_assert(length(typename COMMAND_SPEC::SHAPE{}) == 2 && get<1>(typename COMMAND_SPEC::SHAPE{}) % 4 == 0);
        static_assert(INSTANCES * AGENTS == SPEC::INSTANCES);
        static_assert(length(typename PARAMETER_SPEC::SHAPE{}) == 1 && get<0>(typename PARAMETER_SPEC::SHAPE{}) == INSTANCES);
        static_assert(length(typename PHYSICAL_SPEC::SHAPE{}) == 1 && get<0>(typename PHYSICAL_SPEC::SHAPE{}) == INSTANCES);
        devices::cuda::TAG<devices::CUDA<DEV_SPEC>, true> tag_device{};
        rl::environments::hyperdrone::controllers::raptor::cuda::prepare_kernel<<<(SPEC::INSTANCES + 31) / 32, 32, 0, device.stream>>>(tag_device, controller, state, buffer, parameters, physical_states, commands);
        check_status(device);
        evaluate_step(device, controller, state, buffer, rng);
        return reshape_row_major(device, state.previous_actions, tensor::Shape<TI, INSTANCES, AGENTS * 4>{});
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
