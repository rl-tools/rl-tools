#include "../../../../../../version.h"
#include "../../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_YAW_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_YAW_OPERATIONS_CUDA_H

#include "operations_generic.h"
#include "../../../../../../containers/tensor/operations_cuda.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::hyperdrone::tasks::target_frame::yaw::cuda{
        template<typename DEVICE, typename STATE_SPEC, typename TARGET_SPEC>
        __global__ void targets_kernel(DEVICE device, const Tensor<STATE_SPEC> states, Tensor<TARGET_SPEC> targets){
            using TI = typename STATE_SPEC::TI;
            using T = typename STATE_SPEC::T::T;
            const TI instance_i = blockIdx.x * blockDim.x + threadIdx.x;
            if(instance_i < get<0>(typename STATE_SPEC::SHAPE{})){
                T cosine, sine;
                yaw_target(device, get_ref(device, states, instance_i), cosine, sine);
                set(device, targets, static_cast<typename TARGET_SPEC::T>(cosine), instance_i, 0);
                set(device, targets, static_cast<typename TARGET_SPEC::T>(sine), instance_i, 1);
            }
        }
    }

    template<typename DEV_SPEC, typename STATE_SPEC, typename TARGET_SPEC>
    void yaw_targets(devices::CUDA<DEV_SPEC>& device, const Tensor<STATE_SPEC>& states, Tensor<TARGET_SPEC>& targets){
        static_assert(length(typename STATE_SPEC::SHAPE{}) == 1);
        static_assert(length(typename TARGET_SPEC::SHAPE{}) == 2);
        static_assert(get<0>(typename STATE_SPEC::SHAPE{}) == get<0>(typename TARGET_SPEC::SHAPE{}));
        static_assert(get<1>(typename TARGET_SPEC::SHAPE{}) == 2);
        using DEVICE = devices::CUDA<DEV_SPEC>;
        constexpr typename STATE_SPEC::TI BLOCK_SIZE = 32;
        constexpr auto N_BLOCKS = (get<0>(typename STATE_SPEC::SHAPE{}) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::tasks::target_frame::yaw::cuda::targets_kernel<<<N_BLOCKS, BLOCK_SIZE, 0, device.stream>>>(tag_device, states, targets);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
