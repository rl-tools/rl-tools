#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_GRU_RESET_MASK_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_GRU_RESET_MASK_H

#include "../../../rl_tools.h"
#include "../../../containers/matrix/matrix.h"
#include "../../../containers/tensor/tensor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::layers::gru::mode{
    template <typename MASK>
    struct ResetMaskSize;
    template <typename SPEC>
    struct ResetMaskSize<Matrix<SPEC>>{
        static_assert(SPEC::ROWS == 1, "The reset mask for GRU layers must have a single row.");
        static constexpr auto VALUE = SPEC::COLS;
    };
    template <typename SPEC>
    struct ResetMaskSize<Tensor<SPEC>>{
        static_assert(length(typename SPEC::SHAPE{}) == 1, "The reset mask for GRU layers must be a rank-1 tensor.");
        static constexpr auto VALUE = get<0>(typename SPEC::SHAPE{});
    };
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool reset_mask_value(DEVICE&, const Matrix<SPEC>& mask, typename DEVICE::index_t batch_i){
        return get(mask, 0, batch_i);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool reset_mask_value(DEVICE& device, const Tensor<SPEC>& mask, typename DEVICE::index_t batch_i){
        return get(device, mask, batch_i);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
