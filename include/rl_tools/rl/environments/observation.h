#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_OBSERVATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_OBSERVATION_H

#include "../../containers/tensor/tensor.h"
#include "../../utils/generic/typing.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::observation{

    // Image observation base
    template <typename T_TI, T_TI T_HEIGHT, T_TI T_WIDTH, T_TI T_CHANNELS>
    struct Image{
        using TI = T_TI;
        static constexpr TI HEIGHT = T_HEIGHT;
        static constexpr TI WIDTH = T_WIDTH;
        static constexpr TI CHANNELS = T_CHANNELS;
        static constexpr TI DIM = HEIGHT * WIDTH * CHANNELS;
        using SHAPE = tensor::Shape<TI, HEIGHT, WIDTH, CHANNELS>;
    };

    // Compose: multiple observations (each stored as separate shaped tensor)
    template <typename T_A, typename T_B>
    struct Compose{
        using A = T_A;
        using B = T_B;
        // No single DIM or SHAPE - this is fundamentally multi-tensor
    };

    template <typename T> struct is_compose : utils::typing::false_type {};
    template <typename A, typename B> struct is_compose<Compose<A, B>> : utils::typing::true_type {};
    template <typename T> static constexpr bool is_compose_v = is_compose<T>::value;
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
