#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_MIXED_IMITATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_MIXED_IMITATION_H

#include "../../../mode/mode.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::ppo::mixed_imitation{
    template <typename T_BASE = mode::Default<>, typename T_SPEC = bool>
    struct FixedWeight: T_BASE{
        using BASE = T_BASE;
        using SPEC = T_SPEC;
    };
    template <typename T_BASE = mode::Default<>, typename T_SPEC = bool>
    struct FixedNormRatio: T_BASE{
        using BASE = T_BASE;
        using SPEC = T_SPEC;
    };
    template <typename TYPE_POLICY, typename TI>
    struct DefaultParameters{
        using T = typename TYPE_POLICY::DEFAULT;
        static constexpr TI TEACHER_FORCING_UPDATES = 10;
        static constexpr T IMITATION_WEIGHT = 1;
        static constexpr T OUTPUT_GRADIENT_NORM_RATIO = 1;
        static constexpr T MAX_IMITATION_WEIGHT = 1000;
        static constexpr T NORM_EPSILON = 1e-12;
        using BALANCING_MODE = Mode<FixedWeight<>>;
    };
    template <typename T_TYPE_POLICY, typename T_TI, T_TI T_BATCH_SIZE, T_TI T_GROUP_SIZE, T_TI T_IMITATION_PER_GROUP, typename T_PARAMETERS = DefaultParameters<T_TYPE_POLICY, T_TI>>
    struct Specification{
        using TYPE_POLICY = T_TYPE_POLICY;
        using T = typename TYPE_POLICY::DEFAULT;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
        static constexpr TI BATCH_SIZE = T_BATCH_SIZE;
        static constexpr TI GROUP_SIZE = T_GROUP_SIZE;
        static constexpr TI IMITATION_PER_GROUP = T_IMITATION_PER_GROUP;
        static_assert(GROUP_SIZE > 0 && BATCH_SIZE > 0);
        static_assert(IMITATION_PER_GROUP < GROUP_SIZE, "Each group must retain RL samples");
        static_assert(BATCH_SIZE % GROUP_SIZE == 0, "Every minibatch must contain complete groups");
        static constexpr TI IMITATION_BATCH_SIZE = BATCH_SIZE / GROUP_SIZE * IMITATION_PER_GROUP;
        static constexpr TI RL_BATCH_SIZE = BATCH_SIZE - IMITATION_BATCH_SIZE;
        static_assert(PARAMETERS::IMITATION_WEIGHT > 0 && PARAMETERS::OUTPUT_GRADIENT_NORM_RATIO > 0);
        static_assert(PARAMETERS::MAX_IMITATION_WEIGHT >= PARAMETERS::IMITATION_WEIGHT && PARAMETERS::NORM_EPSILON > 0);
    };
    template <typename SPEC>
    struct Metrics{
        using T = typename SPEC::T;
        T imitation_mse = 0;
        T rl_output_gradient_norm = 0;
        T imitation_output_gradient_norm = 0;
        T weighted_imitation_output_gradient_norm = 0;
        T imitation_weight = 0;
        T output_gradient_norm_ratio = 0;
        bool ratio_valid = false;
        bool balancing_fallback = false;
        bool weight_clamped = false;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
