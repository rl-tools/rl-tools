#pragma once

namespace rl_tools::rendering::raytracing::yaw_prediction {

#ifndef ABLATION_CHANNEL_MULTIPLIER
#define ABLATION_CHANNEL_MULTIPLIER 1
#endif

#ifndef ABLATION_USE_CROSS_CONV
#define ABLATION_USE_CROSS_CONV 1
#endif

    template <typename TI, TI T_CHANNEL_MULTIPLIER = ABLATION_CHANNEL_MULTIPLIER, bool T_USE_CROSS_CONV = (ABLATION_USE_CROSS_CONV != 0)>
    struct ModelConfig {
        static constexpr TI CHANNEL_MULTIPLIER = T_CHANNEL_MULTIPLIER;
        static constexpr bool USE_CROSS_CONV = T_USE_CROSS_CONV;
        static constexpr TI EARLY_CH_1 = 32 * CHANNEL_MULTIPLIER;
        static constexpr TI EARLY_CH_2 = 64 * CHANNEL_MULTIPLIER;
        static constexpr TI LATE_1X1_CH = 128 * CHANNEL_MULTIPLIER;
        static constexpr TI LATE_CH = 256 * CHANNEL_MULTIPLIER;
        static constexpr TI HEAD_1X1_CH = 256 * CHANNEL_MULTIPLIER;
        static constexpr TI HEAD_DENSE_CH = 128 * CHANNEL_MULTIPLIER;
    };
}
