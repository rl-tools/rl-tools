#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_OBSERVATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_OBSERVATION_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::observation {
    // channel blocks of the composed visual observation: the per-pixel channel layout is
    // [stacked frames | target frame | zero padding], the composed channel count is what the
    // policy CNN input shape derives from
    template <typename T_TI, T_TI T_N, T_TI T_STRIDE = 1>
    struct ImageStack {
        using TI = T_TI;
        static constexpr TI N = T_N;
        static constexpr TI STRIDE = T_STRIDE;
    };
    template <typename T_TI>
    struct TargetImage {
        using TI = T_TI;
    };
    template <typename T_TI, T_TI T_TO>
    struct PadChannelsTo {
        using TI = T_TI;
        static constexpr TI TO = T_TO;
    };

    namespace channels {
        template <typename TI_VALUE>
        struct Traits {
            using TI = TI_VALUE;
            TI stack_n = 1;
            TI stack_stride = 1;
            bool has_target = false;
            TI pad_to = 0;
        };
        template <typename TI>
        constexpr Traits<TI> fold(Traits<TI> traits){
            return traits;
        }
        template <typename TI, typename... REST, TI N, TI STRIDE>
        constexpr Traits<TI> fold(Traits<TI> traits, ImageStack<TI, N, STRIDE>, REST... rest){
            traits.stack_n = N;
            traits.stack_stride = STRIDE;
            return fold(traits, rest...);
        }
        template <typename TI, typename... REST>
        constexpr Traits<TI> fold(Traits<TI> traits, TargetImage<TI>, REST... rest){
            traits.has_target = true;
            return fold(traits, rest...);
        }
        template <typename TI, typename... REST, TI TO>
        constexpr Traits<TI> fold(Traits<TI> traits, PadChannelsTo<TI, TO>, REST... rest){
            traits.pad_to = TO;
            return fold(traits, rest...);
        }
    }

    template <typename T_FIRST, typename... T_BLOCKS>
    struct Channels {
        using TI = typename T_FIRST::TI;
        static constexpr channels::Traits<TI> TRAITS = channels::fold(channels::Traits<TI>{}, T_FIRST{}, T_BLOCKS{}...);
        static constexpr TI STACK_N = TRAITS.stack_n;
        static constexpr TI STACK_STRIDE = TRAITS.stack_stride;
        static constexpr bool HAS_TARGET = TRAITS.has_target;
        static constexpr TI PAD_TO = TRAITS.pad_to;
        template <TI IMAGE_CHANNELS>
        static constexpr TI LOGICAL_CHANNELS = STACK_N * IMAGE_CHANNELS + (HAS_TARGET ? IMAGE_CHANNELS : 0);
        template <TI IMAGE_CHANNELS>
        static constexpr TI CHANNELS = PAD_TO == 0 ? LOGICAL_CHANNELS<IMAGE_CHANNELS> : ((LOGICAL_CHANNELS<IMAGE_CHANNELS> + PAD_TO - 1) / PAD_TO) * PAD_TO;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
