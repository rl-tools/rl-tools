#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_TYPES_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_TYPES_H

#include "../rl_tools.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering{
        struct SceneLight {
            int type; // 0=directional, 1=point, 2=spot
            float position[3];
            float direction[3];
            float color[3];
            float attenuation_constant;
            float attenuation_linear;
            float attenuation_quadratic;
            float cos_inner_cone;
            float cos_outer_cone;
        };

        struct CollisionResult {
            float distance;
            int hit;
        };

        template <
            bool T_LOAD_TEXTURES,
            bool T_NORMAL_SHADING,
            bool T_METALLIC_REFLECTIONS,
            bool T_SRGB_OUTPUT,
            bool T_CHECKER_BACKGROUND,
            bool T_PBR_SHADING,
            bool T_PUNCTUAL_LIGHT_SHADOWS = false
        >
        struct ShadingOptions{
            static constexpr bool LOAD_TEXTURES = T_LOAD_TEXTURES;
            static constexpr bool NORMAL_SHADING = T_NORMAL_SHADING;
            static constexpr bool METALLIC_REFLECTIONS = T_METALLIC_REFLECTIONS;
            static constexpr bool SRGB_OUTPUT = T_SRGB_OUTPUT;
            static constexpr bool CHECKER_BACKGROUND = T_CHECKER_BACKGROUND;
            static constexpr bool PBR_SHADING = T_PBR_SHADING;
            static constexpr bool PUNCTUAL_LIGHT_SHADOWS = T_PUNCTUAL_LIGHT_SHADOWS;
        };

        using Medium = ShadingOptions<true, true, true, true, true, false>;
        using High = ShadingOptions<true, true, true, true, false, true>;
        using VeryHigh = ShadingOptions<true, true, true, true, false, true, true>;
        using Low = ShadingOptions<false, false, false, false, false, false>;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
