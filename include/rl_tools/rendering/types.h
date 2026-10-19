#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_TYPES_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_TYPES_H

#include "../rl_tools.h"
#include "../utils/generic/typing.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    // Byte copies keep the IEEE-754 miss sentinel defined under fast-math without aliasing float storage.
    RL_TOOLS_FUNCTION_PLACEMENT inline void store_miss(float& value){
        static_assert(sizeof(float) == 4 && sizeof(unsigned int) == 4, "rendering requires 32-bit float and unsigned int");
        const unsigned int bits = 0x7f800000u;
        const auto* source = reinterpret_cast<const unsigned char*>(&bits);
        auto* destination = reinterpret_cast<unsigned char*>(&value);
        for(unsigned int i = 0; i < sizeof(value); i++){
            destination[i] = source[i];
        }
    }

    RL_TOOLS_FUNCTION_PLACEMENT inline bool is_miss(const float& value){
        unsigned int bits = 0;
        const auto* source = reinterpret_cast<const unsigned char*>(&value);
        auto* destination = reinterpret_cast<unsigned char*>(&bits);
        for(unsigned int i = 0; i < sizeof(value); i++){
            destination[i] = source[i];
        }
        return bits == 0x7f800000u;
    }

    namespace rendering{
        // Read exponent bits through a reference: fast-math may assume a floating-point value parameter is finite before inspecting its bits.
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT bool is_finite(const T& value){
            static_assert(sizeof(T) == 4 || sizeof(T) == 8, "is_finite expects an IEEE-754 single or double");
            using BITS = utils::typing::conditional_t<sizeof(T) == 4, unsigned int, unsigned long long>;
            static_assert(sizeof(BITS) == sizeof(T));
            BITS bits;
            auto* destination = reinterpret_cast<unsigned char*>(&bits);
            const auto* source = reinterpret_cast<const unsigned char*>(&value);
            for(unsigned byte = 0; byte < sizeof(T); byte++){
                destination[byte] = source[byte];
            }
            constexpr BITS EXPONENT = sizeof(T) == 4 ? (BITS)0x7f800000u : (BITS)0x7ff0000000000000ull;
            return (bits & EXPONENT) != EXPONENT;
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT bool is_finite(const T* values, unsigned count){
            for(unsigned i = 0; i < count; i++){
                if(!is_finite(values[i])){
                    return false;
                }
            }
            return true;
        }

        // a part's articulation relative to its parent part (the assembly base for a root part),
        // in the part frame: revolute turns coordinates[coordinate] radians about axis through
        // pivot, prismatic slides coordinates[coordinate] metres along axis; at zero coordinates
        // every part sits at its rest frame
        struct Joint {
            enum class Type { FIXED, REVOLUTE, PRISMATIC };
            Type type = Type::FIXED;
            float axis[3] = {0, 0, 1};
            float pivot[3] = {0, 0, 0};
            size_t coordinate = 0;
        };

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
        using Low = ShadingOptions<false, false, false, true, false, false>;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
