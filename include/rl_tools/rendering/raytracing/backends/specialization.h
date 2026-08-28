#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_SPECIALIZATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_BACKENDS_SPECIALIZATION_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rendering::raytracing::backends{
        // single source of the shader specialization/function-constant values consumed by the
        // Vulkan, Metal, and WebGPU pipelines: the derivation logic lives once so the backends
        // cannot drift (binding indices and constant names stay per backend)
        template <typename SPEC>
        struct SpecializationConstants{
            static constexpr bool SRGB_OUTPUT = SPEC::SHADING::SRGB_OUTPUT;
            static constexpr bool MOTION_BLUR = SPEC::ENABLE_MOTION_BLUR;
            static constexpr int MOTION_SAMPLES = SPEC::ENABLE_MOTION_BLUR ? (int)SPEC::MOTION_BLUR_SAMPLES : 1;
            static constexpr int AA_GRID = SPEC::ENABLE_ANTI_ALIASING ? (int)SPEC::ANTI_ALIASING_GRID_SIZE : 1;
            static constexpr bool CHECKER_BACKGROUND = SPEC::SHADING::CHECKER_BACKGROUND;
            static constexpr bool LOAD_TEXTURES = SPEC::SHADING::LOAD_TEXTURES;
            static constexpr bool NORMAL_SHADING = SPEC::SHADING::NORMAL_SHADING;
            static constexpr bool METALLIC_REFLECTIONS = SPEC::SHADING::METALLIC_REFLECTIONS;
            static constexpr bool PBR_SHADING = SPEC::SHADING::PBR_SHADING;
            static constexpr bool PUNCTUAL_LIGHT_SHADOWS = SPEC::SHADING::PUNCTUAL_LIGHT_SHADOWS;
            static constexpr int OVERLAY_COUNT = SPEC::ENABLE_OVERLAYS ? (int)SPEC::MAX_OVERLAYS_PER_CAMERA : 0;
            static constexpr bool SEMANTIC_SEGMENTATION = SPEC::SEMANTIC_SEGMENTATION;
            static constexpr bool HAS_OBSERVATION = SPEC::HAS_OBSERVATION;
            static constexpr bool DYNAMIC_MOTION_BLUR = SPEC::ENABLE_DYNAMIC_MOTION_BLUR;
            static constexpr bool RESOLVE_RGB = SPEC::ENABLE_DYNAMIC_MOTION_BLUR && SPEC::HAS_RGB;
            static constexpr bool RESOLVE_DEPTH = SPEC::ENABLE_DYNAMIC_MOTION_BLUR && SPEC::HAS_DEPTH;
            // WGSL-only consumers (the other shader backends derive these in-shader)
            static constexpr int NUM_OVERLAYS = (int)SPEC::NUM_OVERLAYS;
            static constexpr int OVERLAY_CAPACITY = (int)SPEC::MAX_OVERLAY_INSTANCES;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
