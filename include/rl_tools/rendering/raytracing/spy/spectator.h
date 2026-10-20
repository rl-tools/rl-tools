#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SPY_SPECTATOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SPY_SPECTATOR_H

#include "../../../devices/cpu.h"
#include "../renderer.h"

#include <array>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::spy {
    template <typename SOURCE_CONFIG>
    struct SpectatorConfig: rendering::raytracing::config::Default<typename SOURCE_CONFIG::T, typename SOURCE_CONFIG::TI> {
        using TI = typename SOURCE_CONFIG::TI;
        using SHADING = typename SOURCE_CONFIG::SHADING;
        static constexpr TI CAM_WIDTH = 640;
        static constexpr TI CAM_HEIGHT = 480;
        static constexpr TI NUM_CAMERAS = 1;
        static constexpr TI NUM_PROBES = 0;
        static constexpr TI NUM_OVERLAYS = SOURCE_CONFIG::NUM_OVERLAYS;
        static constexpr TI MAX_OVERLAY_INSTANCES = SOURCE_CONFIG::MAX_OVERLAY_INSTANCES;
        static constexpr TI MAX_OVERLAYS_PER_CAMERA = SOURCE_CONFIG::MAX_OVERLAYS_PER_CAMERA;
    };

    template <typename T_SPEC>
    struct SpectatorJob {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        const rendering::raytracing::Scene* scene = nullptr;
        const rendering::raytracing::AssetPool* pool = nullptr;
        rendering::SceneMetadata<T> metadata;
        size_t view = 0;
        size_t scene_version = 0;
        rendering::raytracing::Camera<T> camera{};
        std::array<rendering::raytracing::OverlaySnapshot<SPEC>, SPEC::MAX_OVERLAYS_PER_CAMERA> overlays{};
        std::array<rendering::raytracing::OverlayIndex, SPEC::MAX_OVERLAYS_PER_CAMERA> overlay_indices{};
        size_t num_overlays = 0;
        double captured_at = 0;
    };

    struct SpectatorImage {
        std::vector<uint32_t> pixels;
        size_t view = 0;
        size_t scene_version = 0;
        double captured_at = 0;
    };

    template <typename T_CONFIG>
    struct Spectator {
        using CONFIG = T_CONFIG;
        std::thread worker;
        std::mutex mutex;
        std::condition_variable wake;
        SpectatorJob<rendering::raytracing::Specification<CONFIG>> request;
        SpectatorImage result;
        std::exception_ptr error;
        bool stopping = false;
        bool requested = false;
        bool working = false;
        bool ready = false;
    };

    template <typename CONFIG, typename BACKEND = rendering::raytracing::backends::Default>
    struct SpectatorBuffer {
        using SPEC = rendering::raytracing::Specification<CONFIG>;
        devices::DefaultCPU device;
        rendering::raytracing::Renderer<SPEC, BACKEND> renderer;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
