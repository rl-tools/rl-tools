#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_SPECTATOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_SPY_SPECTATOR_H

#include "../../../../devices/cpu.h"
#include "../../../../rendering/raytracing/renderer.h"

#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::spy {
    template <typename SOURCE_CONFIG>
    struct SpectatorConfig: rendering::raytracing::config::Default<typename SOURCE_CONFIG::T, typename SOURCE_CONFIG::TI> {
        using TI = typename SOURCE_CONFIG::TI;
        using SHADING = typename SOURCE_CONFIG::SHADING;
        static constexpr TI CAM_WIDTH = 640;
        static constexpr TI CAM_HEIGHT = 480;
        static constexpr TI NUM_CAMERAS = 1;
        static constexpr TI NUM_PROBES = 0;
        static constexpr TI NUM_OVERLAYS = SOURCE_CONFIG::NUM_OVERLAYS > 0 ? 1 : 0;
        static constexpr TI MAX_OVERLAY_INSTANCES = SOURCE_CONFIG::MAX_OVERLAY_INSTANCES;
        static constexpr TI MAX_OVERLAYS_PER_CAMERA = NUM_OVERLAYS;
    };

    struct OverlaySlot {
        uint32_t object = 0;
        bool active = false;
        float transform[12] = {};
    };

    template <typename T>
    struct SpectatorJob {
        const rendering::raytracing::Scene* scene = nullptr;
        const rendering::raytracing::AssetPool* pool = nullptr;
        rendering::SceneMetadata<T> metadata;
        size_t member = 0;
        size_t instance = 0;
        size_t scene_version = 0;
        rendering::raytracing::Camera<T> camera{};
        std::vector<OverlaySlot> overlay;
        double captured_at = 0;
    };

    struct SpectatorImage {
        std::vector<uint32_t> pixels;
        size_t member = 0;
        size_t instance = 0;
        size_t scene_version = 0;
        double captured_at = 0;
    };

    template <typename T_CONFIG>
    struct Spectator {
        using CONFIG = T_CONFIG;
        std::thread worker;
        std::mutex mutex;
        std::condition_variable wake;
        SpectatorJob<typename CONFIG::T> request;
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
