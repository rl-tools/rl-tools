#pragma once
// the renderer configuration of this JIT module, shared by the bindings (renderer.cpp) and the
// device kernels (set_pose_device.cu)
#ifndef HYPERLUX_NUM_PROBES
#define HYPERLUX_NUM_PROBES 1
#endif
#include "verbosity.h"
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/hyperlux/operations_cpu_mux.h>
#include HYPERLUX_CAMERA_CONFIG_HEADER

#ifndef HYPERLUX_WIDTH
#error "HYPERLUX_WIDTH must be defined"
#endif
#ifndef HYPERLUX_HEIGHT
#error "HYPERLUX_HEIGHT must be defined"
#endif
#ifndef HYPERLUX_NUM_CAMERAS
#error "HYPERLUX_NUM_CAMERAS must be defined"
#endif
#ifndef HYPERLUX_SHADING
#define HYPERLUX_SHADING 2
#endif
#ifndef HYPERLUX_MB_SAMPLES
#define HYPERLUX_MB_SAMPLES 1
#endif
#ifndef HYPERLUX_AA_GRID
#define HYPERLUX_AA_GRID 1
#endif
#ifndef HYPERLUX_NUM_OVERLAYS
#define HYPERLUX_NUM_OVERLAYS 0
#endif
#ifndef HYPERLUX_MAX_OVERLAY_INSTANCES
#define HYPERLUX_MAX_OVERLAY_INSTANCES 0
#endif
#ifndef HYPERLUX_MAX_OVERLAYS_PER_CAMERA
#define HYPERLUX_MAX_OVERLAYS_PER_CAMERA 0
#endif
#ifndef HYPERLUX_DYNAMIC_MB
#define HYPERLUX_DYNAMIC_MB 0
#endif

namespace rlt = rl_tools;
namespace rro = rlt::hyperlux::outputs;
namespace rrt = rl_tools::hyperlux;

// Each configuration needs a distinct C++ type in nanobind's shared type registry.
namespace HYPERLUX_MODULE_NAME {
    using DEVICE = rlt::devices::DEVICE_FACTORY<hyperlux::CPUSpecification>;
    using T = float;
    using TI = typename DEVICE::index_t;

    template <int SHADING_ID> struct ShadingSelector;
    template <> struct ShadingSelector<0> { using type = rrt::Low; };
    template <> struct ShadingSelector<1> { using type = rrt::Medium; };
    template <> struct ShadingSelector<2> { using type = rrt::High; };
    template <> struct ShadingSelector<3> { using type = rrt::VeryHigh; };

    static constexpr bool PROBES_ENABLED = rro::Requirements<hyperlux_camera_config::OUTPUTS>::HAS_PROBES;

    struct RendererConfiguration: rrt::config::Default<T, TI>{
        static constexpr TI CAM_WIDTH = HYPERLUX_WIDTH;
        static constexpr TI CAM_HEIGHT = HYPERLUX_HEIGHT;
        static constexpr TI NUM_CAMERAS = HYPERLUX_NUM_CAMERAS;
        static constexpr TI NUM_PROBES = HYPERLUX_NUM_PROBES;
        using SHADING = typename ShadingSelector<HYPERLUX_SHADING>::type;
        using OUTPUTS = hyperlux_camera_config::OUTPUTS;
        static constexpr bool ENABLE_MOTION_BLUR = HYPERLUX_MB_SAMPLES > 1;
        static constexpr TI MOTION_BLUR_SAMPLES = HYPERLUX_MB_SAMPLES;
        static constexpr bool ENABLE_ANTI_ALIASING = HYPERLUX_AA_GRID > 1;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = HYPERLUX_AA_GRID;
        static constexpr TI NUM_OVERLAYS = HYPERLUX_NUM_OVERLAYS;
        static constexpr TI MAX_OVERLAY_INSTANCES = HYPERLUX_MAX_OVERLAY_INSTANCES;
        static constexpr TI MAX_OVERLAYS_PER_CAMERA = HYPERLUX_MAX_OVERLAYS_PER_CAMERA;
        static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = HYPERLUX_DYNAMIC_MB != 0;
    };

    using RENDERER_SPEC = rrt::Specification<RendererConfiguration>;
}
