// exercised by test_spec_header_escape_hatch: a user-pinned World specification reached
// through EnvConfig(spec_header=...) — the header includes what it needs and defines
// hyperdrone_env_user::WORLD
#pragma once
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <cstddef>

namespace hyperdrone_env_user {
    using T = float;
    using TI = std::size_t;
    struct SPEC: rl_tools::rl::environments::hyperdrone::presets::X500FPV<T, TI> {
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI INSTANCES_PER_WORLD = 2;
        static constexpr TI MAX_DRONES = 3;
        static constexpr TI CAM_WIDTH = 16;
        static constexpr TI CAM_HEIGHT = 16;
        using SHADING = rl_tools::rendering::raytracing::Low;
    };
    using WORLD = rl_tools::rl::environments::hyperdrone::World<SPEC>;
}
