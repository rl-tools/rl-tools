#pragma once

#include <algorithm>
#include <cmath>

namespace rl_tools::rendering::raytracing::interactive {
    struct InputState {
        bool forward = false, backward = false, left = false, right = false;
        bool up = false, down = false;
        float yaw = 0;
        float pitch = 0;
        bool quit = false;
        bool capture_pose = false;
        bool cycle_view = false;
    };

    inline void look(InputState& input, float dx, float dy) {
        constexpr float sensitivity = 0.002f;
        constexpr float max_pitch = static_cast<float>(M_PI) * 0.49f;
        input.yaw -= dx * sensitivity;
        input.pitch = std::min(std::max(input.pitch - dy * sensitivity, -max_pitch), max_pitch);
    }
}
