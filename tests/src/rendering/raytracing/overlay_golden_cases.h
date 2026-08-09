#ifndef TESTS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_CASES_H
#define TESTS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_CASES_H

#include "overlay_scenario_cases.h"

#include <array>
#include <cstddef>

namespace overlay_goldens {
    struct View {
        const char* id;
        float position[3];
        float look_at[3];
        float up[3];
    };

    static constexpr std::array<View, 6> VIEWS = {{
        {"centered", {0.0f, 0.0f, 0.0f}, {4.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {"left_offset", {0.0f, -0.8f, 0.2f}, {4.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {"right_offset", {0.0f, 0.8f, -0.2f}, {4.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {"elevated", {0.0f, 0.0f, 1.2f}, {4.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {"close_low", {1.0f, 0.0f, -0.6f}, {4.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
        {"oblique", {0.0f, -0.6f, 0.8f}, {4.0f, 0.5f, -0.2f}, {0.05f, 0.0f, 0.9987f}},
    }};

    enum class CaptureState {
        INITIAL,
        UPDATED,
    };

    inline const char* capture_state_id(CaptureState state){
        return state == CaptureState::INITIAL ? "initial" : "updated";
    }

    inline std::size_t capture_state_count(overlay_scenarios::Scenario scenario){
        return scenario == overlay_scenarios::Scenario::SHARED_SCENE_NO_DYNAMIC ? 1 : 2;
    }
}

#endif
