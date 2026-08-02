#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_CASES_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_CASES_H

#include <rl_tools/rendering/raytracing/renderer.h>

namespace golden {
    template <typename T>
    struct Pose {
        T position[3];
        T look_at[3];
        T up[3];
    };

    template <typename T_T, typename T_TI>
    struct Cases {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI CAM_WIDTH = 256;
        static constexpr TI CAM_HEIGHT = 256;
        static constexpr TI NUM_CAMERAS = 4;
        static constexpr TI NUM_PROBES = 64;

        // FLU frame. Scene: ProcTHOR-Train-1.glb, bbox x [-15.9, 0], y [-17.6, 0.1], z [0, 2.65].
        // Pose 0: exterior overview; pose 1: interior (interactive viewer pos [-6.28, -4.18, 1.99] quat wxyz [0.954, 0.035, 0.124, -0.270]); pose 2: pose 1 yawed +90 deg; pose 3: floor close-up.
        static constexpr Pose<T> POSES[NUM_CAMERAS] = {
            {{  3.4,   2.6,  9.2 }, { -7.94,  -8.73,   1.30  }, {0,      0,      1     }},
            {{ -6.28, -4.18, 1.99}, { -5.4566, -4.6867, 1.7344}, {0.2178, -0.1338, 0.9668}},
            {{ -6.28, -4.18, 1.99}, { -5.7733, -3.3566, 1.7344}, {0.1338,  0.2178, 0.9668}},
            {{-10.0,  -1.5,  1.2 }, { -9.42,  -0.92,   0.62  }, {0,      0,      1     }}
        };
        static constexpr T MOTION_BLUR_DELTA[3] = {0.2, 0.1, 0};

        template <typename SHADING, bool ENABLE_MOTION_BLUR = false, TI MOTION_BLUR_SAMPLES = 1, bool ENABLE_ANTI_ALIASING = false, TI ANTI_ALIASING_GRID_SIZE = 1, rl_tools::rendering::raytracing::OutputMode OUTPUT_MODE = rl_tools::rendering::raytracing::OutputMode::RGB>
        using Specification = rl_tools::rendering::raytracing::Specification<T, TI, CAM_WIDTH, CAM_HEIGHT, NUM_CAMERAS, NUM_PROBES, SHADING, ENABLE_MOTION_BLUR, MOTION_BLUR_SAMPLES, ENABLE_ANTI_ALIASING, ANTI_ALIASING_GRID_SIZE, OUTPUT_MODE>;

        using LOW_RGB = Specification<rl_tools::rendering::raytracing::Low>;
        using MEDIUM_RGB = Specification<rl_tools::rendering::raytracing::Medium>;
        using HIGH_RGB = Specification<rl_tools::rendering::raytracing::High>;
        using VERY_HIGH_RGB = Specification<rl_tools::rendering::raytracing::VeryHigh>;
        using HIGH_RGB_AA2 = Specification<rl_tools::rendering::raytracing::High, false, 1, true, 2>;
        using HIGH_RGB_MB4 = Specification<rl_tools::rendering::raytracing::High, true, 4>;
        using LOW_RGBD = Specification<rl_tools::rendering::raytracing::Low, false, 1, false, 1, rl_tools::rendering::raytracing::OutputMode::RGBD>;
        using HIGH_RGBD = Specification<rl_tools::rendering::raytracing::High, false, 1, false, 1, rl_tools::rendering::raytracing::OutputMode::RGBD>;
    };
}

#endif
