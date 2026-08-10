#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_CASES_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_CASES_H

#include <rl_tools/rendering/raytracing/renderer.h>

namespace golden {
    template <typename T>
    struct Pose {
        const char* id; // subfolder name for golden and backend-frame outputs
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
        static constexpr TI NUM_CAMERAS = 12;
        static constexpr TI NUM_PROBES = 64;

        // FLU frame. Scene: ProcTHOR-Train-1.glb, bbox x [-15.9, 0], y [-17.6, 0.1], z [0, 2.65].
        // Pose 00: exterior overview; pose 01: interior (interactive viewer pos [-6.28, -4.18, 1.99] quat wxyz [0.954, 0.035, 0.124, -0.270]); pose 02: pose 01 yawed +90 deg; pose 03: floor close-up.
        // Poses 04-11: recorded navigation poses; look_at/up derived from the recorded quaternion (forward_body = (1, 0, 0), up_body = (0, 0, 1)).
        static constexpr Pose<T> POSES[NUM_CAMERAS] = {
            {"00", {  3.4,   2.6,  9.2 }, { -7.94,  -8.73,   1.30  }, {0,      0,      1     }},
            {"01", { -6.28, -4.18, 1.99}, { -5.4566, -4.6867, 1.7344}, {0.2178, -0.1338, 0.9668}},
            {"02", { -6.28, -4.18, 1.99}, { -5.7733, -3.3566, 1.7344}, {0.1338,  0.2178, 0.9668}},
            {"03", {-10.0,  -1.5,  1.2 }, { -9.42,  -0.92,   0.62  }, {0,      0,      1     }},
            {"04", {-8.623972, -13.78547, 1.378925}, {-7.87335, -14.38384, 1.098727}, {0.2190995, -0.1746609, 0.9599422}}, // yaw=5.61016846 pitch=-0.284000546 quat_wxyz=(0.934412777,0.0467301309,0.133585945,-0.326869905)
            {"05", {-8.851608, -14.26024, 1.378925}, {-9.560421, -14.81557, 0.9439588}, {-0.3423963, -0.2682541, 0.9004468}}, // yaw=3.80616713 pitch=-0.450000554 quat_wxyz=(-0.317983687,-0.210902423,-0.0727787316,0.921471536)
            {"06", {-9.23895, -14.42166, 1.378925}, {-9.527512, -13.48674, 1.172421}, {-0.06090194, 0.1973191, 0.9784458}}, // yaw=1.87016726 pitch=-0.208000466 quat_wxyz=(0.590543032,-0.0835328847,0.0616390146,0.800301135)
            {"07", {-6.75248, -11.95607, 1.378925}, {-5.834495, -11.78127, 1.022909}, {0.3497314, 0.06659576, 0.9344801}}, // yaw=0.18816705 pitch=-0.364000559 quat_wxyz=(0.979134083,-0.0170037393,0.180196688,0.0923931599)
            {"08", {-5.578073, -11.17941, 1.378925}, {-6.497158, -10.9725, 1.043552}, {-0.3271845, 0.07365467, 0.9420856}}, // yaw=2.92016721 pitch=-0.342000574 quat_wxyz=(0.10887526,-0.169126302,0.0188013148,0.979381979)
            {"09", {-6.62411, -8.327263, 1.378925}, {-6.43543, -7.359907, 1.209742}, {0.0323884, 0.1660538, 0.9855847}}, // yaw=1.37816715 pitch=-0.170000598 quat_wxyz=(0.769042492,-0.0539806932,0.0655267239,0.633534551)
            {"10", {-6.529247, -6.471247, 1.378925}, {-7.501519, -6.577228, 1.170464}, {-0.2072329, -0.02258914, 0.9780308}}, // yaw=3.25016761 pitch=-0.210000545 quat_wxyz=(-0.0539619774,-0.104653038,-0.00568693737,0.993027449)
            {"11", {-6.843887, -4.858216, 1.378925}, {-6.422187, -3.968547, 1.203832}, {0.07499529, 0.1582194, 0.9845518}}  // yaw=1.12816787 pitch=-0.176000521 quat_wxyz=(0.841808677,-0.0469879285,0.074271217,0.532573104)
        };
        static constexpr T MOTION_BLUR_DELTA[3] = {0.2, 0.1, 0};

        // dynamic-motion-blur overlay drive: a cube in the pose-01 living room (free space per
        // the scene AABBs), translated and spun across the shutter — slerp-exact (< 180 deg)
        static constexpr T OVERLAY_HALF_EXTENT = 0.45;
        static constexpr T OVERLAY_POSITION_CLOSE[3] = {-5.0, -5.2, 1.5};
        static constexpr T OVERLAY_POSITION_OPEN[3] = {-5.35, -5.45, 1.35};
        static constexpr T OVERLAY_SPIN_CLOSE = 0.6;
        static constexpr T OVERLAY_SPIN_OPEN = -0.6;

        template <typename T_SHADING, bool T_ENABLE_MOTION_BLUR = false, TI T_MOTION_BLUR_SAMPLES = 1, bool T_ENABLE_ANTI_ALIASING = false, TI T_ANTI_ALIASING_GRID_SIZE = 1, bool T_OUTPUT_DEPTH = false>
        struct Config: rl_tools::rendering::raytracing::config::Default<T, TI>{
            static constexpr TI CAM_WIDTH = Cases::CAM_WIDTH, CAM_HEIGHT = Cases::CAM_HEIGHT, NUM_CAMERAS = Cases::NUM_CAMERAS, NUM_PROBES = Cases::NUM_PROBES;
            using SHADING = T_SHADING;
            static constexpr bool OUTPUT_DEPTH = T_OUTPUT_DEPTH;
            static constexpr bool ENABLE_MOTION_BLUR = T_ENABLE_MOTION_BLUR;
            static constexpr TI MOTION_BLUR_SAMPLES = T_MOTION_BLUR_SAMPLES;
            static constexpr bool ENABLE_ANTI_ALIASING = T_ENABLE_ANTI_ALIASING;
            static constexpr TI ANTI_ALIASING_GRID_SIZE = T_ANTI_ALIASING_GRID_SIZE;
        };
        template <typename T_SHADING, bool T_ENABLE_MOTION_BLUR = false, TI T_MOTION_BLUR_SAMPLES = 1, bool T_ENABLE_ANTI_ALIASING = false, TI T_ANTI_ALIASING_GRID_SIZE = 1, bool T_OUTPUT_DEPTH = false>
        using Specification = rl_tools::rendering::raytracing::Specification<Config<T_SHADING, T_ENABLE_MOTION_BLUR, T_MOTION_BLUR_SAMPLES, T_ENABLE_ANTI_ALIASING, T_ANTI_ALIASING_GRID_SIZE, T_OUTPUT_DEPTH>>;

        template <typename T_SHADING, TI T_MOTION_BLUR_SAMPLES>
        struct DynamicOverlayConfig: Config<T_SHADING, true, T_MOTION_BLUR_SAMPLES>{
            static constexpr TI NUM_OVERLAYS = 1;
            static constexpr TI MAX_OVERLAY_INSTANCES = 4;
            static constexpr TI MAX_OVERLAYS_PER_CAMERA = 1;
            static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = true;
        };

        using LOW_RGB = Specification<rl_tools::rendering::raytracing::Low>;
        using MEDIUM_RGB = Specification<rl_tools::rendering::raytracing::Medium>;
        using HIGH_RGB = Specification<rl_tools::rendering::raytracing::High>;
        using VERY_HIGH_RGB = Specification<rl_tools::rendering::raytracing::VeryHigh>;
        using HIGH_RGB_AA2 = Specification<rl_tools::rendering::raytracing::High, false, 1, true, 2>;
        using HIGH_RGB_MB4 = Specification<rl_tools::rendering::raytracing::High, true, 4>;
        using HIGH_RGB_MB4_DYNAMIC = rl_tools::rendering::raytracing::Specification<DynamicOverlayConfig<rl_tools::rendering::raytracing::High, 4>>;
        using LOW_RGBD = Specification<rl_tools::rendering::raytracing::Low, false, 1, false, 1, true>;
        using HIGH_RGBD = Specification<rl_tools::rendering::raytracing::High, false, 1, false, 1, true>;
    };
}

#endif
