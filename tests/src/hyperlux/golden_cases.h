#ifndef TESTS_HYPERLUX_GOLDEN_CASES_H
#define TESTS_HYPERLUX_GOLDEN_CASES_H

// The corpus is view x state x output. The state is what happens between the previous instant
// (shutter open, the flow "previous") and the current one (shutter close): whether the camera
// moves and whether overlays are absent, present or moving. A Case renders one specification
// under one state; every output the specification produces is filed under <view>/<state>/<output>/.

#include <rl_tools/hyperlux/hyperlux.h>

#include <string>
#include <type_traits>

namespace golden {
    enum class Camera { FIXED, MOVING };
    enum class Overlays { NONE, FIXED, MOVING };

    constexpr const char* state_name(Camera camera, Overlays overlays){
        if(overlays == Overlays::NONE){
            return camera == Camera::FIXED ? "static" : "camera";
        }
        if(overlays == Overlays::FIXED){
            return camera == Camera::FIXED ? "overlays" : "camera-overlays";
        }
        return camera == Camera::FIXED ? "objects" : "camera-objects";
    }

    template <typename T>
    struct Pose {
        const char* id;
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
        static constexpr T FOV = 80;

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

        // camera motion: the previous instant is the pose translated by -CAMERA_TRANSLATION and yawed by -CAMERA_YAW about world z
        static constexpr T CAMERA_TRANSLATION[3] = {0.2, 0.1, 0};
        static constexpr T CAMERA_YAW = 0.03;

        // overlays: overlay 0 is a cube in the pose-01 living room (free space per the scene AABBs)
        // attached to every camera; overlay 1 + i is a smaller cube private to camera i, placed in
        // its frustum so that every view sees an object. Motion between the instants is a
        // translation plus a spin about world z (slerp-exact, < 180 deg).
        static constexpr TI NUM_OVERLAYS = 1 + NUM_CAMERAS;
        static constexpr TI MAX_OVERLAY_INSTANCES = 2;
        static constexpr TI MAX_OVERLAYS_PER_CAMERA = 2;
        static constexpr T SHARED_HALF_EXTENT = 0.45;
        static constexpr T SHARED_POSITION_CLOSE[3] = {-5.0, -5.2, 1.5};
        static constexpr T SHARED_POSITION_OPEN[3] = {-5.35, -5.45, 1.35};
        static constexpr T SHARED_SPIN_CLOSE = 0.6;
        static constexpr T SHARED_SPIN_OPEN = -0.6;
        static constexpr T PRIVATE_HALF_EXTENT = 0.12;
        static constexpr T PRIVATE_AHEAD = 0.9;   // along the view axis (pose 03 looks at the floor 1.1 m ahead)
        static constexpr T PRIVATE_ASIDE = -0.25; // along the image horizontal axis, negative is left
        static constexpr T PRIVATE_BELOW = 0.15;  // against the image vertical axis
        static constexpr T PRIVATE_TRANSLATION = 0.12; // the open instant sits this far above the close one
        static constexpr T PRIVATE_SPIN_CLOSE = 0.4;
        static constexpr T PRIVATE_SPIN_OPEN = -0.4;

        template <bool T_OVERLAYS>
        struct Base: rl_tools::hyperlux::config::Default<T, TI>{
            static constexpr TI CAM_WIDTH = Cases::CAM_WIDTH, CAM_HEIGHT = Cases::CAM_HEIGHT, NUM_CAMERAS = Cases::NUM_CAMERAS, NUM_PROBES = Cases::NUM_PROBES;
            static constexpr TI NUM_OVERLAYS = T_OVERLAYS ? Cases::NUM_OVERLAYS : 0;
            static constexpr TI MAX_OVERLAY_INSTANCES = T_OVERLAYS ? Cases::MAX_OVERLAY_INSTANCES : 0;
            static constexpr TI MAX_OVERLAYS_PER_CAMERA = T_OVERLAYS ? Cases::MAX_OVERLAYS_PER_CAMERA : 0;
        };
        template <typename T_SHADING, TI T_MOTION_BLUR_SAMPLES = 1, TI T_ANTI_ALIASING_GRID_SIZE = 1, bool T_OUTPUT_DEPTH = false, bool T_OVERLAYS = false>
        struct RGBConfig: Base<T_OVERLAYS>{
            using SHADING = T_SHADING;
            using OUTPUTS = typename rl_tools::hyperlux::outputs::Concat<
                rl_tools::hyperlux::outputs::List<rl_tools::hyperlux::outputs::RGB<>>,
                rl_tools::hyperlux::outputs::If<(T_OUTPUT_DEPTH), rl_tools::hyperlux::outputs::Depth<>>,
                rl_tools::hyperlux::outputs::List<rl_tools::hyperlux::outputs::Probes>
            >::type;
            static constexpr bool ENABLE_MOTION_BLUR = T_MOTION_BLUR_SAMPLES > 1;
            static constexpr TI MOTION_BLUR_SAMPLES = T_MOTION_BLUR_SAMPLES;
            static constexpr bool ENABLE_DYNAMIC_MOTION_BLUR = T_OVERLAYS && T_MOTION_BLUR_SAMPLES > 1;
            static constexpr bool ENABLE_ANTI_ALIASING = T_ANTI_ALIASING_GRID_SIZE > 1;
            static constexpr TI ANTI_ALIASING_GRID_SIZE = T_ANTI_ALIASING_GRID_SIZE;
        };
        // normals and segmentation are single-sample and independent of shading, AA and MB; the
        // segmentation-capable cases load the scene as one instance per GLB root node (ids pinned
        // by node order) instead of the welded single object the other cases use
        template <bool T_OVERLAYS>
        struct GeometryConfig: Base<T_OVERLAYS>{
            using SHADING = rl_tools::hyperlux::Low;
            using OUTPUTS = rl_tools::hyperlux::outputs::List<
                rl_tools::hyperlux::outputs::Normals<>,
                rl_tools::hyperlux::outputs::Segmentation<>,
                rl_tools::hyperlux::outputs::Probes>;
        };
        template <bool T_OVERLAYS>
        struct SemanticConfig: GeometryConfig<T_OVERLAYS>{
            using OUTPUTS = rl_tools::hyperlux::outputs::List<
                rl_tools::hyperlux::outputs::Segmentation<rl_tools::hyperlux::outputs::format::UInt32, rl_tools::hyperlux::outputs::labels::Semantic>,
                rl_tools::hyperlux::outputs::Probes>;
        };
        template <bool T_OVERLAYS>
        struct FlowConfig: Base<T_OVERLAYS>{
            using SHADING = rl_tools::hyperlux::Low;
            using OUTPUTS = rl_tools::hyperlux::outputs::List<
                rl_tools::hyperlux::outputs::Flow<>,
                rl_tools::hyperlux::outputs::FlowStatus<>,
                rl_tools::hyperlux::outputs::Flow<rl_tools::hyperlux::outputs::direction::Forward>,
                rl_tools::hyperlux::outputs::FlowStatus<rl_tools::hyperlux::outputs::direction::Forward>,
                rl_tools::hyperlux::outputs::Probes>;
        };
        template <typename CONFIG>
        using Spec = rl_tools::hyperlux::Specification<CONFIG>;

        // PROBES: the case that also files the collision probes (single-sample, shading independent)
        template <typename T_SPEC, Camera T_CAMERA, Overlays T_OVERLAYS, bool T_PROBES = false>
        struct Case {
            using SPEC = T_SPEC;
            static constexpr Camera CAMERA = T_CAMERA;
            static constexpr Overlays OVERLAYS = T_OVERLAYS;
            static constexpr bool PROBES = T_PROBES;
            static constexpr const char* STATE = state_name(CAMERA, OVERLAYS);
            static_assert((OVERLAYS != Overlays::NONE) == SPEC::ENABLE_OVERLAYS, "overlays are enabled exactly in the overlay states");
            static_assert(CAMERA == Camera::FIXED || SPEC::HAS_CAMERA_PAIR || SPEC::HAS_FLOW, "camera motion needs a two-instant output");
            static_assert(OVERLAYS != Overlays::MOVING || SPEC::ENABLE_DYNAMIC_MOTION_BLUR || SPEC::HAS_FLOW_OVERLAYS, "moving overlays need a two-instant output");
        };

        template <typename... T_CASES>
        struct List {};

        using Low = rl_tools::hyperlux::Low;
        using Medium = rl_tools::hyperlux::Medium;
        using High = rl_tools::hyperlux::High;
        using VeryHigh = rl_tools::hyperlux::VeryHigh;
        using ALL = List<
            Case<Spec<RGBConfig<Low>>, Camera::FIXED, Overlays::NONE, true>,
            Case<Spec<RGBConfig<Medium>>, Camera::FIXED, Overlays::NONE>,
            Case<Spec<RGBConfig<High, 1, 1, true>>, Camera::FIXED, Overlays::NONE>,
            Case<Spec<RGBConfig<VeryHigh>>, Camera::FIXED, Overlays::NONE>,
            Case<Spec<RGBConfig<High, 1, 2>>, Camera::FIXED, Overlays::NONE>,
            Case<Spec<GeometryConfig<false>>, Camera::FIXED, Overlays::NONE>,
            Case<Spec<SemanticConfig<false>>, Camera::FIXED, Overlays::NONE>,
            Case<Spec<RGBConfig<High, 4>>, Camera::MOVING, Overlays::NONE>,
            Case<Spec<FlowConfig<false>>, Camera::MOVING, Overlays::NONE>,
            Case<Spec<RGBConfig<High, 1, 1, true, true>>, Camera::FIXED, Overlays::FIXED>,
            Case<Spec<GeometryConfig<true>>, Camera::FIXED, Overlays::FIXED>,
            Case<Spec<SemanticConfig<true>>, Camera::FIXED, Overlays::FIXED>,
            Case<Spec<RGBConfig<High, 4, 1, false, true>>, Camera::MOVING, Overlays::FIXED>,
            Case<Spec<FlowConfig<true>>, Camera::MOVING, Overlays::FIXED>,
            Case<Spec<RGBConfig<High, 4, 1, false, true>>, Camera::FIXED, Overlays::MOVING>,
            Case<Spec<FlowConfig<true>>, Camera::FIXED, Overlays::MOVING>,
            Case<Spec<RGBConfig<High, 4, 1, false, true>>, Camera::MOVING, Overlays::MOVING>,
            Case<Spec<FlowConfig<true>>, Camera::MOVING, Overlays::MOVING>
        >;

        // the RGB output directory: <tier>[-aa<grid>][-mb<samples>]
        template <typename SPEC>
        static std::string rgb_output(){
            using SHADING = typename SPEC::SHADING;
            static_assert(std::is_same_v<SHADING, Low> || std::is_same_v<SHADING, Medium> || std::is_same_v<SHADING, High> || std::is_same_v<SHADING, VeryHigh>, "unnamed shading tier");
            std::string name = std::is_same_v<SHADING, Low> ? "low" : std::is_same_v<SHADING, Medium> ? "medium" : std::is_same_v<SHADING, High> ? "high" : "very-high";
            if constexpr(SPEC::ENABLE_ANTI_ALIASING){
                name += "-aa" + std::to_string(SPEC::ANTI_ALIASING_GRID_SIZE);
            }
            if constexpr(SPEC::ENABLE_MOTION_BLUR){
                name += "-mb" + std::to_string(SPEC::MOTION_BLUR_SAMPLES);
            }
            return name;
        }

        // <state>/<primary output>, for logs and test names
        template <typename CASE>
        static std::string name(){
            using SPEC = typename CASE::SPEC;
            std::string primary;
            if constexpr(SPEC::HAS_RGB){
                primary = rgb_output<SPEC>();
            }
            else if constexpr(SPEC::HAS_FLOW){
                primary = "flow";
            }
            else if constexpr(SPEC::HAS_SEMANTIC_SEGMENTATION){
                primary = "segmentation-semantic";
            }
            else{
                primary = "geometry";
            }
            return std::string(CASE::STATE) + "/" + primary;
        }

        // single-camera pseudo-spec for the per-pose grid writers (detail::write_*_grid_png)
        struct SingleCamera{
            using TI = Cases::TI;
            static constexpr TI NUM_CAMERAS = 1;
            static constexpr TI CAM_WIDTH = Cases::CAM_WIDTH;
            static constexpr TI CAM_HEIGHT = Cases::CAM_HEIGHT;
            static constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
            static constexpr TI GRID_COLS = 1;
            static constexpr TI GRID_ROWS = 1;
        };
    };
}

#endif
