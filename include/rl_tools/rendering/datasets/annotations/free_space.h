#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_ANNOTATIONS_FREE_SPACE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_ANNOTATIONS_FREE_SPACE_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::annotations {
    template <typename T>
    struct FreeSpacePosition {
        T position[3];
        T yaw;
        T score;
    };

    template <typename T_T, typename T_TI, T_TI T_MAX_POSITIONS = 256>
    struct FreeSpaceSpecification {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI MAX_POSITIONS = T_MAX_POSITIONS;
    };

    // task-facing, graphics-agnostic free-space poses for spawning. Plain trivially-copyable
    // data — consumers mirror it into device memory for on-device sampling. The annotation
    // product is the interface: producers (the probe-based annotate, later dataset-native
    // sources) are interchangeable behind it.
    template <typename T_SPEC>
    struct FreeSpace {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        FreeSpacePosition<T> positions[SPEC::MAX_POSITIONS];
        TI num_positions = 0;
    };

    // defaults pin the historical constants — annotation output with a default-constructed
    // parameters struct (plus fov/aspect) is bit-identical to the pre-parameterization pass
    template <typename T_T, typename T_TI>
    struct FreeSpaceParameters {
        using T = T_T;
        using TI = T_TI;
        T fov = 0;                                    // degrees; required — no meaningful default
        T aspect = 0;                                 // required
        T min_clearance = 1.0;                        // candidate acceptance + search-volume margin
        TI max_candidates_tested = 4096;
        TI min_required_positions = 50;               // early-out once this many candidates accepted
        T look_ahead = 1.0;                           // yaw-facing look-at distance of the candidate camera
        T hit_ratio_threshold = 0.72;                 // acceptance: fraction of probes that must hit
        T average_distance_threshold = 0.45;          // acceptance: normalized mean hit distance upper bound
        T near_hit_distance = 0.25;                   // hits closer than this count as "near"
        T forward_open_minimum = 0.6;                 // forward probe distance window for the openness bonus
        T forward_open_maximum = 6.0;
        T score_weight_hit_ratio = 2.0;
        T score_weight_average_distance = 1.1;
        T score_weight_near_ratio = 0.8;
        T score_forward_open_bonus = 0.15;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
