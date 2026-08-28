#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_ANNOTATIONS_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_ANNOTATIONS_OPERATIONS_CPU_H

#include "free_space.h"
#include "../operations_cpu.h"
#include "../../camera.h"
#include "../../../containers/tensor/tensor.h"

#include <array>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::annotations {
    template <typename T, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT T radical_inverse(TI n, TI base) {
        T inv_base = static_cast<T>(1) / static_cast<T>(base);
        T inv = inv_base;
        T result = static_cast<T>(0);
        while (n > 0) {
            const TI digit = n % base;
            result += static_cast<T>(digit) * inv;
            inv *= inv_base;
            n /= base;
        }
        return result;
    }

    // probe-driven free-space annotation: Halton-sampled candidate poses inside the scene AABB,
    // scored by the probe rays' hit statistics. PROBE is any probe-capable renderer — injected so
    // the datasets layer stays independent of the raytracing implementation. Consumes the probe
    // target's camera tensor (candidate batches are staged into it); producers that rely on
    // persistent cameras must rewrite them after annotation.
    template <typename DEVICE, typename SPEC, typename METADATA_T, typename PROBE, typename PARAMETERS_T, typename PARAMETERS_TI>
    void annotate(DEVICE& device, FreeSpace<SPEC>& annotations, const rendering::SceneMetadata<METADATA_T>& metadata, PROBE& probe_target, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI NUM_CAMERAS = PROBE::SPEC::NUM_CAMERAS;
        constexpr TI NUM_PROBES = PROBE::SPEC::NUM_PROBES;

        const T center[3] = {(T)metadata.center[0], (T)metadata.center[1], (T)metadata.center[2]};
        const T half_extent[3] = {(T)metadata.half_extent[0], (T)metadata.half_extent[1], (T)metadata.half_extent[2]};

        RL_TOOLS_RENDERING_DATASETS_LOG("annotations::free_space: center=[" << center[0] << "," << center[1] << "," << center[2] << "] half_extent=[" << half_extent[0] << "," << half_extent[1] << "," << half_extent[2] << "]");
        utils::assert_exit(device, data(collision_results(device, probe_target)) != nullptr, "annotations::free_space: collision results buffer is null");
        utils::assert_exit(device, parameters.fov > 0 && parameters.aspect > 0, "annotations::free_space: parameters.fov and parameters.aspect must be set");

        struct Candidate {
            FreeSpacePosition<T> position;
            T score;
        };

        constexpr T PI = static_cast<T>(3.14159265358979323846);
        const T min_clearance = (T)parameters.min_clearance;
        const TI max_batches = ((TI)parameters.max_candidates_tested + NUM_CAMERAS - 1) / NUM_CAMERAS;
        const T max_half = std::max({half_extent[0], half_extent[1], half_extent[2]});
        const T max_dist = max_half > static_cast<T>(1)
            ? static_cast<T>(2) * max_half
            : static_cast<T>(20);
        const T search_half_extent[3] = {
            std::max(static_cast<T>(0), half_extent[0] - min_clearance),
            std::max(static_cast<T>(0), half_extent[1] - min_clearance),
            std::max(static_cast<T>(0), half_extent[2] - min_clearance)
        };

        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(max_batches) * static_cast<size_t>(NUM_CAMERAS));

        std::array<FreeSpacePosition<T>, NUM_CAMERAS> batch_positions{};
        Tensor<typename decltype(probe_target.cameras)::SPEC> camera_staging;
        malloc(device, camera_staging);
        Tensor<typename decltype(probe_target.collision_results)::SPEC> probe_staging;
        malloc(device, probe_staging);
        TI total_tested = 0, no_hits = 0, failed_hit_ratio = 0, failed_avg_dist = 0, failed_min_dist = 0;
        T best_min_hit_dist = 0;

        for (TI batch_i = 0; batch_i < max_batches && candidates.size() < (size_t)parameters.min_required_positions; batch_i++) {
            for (TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++) {
                const TI candidate_i = batch_i * NUM_CAMERAS + camera_i + 1;
                const T hx = radical_inverse<T>(candidate_i, static_cast<TI>(2));
                const T hy = radical_inverse<T>(candidate_i, static_cast<TI>(3));
                const T hz = radical_inverse<T>(candidate_i, static_cast<TI>(5));
                const T hw = radical_inverse<T>(candidate_i, static_cast<TI>(7));

                const T yaw = static_cast<T>(2) * PI * hw;

                auto& pos = batch_positions[camera_i];
                pos.position[0] = center[0] + (hx * static_cast<T>(2) - static_cast<T>(1)) * search_half_extent[0];
                pos.position[1] = center[1] + (hy * static_cast<T>(2) - static_cast<T>(1)) * search_half_extent[1];
                pos.position[2] = center[2] + (hz * static_cast<T>(2) - static_cast<T>(1)) * search_half_extent[2];
                pos.yaw = yaw;
                pos.score = static_cast<T>(0);

                const T cam_position[3] = {pos.position[0], pos.position[1], pos.position[2]};
                const T cam_look_at[3] = {
                    pos.position[0] + (T)parameters.look_ahead * std::cos(pos.yaw),
                    pos.position[1] + (T)parameters.look_ahead * std::sin(pos.yaw),
                    pos.position[2]
                };
                const T cam_up[3] = {0, 0, 1};
                set(device, camera_staging, make_camera_data(cam_position, cam_look_at, cam_up, (T)parameters.fov, (T)parameters.aspect), camera_i);
            }

            copy(device, probe_target.device, camera_staging, cameras(device, probe_target));
            probe(device, probe_target);

            copy(probe_target.device, device, collision_results(device, probe_target), probe_staging);
            const rendering::CollisionResult* probe_results = data(probe_staging);
            for (TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++) {
                const rendering::CollisionResult* camera_probes = probe_results + static_cast<size_t>(camera_i) * static_cast<size_t>(NUM_PROBES);
                TI hit_count = 0;
                TI very_near_hit_count = 0;
                // finite sentinel: infinity() is undefined behavior under -ffast-math and breaks the min tracking on Apple clang
                T min_hit_dist = std::numeric_limits<T>::max();
                T sum_hit_dist = static_cast<T>(0);

                for (TI probe_i = 0; probe_i < NUM_PROBES; probe_i++) {
                    const rendering::CollisionResult& probe_result = camera_probes[probe_i];
                    if (probe_result.hit) {
                        const T dist = static_cast<T>(probe_result.distance);
                        hit_count++;
                        sum_hit_dist += dist;
                        if (dist < min_hit_dist) {
                            min_hit_dist = dist;
                        }
                        if (dist < (T)parameters.near_hit_distance) {
                            very_near_hit_count++;
                        }
                    }
                }

                total_tested++;
                if (hit_count == 0) {
                    no_hits++;
                    continue;
                }

                const T hit_ratio = static_cast<T>(hit_count) / static_cast<T>(NUM_PROBES);
                const T avg_hit_dist = sum_hit_dist / static_cast<T>(hit_count);
                const T avg_dist_norm = avg_hit_dist / max_dist;
                const T near_ratio = static_cast<T>(very_near_hit_count) / static_cast<T>(NUM_PROBES);
                const T forward_dist = static_cast<T>(camera_probes[0].distance);
                const bool forward_open = camera_probes[0].hit && forward_dist > (T)parameters.forward_open_minimum && forward_dist < (T)parameters.forward_open_maximum;

                const T score =
                    (T)parameters.score_weight_hit_ratio * hit_ratio
                    - (T)parameters.score_weight_average_distance * avg_dist_norm
                    - (T)parameters.score_weight_near_ratio * near_ratio
                    + (forward_open ? (T)parameters.score_forward_open_bonus : static_cast<T>(0));

                if(hit_ratio <= (T)parameters.hit_ratio_threshold) failed_hit_ratio++;
                else if(avg_dist_norm >= (T)parameters.average_distance_threshold) failed_avg_dist++;
                else if(min_hit_dist <= min_clearance){ failed_min_dist++; if(min_hit_dist > best_min_hit_dist) best_min_hit_dist = min_hit_dist; }

                const bool free_space_like =
                    hit_ratio > (T)parameters.hit_ratio_threshold &&
                    avg_dist_norm < (T)parameters.average_distance_threshold &&
                    min_hit_dist > min_clearance;

                if (free_space_like) {
                    auto pos = batch_positions[camera_i];
                    pos.score = score;
                    candidates.push_back({pos, score});
                }
            }
        }
        free(device, camera_staging);
        free(device, probe_staging);

        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });

        const TI take_n = std::min(static_cast<TI>(candidates.size()), SPEC::MAX_POSITIONS);
        for (TI i = 0; i < take_n; i++) {
            annotations.positions[i] = candidates[i].position;
        }
        annotations.num_positions = take_n;

        RL_TOOLS_RENDERING_DATASETS_LOG("annotations::free_space: tested=" << total_tested << " no_hits=" << no_hits << " failed_hit_ratio=" << failed_hit_ratio << " failed_avg_dist=" << failed_avg_dist << " failed_min_dist=" << failed_min_dist << " (best_min_hit=" << best_min_hit_dist << ") accepted=" << annotations.num_positions);
        utils::assert_exit(device, annotations.num_positions > 0, "annotations::free_space: no valid positions found");
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT FreeSpacePosition<typename SPEC::T> sample_free_position(DEVICE& device, const FreeSpace<SPEC>& annotations, RNG& rng) {
        using TI = typename SPEC::TI;
        if (annotations.num_positions == 0) {
            return {};
        }
        const TI index = random::uniform_int_distribution(device.random, static_cast<TI>(0), static_cast<TI>(annotations.num_positions - 1), rng);
        return annotations.positions[index];
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
