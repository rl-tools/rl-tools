#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SCENE_PROCTHOR_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SCENE_PROCTHOR_OPERATIONS_CPU_H

#include "scene.h"
#include "../../renderer.h"
#include "../../operations_cpu_mux.h"

#include <array>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::scene::procthor {
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

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T frac(T x) {
        return x - std::floor(x);
    }

    template <typename DEVICE, typename SCENE_SPEC, typename RENDERER_SPEC, typename RENDER_DEVICE>
    void precompute_indoor_positions(DEVICE& device, Scene<SCENE_SPEC>& scene, rendering::raytracing::Renderer<RENDERER_SPEC, RENDER_DEVICE>& renderer, typename SCENE_SPEC::T fov, typename SCENE_SPEC::T aspect) {
        using T = typename SCENE_SPEC::T;
        using TI = typename SCENE_SPEC::TI;
        constexpr TI NUM_CAMERAS = RENDERER_SPEC::NUM_CAMERAS;
        constexpr TI NUM_PROBES = RENDERER_SPEC::NUM_PROBES;

        scene.config.scene_center[0] = renderer.scene_center[0];
        scene.config.scene_center[1] = renderer.scene_center[1];
        scene.config.scene_center[2] = renderer.scene_center[2];
        scene.config.scene_radius = renderer.camera_radius;

        const T center[3] = {renderer.scene_center[0], renderer.scene_center[1], renderer.scene_center[2]};
        const T half_extent[3] = {renderer.scene_half_extent[0], renderer.scene_half_extent[1], renderer.scene_half_extent[2]};

        RL_TOOLS_RENDERING_RAYTRACING_LOG("precompute_indoor_positions: center=[" << center[0] << "," << center[1] << "," << center[2] << "] half_extent=[" << half_extent[0] << "," << half_extent[1] << "," << half_extent[2] << "]");
        utils::assert_exit(device, data(collision_results(device, renderer)) != nullptr, "precompute_indoor_positions: collision results buffer is null");

        struct Candidate {
            IndoorPosition<T> position;
            T score;
        };

        constexpr T PI = static_cast<T>(3.14159265358979323846);
        constexpr T MIN_CLEARANCE = static_cast<T>(1.0);
        constexpr TI MAX_TOTAL_TESTED = 4096;
        constexpr TI MAX_BATCHES = (MAX_TOTAL_TESTED + NUM_CAMERAS - 1) / NUM_CAMERAS;
        constexpr TI MIN_REQUIRED_POSITIONS = 50;
        const T max_half = std::max({half_extent[0], half_extent[1], half_extent[2]});
        const T max_dist = max_half > static_cast<T>(1)
            ? static_cast<T>(2) * max_half
            : static_cast<T>(20);
        const T look_ahead = static_cast<T>(1);
        const T search_half_extent[3] = {
            std::max(static_cast<T>(0), half_extent[0] - MIN_CLEARANCE),
            std::max(static_cast<T>(0), half_extent[1] - MIN_CLEARANCE),
            std::max(static_cast<T>(0), half_extent[2] - MIN_CLEARANCE)
        };

        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(MAX_BATCHES) * static_cast<size_t>(NUM_CAMERAS));

        std::array<IndoorPosition<T>, NUM_CAMERAS> batch_positions{};
        std::vector<rendering::raytracing::Camera<T>> camera_staging(NUM_CAMERAS);
        TI total_tested = 0, no_hits = 0, failed_hit_ratio = 0, failed_avg_dist = 0, failed_min_dist = 0;
        T best_min_hit_dist = 0;

        for (TI batch_i = 0; batch_i < MAX_BATCHES && candidates.size() < MIN_REQUIRED_POSITIONS; batch_i++) {
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
                    pos.position[0] + look_ahead * std::cos(pos.yaw),
                    pos.position[1] + look_ahead * std::sin(pos.yaw),
                    pos.position[2]
                };
                const T cam_up[3] = {0, 0, 1};
                camera_staging[camera_i] = make_camera_data(cam_position, cam_look_at, cam_up, fov, aspect);
            }

            copy_to_renderer(device, renderer, camera_staging.data(), data(cameras(device, renderer)), NUM_CAMERAS);
            probe(device, renderer);

            std::vector<rendering::raytracing::CollisionResult> probe_staging((size_t)NUM_CAMERAS * NUM_PROBES);
            copy_from_renderer(device, renderer, data(collision_results(device, renderer)), probe_staging.data(), probe_staging.size());
            const rendering::raytracing::CollisionResult* probe_results = probe_staging.data();
            for (TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++) {
                const rendering::raytracing::CollisionResult* camera_probes = probe_results + static_cast<size_t>(camera_i) * static_cast<size_t>(NUM_PROBES);
                TI hit_count = 0;
                TI very_near_hit_count = 0;
                // finite sentinel: infinity() is undefined behavior under -ffast-math and breaks the min tracking on Apple clang
                T min_hit_dist = std::numeric_limits<T>::max();
                T sum_hit_dist = static_cast<T>(0);

                for (TI probe_i = 0; probe_i < NUM_PROBES; probe_i++) {
                    const rendering::raytracing::CollisionResult& probe = camera_probes[probe_i];
                    if (probe.hit) {
                        const T dist = static_cast<T>(probe.distance);
                        hit_count++;
                        sum_hit_dist += dist;
                        if (dist < min_hit_dist) {
                            min_hit_dist = dist;
                        }
                        if (dist < static_cast<T>(0.25)) {
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
                const bool forward_open = camera_probes[0].hit && forward_dist > static_cast<T>(0.6) && forward_dist < static_cast<T>(6.0);

                const T score =
                    static_cast<T>(2.0) * hit_ratio
                    - static_cast<T>(1.1) * avg_dist_norm
                    - static_cast<T>(0.8) * near_ratio
                    + (forward_open ? static_cast<T>(0.15) : static_cast<T>(0));

                if(hit_ratio <= static_cast<T>(0.72)) failed_hit_ratio++;
                else if(avg_dist_norm >= static_cast<T>(0.45)) failed_avg_dist++;
                else if(min_hit_dist <= MIN_CLEARANCE){ failed_min_dist++; if(min_hit_dist > best_min_hit_dist) best_min_hit_dist = min_hit_dist; }

                const bool indoor_like =
                    hit_ratio > static_cast<T>(0.72) &&
                    avg_dist_norm < static_cast<T>(0.45) &&
                    min_hit_dist > MIN_CLEARANCE;

                if (indoor_like) {
                    auto pos = batch_positions[camera_i];
                    pos.score = score;
                    candidates.push_back({pos, score});
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });

        const TI take_n = std::min(static_cast<TI>(candidates.size()), SCENE_SPEC::MAX_INDOOR_POSITIONS);
        for (TI i = 0; i < take_n; i++) {
            scene.indoor_positions[i] = candidates[i].position;
        }
        scene.num_indoor_positions = take_n;

        RL_TOOLS_RENDERING_RAYTRACING_LOG("precompute_indoor_positions: tested=" << total_tested << " no_hits=" << no_hits << " failed_hit_ratio=" << failed_hit_ratio << " failed_avg_dist=" << failed_avg_dist << " failed_min_dist=" << failed_min_dist << " (best_min_hit=" << best_min_hit_dist << ") accepted=" << scene.num_indoor_positions);
        utils::assert_exit(device, scene.num_indoor_positions > 0, "precompute_indoor_positions: no valid indoor positions found");
    }

    template <typename DEVICE, typename SCENE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT IndoorPosition<typename SCENE_SPEC::T> sample_indoor_position(DEVICE& device, const Scene<SCENE_SPEC>& scene, RNG& rng) {
        using TI = typename SCENE_SPEC::TI;
        if (scene.num_indoor_positions == 0) {
            return {};
        }
        const TI index = random::uniform_int_distribution(device.random, static_cast<TI>(0), static_cast<TI>(scene.num_indoor_positions - 1), rng);
        return scene.indoor_positions[index];
    }

    template <typename DEVICE, typename RENDERER_SPEC, typename RENDER_DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT typename RENDERER_SPEC::T evaluate_clearance(DEVICE& device, rendering::raytracing::Renderer<RENDERER_SPEC, RENDER_DEVICE>& renderer, typename RENDERER_SPEC::TI camera_index) {
        using T = typename RENDERER_SPEC::T;
        constexpr auto NUM_PROBES = RENDERER_SPEC::NUM_PROBES;
        if (data(renderer.collision_results) == nullptr) {
            return std::numeric_limits<T>::max();
        }
        std::vector<rendering::raytracing::CollisionResult> probe_staging(NUM_PROBES);
        copy_from_renderer(device, renderer, data(renderer.collision_results) + (size_t)camera_index * NUM_PROBES, probe_staging.data(), NUM_PROBES);
        const rendering::raytracing::CollisionResult* camera_probes = probe_staging.data();
        T min_dist = std::numeric_limits<T>::max();
        for (typename RENDERER_SPEC::TI probe_i = 0; probe_i < NUM_PROBES; probe_i++) {
            if (camera_probes[probe_i].hit) {
                T dist = static_cast<T>(camera_probes[probe_i].distance);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
        }
        return min_dist;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
