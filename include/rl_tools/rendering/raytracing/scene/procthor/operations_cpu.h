#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SCENE_PROCTHOR_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SCENE_PROCTHOR_OPERATIONS_CPU_H

#include "scene.h"
#include "../../renderer.h"
#include "../../backends/optix/operations_cuda.h"

#include <array>
#include <cmath>
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

    template <typename DEVICE, typename SCENE_SPEC, typename RENDERER_SPEC>
    void precompute_indoor_positions(DEVICE& device, Scene<SCENE_SPEC>& scene, rendering::raytracing::Renderer<RENDERER_SPEC>& renderer, typename SCENE_SPEC::T eye_height, typename SCENE_SPEC::T cos_fov, typename SCENE_SPEC::T aspect) {
        using T = typename SCENE_SPEC::T;
        using TI = typename SCENE_SPEC::TI;
        constexpr TI NUM_CAMERAS = RENDERER_SPEC::NUM_CAMERAS;
        constexpr TI NUM_PROBES = RENDERER_SPEC::NUM_PROBES;

        scene.config.scene_center[0] = renderer.scene_center[0];
        scene.config.scene_center[1] = renderer.scene_center[1];
        scene.config.scene_center[2] = renderer.scene_center[2];
        scene.config.scene_radius = renderer.camera_radius;

        if (renderer.collision_results_buffer == nullptr) {
            constexpr T PI = static_cast<T>(3.14159265358979323846);
            const T center_x = renderer.scene_center[0];
            const T center_z = renderer.scene_center[2];
            const T search_radius = renderer.camera_radius > static_cast<T>(1)
                ? static_cast<T>(0.95) * renderer.camera_radius
                : static_cast<T>(8);
            const TI take_n = SCENE_SPEC::MAX_INDOOR_POSITIONS < NUM_CAMERAS ? SCENE_SPEC::MAX_INDOOR_POSITIONS : NUM_CAMERAS;
            for (TI i = 0; i < take_n; i++) {
                const T u = radical_inverse<T>(i + 1, static_cast<TI>(2));
                const T v = radical_inverse<T>(i + 1, static_cast<TI>(3));
                const T radius = search_radius * std::sqrt(u);
                const T angle = static_cast<T>(2) * PI * v;

                auto& pos = scene.indoor_positions[i];
                pos.position[0] = center_x + radius * std::cos(angle);
                pos.position[1] = static_cast<T>(0);
                pos.position[2] = center_z + radius * std::sin(angle);
                pos.yaw = static_cast<T>(2) * PI * frac(static_cast<T>(0.61803398875) * static_cast<T>(i + 1));
                pos.score = static_cast<T>(0);
            }
            scene.num_indoor_positions = take_n;
            return;
        }

        struct Candidate {
            IndoorPosition<T> position;
            T score;
        };

        constexpr T PI = static_cast<T>(3.14159265358979323846);
        constexpr TI NUM_BATCHES = 8;
        const T center_x = renderer.scene_center[0];
        const T center_z = renderer.scene_center[2];
        const T search_radius = renderer.camera_radius > static_cast<T>(1)
            ? static_cast<T>(0.95) * renderer.camera_radius
            : static_cast<T>(8);
        const T max_dist = renderer.camera_radius > static_cast<T>(1)
            ? static_cast<T>(2) * renderer.camera_radius
            : static_cast<T>(20);
        const T look_ahead = static_cast<T>(1);

        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(NUM_BATCHES) * static_cast<size_t>(NUM_CAMERAS));

        std::array<CameraData, NUM_CAMERAS> cameras{};
        std::array<IndoorPosition<T>, NUM_CAMERAS> batch_positions{};

        for (TI batch_i = 0; batch_i < NUM_BATCHES; batch_i++) {
            for (TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++) {
                const TI candidate_i = batch_i * NUM_CAMERAS + camera_i + 1;
                const T u = radical_inverse<T>(candidate_i, static_cast<TI>(2));
                const T v = radical_inverse<T>(candidate_i, static_cast<TI>(3));
                const T w = radical_inverse<T>(candidate_i, static_cast<TI>(5));

                const T radius = search_radius * std::sqrt(u);
                const T angle = static_cast<T>(2) * PI * v;
                const T yaw = static_cast<T>(2) * PI * frac(v + static_cast<T>(0.37) * w);

                auto& pos = batch_positions[camera_i];
                pos.position[0] = center_x + radius * std::cos(angle);
                pos.position[1] = static_cast<T>(0);
                pos.position[2] = center_z + radius * std::sin(angle);
                pos.yaw = yaw;
                pos.score = static_cast<T>(0);

                cameras[camera_i] = make_camera_data(
                    owl::vec3f(pos.position[0], pos.position[1] + eye_height, pos.position[2]),
                    owl::vec3f(
                        pos.position[0] + look_ahead * std::cos(pos.yaw),
                        pos.position[1] + eye_height,
                        pos.position[2] + look_ahead * std::sin(pos.yaw)
                    ),
                    owl::vec3f(0.f, 1.f, 0.f),
                    cos_fov,
                    aspect
                );
            }

            set_cameras(device, renderer, cameras.data(), NUM_CAMERAS);
            render(device, renderer);

            const rendering::raytracing::CollisionResult* probe_results = read_collision_results(device, renderer);
            for (TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++) {
                const rendering::raytracing::CollisionResult* camera_probes = probe_results + static_cast<size_t>(camera_i) * static_cast<size_t>(NUM_PROBES);
                TI hit_count = 0;
                TI very_near_hit_count = 0;
                T min_hit_dist = std::numeric_limits<T>::infinity();
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

                if (hit_count == 0) {
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

                const bool indoor_like =
                    hit_ratio > static_cast<T>(0.72) &&
                    avg_dist_norm < static_cast<T>(0.45) &&
                    min_hit_dist > static_cast<T>(0.18);

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

        if (scene.num_indoor_positions == 0) {
            const TI fallback_n = SCENE_SPEC::MAX_INDOOR_POSITIONS < NUM_CAMERAS ? SCENE_SPEC::MAX_INDOOR_POSITIONS : NUM_CAMERAS;
            for (TI i = 0; i < fallback_n; i++) {
                const T u = radical_inverse<T>(i + 1, static_cast<TI>(2));
                const T v = radical_inverse<T>(i + 1, static_cast<TI>(3));
                const T radius = search_radius * std::sqrt(u);
                const T angle = static_cast<T>(2) * PI * v;

                auto& pos = scene.indoor_positions[i];
                pos.position[0] = center_x + radius * std::cos(angle);
                pos.position[1] = static_cast<T>(0);
                pos.position[2] = center_z + radius * std::sin(angle);
                pos.yaw = static_cast<T>(2) * PI * frac(static_cast<T>(0.61803398875) * static_cast<T>(i + 1));
                pos.score = static_cast<T>(0);
            }
            scene.num_indoor_positions = fallback_n;
        }
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

    template <typename DEVICE, typename RENDERER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT typename RENDERER_SPEC::T evaluate_clearance(DEVICE& device, rendering::raytracing::Renderer<RENDERER_SPEC>& renderer, typename RENDERER_SPEC::TI camera_index) {
        using T = typename RENDERER_SPEC::T;
        constexpr auto NUM_PROBES = RENDERER_SPEC::NUM_PROBES;
        const rendering::raytracing::CollisionResult* results = read_collision_results(device, renderer);
        if (results == nullptr) {
            return std::numeric_limits<T>::infinity();
        }
        const rendering::raytracing::CollisionResult* camera_probes = results + static_cast<size_t>(camera_index) * static_cast<size_t>(NUM_PROBES);
        T min_dist = std::numeric_limits<T>::infinity();
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
