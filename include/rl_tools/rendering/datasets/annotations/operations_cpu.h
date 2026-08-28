#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_ANNOTATIONS_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_ANNOTATIONS_OPERATIONS_CPU_H

#include "free_space.h"
#include "cache.h"
#include "../operations_cpu.h"
#include "../../camera.h"
#include "../../../containers/tensor/tensor.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
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

    namespace free_space {
        // bump on any output-affecting change to the scan (candidate sequence, probe-direction
        // assumptions, scoring, SceneMetadata consumption) — cache keys include it
        constexpr std::uint32_t VERSION = 1;

        template <typename T>
        struct ScanEntry {
            std::uint64_t candidate;
            FreeSpacePosition<T> position;
        };

        // resumable progress over the deterministic Halton candidate sequence: the scan is
        // candidate-exact (independent of the probe vehicle's batch width), so the accepted list
        // is a pure prefix-monotone function of (artifact, scoring parameters, NUM_PROBES) — a
        // stored scan can be replayed for any smaller request and extended in place for a larger
        // one, bit-identical to a from-scratch run
        template <typename T>
        struct Scan {
            std::uint64_t tested = 0;
            std::vector<ScanEntry<T>> accepted;
        };

        namespace detail {
            inline void append_u32(std::string& out, std::uint32_t value) {
                for (unsigned i = 0; i < 4; i++) {
                    out += static_cast<char>((value >> (8u * i)) & 0xFFu);
                }
            }
            inline void append_u64(std::string& out, std::uint64_t value) {
                for (unsigned i = 0; i < 8; i++) {
                    out += static_cast<char>((value >> (8u * i)) & 0xFFu);
                }
            }
            inline void append_f64(std::string& out, double value) {
                std::uint64_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                append_u64(out, bits);
            }
            inline void write_u32(std::ostream& out, std::uint32_t value) {
                std::string bytes;
                append_u32(bytes, value);
                out.write(bytes.data(), 4);
            }
            inline void write_u64(std::ostream& out, std::uint64_t value) {
                std::string bytes;
                append_u64(bytes, value);
                out.write(bytes.data(), 8);
            }
            inline void write_f64(std::ostream& out, double value) {
                std::uint64_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                write_u64(out, bits);
            }
            inline bool read_u32(std::istream& in, std::uint32_t& value) {
                unsigned char bytes[4];
                if (!in.read(reinterpret_cast<char*>(bytes), 4)) {
                    return false;
                }
                value = 0;
                for (unsigned i = 0; i < 4; i++) {
                    value |= std::uint32_t(bytes[i]) << (8u * i);
                }
                return true;
            }
            inline bool read_u64(std::istream& in, std::uint64_t& value) {
                unsigned char bytes[8];
                if (!in.read(reinterpret_cast<char*>(bytes), 8)) {
                    return false;
                }
                value = 0;
                for (unsigned i = 0; i < 8; i++) {
                    value |= std::uint64_t(bytes[i]) << (8u * i);
                }
                return true;
            }
            inline bool read_f64(std::istream& in, double& value) {
                std::uint64_t bits;
                if (!read_u64(in, bits)) {
                    return false;
                }
                std::memcpy(&value, &bits, sizeof(value));
                return true;
            }
        }

        // scoring identity: everything that determines per-candidate accept/score decisions. The
        // stopping parameters (min_required_positions, max_candidates_tested) are deliberately
        // excluded — they are applied at replay time against the stored scan
        template <typename T, typename METADATA_T, typename PARAMETERS_T, typename PARAMETERS_TI>
        std::string key(const rendering::SceneMetadata<METADATA_T>& metadata, std::uint64_t num_probes, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters) {
            std::string material = "rl_tools.rendering.datasets.annotations.free_space";
            detail::append_u32(material, VERSION);
            detail::append_u32(material, static_cast<std::uint32_t>(sizeof(T)));
            detail::append_u64(material, num_probes);
            for (unsigned i = 0; i < 3; i++) {
                detail::append_f64(material, static_cast<double>(static_cast<T>(metadata.center[i])));
            }
            for (unsigned i = 0; i < 3; i++) {
                detail::append_f64(material, static_cast<double>(static_cast<T>(metadata.half_extent[i])));
            }
            detail::append_f64(material, static_cast<double>(static_cast<T>(metadata.max_ray_length)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.fov)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.aspect)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.min_clearance)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.look_ahead)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.hit_ratio_threshold)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.average_distance_threshold)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.near_hit_distance)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.forward_open_minimum)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.forward_open_maximum)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.score_weight_hit_ratio)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.score_weight_average_distance)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.score_weight_near_ratio)));
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.score_forward_open_bonus)));
            conta::detail::Sha1 sha1;
            sha1.update(reinterpret_cast<const unsigned char*>(material.data()), material.size());
            return sha1.finalize();
        }

        constexpr char BLOB_MAGIC[4] = {'R', 'L', 'F', 'S'};
        constexpr std::uint32_t BLOB_FORMAT_VERSION = 1;

        template <typename T>
        bool read(const std::string& path, Scan<T>& progress) {
            progress.tested = 0;
            progress.accepted.clear();
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                return false;
            }
            char magic[4];
            if (!file.read(magic, 4) || std::memcmp(magic, BLOB_MAGIC, 4) != 0) {
                return false;
            }
            std::uint32_t format_version;
            if (!detail::read_u32(file, format_version) || format_version != BLOB_FORMAT_VERSION) {
                return false;
            }
            std::uint64_t count;
            if (!detail::read_u64(file, progress.tested) || !detail::read_u64(file, count) || count > progress.tested) {
                return false;
            }
            progress.accepted.reserve(count);
            std::uint64_t previous_candidate = 0;
            for (std::uint64_t entry_i = 0; entry_i < count; entry_i++) {
                ScanEntry<T> entry;
                double values[5];
                if (!detail::read_u64(file, entry.candidate)) {
                    return false;
                }
                for (double& value : values) {
                    if (!detail::read_f64(file, value)) {
                        return false;
                    }
                }
                if (entry.candidate <= previous_candidate || entry.candidate > progress.tested) {
                    return false;
                }
                previous_candidate = entry.candidate;
                entry.position.position[0] = static_cast<T>(values[0]);
                entry.position.position[1] = static_cast<T>(values[1]);
                entry.position.position[2] = static_cast<T>(values[2]);
                entry.position.yaw = static_cast<T>(values[3]);
                entry.position.score = static_cast<T>(values[4]);
                progress.accepted.push_back(entry);
            }
            return true;
        }

        template <typename T>
        bool write(const std::string& path, const Scan<T>& progress) {
            const std::string temporary = path + ".part." + std::to_string(conta::detail::process_id());
            {
                std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
                if (!file.is_open()) {
                    return false;
                }
                file.write(BLOB_MAGIC, 4);
                detail::write_u32(file, BLOB_FORMAT_VERSION);
                detail::write_u64(file, progress.tested);
                detail::write_u64(file, static_cast<std::uint64_t>(progress.accepted.size()));
                for (const ScanEntry<T>& entry : progress.accepted) {
                    detail::write_u64(file, entry.candidate);
                    detail::write_f64(file, static_cast<double>(entry.position.position[0]));
                    detail::write_f64(file, static_cast<double>(entry.position.position[1]));
                    detail::write_f64(file, static_cast<double>(entry.position.position[2]));
                    detail::write_f64(file, static_cast<double>(entry.position.yaw));
                    detail::write_f64(file, static_cast<double>(entry.position.score));
                }
                if (!file.good()) {
                    return false;
                }
            }
            std::error_code error_code;
            std::filesystem::rename(temporary, path, error_code);
            if (error_code) {
                std::filesystem::remove(temporary, error_code);
                return false;
            }
            return true;
        }

        template <typename T, typename PARAMETERS_T, typename PARAMETERS_TI>
        bool sufficient(const Scan<T>& progress, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters) {
            return progress.tested >= static_cast<std::uint64_t>(parameters.max_candidates_tested)
                || progress.accepted.size() >= static_cast<std::size_t>(parameters.min_required_positions);
        }

        // probe-driven candidate scan: Halton-sampled poses inside the scene AABB, scored by the
        // probe rays' hit statistics. PROBE is any probe-capable renderer — injected so the
        // datasets layer stays independent of the raytracing implementation; its batch width is
        // pure transport (results are processed candidate-by-candidate in index order, stopping
        // exactly at the requested acceptance count). Consumes the probe target's camera tensor;
        // producers that rely on persistent cameras must rewrite them after annotation
        template <typename T, typename DEVICE, typename METADATA_T, typename PROBE, typename PARAMETERS_T, typename PARAMETERS_TI>
        void scan(DEVICE& device, Scan<T>& progress, const rendering::SceneMetadata<METADATA_T>& metadata, PROBE& probe_target, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters) {
            constexpr std::uint64_t NUM_CAMERAS = static_cast<std::uint64_t>(PROBE::SPEC::NUM_CAMERAS);
            constexpr std::uint64_t NUM_PROBES = static_cast<std::uint64_t>(PROBE::SPEC::NUM_PROBES);

            const T center[3] = {(T)metadata.center[0], (T)metadata.center[1], (T)metadata.center[2]};
            const T half_extent[3] = {(T)metadata.half_extent[0], (T)metadata.half_extent[1], (T)metadata.half_extent[2]};

            RL_TOOLS_RENDERING_DATASETS_LOG("annotations::free_space: center=[" << center[0] << "," << center[1] << "," << center[2] << "] half_extent=[" << half_extent[0] << "," << half_extent[1] << "," << half_extent[2] << "]");
            utils::assert_exit(device, data(collision_results(device, probe_target)) != nullptr, "annotations::free_space: collision results buffer is null");
            utils::assert_exit(device, parameters.fov > 0 && parameters.aspect > 0, "annotations::free_space: parameters.fov and parameters.aspect must be set");

            constexpr T PI = static_cast<T>(3.14159265358979323846);
            const T min_clearance = (T)parameters.min_clearance;
            const std::uint64_t max_candidates = static_cast<std::uint64_t>(parameters.max_candidates_tested);
            const std::uint64_t min_required = static_cast<std::uint64_t>(parameters.min_required_positions);
            const T max_half = std::max({half_extent[0], half_extent[1], half_extent[2]});
            const T max_dist = max_half > static_cast<T>(1)
                ? static_cast<T>(2) * max_half
                : static_cast<T>(20);
            const T search_half_extent[3] = {
                std::max(static_cast<T>(0), half_extent[0] - min_clearance),
                std::max(static_cast<T>(0), half_extent[1] - min_clearance),
                std::max(static_cast<T>(0), half_extent[2] - min_clearance)
            };

            std::array<FreeSpacePosition<T>, NUM_CAMERAS> batch_positions{};
            Tensor<typename decltype(probe_target.cameras)::SPEC> camera_staging;
            malloc(device, camera_staging);
            Tensor<typename decltype(probe_target.collision_results)::SPEC> probe_staging;
            malloc(device, probe_staging);
            std::uint64_t fresh_tested = 0, no_hits = 0, failed_hit_ratio = 0, failed_avg_dist = 0, failed_min_dist = 0;
            T best_min_hit_dist = 0;

            while (progress.tested < max_candidates && progress.accepted.size() < min_required) {
                const std::uint64_t batch_first = progress.tested + 1;
                for (std::uint64_t camera_i = 0; camera_i < NUM_CAMERAS; camera_i++) {
                    const std::uint64_t candidate_i = batch_first + camera_i;
                    const T hx = radical_inverse<T>(candidate_i, static_cast<std::uint64_t>(2));
                    const T hy = radical_inverse<T>(candidate_i, static_cast<std::uint64_t>(3));
                    const T hz = radical_inverse<T>(candidate_i, static_cast<std::uint64_t>(5));
                    const T hw = radical_inverse<T>(candidate_i, static_cast<std::uint64_t>(7));

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
                for (std::uint64_t camera_i = 0; camera_i < NUM_CAMERAS && progress.tested < max_candidates && progress.accepted.size() < min_required; camera_i++) {
                    const rendering::CollisionResult* camera_probes = probe_results + static_cast<size_t>(camera_i) * static_cast<size_t>(NUM_PROBES);
                    std::uint64_t hit_count = 0;
                    std::uint64_t very_near_hit_count = 0;
                    // finite sentinel: infinity() is undefined behavior under -ffast-math and breaks the min tracking on Apple clang
                    T min_hit_dist = std::numeric_limits<T>::max();
                    T sum_hit_dist = static_cast<T>(0);

                    for (std::uint64_t probe_i = 0; probe_i < NUM_PROBES; probe_i++) {
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

                    progress.tested = batch_first + camera_i;
                    fresh_tested++;
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
                        ScanEntry<T> entry;
                        entry.candidate = progress.tested;
                        entry.position = batch_positions[camera_i];
                        entry.position.score = score;
                        progress.accepted.push_back(entry);
                    }
                }
            }
            free(device, camera_staging);
            free(device, probe_staging);

            RL_TOOLS_RENDERING_DATASETS_LOG("annotations::free_space: tested=" << fresh_tested << " no_hits=" << no_hits << " failed_hit_ratio=" << failed_hit_ratio << " failed_avg_dist=" << failed_avg_dist << " failed_min_dist=" << failed_min_dist << " (best_min_hit=" << best_min_hit_dist << ") accepted=" << progress.accepted.size());
        }

        // replay the requested stopping rule against a (possibly longer) stored scan: take
        // acceptances in scan order up to the request's candidate budget and acceptance count,
        // then rank by score (stable — ties keep scan order for cross-platform determinism)
        template <typename DEVICE, typename SPEC, typename PARAMETERS_T, typename PARAMETERS_TI>
        void finalize(DEVICE& device, const Scan<typename SPEC::T>& progress, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters, FreeSpace<SPEC>& annotations) {
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            // the stopping rule is owned by the parameters, never by the product's capacity —
            // this keeps stored scans loadable into any sufficiently large FreeSpace<SPEC>
            utils::assert_exit(device, static_cast<std::uint64_t>(SPEC::MAX_POSITIONS) >= static_cast<std::uint64_t>(parameters.min_required_positions), "annotations::free_space: MAX_POSITIONS must cover min_required_positions");
            std::vector<FreeSpacePosition<T>> selected;
            for (const ScanEntry<T>& entry : progress.accepted) {
                if (entry.candidate > static_cast<std::uint64_t>(parameters.max_candidates_tested)) {
                    break;
                }
                selected.push_back(entry.position);
                if (selected.size() >= static_cast<std::size_t>(parameters.min_required_positions)) {
                    break;
                }
            }
            std::stable_sort(selected.begin(), selected.end(), [](const FreeSpacePosition<T>& a, const FreeSpacePosition<T>& b) {
                return a.score > b.score;
            });
            const TI take_n = std::min(static_cast<TI>(selected.size()), SPEC::MAX_POSITIONS);
            for (TI i = 0; i < take_n; i++) {
                annotations.positions[i] = selected[i];
            }
            annotations.num_positions = take_n;
            utils::assert_exit(device, annotations.num_positions > 0, "annotations::free_space: no valid positions found");
        }
    }

    template <typename DEVICE, typename SPEC, typename METADATA_T, typename PROBE, typename PARAMETERS_T, typename PARAMETERS_TI>
    void annotate(DEVICE& device, FreeSpace<SPEC>& annotations, const rendering::SceneMetadata<METADATA_T>& metadata, PROBE& probe_target, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters) {
        free_space::Scan<typename SPEC::T> progress;
        free_space::scan(device, progress, metadata, probe_target, parameters);
        free_space::finalize(device, progress, parameters, annotations);
    }

    // cached kernel: entries are keyed on (artifact content hash x scoring identity) and store
    // scan progress, so a request for fewer positions replays the stored scan (no probe launches
    // at all) and a request for more resumes it where it stopped
    template <typename DEVICE, typename SPEC, typename METADATA_T, typename PROBE, typename PARAMETERS_T, typename PARAMETERS_TI>
    void annotate(DEVICE& device, FreeSpace<SPEC>& annotations, const rendering::SceneMetadata<METADATA_T>& metadata, PROBE& probe_target, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters, const Cache& cache) {
        using T = typename SPEC::T;
        if (cache.directory.empty() || metadata.content_hash.empty()) {
            annotate(device, annotations, metadata, probe_target, parameters);
            return;
        }
        const std::string directory = cache.directory + "/free_space";
        std::filesystem::create_directories(directory);
        const std::string path = directory + "/" + metadata.content_hash + "-" + free_space::key<T>(metadata, static_cast<std::uint64_t>(PROBE::SPEC::NUM_PROBES), parameters) + ".bin";
        free_space::Scan<T> progress;
        free_space::read(path, progress);
        if (!free_space::sufficient(progress, parameters)) {
            free_space::scan(device, progress, metadata, probe_target, parameters);
            if (!free_space::write(path, progress)) {
                RL_TOOLS_RENDERING_DATASETS_LOG_ERR("annotations::free_space: failed to write cache entry " << path);
            }
        }
        else {
            RL_TOOLS_RENDERING_DATASETS_LOG("annotations::free_space: cache hit (tested=" << progress.tested << " accepted=" << progress.accepted.size() << "): " << path);
        }
        free_space::finalize(device, progress, parameters, annotations);
    }

    // dataset-mediated annotation: the interposition point for dataset wrappers (subset or
    // re-score the table, swap parameters per scene, tap dataset-native sources). The default
    // forwards to the artifact-keyed kernel; call sites must invoke it unqualified so wrapper
    // overloads are found through ADL on the dataset type
    template <typename DEVICE, typename DATASET, typename INDEX_TI, typename SPEC, typename METADATA_T, typename PROBE, typename PARAMETERS_T, typename PARAMETERS_TI>
    void annotate(DEVICE& device, const DATASET& dataset, const typename DATASET::Corpus& corpus, INDEX_TI index, FreeSpace<SPEC>& annotations, const rendering::SceneMetadata<METADATA_T>& metadata, PROBE& probe_target, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters, const Cache& cache) {
        annotate(device, annotations, metadata, probe_target, parameters, cache);
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
