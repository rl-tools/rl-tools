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
        // bump on any output-affecting change (candidate sequence, probe camera construction,
        // probe-direction assumptions, scoring, SceneMetadata consumption) — cache keys include it
        constexpr std::uint32_t VERSION = 1;

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
            inline bool read_bytes(std::istream& in, unsigned char* bytes, std::size_t size) {
                return static_cast<bool>(in.read(reinterpret_cast<char*>(bytes), std::streamsize(size)));
            }
            inline bool read_u32(std::istream& in, std::uint32_t& value) {
                unsigned char bytes[4];
                if (!read_bytes(in, bytes, 4)) {
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
                if (!read_bytes(in, bytes, 8)) {
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

        // cache identity: every input the table depends on — the algorithm version, the scalar
        // type, the probe count, the effective metadata values, and all parameters
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
            detail::append_f64(material, static_cast<double>(static_cast<T>(parameters.min_clearance)));
            detail::append_u64(material, static_cast<std::uint64_t>(parameters.max_candidates_tested));
            detail::append_u64(material, static_cast<std::uint64_t>(parameters.min_required_positions));
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
        constexpr std::uint32_t BLOB_FORMAT_VERSION = 2;

        template <typename T>
        bool read(const std::string& path, std::vector<FreeSpacePosition<T>>& table) {
            table.clear();
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                return false;
            }
            char magic[4];
            if (!file.read(magic, 4) || std::memcmp(magic, BLOB_MAGIC, 4) != 0) {
                return false;
            }
            std::uint32_t format_version;
            std::uint64_t count;
            if (!detail::read_u32(file, format_version) || format_version != BLOB_FORMAT_VERSION || !detail::read_u64(file, count)) {
                return false;
            }
            table.reserve(count);
            for (std::uint64_t position_i = 0; position_i < count; position_i++) {
                double values[5];
                for (double& value : values) {
                    if (!detail::read_f64(file, value)) {
                        table.clear();
                        return false;
                    }
                }
                FreeSpacePosition<T> position;
                position.position[0] = static_cast<T>(values[0]);
                position.position[1] = static_cast<T>(values[1]);
                position.position[2] = static_cast<T>(values[2]);
                position.yaw = static_cast<T>(values[3]);
                position.score = static_cast<T>(values[4]);
                table.push_back(position);
            }
            return true;
        }

        template <typename T>
        bool write(const std::string& path, const std::vector<FreeSpacePosition<T>>& table) {
            const std::string temporary = path + ".part." + std::to_string(conta::detail::process_id());
            {
                std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
                if (!file.is_open()) {
                    return false;
                }
                file.write(BLOB_MAGIC, 4);
                std::string header;
                detail::append_u32(header, BLOB_FORMAT_VERSION);
                detail::append_u64(header, static_cast<std::uint64_t>(table.size()));
                file.write(header.data(), std::streamsize(header.size()));
                std::string body;
                for (const FreeSpacePosition<T>& position : table) {
                    detail::append_f64(body, static_cast<double>(position.position[0]));
                    detail::append_f64(body, static_cast<double>(position.position[1]));
                    detail::append_f64(body, static_cast<double>(position.position[2]));
                    detail::append_f64(body, static_cast<double>(position.yaw));
                    detail::append_f64(body, static_cast<double>(position.score));
                }
                file.write(body.data(), std::streamsize(body.size()));
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

        // probe-driven candidate scan: Halton-sampled poses inside the scene AABB, scored by the
        // probe rays' hit statistics; the result is the score-ranked table. PROBE is any
        // probe-capable renderer — injected so the datasets layer stays independent of the
        // raytracing implementation; its batch width is pure transport (results are processed
        // candidate-by-candidate in index order, stopping exactly at the requested count). The
        // probe results depend only on candidate position and yaw: the probe set is the fixed
        // world-frame direction spiral plus the camera-forward ray, so the staged cameras use
        // pinned internal fov/aspect. Consumes the probe target's camera tensor — producers that
        // rely on persistent cameras must rewrite them after annotation
        template <typename T, typename DEVICE, typename METADATA_T, typename PROBE, typename PARAMETERS_T, typename PARAMETERS_TI>
        void scan(DEVICE& device, std::vector<FreeSpacePosition<T>>& table, const rendering::SceneMetadata<METADATA_T>& metadata, PROBE& probe_target, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters) {
            constexpr std::uint64_t NUM_CAMERAS = static_cast<std::uint64_t>(PROBE::SPEC::NUM_CAMERAS);
            constexpr std::uint64_t NUM_PROBES = static_cast<std::uint64_t>(PROBE::SPEC::NUM_PROBES);
            constexpr T PROBE_CAMERA_FOV = 90;
            constexpr T PROBE_CAMERA_ASPECT = 1;
            constexpr T PROBE_CAMERA_LOOK_AHEAD = 1;

            const T center[3] = {(T)metadata.center[0], (T)metadata.center[1], (T)metadata.center[2]};
            const T half_extent[3] = {(T)metadata.half_extent[0], (T)metadata.half_extent[1], (T)metadata.half_extent[2]};

            RL_TOOLS_RENDERING_DATASETS_LOG("annotations::free_space: center=[" << center[0] << "," << center[1] << "," << center[2] << "] half_extent=[" << half_extent[0] << "," << half_extent[1] << "," << half_extent[2] << "]");
            utils::assert_exit(device, data(collision_results(device, probe_target)) != nullptr, "annotations::free_space: collision results buffer is null");

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
            std::uint64_t tested = 0, no_hits = 0, failed_hit_ratio = 0, failed_avg_dist = 0, failed_min_dist = 0;
            T best_min_hit_dist = 0;
            table.clear();

            while (tested < max_candidates && table.size() < min_required) {
                const std::uint64_t batch_first = tested + 1;
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
                        pos.position[0] + PROBE_CAMERA_LOOK_AHEAD * std::cos(pos.yaw),
                        pos.position[1] + PROBE_CAMERA_LOOK_AHEAD * std::sin(pos.yaw),
                        pos.position[2]
                    };
                    const T cam_up[3] = {0, 0, 1};
                    set(device, camera_staging, make_camera_data(cam_position, cam_look_at, cam_up, PROBE_CAMERA_FOV, PROBE_CAMERA_ASPECT), camera_i);
                }

                copy(device, probe_target.device, camera_staging, cameras(device, probe_target));
                probe(device, probe_target);

                copy(probe_target.device, device, collision_results(device, probe_target), probe_staging);
                const rendering::CollisionResult* probe_results = data(probe_staging);
                for (std::uint64_t camera_i = 0; camera_i < NUM_CAMERAS && tested < max_candidates && table.size() < min_required; camera_i++) {
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

                    tested++;
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
                        FreeSpacePosition<T> position = batch_positions[camera_i];
                        position.score = score;
                        table.push_back(position);
                    }
                }
            }
            free(device, camera_staging);
            free(device, probe_staging);

            // stable: ties keep candidate order, for cross-platform determinism
            std::stable_sort(table.begin(), table.end(), [](const FreeSpacePosition<T>& a, const FreeSpacePosition<T>& b) {
                return a.score > b.score;
            });

            RL_TOOLS_RENDERING_DATASETS_LOG("annotations::free_space: tested=" << tested << " no_hits=" << no_hits << " failed_hit_ratio=" << failed_hit_ratio << " failed_avg_dist=" << failed_avg_dist << " failed_min_dist=" << failed_min_dist << " (best_min_hit=" << best_min_hit_dist << ") accepted=" << table.size());
        }

        template <typename DEVICE, typename SPEC, typename PARAMETERS_T, typename PARAMETERS_TI>
        void fill(DEVICE& device, const std::vector<FreeSpacePosition<typename SPEC::T>>& table, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters, FreeSpace<SPEC>& annotations) {
            using TI = typename SPEC::TI;
            // capacity must never shape the result — this keeps the table (and its cache blob)
            // loadable into any sufficiently large FreeSpace<SPEC>
            utils::assert_exit(device, static_cast<std::uint64_t>(SPEC::MAX_POSITIONS) >= static_cast<std::uint64_t>(parameters.min_required_positions), "annotations::free_space: MAX_POSITIONS must cover min_required_positions");
            annotations.num_positions = static_cast<TI>(table.size());
            for (TI position_i = 0; position_i < annotations.num_positions; position_i++) {
                annotations.positions[position_i] = table[position_i];
            }
            utils::assert_exit(device, annotations.num_positions > 0, "annotations::free_space: no valid positions found");
        }
    }

    template <typename DEVICE, typename SPEC, typename METADATA_T, typename PROBE, typename PARAMETERS_T, typename PARAMETERS_TI>
    void annotate(DEVICE& device, FreeSpace<SPEC>& annotations, const rendering::SceneMetadata<METADATA_T>& metadata, PROBE& probe_target, const FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters) {
        std::vector<FreeSpacePosition<typename SPEC::T>> table;
        free_space::scan(device, table, metadata, probe_target, parameters);
        free_space::fill(device, table, parameters, annotations);
    }

    // cached kernel: entries are keyed on (artifact content hash x annotation identity); a hit
    // loads the table without any probe launches
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
        std::vector<FreeSpacePosition<T>> table;
        if (free_space::read(path, table)) {
            RL_TOOLS_RENDERING_DATASETS_LOG("annotations::free_space: cache hit (" << table.size() << " positions): " << path);
        }
        else {
            free_space::scan(device, table, metadata, probe_target, parameters);
            if (!free_space::write(path, table)) {
                RL_TOOLS_RENDERING_DATASETS_LOG_ERR("annotations::free_space: failed to write cache entry " << path);
            }
        }
        free_space::fill(device, table, parameters, annotations);
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
