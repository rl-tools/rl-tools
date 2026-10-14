#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_SAVE_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_SAVE_CPU_H

#include "operations_cpu_common.h"

// Opt-in save verbs (deliberately not part of operations_cpu_mux.h): include this header after
// the backend operations header in targets that write debug images/binaries. The verbs are
// backend-generic — copy(renderer.device, device, ...) orders the readback after the renderer's
// in-flight work and bridges the backend memory domain.
//
// STATIC gives the stb implementation internal linkage so multiple TUs of one binary may include
// this header without duplicate-symbol link errors; RL_TOOLS_STB_IMAGE_WRITE_PROVIDED arbitrates
// with other stb-write-providing headers (e.g. the test golden_io.h) so the implementation lands
// exactly once per TU regardless of include order (stb's implementation section has no include
// guard).
#ifndef RL_TOOLS_STB_IMAGE_WRITE_PROVIDED
#define RL_TOOLS_STB_IMAGE_WRITE_PROVIDED
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

#include <vector>
#include <type_traits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rendering::raytracing::detail{
        template <typename SPEC, typename SPEC::TI CAMERAS = SPEC::NUM_CAMERAS, typename DEVICE>
        void write_grid_png(DEVICE& device, const uint32_t* fb, const char* filename){
            using TI = typename SPEC::TI;
            constexpr TI cam_pixels = SPEC::CAM_PIXELS;
            constexpr int grid_width = SPEC::GRID_COLS * SPEC::CAM_WIDTH;
            constexpr int grid_height = SPEC::GRID_ROWS * SPEC::CAM_HEIGHT;
            std::vector<uint32_t> grid_image(grid_width * grid_height, 0);

            for(int i = 0; i < (int)CAMERAS; i++){
                int col = i % SPEC::GRID_COLS;
                int row = i / SPEC::GRID_COLS;
                int offset_x = col * SPEC::CAM_WIDTH;
                int offset_y = row * SPEC::CAM_HEIGHT;

                for(int y = 0; y < (int)SPEC::CAM_HEIGHT; y++){
                    memcpy(&grid_image[(offset_y + y) * grid_width + offset_x],
                           &fb[i * cam_pixels + y * SPEC::CAM_WIDTH],
                           SPEC::CAM_WIDTH * sizeof(uint32_t));
                }
            }

            stbi_write_png(filename, grid_width, grid_height, 4,
                           grid_image.data(), grid_width * sizeof(uint32_t));
            log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "Written grid image (", SPEC::GRID_COLS, "x", SPEC::GRID_ROWS,
                   " cameras, ", grid_width, "x", grid_height, " px) to ", filename);
        }

        template <typename SPEC, typename SPEC::TI CAMERAS = SPEC::NUM_CAMERAS, typename DEVICE>
        void write_segmentation_grid_png(DEVICE& device, const uint32_t* segmentation, const char* filename){
            constexpr typename SPEC::TI num_pixels = CAMERAS * SPEC::CAM_PIXELS;
            std::vector<uint32_t> colored(num_pixels);
            for(size_t pixel_i = 0; pixel_i < (size_t)num_pixels; pixel_i++){
                colored[pixel_i] = segmentation_color(device, segmentation[pixel_i]);
            }
            write_grid_png<SPEC, CAMERAS>(device, colored.data(), filename);
        }

        template <typename SPEC, typename SPEC::TI CAMERAS = SPEC::NUM_CAMERAS, typename DEVICE>
        void write_normals_grid_png(DEVICE& device, const float* normals_host, const char* filename){
            constexpr typename SPEC::TI num_pixels = CAMERAS * SPEC::CAM_PIXELS;
            std::vector<uint32_t> colored(num_pixels);
            for(size_t pixel_i = 0; pixel_i < (size_t)num_pixels; pixel_i++){
                colored[pixel_i] = normal_color(device, &normals_host[pixel_i * 3]);
            }
            write_grid_png<SPEC, CAMERAS>(device, colored.data(), filename);
        }

        // direction → hue, magnitude → saturation against the per-image maximum. Advisory
        // review image only (the normalization is content-dependent, like depth.png); the
        // machine-compared flow target is the float binary.
        template <typename SPEC, typename SPEC::TI CAMERAS = SPEC::NUM_CAMERAS, typename DEVICE>
        void write_flow_grid_png(DEVICE& device, const float* flow_host, const char* filename){
            constexpr typename SPEC::TI num_pixels = CAMERAS * SPEC::CAM_PIXELS;
            float max_magnitude = 0;
            for(size_t pixel_i = 0; pixel_i < (size_t)num_pixels; pixel_i++){
                const float u = flow_host[pixel_i * 2 + 0];
                const float v = flow_host[pixel_i * 2 + 1];
                max_magnitude = std::max(max_magnitude, std::sqrt(u * u + v * v));
            }
            std::vector<uint32_t> colored(num_pixels);
            for(size_t pixel_i = 0; pixel_i < (size_t)num_pixels; pixel_i++){
                colored[pixel_i] = flow_color(device, flow_host[pixel_i * 2 + 0], flow_host[pixel_i * 2 + 1], max_magnitude);
            }
            write_grid_png<SPEC, CAMERAS>(device, colored.data(), filename);
        }

        template <typename SPEC, typename SPEC::TI CAMERAS = SPEC::NUM_CAMERAS, typename DEVICE>
        void write_flow_status_grid_png(DEVICE& device, const uint32_t* status_host, const char* filename){
            constexpr typename SPEC::TI num_pixels = CAMERAS * SPEC::CAM_PIXELS;
            std::vector<uint32_t> colored(num_pixels);
            for(size_t pixel_i = 0; pixel_i < (size_t)num_pixels; pixel_i++){
                colored[pixel_i] = flow_status_color(device, status_host[pixel_i]);
            }
            write_grid_png<SPEC, CAMERAS>(device, colored.data(), filename);
        }

        template <typename SPEC, typename SPEC::TI CAMERAS = SPEC::NUM_CAMERAS, typename DEVICE>
        void write_depth_grid_png(DEVICE& device, const float* depth_host, const char* filename){
            using TI = typename SPEC::TI;
            constexpr TI cam_pixels = SPEC::CAM_PIXELS;
            constexpr int grid_width = SPEC::GRID_COLS * SPEC::CAM_WIDTH;
            constexpr int grid_height = SPEC::GRID_ROWS * SPEC::CAM_HEIGHT;
            std::vector<uint32_t> grid_image(grid_width * grid_height, 0);
            constexpr size_t depth_count = (size_t)CAMERAS * SPEC::CAM_PIXELS;
            float min_valid_depth = std::numeric_limits<float>::max();
            float max_valid_depth = std::numeric_limits<float>::lowest();
            for(size_t depth_i = 0; depth_i < depth_count; depth_i++){
                const float depth = depth_host[depth_i];
                if(!is_miss(depth) && depth >= 0.f){
                    min_valid_depth = std::min(min_valid_depth, depth);
                    max_valid_depth = std::max(max_valid_depth, depth);
                }
            }
            const bool has_valid_depth = min_valid_depth <= max_valid_depth;
            const float valid_depth_range = has_valid_depth ? max_valid_depth - min_valid_depth : 0.f;

            for(int i = 0; i < (int)CAMERAS; i++){
                int col = i % SPEC::GRID_COLS;
                int row = i / SPEC::GRID_COLS;
                int offset_x = col * SPEC::CAM_WIDTH;
                int offset_y = row * SPEC::CAM_HEIGHT;

                for(int y = 0; y < (int)SPEC::CAM_HEIGHT; y++){
                    for(int x = 0; x < (int)SPEC::CAM_WIDTH; x++){
                        const float depth = depth_host[i * cam_pixels + y * SPEC::CAM_WIDTH + x];
                        uint8_t value = 0;
                        if(has_valid_depth && !is_miss(depth) && depth >= 0.f){
                            const float normalized = fminf(fmaxf((depth - min_valid_depth) / (valid_depth_range + 1e-6f), 0.f), 1.f);
                            value = static_cast<uint8_t>(normalized * 255.f);
                        }
                        grid_image[(offset_y + y) * grid_width + offset_x + x] =
                            (0xFFu << 24) | (uint32_t(value) << 16) | (uint32_t(value) << 8) | uint32_t(value);
                    }
                }
            }

            stbi_write_png(filename, grid_width, grid_height, 4,
                           grid_image.data(), grid_width * sizeof(uint32_t));
            log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "Written depth image (", SPEC::GRID_COLS, "x", SPEC::GRID_ROWS,
                   " cameras, ", grid_width, "x", grid_height, " px) to ", filename);
        }

        template <typename SPEC, typename SPEC::TI CAMERAS = SPEC::NUM_CAMERAS, typename DEVICE>
        void write_depth_bin(DEVICE& device, const float* depth_host, const char* filename){
            constexpr size_t depth_count = (size_t)CAMERAS * SPEC::CAM_PIXELS;
            FILE* f = fopen(filename, "wb");
            if(f){
                int nc = CAMERAS;
                int h = SPEC::CAM_HEIGHT;
                int w = SPEC::CAM_WIDTH;
                fwrite(&nc, sizeof(int), 1, f);
                fwrite(&h, sizeof(int), 1, f);
                fwrite(&w, sizeof(int), 1, f);
                fwrite(depth_host, sizeof(float), depth_count, f);
                fclose(f);
                log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "Written depth data (", depth_count, " values) to ", filename);
            }
        }

        template <typename SPEC, typename SPEC::TI CAMERAS = SPEC::NUM_CAMERAS, typename DEVICE>
        void write_probes_bin_and_log(DEVICE& device, const CollisionResult* probe_results, const char* filename){
            int total_hits = 0;
            float min_hit_dist = 1e30f, max_hit_dist = 0.f;
            double sum_hit_dist = 0.0;
            for(int i = 0; i < (int)(CAMERAS * SPEC::NUM_PROBES); i++){
                if(probe_results[i].hit){
                    total_hits++;
                    sum_hit_dist += probe_results[i].distance;
                    if(probe_results[i].distance < min_hit_dist){
                        min_hit_dist = probe_results[i].distance;
                    }
                    if(probe_results[i].distance > max_hit_dist){
                        max_hit_dist = probe_results[i].distance;
                    }
                }
            }

            log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "=== COLLISION PROBE RESULTS ===");
            log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "  Total probes:  ", CAMERAS * SPEC::NUM_PROBES);
            log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "  Hits:          ", total_hits,
                   " (", (100.0 * total_hits / (CAMERAS * SPEC::NUM_PROBES)), "%)");
            log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "  Misses:        ", (CAMERAS * SPEC::NUM_PROBES - total_hits));
            if(total_hits > 0){
                log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "  Min hit dist:  ", min_hit_dist);
                log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "  Max hit dist:  ", max_hit_dist);
                log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "  Avg hit dist:  ", (sum_hit_dist / total_hits));
            }
            log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "===============================");

            {
                FILE* f = fopen(filename, "wb");
                if(f){
                    int nc = CAMERAS, np = SPEC::NUM_PROBES;
                    fwrite(&nc, sizeof(int), 1, f);
                    fwrite(&np, sizeof(int), 1, f);
                    fwrite(probe_results, sizeof(CollisionResult),
                           (size_t)CAMERAS * SPEC::NUM_PROBES, f);
                    fclose(f);
                    log(device, device.logger, devices::logging::Verbosity::info, rendering::raytracing::LOG_PREFIX, "Written probe data (", CAMERAS * SPEC::NUM_PROBES,
                           " results) to ", filename);
                }
            }
        }

        template <typename SPEC, typename OUTPUT>
        constexpr typename SPEC::TI ROWS = outputs::TensorSpecification<SPEC, OUTPUT>::SHAPE::template GET<0>;

        template <typename DEVICE, typename SPEC, typename BACKEND, typename TENSOR, typename ELEMENT>
        void save_staging(DEVICE& device, Renderer<SPEC, BACKEND>& renderer, const TENSOR& tensor, std::vector<ELEMENT>& storage){
            using TENSOR_SPEC = typename TENSOR::SPEC;
            static_assert(std::is_same<ELEMENT, typename TENSOR_SPEC::T>::value, "save staging element type must match the tensor element type");
            storage.resize(TENSOR_SPEC::SIZE);
            Tensor<tensor::Specification<ELEMENT, typename TENSOR_SPEC::TI, typename TENSOR_SPEC::SHAPE>> staging;
            staging._data = storage.data();
            copy(renderer.device, device, tensor, staging);
        }
    }

    template <typename DEVICE, typename SPEC, typename BACKEND, typename OUTPUT>
    void save_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, OUTPUT descriptor, const char* filename) {
        static_assert(utils::typing::is_same_v<typename OUTPUT::FORMAT, rendering::raytracing::outputs::format::RGBA8>, "save_image requires an RGBA8 representation");
        std::vector<uint32_t> pixels;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, descriptor), pixels);
        rendering::raytracing::detail::write_grid_png<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, pixels.data(), filename);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void save_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const char* filename){
        using OUTPUT = rendering::raytracing::outputs::RGB<>;
        static_assert(SPEC::HAS_RGB, "save_image requires an RGB-capable renderer specification");
        std::vector<uint32_t> frame_buffer_host;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, OUTPUT{}), frame_buffer_host);
        rendering::raytracing::detail::write_grid_png<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, frame_buffer_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void save_segmentation_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const char* filename){
        using OUTPUT = rendering::raytracing::outputs::Segmentation<rendering::raytracing::outputs::format::UInt32, utils::typing::conditional_t<SPEC::HAS_SEMANTIC_SEGMENTATION, rendering::raytracing::outputs::labels::Semantic, rendering::raytracing::outputs::labels::Instance>>;
        static_assert(SPEC::HAS_SEGMENTATION, "save_segmentation_image requires a segmentation-capable renderer specification");
        std::vector<uint32_t> segmentation_host;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, OUTPUT{}), segmentation_host);
        rendering::raytracing::detail::write_segmentation_grid_png<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, segmentation_host.data(), filename);
        std::ofstream metadata(std::string(filename) + ".json");
        metadata << "{\"segmentation_mode\":\"" << (SPEC::HAS_SEMANTIC_SEGMENTATION ? "semantic" : "instance") << "\",\"background_id\":4294967295}\n";
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void save_normals_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const char* filename){
        using OUTPUT = rendering::raytracing::outputs::Normals<>;
        static_assert(SPEC::HAS_NORMALS, "save_normals_image requires a normals-capable renderer specification");
        std::vector<float> normals_host;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, OUTPUT{}), normals_host);
        rendering::raytracing::detail::write_normals_grid_png<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, normals_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void save_flow_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const char* filename){
        using OUTPUT = rendering::raytracing::outputs::Flow<>;
        static_assert(SPEC::HAS_FLOW, "save_flow_image requires a flow-capable renderer specification");
        std::vector<float> flow_host;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, OUTPUT{}), flow_host);
        rendering::raytracing::detail::write_flow_grid_png<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, flow_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void save_flow_status_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const char* filename){
        using OUTPUT = rendering::raytracing::outputs::FlowStatus<>;
        static_assert(SPEC::HAS_FLOW, "save_flow_status_image requires a flow-capable renderer specification");
        std::vector<uint32_t> status_host;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, OUTPUT{}), status_host);
        rendering::raytracing::detail::write_flow_status_grid_png<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, status_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void save_flow_forward_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const char* filename){
        using OUTPUT = rendering::raytracing::outputs::Flow<rendering::raytracing::outputs::direction::Forward>;
        static_assert(SPEC::HAS_FLOW_FORWARD, "save_flow_forward_image requires a forward-flow renderer specification");
        std::vector<float> flow_host;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, OUTPUT{}), flow_host);
        rendering::raytracing::detail::write_flow_grid_png<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, flow_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void save_flow_forward_status_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const char* filename){
        using OUTPUT = rendering::raytracing::outputs::FlowStatus<rendering::raytracing::outputs::direction::Forward>;
        static_assert(SPEC::HAS_FLOW_FORWARD, "save_flow_forward_status_image requires a forward-flow renderer specification");
        std::vector<uint32_t> status_host;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, OUTPUT{}), status_host);
        rendering::raytracing::detail::write_flow_status_grid_png<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, status_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void save_depth_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const char* filename){
        using OUTPUT = rendering::raytracing::outputs::Depth<>;
        static_assert(SPEC::HAS_DEPTH, "save_depth_image requires a depth-capable renderer specification");
        std::vector<float> depth_host;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, OUTPUT{}), depth_host);
        rendering::raytracing::detail::write_depth_grid_png<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, depth_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void save_depth(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const char* filename){
        using OUTPUT = rendering::raytracing::outputs::Depth<>;
        static_assert(SPEC::HAS_DEPTH, "save_depth requires a depth-capable renderer specification");
        std::vector<float> depth_host;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, OUTPUT{}), depth_host);
        rendering::raytracing::detail::write_depth_bin<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, depth_host.data(), filename);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void save_probes(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const char* filename){
        using OUTPUT = rendering::raytracing::outputs::Probes;

        std::vector<rendering::raytracing::CollisionResult> collision_results_host;
        rendering::raytracing::detail::save_staging(device, renderer, output(device, renderer, OUTPUT{}), collision_results_host);
        rendering::raytracing::detail::write_probes_bin_and_log<SPEC, rendering::raytracing::detail::ROWS<SPEC, OUTPUT>>(device, collision_results_host.data(), filename);

    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
