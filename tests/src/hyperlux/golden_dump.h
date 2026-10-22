#ifndef TESTS_HYPERLUX_GOLDEN_DUMP_H
#define TESTS_HYPERLUX_GOLDEN_DUMP_H

// Raw per-case dumps of golden::Rendered: one multi-camera binary per output plus meta.json. The
// wasm reference module writes them (generate_golden --dump-dir) and the native comparator loads
// them in place of an in-process render (RL_TOOLS_HYPERLUX_GOLDEN_DUMP_DIRECTORY).
#include "golden_io.h"
#include "golden_layout.h"
#include "golden_manifest.h"
#include "golden_render.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace golden {
    namespace dump {
        inline std::string binary_path(const std::string& directory, const char* output){
            return layout::join(directory, std::string(output) + ".bin");
        }
        inline std::string meta_path(const std::string& directory){
            return layout::join(directory, "meta.json");
        }
    }

    template <typename CASE, typename CASES>
    bool write_rendered(const std::string& directory, const Rendered<typename CASES::T>& rendered){
        using SPEC = typename CASE::SPEC;
        static_assert(std::is_same_v<typename CASES::T, float>, "dumps store IEEE-754 32-bit floats");
        std::filesystem::create_directories(directory);
        bool ok = true;
        if constexpr(SPEC::HAS_RGB){
            ok &= write_multi_camera_uint32_bin(dump::binary_path(directory, "frame_buffer"), rendered.frame_buffer.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        }
        if constexpr(SPEC::HAS_DEPTH){
            ok &= write_multi_camera_float_bin(dump::binary_path(directory, "depth"), rendered.depth_buffer.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        }
        if constexpr(SPEC::HAS_NORMALS){
            ok &= write_multi_camera_float_bin(dump::binary_path(directory, "normals"), rendered.normals.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, 3);
        }
        if constexpr(SPEC::HAS_SEGMENTATION){
            ok &= write_multi_camera_uint32_bin(dump::binary_path(directory, "segmentation"), rendered.segmentation.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        }
        if constexpr(SPEC::HAS_FLOW){
            ok &= write_multi_camera_float_bin(dump::binary_path(directory, "flow"), rendered.flow.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, 2);
            ok &= write_multi_camera_uint32_bin(dump::binary_path(directory, "flow_status"), rendered.flow_status.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        }
        if constexpr(SPEC::HAS_FLOW_FORWARD){
            ok &= write_multi_camera_float_bin(dump::binary_path(directory, "flow_forward"), rendered.flow_forward.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, 2);
            ok &= write_multi_camera_uint32_bin(dump::binary_path(directory, "flow_forward_status"), rendered.flow_forward_status.data(), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT);
        }
        if constexpr(CASE::PROBES){
            std::vector<float> probes(rendered.probes.size() * 2);
            for(size_t probe_i = 0; probe_i < rendered.probes.size(); probe_i++){
                probes[probe_i * 2 + 0] = rendered.probes[probe_i].distance;
                probes[probe_i * 2 + 1] = rendered.probes[probe_i].hit != 0 ? 1.f : 0.f;
            }
            ok &= write_multi_camera_float_bin(dump::binary_path(directory, "probes"), probes.data(), SPEC::NUM_CAMERAS, SPEC::NUM_PROBES, 1, 2);
        }
        nlohmann::json classes = nlohmann::json::object();
        for(const auto& [id, name] : rendered.semantic_classes){
            classes[std::to_string(id)] = name;
        }
        ok &= write_json(dump::meta_path(directory), {{"num_instances", rendered.num_instances}, {"max_depth", rendered.max_depth}, {"semantic_classes", classes}});
        return ok;
    }

    template <typename CASE, typename CASES>
    bool load_rendered(const std::string& directory, Rendered<typename CASES::T>& rendered){
        using SPEC = typename CASE::SPEC;
        static_assert(std::is_same_v<typename CASES::T, float>, "dumps store IEEE-754 32-bit floats");
        rendered = {};
        bool ok = true;
        if constexpr(SPEC::HAS_RGB){
            ok &= load_multi_camera_uint32_bin(dump::binary_path(directory, "frame_buffer"), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.frame_buffer);
        }
        if constexpr(SPEC::HAS_DEPTH){
            ok &= load_multi_camera_float_bin(dump::binary_path(directory, "depth"), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.depth_buffer);
        }
        if constexpr(SPEC::HAS_NORMALS){
            ok &= load_multi_camera_float_bin(dump::binary_path(directory, "normals"), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.normals, 3);
        }
        if constexpr(SPEC::HAS_SEGMENTATION){
            ok &= load_multi_camera_uint32_bin(dump::binary_path(directory, "segmentation"), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.segmentation);
        }
        if constexpr(SPEC::HAS_FLOW){
            ok &= load_multi_camera_float_bin(dump::binary_path(directory, "flow"), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.flow, 2);
            ok &= load_multi_camera_uint32_bin(dump::binary_path(directory, "flow_status"), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.flow_status);
        }
        if constexpr(SPEC::HAS_FLOW_FORWARD){
            ok &= load_multi_camera_float_bin(dump::binary_path(directory, "flow_forward"), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.flow_forward, 2);
            ok &= load_multi_camera_uint32_bin(dump::binary_path(directory, "flow_forward_status"), SPEC::NUM_CAMERAS, SPEC::CAM_WIDTH, SPEC::CAM_HEIGHT, rendered.flow_forward_status);
        }
        if constexpr(CASE::PROBES){
            std::vector<float> probes;
            ok &= load_multi_camera_float_bin(dump::binary_path(directory, "probes"), SPEC::NUM_CAMERAS, SPEC::NUM_PROBES, 1, probes, 2);
            rendered.probes.resize(probes.size() / 2);
            for(size_t probe_i = 0; probe_i < rendered.probes.size(); probe_i++){
                const uint32_t hit_bits = float_bits(probes[probe_i * 2 + 1]);
                if(hit_bits != 0 && hit_bits != 0x3f800000u){
                    return false;
                }
                rendered.probes[probe_i].distance = probes[probe_i * 2 + 0];
                rendered.probes[probe_i].hit = probes[probe_i * 2 + 1] != 0.f ? 1 : 0;
            }
        }
        std::ifstream input(dump::meta_path(directory));
        if(!input){
            return false;
        }
        const nlohmann::json meta = nlohmann::json::parse(input, nullptr, false);
        if(meta.is_discarded() || !meta.contains("num_instances") || !meta.contains("max_depth") || !meta.contains("semantic_classes")){
            return false;
        }
        rendered.num_instances = meta["num_instances"].get<size_t>();
        rendered.max_depth = meta["max_depth"].get<float>();
        for(const auto& [id, name] : meta["semantic_classes"].items()){
            rendered.semantic_classes[(uint32_t)std::stoul(id)] = name.get<std::string>();
        }
        return ok;
    }
}

#endif
