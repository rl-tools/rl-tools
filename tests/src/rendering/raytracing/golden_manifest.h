#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_MANIFEST_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_MANIFEST_H

// The corpus carries a manifest.json with its golden_version; the comparators refuse any other
// version, so a stale checkout fails instead of being compared against the wrong target. Bump the
// version together with a row in GOLDEN_LEDGER.md and the DATA_REVISION in download_data.sh (both
// in this directory); the generator then writes golden-v<new>.* next to the earlier generations,
// which stay in place.

#include "golden_layout.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace golden {
    static constexpr uint32_t PROCTHOR_TRAIN_1_GOLDEN_VERSION = 4;
    static constexpr const char* PROCTHOR_TRAIN_1_SCENE = "conta:75d1d67a17ecd16bc3a214baa895240da8060df7";
    // the backend that renders the canonical corpus and is held to bit-exact reproduction
    static constexpr const char* REFERENCE_BACKEND = "generic";

    inline nlohmann::json procthor_manifest(const std::string& backend, const nlohmann::json& environment){
        return {
            {"golden_version", PROCTHOR_TRAIN_1_GOLDEN_VERSION},
            {"reference_backend", backend},
            {"scene", PROCTHOR_TRAIN_1_SCENE},
            {"generation_environment", environment}
        };
    }

    inline bool write_json(const std::string& path, const nlohmann::json& content){
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream output(path);
        output << content.dump(2) << "\n";
        return output.good();
    }

    template <typename T>
    bool write_view(const std::string& path, const T position[3], const T look_at[3], const T up[3], T fov_degrees, int width, int height){
        return write_json(path, {
            {"position", {position[0], position[1], position[2]}},
            {"look_at", {look_at[0], look_at[1], look_at[2]}},
            {"up", {up[0], up[1], up[2]}},
            {"fov_degrees", fov_degrees},
            {"width", width},
            {"height", height}
        });
    }

    inline std::string manifest_error(const std::string& path){
        std::ifstream input(path);
        const nlohmann::json manifest = nlohmann::json::parse(input, nullptr, false);
        if(!manifest.is_object()){
            return "missing or malformed manifest";
        }
        if(!manifest.contains("golden_version") || !manifest["golden_version"].is_number_unsigned()
           || manifest["golden_version"] != PROCTHOR_TRAIN_1_GOLDEN_VERSION){
            return "golden corpus version mismatch";
        }
        if(!manifest.contains("reference_backend") || manifest["reference_backend"] != REFERENCE_BACKEND){
            return "golden reference backend mismatch";
        }
        if(!manifest.contains("scene") || manifest["scene"] != PROCTHOR_TRAIN_1_SCENE){
            return "golden scene mismatch";
        }
        if(!manifest.contains("generation_environment") || !manifest["generation_environment"].is_object()){
            return "missing generation environment";
        }
        return "";
    }
}

#endif
