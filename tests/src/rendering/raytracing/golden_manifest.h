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
    static constexpr uint32_t PROCTHOR_TRAIN_1_GOLDEN_VERSION = 2;
    static constexpr const char* PROCTHOR_TRAIN_1_SCENE = "conta:33b712cad9e5ccf39b238a3f26fdd83e4e4a4d43";
    // the backend that renders the canonical corpus and is held to bit-exact reproduction
    static constexpr const char* REFERENCE_BACKEND = "optix";

    inline nlohmann::json procthor_manifest(const std::string& backend){
        return {{"golden_version", PROCTHOR_TRAIN_1_GOLDEN_VERSION}, {"reference_backend", backend}};
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

    // 0 when the file is missing or malformed
    inline uint32_t manifest_version(const std::string& path){
        std::ifstream input(path);
        const nlohmann::json manifest = nlohmann::json::parse(input, nullptr, false);
        return manifest.is_object() ? manifest.value("golden_version", 0u) : 0u;
    }
}

#endif
