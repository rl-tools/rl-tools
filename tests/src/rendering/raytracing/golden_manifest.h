#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_MANIFEST_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_MANIFEST_H

// Golden corpus versions. Each published corpus carries a manifest.json with its golden_version;
// the comparators refuse any other version, so a stale checkout fails instead of being compared
// against the wrong target. Bump a version together with a row in GOLDEN_LEDGER.md and the
// DATA_REVISION in download_data.sh (both in this directory).

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <string>

namespace golden {
    static constexpr uint32_t PROCTHOR_STATIC_SCENE_GOLDEN_VERSION = 2;
    static constexpr uint32_t OVERLAY_GOLDEN_VERSION = 1;
    // the backend that renders the canonical corpora and is held to bit-exact reproduction
    static constexpr const char* REFERENCE_BACKEND = "optix";

    inline nlohmann::json procthor_manifest(const std::string& backend){
        return {{"golden_version", PROCTHOR_STATIC_SCENE_GOLDEN_VERSION}, {"reference_backend", backend}};
    }

    inline bool write_manifest(const std::string& path, const nlohmann::json& manifest){
        std::ofstream output(path);
        output << manifest.dump(2) << "\n";
        return output.good();
    }

    // 0 when the file is missing or malformed
    inline uint32_t manifest_version(const std::string& path){
        std::ifstream input(path);
        const nlohmann::json manifest = nlohmann::json::parse(input, nullptr, false);
        return manifest.is_object() ? manifest.value("golden_version", 0u) : 0u;
    }
}

#endif
