#ifndef TESTS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_MANIFEST_H
#define TESTS_RENDERING_RAYTRACING_OVERLAY_GOLDEN_MANIFEST_H

#include "golden_io.h"
#include "overlay_golden_cases.h"

#include <nlohmann/json.hpp>

#include <string>

namespace overlay_goldens {
    template <typename SPEC>
    nlohmann::json expected_manifest(){
        nlohmann::json manifest;
        manifest["schema_version"] = 2;
        manifest["binary_format_version"] = golden::MULTI_CAMERA_BINARY_VERSION;
        manifest["reference_backend"] = "optix";
        manifest["outputs"] = nlohmann::json::array({"rgb", "depth", "segmentation", "normals"});
        manifest["num_cameras"] = (std::size_t)SPEC::NUM_CAMERAS;
        manifest["width"] = (std::size_t)SPEC::CAM_WIDTH;
        manifest["height"] = (std::size_t)SPEC::CAM_HEIGHT;
        manifest["camera_grid"] = "2x2 row-major logical cameras 0,1,2,3";
        auto views = nlohmann::json::array();
        for(const auto& view : VIEWS){
            views.push_back(view.id);
        }
        manifest["views"] = views;
        auto scenarios = nlohmann::json::array();
        auto capture_states = nlohmann::json::object();
        for(const auto scenario : overlay_scenarios::SCENARIOS){
            const std::string id = overlay_scenarios::scenario_id(scenario);
            scenarios.push_back(id);
            capture_states[id] = capture_state_count(scenario) == 1
                ? nlohmann::json::array({"initial"})
                : nlohmann::json::array({"initial", "updated"});
        }
        manifest["scenarios"] = scenarios;
        manifest["capture_states"] = capture_states;
        return manifest;
    }
}

#endif
