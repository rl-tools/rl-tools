#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_PROCTHOR_CONVERSION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_PROCTHOR_CONVERSION_H

#include <string>

// scene_instance.json (ai2thor-hab / HSSD) → self-contained GLB conversion. Declaration-only:
// the implementation is the rl_tools_procthor_conversion library (src/rendering/procthor2glb/)
// with its heavy dependencies (tinygltf, basisu, stb) — targets using the AI2ThorHab dataset
// source (or the procthor2glb CLI) link it.
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::procthor::conversion {
    // part of the loader's conversion-cache key: bump on any change to the conversion output
    inline constexpr int VERSION = 1;

    enum class Dataset { AI2THOR_HAB, HSSD };

    struct Options {
        std::string scene_instance_path;
        std::string output_glb_path;          // empty: <scene_name>.glb next to the working directory
        bool normalize = false;               // decode KTX2, dequantize meshes, bake texture transforms
        Dataset dataset = Dataset::AI2THOR_HAB;
        std::string hssd_lighting_path;       // explicit HSSD lighting JSON instead of scene default_lighting
    };

    bool scene_instance_to_glb(const Options& options, std::string& error);
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
