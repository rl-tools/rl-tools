// procthor2glb: Convert a Habitat ai2thor-hab or HSSD scene_instance.json into a
// single self-contained GLB file. Thin CLI over the rl_tools_procthor_conversion
// library (conversion.cpp), which the AI2ThorHab dataset source consumes as well.

#include <rl_tools/rendering/datasets/procthor/conversion.h>

#include <iostream>
#include <string>

namespace conversion = rl_tools::rendering::datasets::procthor::conversion;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: procthor2glb <scene_instance.json> [-o output.glb] [--normalize] [--hssd] [--hssd-lighting file.json]\n";
        return 1;
    }

    conversion::Options options;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        const std::string hssdLightingPrefix = "--hssd-lighting=";
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            options.output_glb_path = argv[++i];
        } else if (arg == "--normalize") {
            options.normalize = true;
        } else if (arg == "--hssd") {
            options.dataset = conversion::Dataset::HSSD;
        } else if (arg == "--hssd-lighting") {
            if (i + 1 >= argc) {
                std::cerr << "error: missing value for --hssd-lighting\n";
                return 1;
            }
            options.hssd_lighting_path = argv[++i];
        } else if (arg.compare(0, hssdLightingPrefix.size(), hssdLightingPrefix) == 0) {
            options.hssd_lighting_path = arg.substr(hssdLightingPrefix.size());
        } else if (arg == "-h" || arg == "--help") {
            std::cout
                << "Usage: procthor2glb <scene_instance.json> [-o output.glb] "
                   "[--normalize] [--hssd] [--hssd-lighting file.json]\n"
                << "\nConvert a Habitat ai2thor-hab or HSSD scene to a "
                   "self-contained GLB.\n\n"
                << "Options:\n"
                << "  -o, --output <file>  Output GLB path (default: "
                   "<scene_name>.glb)\n"
                << "  --normalize          Decode KTX2 to PNG, dequantize meshes,\n"
                << "                       and bake texture transforms into UVs\n"
                << "                       for maximum viewer compatibility\n"
                << "  --hssd               Resolve stages/ and objects/ in an HSSD\n"
                << "                       dataset checkout\n"
                << "  --hssd-lighting <file>\n"
                << "                       Import an explicit HSSD/Habitat lighting\n"
                << "                       JSON config instead of scene default_lighting\n"
                << "  -h, --help           Show this help message\n";
            return 0;
        } else {
            options.scene_instance_path = arg;
        }
    }

    if (options.scene_instance_path.empty()) {
        std::cerr << "error: no input scene_instance.json specified\n";
        return 1;
    }

    std::string error;
    if (!conversion::scene_instance_to_glb(options, error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
    }
    return 0;
}
