// bulk pre-warm of the free-space annotation cache: enumerate a dataset, materialize each scene
// and run the cached annotate so training runs start from warm cache entries. The probe batch
// width is pure transport (the scan is candidate-exact), so this tool's wide probe vehicle
// produces bit-identical tables to any consumer with matching --probes and scoring parameters
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>
#include <rl_tools/rendering/datasets/annotations/operations_cpu.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace rlt = rl_tools;
using rl_tools::load; // makes the template name visible at global scope so load<SHADING, HAS_RGB>(...) parses; dataset dispatch still happens via ADL

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;

struct Options {
    std::string glb_directory;
    std::string ai2thorhab_root;
    std::string split = "Train";
    std::string cache_directory = rlt::rendering::datasets::annotations::default_cache_directory();
    TI probes = 1;
    TI min_required = 50;
    TI max_candidates = 4096;
    size_t first = 0;
    size_t count = 0;
};

template <TI T_NUM_PROBES>
struct PROBE_CONFIG: rlt::rendering::raytracing::config::Default<T, TI> {
    static constexpr TI CAM_WIDTH = 32;
    static constexpr TI CAM_HEIGHT = 32;
    static constexpr TI NUM_CAMERAS = 64;
    static constexpr TI NUM_PROBES = T_NUM_PROBES;
    using SHADING = rlt::rendering::raytracing::Low;
    static constexpr bool OUTPUT_RGB = true;
    static constexpr bool OUTPUT_DEPTH = false;
    static constexpr bool OUTPUT_SEGMENTATION = false;
};

template <TI NUM_PROBES, typename DATASET>
int run(DEVICE& device, const DATASET& dataset, const Options& options) {
    using RENDERER = rlt::rendering::raytracing::Renderer<rlt::rendering::raytracing::Specification<PROBE_CONFIG<NUM_PROBES>>>;
    using ANNOTATIONS_SPEC = rlt::rendering::datasets::annotations::FreeSpaceSpecification<T, TI, 4096>;
    typename DATASET::Corpus corpus;
    enumerate(device, dataset, corpus);
    const size_t first = options.first;
    if (first >= corpus.references.size()) {
        std::cerr << "--first " << first << " out of range (corpus has " << corpus.references.size() << " scenes)\n";
        return 1;
    }
    const size_t count = options.count == 0 ? corpus.references.size() - first : std::min(options.count, corpus.references.size() - first);
    std::cout << "corpus: " << corpus.references.size() << " scenes; annotating [" << first << ", " << (first + count) << ") into " << options.cache_directory << "\n";

    const rlt::rendering::datasets::annotations::Cache cache{options.cache_directory};
    rlt::rendering::datasets::annotations::FreeSpaceParameters<T, TI> parameters{};
    parameters.min_required_positions = options.min_required;
    parameters.max_candidates_tested = options.max_candidates;

    auto annotations = std::make_unique<rlt::rendering::datasets::annotations::FreeSpace<ANNOTATIONS_SPEC>>();
    for (size_t scene_i = first; scene_i < first + count; scene_i++) {
        rlt::rendering::Bundle<T> bundle;
        if (!load<rlt::rendering::Low, true>(device, dataset, corpus, scene_i, bundle)) {
            std::cerr << "failed to load scene " << scene_i << ": " << corpus.references[scene_i] << "\n";
            return 1;
        }
        auto renderer = std::make_unique<RENDERER>();
        rlt::malloc(device, *renderer);
        rlt::init(device, *renderer, bundle);
        rlt::generate_probe_directions(device, *renderer);
        rlt::rendering::datasets::annotations::annotate(device, *annotations, bundle.metadata, *renderer, parameters, cache);
        std::cout << "scene[" << scene_i << "] " << bundle.metadata.content_hash << ": positions=" << annotations->num_positions << "\n";
        rlt::free(device, *renderer);
    }
    return 0;
}

template <typename DATASET>
int run(DEVICE& device, const DATASET& dataset, const Options& options) {
    switch (options.probes) {
        case 1: return run<1>(device, dataset, options);
        case 8: return run<8>(device, dataset, options);
        case 64: return run<64>(device, dataset, options);
        default:
            std::cerr << "supported --probes values: 1, 8, 64 (must match the consumer's NUM_PROBES)\n";
            return 1;
    }
}

int main(int argc, char** argv) {
    Options options;
    for (int arg_i = 1; arg_i < argc; arg_i++) {
        const std::string argument = argv[arg_i];
        auto value = [&]() -> std::string {
            if (arg_i + 1 >= argc) {
                std::cerr << argument << " requires a value\n";
                std::exit(1);
            }
            return argv[++arg_i];
        };
        if (argument == "--glb") { options.glb_directory = value(); }
        else if (argument == "--ai2thorhab") { options.ai2thorhab_root = value(); }
        else if (argument == "--split") { options.split = value(); }
        else if (argument == "--cache") { options.cache_directory = value(); }
        else if (argument == "--probes") { options.probes = (TI)std::atoll(value().c_str()); }
        else if (argument == "--min-required") { options.min_required = (TI)std::atoll(value().c_str()); }
        else if (argument == "--max-candidates") { options.max_candidates = (TI)std::atoll(value().c_str()); }
        else if (argument == "--first") { options.first = (size_t)std::atoll(value().c_str()); }
        else if (argument == "--count") { options.count = (size_t)std::atoll(value().c_str()); }
        else {
            std::cerr << "usage: rendering_datasets_annotate (--glb DIR | --ai2thorhab ROOT [--split S]) [--probes 1|8|64] [--min-required N] [--max-candidates N] [--cache DIR] [--first I] [--count N]\n";
            return argument == "--help" ? 0 : 1;
        }
    }
    if (options.glb_directory.empty() == options.ai2thorhab_root.empty()) {
        std::cerr << "exactly one of --glb/--ai2thorhab is required\n";
        return 1;
    }
    if (options.min_required > 4096) {
        std::cerr << "--min-required must be <= 4096\n";
        return 1;
    }

    DEVICE device;
    rlt::init(device);
    if (!options.glb_directory.empty()) {
        return run(device, rlt::rendering::datasets::procthor::GLB{options.glb_directory, {}}, options);
    }
    rlt::rendering::datasets::procthor::AI2ThorHab dataset;
    dataset.root = options.ai2thorhab_root;
    dataset.split = options.split;
    return run(device, dataset, options);
}
