#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_DATASETS_PROCTHOR_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_DATASETS_PROCTHOR_OPERATIONS_CPU_H

#include "procthor.h"
#include "catalog.h"
#include "conversion.h"
#include "../operations_cpu.h"
#include "../glb/operations_cpu.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <filesystem>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::datasets::procthor {
    template <typename DEVICE>
    void enumerate(DEVICE& device, const GLB& dataset, Corpus& corpus) {
        using rl_tools::enumerate;
        enumerate(device, hyperlux::datasets::glb::Dataset{dataset.directory, dataset.references}, corpus);
    }

    template <typename SHADING = hyperlux::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T, typename MODE = mode::Default<>>
    bool load(DEVICE& device, const GLB& dataset, const Corpus& corpus, size_t index, hyperlux::Bundle<T>& bundle, const Mode<MODE>& mode = Mode<MODE>{}) {
        utils::assert_exit(device, index < corpus.references.size(), "datasets::procthor: corpus index out of range");
        using rl_tools::load;
        return load<SHADING, HAS_RGB>(device, bundle, corpus.references[index], mode);
    }


    template <typename DEVICE>
    void enumerate(DEVICE& device, const AI2ThorHab& dataset, Corpus& corpus) {
        corpus.references.clear();
        const std::string prefix = "ProcTHOR-" + dataset.split + "-";
        const std::string suffix = ".scene_instance.json";
        const std::filesystem::path scenes_root = std::filesystem::path(dataset.root) / "configs" / "scenes" / "ProcTHOR";
        utils::assert_exit(device, std::filesystem::is_directory(scenes_root), "datasets::procthor::AI2ThorHab: configs/scenes/ProcTHOR not found under root");
        for (const auto& entry : std::filesystem::recursive_directory_iterator(scenes_root)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string filename = entry.path().filename().string();
            if (filename.rfind(prefix, 0) == 0 && filename.size() > suffix.size() && filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0) {
                corpus.references.push_back(entry.path().string());
            }
        }
        std::sort(corpus.references.begin(), corpus.references.end(), hyperlux::datasets::natural_less);
        utils::assert_exit(device, !corpus.references.empty(), "datasets::procthor::AI2ThorHab: no scene_instance.json files found for split");
    }

    template <typename SHADING = hyperlux::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T, typename MODE = mode::Default<>>
    bool load(DEVICE& device, const AI2ThorHab& dataset, const Corpus& corpus, size_t index, hyperlux::Bundle<T>& bundle, const Mode<MODE>& mode = Mode<MODE>{}) {
        utils::assert_exit(device, index < corpus.references.size(), "datasets::procthor: corpus index out of range");
        const std::string& source = corpus.references[index];
        std::string source_hash;
        utils::assert_exit(device, hyperlux::datasets::content_hash(device, source, source_hash), "datasets::procthor::AI2ThorHab: failed to hash scene_instance");
        conversion::Options semantic_options;
        semantic_options.scene_instance_path = source;
        std::string semantic_source, semantic_error;
        if(!conversion::semantic_source(semantic_options, semantic_source, semantic_error)){
            log(device, device.logger, devices::logging::Verbosity::error, hyperlux::datasets::LOG_PREFIX, semantic_error);
            return false;
        }
        conta::detail::Sha1 fingerprint;
        fingerprint.update(reinterpret_cast<const unsigned char*>(source_hash.data()), source_hash.size());
        fingerprint.update(reinterpret_cast<const unsigned char*>(semantic_source.data()), semantic_source.size());
        source_hash = fingerprint.finalize();
        std::string cache_directory = dataset.cache_directory;
        if (cache_directory.empty()) {
            const char* home = std::getenv("HOME");
            cache_directory = (home != nullptr ? std::string(home) + "/.cache" : std::string(".")) + "/rl_tools/procthor_glb";
        }
        std::filesystem::create_directories(cache_directory);
        std::string scene_name = std::filesystem::path(source).stem().string();
        const std::string scene_instance_suffix = ".scene_instance";
        if(scene_name.size() > scene_instance_suffix.size() && scene_name.compare(scene_name.size() - scene_instance_suffix.size(), scene_instance_suffix.size(), scene_instance_suffix) == 0){
            scene_name = scene_name.substr(0, scene_name.size() - scene_instance_suffix.size());
        }
        const std::string cache_path = cache_directory + "/" + scene_name + "-" + source_hash + "-v" + std::to_string(conversion::VERSION) + (dataset.normalize ? "" : "-raw") + ".glb";
        if (!std::filesystem::exists(cache_path)) {
            log(device, device.logger, devices::logging::Verbosity::info, hyperlux::datasets::LOG_PREFIX, "procthor::AI2ThorHab: converting ", scene_name, " -> ", cache_path);
            conversion::Options options;
            options.scene_instance_path = source;
            options.output_glb_path = cache_path + ".part." + std::to_string(::getpid()); // pid-unique: concurrent loaders converting the same scene must not share a temp file
            options.normalize = dataset.normalize;
            std::string error;
            const bool converted = conversion::scene_instance_to_glb(options, error);
            if (!converted) {
                log(device, device.logger, devices::logging::Verbosity::error, hyperlux::datasets::LOG_PREFIX, "procthor::AI2ThorHab: conversion failed: ", error);
                return false;
            }
            std::filesystem::rename(options.output_glb_path, cache_path);
        }
        using rl_tools::load;
        return load<SHADING, HAS_RGB>(device, bundle, cache_path, mode);
    }
}
namespace rl_tools {
    template <typename DEVICE>
    void enumerate(DEVICE&, const hyperlux::datasets::procthor::Train200&, hyperlux::datasets::procthor::Train200::Corpus& corpus) {
        using DATASET = hyperlux::datasets::procthor::Train200;
        corpus.references.clear();
        corpus.references.reserve(DATASET::NUM_SCENES);
        for (const auto& scene : DATASET::SCENES) {
            corpus.references.emplace_back(scene.reference);
        }
    }

    template <typename SHADING = hyperlux::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T, typename MODE = mode::Default<>>
    bool load(DEVICE& device, const hyperlux::datasets::procthor::Train200&, const hyperlux::datasets::procthor::Train200::Corpus& corpus, size_t index, hyperlux::Bundle<T>& bundle, const Mode<MODE>& mode = Mode<MODE>{}) {
        using LOADER = hyperlux::datasets::procthor::Train200::LOADER;
        return load<SHADING, HAS_RGB>(device, LOADER{}, corpus, index, bundle, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
