#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_PROCTHOR_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_PROCTHOR_OPERATIONS_CPU_H

#include "procthor.h"
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
namespace rl_tools::rendering::datasets::procthor {
    template <typename DEVICE>
    void enumerate(DEVICE& device, const GLB& dataset, Corpus& corpus) {
        corpus.references.clear();
        if(!dataset.references.empty()){
            corpus.references = dataset.references;
            return;
        }
        for (const auto& entry : std::filesystem::directory_iterator(dataset.directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".glb") {
                corpus.references.push_back(entry.path().string());
            }
        }
        std::sort(corpus.references.begin(), corpus.references.end());
        utils::assert_exit(device, !corpus.references.empty(), "datasets::procthor::GLB: no .glb scenes found in directory");
    }

    template <typename SHADING = rendering::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T>
    bool load(DEVICE& device, const GLB& dataset, const Corpus& corpus, size_t index, rendering::Bundle<T>& bundle) {
        utils::assert_exit(device, index < corpus.references.size(), "datasets::procthor: corpus index out of range");
        return rl_tools::load<SHADING, HAS_RGB>(device, bundle, corpus.references[index]);
    }


    // natural (version) order: digit runs compare numerically, mirroring the sort -V corpus
    // conventions — the deterministic enumeration order is part of the dataset contract
    inline bool natural_less(const std::string& a, const std::string& b) {
        size_t i = 0, j = 0;
        while(i < a.size() && j < b.size()){
            if(std::isdigit((unsigned char)a[i]) && std::isdigit((unsigned char)b[j])){
                size_t i_end = i, j_end = j;
                while(i_end < a.size() && std::isdigit((unsigned char)a[i_end])) i_end++;
                while(j_end < b.size() && std::isdigit((unsigned char)b[j_end])) j_end++;
                size_t i_num = i, j_num = j;
                while(i_num < i_end - 1 && a[i_num] == '0') i_num++;
                while(j_num < j_end - 1 && b[j_num] == '0') j_num++;
                const size_t i_len = i_end - i_num, j_len = j_end - j_num;
                if(i_len != j_len) return i_len < j_len;
                const int order = a.compare(i_num, i_len, b, j_num, j_len);
                if(order != 0) return order < 0;
                i = i_end; j = j_end;
            }
            else{
                if(a[i] != b[j]) return (unsigned char)a[i] < (unsigned char)b[j];
                i++; j++;
            }
        }
        return a.size() - i < b.size() - j;
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
        std::sort(corpus.references.begin(), corpus.references.end(), natural_less);
        utils::assert_exit(device, !corpus.references.empty(), "datasets::procthor::AI2ThorHab: no scene_instance.json files found for split");
    }

    template <typename SHADING = rendering::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T>
    bool load(DEVICE& device, const AI2ThorHab& dataset, const Corpus& corpus, size_t index, rendering::Bundle<T>& bundle) {
        utils::assert_exit(device, index < corpus.references.size(), "datasets::procthor: corpus index out of range");
        const std::string& source = corpus.references[index];
        std::string source_hash;
        utils::assert_exit(device, rendering::datasets::content_hash(device, source, source_hash), "datasets::procthor::AI2ThorHab: failed to hash scene_instance");
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
            RL_TOOLS_RENDERING_DATASETS_LOG("procthor::AI2ThorHab: converting " << scene_name << " -> " << cache_path);
            conversion::Options options;
            options.scene_instance_path = source;
            options.output_glb_path = cache_path + ".part." + std::to_string(::getpid()); // pid-unique: concurrent loaders converting the same scene must not share a temp file
            options.normalize = dataset.normalize;
            std::string error;
            const bool converted = conversion::scene_instance_to_glb(options, error);
            if (!converted) {
                RL_TOOLS_RENDERING_DATASETS_LOG_ERR("procthor::AI2ThorHab: conversion failed: " << error);
                return false;
            }
            std::filesystem::rename(options.output_glb_path, cache_path);
        }
        return rl_tools::load<SHADING, HAS_RGB>(device, bundle, cache_path);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
