#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_LAYOUT_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_LAYOUT_H

// Corpus layout: <goldens root>/<corpus>/<view>/<state>/<output>/ holds goldens, backend renders
// and diff-<current golden>-<other image>.png heatmaps; machine-compared binaries live in data/
// so that stepping through the images never lands on a file without a preview.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

namespace golden {
    namespace layout {
        inline std::string join(const std::string& left, const std::string& right){
            return (std::filesystem::path(left) / right).generic_string();
        }

        inline std::string goldens_root(const std::string& test_data_path){
            return join(test_data_path, "rendering/raytracing/goldens");
        }

        inline std::string procthor_directory(const std::string& goldens_root){
            return join(goldens_root, "procthor-train-1");
        }

        inline std::string procthor_view_directory(const std::string& goldens_root, const std::string& pose_id){
            return join(procthor_directory(goldens_root), pose_id);
        }

        inline std::string manifest_path(const std::string& corpus_directory){
            return join(corpus_directory, "manifest.json");
        }

        inline std::string view_path(const std::string& view_directory){
            return join(view_directory, "view.json");
        }

        inline std::string output_directory(const std::string& view_directory, const std::string& state, const std::string& output){
            return join(join(view_directory, state), output);
        }

        inline std::string version_tag(uint32_t version){
            char buffer[16];
            std::snprintf(buffer, sizeof(buffer), "-v%02u", (unsigned)version);
            return buffer;
        }

        // every generation keeps its number; the version pinned in golden_manifest.h is the current one
        inline std::string golden_name(uint32_t version){
            return "golden" + version_tag(version);
        }

        inline std::string golden_png(const std::string& output_directory, uint32_t version){
            return join(output_directory, golden_name(version) + ".png");
        }

        inline std::string golden_bin(const std::string& output_directory, uint32_t version){
            return join(join(output_directory, "data"), golden_name(version) + ".bin");
        }

        inline std::string render_png(const std::string& output_directory, const std::string& backend){
            return join(output_directory, "render-" + backend + ".png");
        }

        inline std::string diff_png(const std::string& output_directory, const std::string& a, const std::string& b){
            return join(output_directory, "diff-" + a + "-" + b + ".png");
        }
    }
}

#endif
