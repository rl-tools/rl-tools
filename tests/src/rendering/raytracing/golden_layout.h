#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_LAYOUT_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_LAYOUT_H

// Corpus layout: <goldens root>/<corpus>/.../<view>/<modality>/ holds everything about one rendered
// quantity of one camera, named so that a directory listing sorts into the order a reviewer flips
// through: historic goldens oldest first (golden-v01.png), the current golden (golden.png), the
// last render of every backend (render-<backend>.png), then the heatmaps in the same order
// (versus-golden-v01.png, versus-render-<backend>.png). Machine-compared binaries live in data/ so
// that stepping through the images never lands on a file without a preview.

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

        inline std::string overlay_directory(const std::string& goldens_root){
            return join(goldens_root, "overlay");
        }

        inline std::string overlay_view_directory(const std::string& goldens_root, const std::string& scenario, const std::string& state, const std::string& view){
            return join(join(join(overlay_directory(goldens_root), scenario), state), view);
        }

        inline std::string manifest_path(const std::string& corpus_directory){
            return join(corpus_directory, "manifest.json");
        }

        inline std::string view_path(const std::string& view_directory){
            return join(view_directory, "view.json");
        }

        inline std::string modality_directory(const std::string& view_directory, const std::string& modality){
            return join(view_directory, modality);
        }

        inline std::string version_tag(uint32_t version){
            char buffer[16];
            std::snprintf(buffer, sizeof(buffer), "-v%02u", (unsigned)version);
            return buffer;
        }

        // version 0 names the current golden, any other version a historic one
        inline std::string golden_png(const std::string& modality_directory, uint32_t version = 0){
            return join(modality_directory, "golden" + (version == 0 ? std::string() : version_tag(version)) + ".png");
        }

        inline std::string golden_bin(const std::string& modality_directory, uint32_t version = 0){
            return join(join(modality_directory, "data"), "golden" + (version == 0 ? std::string() : version_tag(version)) + ".bin");
        }

        inline std::string render_png(const std::string& modality_directory, const std::string& backend){
            return join(modality_directory, "render-" + backend + ".png");
        }

        // name is "render-<backend>" or "golden-vNN": the file the heatmap compares against golden.png
        inline std::string versus_png(const std::string& modality_directory, const std::string& name){
            return join(modality_directory, "versus-" + name + ".png");
        }

        inline std::string historic_name(uint32_t version){
            return "golden" + version_tag(version);
        }
    }
}

#endif
