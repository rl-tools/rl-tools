#ifndef TESTS_RENDERING_RAYTRACING_GOLDEN_LAYOUT_H
#define TESTS_RENDERING_RAYTRACING_GOLDEN_LAYOUT_H

#include <filesystem>
#include <string>

namespace golden {
    namespace layout {
        inline std::string join(const std::string& left, const std::string& right){
            return (std::filesystem::path(left) / right).generic_string();
        }

        struct ScenarioTargetPaths{
            std::string directory;
            std::string rgb_png;
            std::string depth_bin;
            std::string depth_png;
            std::string segmentation_bin;
            std::string segmentation_png;
            std::string normals_png;
        };

        struct ScenarioReviewPaths{
            std::string directory;
            std::string rgb_target_png;
            std::string rgb_current_png;
            std::string rgb_diff_png;
            std::string depth_target_png;
            std::string depth_current_png;
            std::string depth_diff_png;
            std::string segmentation_target_png;
            std::string segmentation_current_png;
            std::string segmentation_diff_png;
            std::string normals_target_png;
            std::string normals_current_png;
            std::string normals_diff_png;
        };

        inline std::string procthor_static_scene_directory(const std::string& golden_root){
            return join(golden_root, "procthor_static_scene");
        }

        inline std::string procthor_pose_directory(const std::string& golden_root, const std::string& pose_id){
            return join(procthor_static_scene_directory(golden_root), pose_id);
        }

        inline std::string legacy_procthor_pose_directory(const std::string& golden_root, const std::string& pose_id){
            return join(golden_root, pose_id);
        }

        inline std::string overlay_directory(const std::string& golden_root){
            return join(golden_root, "overlay");
        }

        inline std::string overlay_manifest_path(const std::string& golden_root){
            return join(overlay_directory(golden_root), "manifest.json");
        }

        inline ScenarioTargetPaths scenario_target_paths(
            const std::string& golden_root,
            const std::string& scenario,
            const std::string& state,
            const std::string& view
        ){
            const std::string directory = join(join(join(overlay_directory(golden_root), scenario), state), view);
            return {
                directory,
                join(directory, "rgb.png"),
                join(directory, "depth.bin"),
                join(directory, "depth.png"),
                join(directory, "segmentation.bin"),
                join(directory, "segmentation.png"),
                join(directory, "normals.png")
            };
        }

        inline ScenarioReviewPaths scenario_review_paths(
            const std::string& artifact_root,
            const std::string& backend,
            const std::string& scenario,
            const std::string& state,
            const std::string& view
        ){
            const std::string directory = join(join(join(join(artifact_root, backend), scenario), state), view);
            return {
                directory,
                join(directory, "rgb_target.png"),
                join(directory, "rgb_current.png"),
                join(directory, "rgb_diff.png"),
                join(directory, "depth_target.png"),
                join(directory, "depth_current.png"),
                join(directory, "depth_diff.png"),
                join(directory, "segmentation_target.png"),
                join(directory, "segmentation_current.png"),
                join(directory, "segmentation_diff.png"),
                join(directory, "normals_target.png"),
                join(directory, "normals_current.png"),
                join(directory, "normals_diff.png")
            };
        }
    }
}

#endif
