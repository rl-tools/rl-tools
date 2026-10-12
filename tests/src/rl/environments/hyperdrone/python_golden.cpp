// Golden-rollout generator sharing the binding's concrete C++ environment (the configuration
// defines and the implementation unit come from CMake). Run from the repo root; regenerate
// deliberately and review the diff.
#include "../../../../../interface/python/hyperdrone/hyperdrone/_native/env/environment.h"

namespace henv = hyperdrone_env_golden;

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {
    constexpr unsigned long long SEED = 1337;
    constexpr int STEPS = 3;

    float golden_action(int step, int instance, int dim){
        return (float)(-1.0 + 2.0 * (((step * 131 + instance * 31 + dim * 7) % 101) / 100.0));
    }

    void write_binary(const std::filesystem::path& path, const void* data, size_t bytes){
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(data), bytes);
    }

    nlohmann::json layout_json(const henv::ObservationLayout& layout){
        nlohmann::json blocks = nlohmann::json::object();
        for(const henv::ObservationBlock& block : layout.blocks){
            blocks[block.name] = {block.offset, block.size};
        }
        return {{"shape", layout.shape}, {"axis", layout.axis}, {"blocks", blocks}};
    }
}

int main(){
    namespace fs = std::filesystem;
    const fs::path scene = fs::absolute("tests/data/ProcTHOR-Train-1.glb");
    if(!fs::exists(scene)){
        std::fprintf(stderr, "scene not found: %s (run from the repo root, with test data downloaded)\n", scene.c_str());
        return 1;
    }
    const fs::path output = fs::path("tests/data/rl/environments/hyperdrone/python_golden");
    fs::create_directories(output);

    henv::Environment env;
    env.init({scene.string()}, {std::nullopt}, {env.parameters_json()}, {}, SEED);

    constexpr size_t total = henv::TOTAL;
    std::array<bool, total> all;
    all.fill(true);
    std::vector<float> observations, observations_privileged, rewards;
    std::vector<uint8_t> terminated;
    std::vector<float> snapshot(total * henv::WORLD::OBSERVATION_DIM);
    std::vector<float> snapshot_privileged(total * henv::WORLD::OBSERVATION_DIM_PRIVILEGED);

    auto record = [&](){
        env.observe(snapshot.data(), true);
        observations.insert(observations.end(), snapshot.begin(), snapshot.end());
        env.observe_privileged(snapshot_privileged.data(), true);
        observations_privileged.insert(observations_privileged.end(), snapshot_privileged.begin(), snapshot_privileged.end());
    };

    env.reset(all.data(), true);
    record();
    std::vector<float> actions(total * henv::WORLD::ACTION_DIM);
    std::vector<float> step_rewards(total);
    std::array<bool, total> step_terminated;
    for(int step_i = 0; step_i < STEPS; step_i++){
        for(size_t instance_i = 0; instance_i < total; instance_i++){
            for(size_t dim_i = 0; dim_i < henv::WORLD::ACTION_DIM; dim_i++){
                actions[instance_i * henv::WORLD::ACTION_DIM + dim_i] = golden_action(step_i, (int)instance_i, (int)dim_i);
            }
        }
        env.step(actions.data(), true);
        record();
        env.rewards(step_rewards.data(), true);
        rewards.insert(rewards.end(), step_rewards.begin(), step_rewards.end());
        env.terminated(step_terminated.data(), true);
        terminated.insert(terminated.end(), step_terminated.begin(), step_terminated.end());
    }

    write_binary(output / "observations.bin", observations.data(), observations.size() * sizeof(float));
    write_binary(output / "observations_privileged.bin", observations_privileged.data(), observations_privileged.size() * sizeof(float));
    write_binary(output / "rewards.bin", rewards.data(), rewards.size() * sizeof(float));
    write_binary(output / "terminated.bin", terminated.data(), terminated.size() * sizeof(uint8_t));

    nlohmann::json manifest;
    manifest["seed"] = SEED;
    manifest["steps"] = STEPS;
    manifest["action_formula"] = "-1 + 2*(((step*131 + instance*31 + dim*7) % 101) / 100)";
    manifest["defines"] = {
        {"HYPERDRONE_ENV_OUTPUT_RGB", HYPERDRONE_ENV_OUTPUT_RGB},
        {"HYPERDRONE_ENV_OUTPUT_DEPTH", HYPERDRONE_ENV_OUTPUT_DEPTH},
        {"HYPERDRONE_ENV_OUTPUT_SEGMENTATION", HYPERDRONE_ENV_OUTPUT_SEGMENTATION},
        {"HYPERDRONE_ENV_SEMANTIC_SEGMENTATION", HYPERDRONE_ENV_SEMANTIC_SEGMENTATION},
        {"HYPERDRONE_ENV_DATASET", HYPERDRONE_ENV_DATASET},
        {"HYPERDRONE_ENV_OUTPUT_NORMALS", HYPERDRONE_ENV_OUTPUT_NORMALS},
        {"HYPERDRONE_ENV_OUTPUT_FLOW", HYPERDRONE_ENV_OUTPUT_FLOW},
        {"HYPERDRONE_ENV_NUM_ENVIRONMENTS", HYPERDRONE_ENV_NUM_ENVIRONMENTS},
        {"HYPERDRONE_ENV_INSTANCES", HYPERDRONE_ENV_INSTANCES},
        {"HYPERDRONE_ENV_CAM_WIDTH", HYPERDRONE_ENV_CAM_WIDTH},
        {"HYPERDRONE_ENV_CAM_HEIGHT", HYPERDRONE_ENV_CAM_HEIGHT},
        {"HYPERDRONE_ENV_SHADING", HYPERDRONE_ENV_SHADING},
        {"HYPERDRONE_ENV_MB_SAMPLES", HYPERDRONE_ENV_MB_SAMPLES},
        {"HYPERDRONE_ENV_PRESET", HYPERDRONE_ENV_PRESET},
        {"HYPERDRONE_ENV_TASK", HYPERDRONE_ENV_TASK},
        {"HYPERDRONE_ENV_N_AGENTS", HYPERDRONE_ENV_N_AGENTS},
        {"HYPERDRONE_ENV_N_CAMERA_AGENTS", HYPERDRONE_ENV_N_CAMERA_AGENTS},
        {"HYPERDRONE_ENV_N_CAMERAS_PER_AGENT", HYPERDRONE_ENV_N_CAMERAS_PER_AGENT},
        {"HYPERDRONE_ENV_DEVICE_CUDA", HYPERDRONE_ENV_DEVICE_CUDA},
    };
    manifest["observation_layout"] = layout_json(henv::observation_layout());
    manifest["observation_layout_privileged"] = layout_json(henv::observation_layout_privileged());
    manifest["total_instances"] = henv::TOTAL;
    manifest["observation_dim"] = henv::WORLD::OBSERVATION_DIM;
    manifest["observation_dim_privileged"] = henv::WORLD::OBSERVATION_DIM_PRIVILEGED;
    manifest["action_dim"] = henv::WORLD::ACTION_DIM;
    manifest["scene"] = scene.filename().string();
    std::ofstream manifest_stream(output / "manifest.json");
    manifest_stream << manifest.dump(4) << "\n";

    std::printf("golden rollout written to %s\n", output.c_str());
    return 0;
}
