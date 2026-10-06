// Golden-rollout generator sharing the binding's concrete C++ environment. Run from the
// repo root; regenerate deliberately and review the diff.
#define HYPERDRONE_ENV_NUM_ENVIRONMENTS 1
#define HYPERDRONE_ENV_INSTANCES 2
#define HYPERDRONE_ENV_CAM_WIDTH 16
#define HYPERDRONE_ENV_CAM_HEIGHT 16
#define HYPERDRONE_ENV_SHADING 0
#define HYPERDRONE_ENV_HISTORY_LENGTH 1
#define HYPERDRONE_ENV_PRESET 0
#define HYPERDRONE_ENV_TASK 0
#define HYPERDRONE_ENV_N_AGENTS 1
#define HYPERDRONE_MODULE_NAME hyperdrone_env_golden
#include "../../../../../interface/python/hyperdrone/hyperdrone/_native/env/environment.h"

namespace henv = hyperdrone_env_golden;

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
}

int main(){
    namespace fs = std::filesystem;
    const fs::path scene_source = fs::absolute("tests/data/ProcTHOR-Train-1.glb");
    if(!fs::exists(scene_source)){
        std::fprintf(stderr, "scene not found: %s (run from the repo root, with test data downloaded)\n", scene_source.c_str());
        return 1;
    }
    const fs::path output = fs::path("interface/python/hyperdrone/tests/env/golden");
    fs::create_directories(output);
    const fs::path scene_directory = fs::temp_directory_path() / "hyperdrone_python_golden_scene";
    fs::remove_all(scene_directory);
    fs::create_directories(scene_directory);
    fs::create_symlink(scene_source, scene_directory / scene_source.filename());

    henv::Environment env;
    env.init({scene_directory.string()}, "", "", SEED);

    const size_t total = henv::TOTAL;
    std::vector<uint8_t> all(total, 1), none(total, 0);
    std::vector<float> observations, observations_privileged, rewards;
    std::vector<uint8_t> terminated;
    std::vector<float> snapshot(total * henv::WORLD::OBSERVATION_DIM);
    std::vector<float> snapshot_privileged(total * henv::WORLD::OBSERVATION_DIM_PRIVILEGED);

    auto record = [&](){
        env.observe(snapshot.data());
        observations.insert(observations.end(), snapshot.begin(), snapshot.end());
        env.observe_privileged(snapshot_privileged.data());
        observations_privileged.insert(observations_privileged.end(), snapshot_privileged.begin(), snapshot_privileged.end());
    };

    env.reset(all.data());
    env.render(all.data());
    record();
    std::vector<float> actions(total * henv::WORLD::ACTION_DIM);
    std::vector<float> step_rewards(total);
    std::vector<uint8_t> step_terminated(total);
    for(int step_i = 0; step_i < STEPS; step_i++){
        for(size_t instance_i = 0; instance_i < total; instance_i++){
            for(size_t dim_i = 0; dim_i < henv::WORLD::ACTION_DIM; dim_i++){
                actions[instance_i * henv::WORLD::ACTION_DIM + dim_i] = golden_action(step_i, (int)instance_i, (int)dim_i);
            }
        }
        env.step(actions.data());
        env.render(none.data());
        record();
        env.read_rewards(step_rewards.data());
        rewards.insert(rewards.end(), step_rewards.begin(), step_rewards.end());
        env.read_terminated(step_terminated.data());
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
    manifest["config_string"] = henv::config_string();
    manifest["observation_layout"] = henv::observation_layout_value(false);
    manifest["observation_layout_privileged"] = henv::observation_layout_value(true);
    manifest["total_instances"] = henv::TOTAL;
    manifest["observation_dim"] = henv::WORLD::OBSERVATION_DIM;
    manifest["observation_dim_privileged"] = henv::WORLD::OBSERVATION_DIM_PRIVILEGED;
    manifest["action_dim"] = henv::WORLD::ACTION_DIM;
    manifest["scene"] = scene_source.filename().string();
    std::ofstream manifest_stream(output / "manifest.json");
    manifest_stream << manifest.dump(4) << "\n";

    std::printf("golden rollout written to %s\n", output.c_str());
    return 0;
}
