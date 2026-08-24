// Golden-rollout generator for the hyperdrone Python binding: compiles the exact env shim
// TU (so the configuration cannot drift from what the binding builds) and records a seeded
// rollout through the C API. The Python test replays it bit-exactly. Run from the repo
// root; regenerate deliberately, review the diff, like every golden in this repository.
#define HYPERDRONE_ENV_NUM_ENVIRONMENTS 1
#define HYPERDRONE_ENV_INSTANCES 2
#define HYPERDRONE_ENV_CAM_WIDTH 16
#define HYPERDRONE_ENV_CAM_HEIGHT 16
#define HYPERDRONE_ENV_SHADING 0
#define HYPERDRONE_ENV_HISTORY_LENGTH 1
#define HYPERDRONE_ENV_PRESET 0
#define HYPERDRONE_ENV_TASK 0
#define HYPERDRONE_ENV_N_AGENTS 1
#include "../../../../../interface/python/hyperdrone/hyperdrone/_native/env/impl.cpp"

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

    hyperdrone::env::Config config;
    void* env = hyperdrone_env_create();
    hyperdrone_env_config(env, &config);
    hyperdrone_env_init(env, scene_directory.c_str(), "", "", SEED);

    const size_t total = config.total_instances;
    std::vector<uint8_t> all(total, 1), none(total, 0);
    std::vector<float> observations, observations_privileged, rewards;
    std::vector<uint8_t> terminated;
    std::vector<float> snapshot(total * config.observation_dim);
    std::vector<float> snapshot_privileged(total * config.observation_dim_privileged);

    auto record = [&](){
        hyperdrone_env_observe(env, snapshot.data());
        observations.insert(observations.end(), snapshot.begin(), snapshot.end());
        hyperdrone_env_observe_privileged(env, snapshot_privileged.data());
        observations_privileged.insert(observations_privileged.end(), snapshot_privileged.begin(), snapshot_privileged.end());
    };

    hyperdrone_env_reset(env, all.data());
    hyperdrone_env_render(env, all.data());
    record();
    std::vector<float> actions(total * config.action_dim);
    std::vector<float> step_rewards(total);
    std::vector<uint8_t> step_terminated(total);
    for(int step_i = 0; step_i < STEPS; step_i++){
        for(size_t instance_i = 0; instance_i < total; instance_i++){
            for(size_t dim_i = 0; dim_i < config.action_dim; dim_i++){
                actions[instance_i * config.action_dim + dim_i] = golden_action(step_i, (int)instance_i, (int)dim_i);
            }
        }
        hyperdrone_env_step(env, actions.data());
        hyperdrone_env_render(env, none.data());
        record();
        hyperdrone_env_rewards(env, step_rewards.data());
        rewards.insert(rewards.end(), step_rewards.begin(), step_rewards.end());
        hyperdrone_env_terminated(env, step_terminated.data());
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
    manifest["config_string"] = hyperdrone_env_config_string();
    manifest["observation_layout"] = hyperdrone_env_observation_layout(0);
    manifest["observation_layout_privileged"] = hyperdrone_env_observation_layout(1);
    manifest["total_instances"] = config.total_instances;
    manifest["observation_dim"] = config.observation_dim;
    manifest["observation_dim_privileged"] = config.observation_dim_privileged;
    manifest["action_dim"] = config.action_dim;
    manifest["scene"] = scene_source.filename().string();
    std::ofstream manifest_stream(output / "manifest.json");
    manifest_stream << manifest.dump(4) << "\n";

    hyperdrone_env_destroy(env);
    std::printf("golden rollout written to %s\n", output.c_str());
    return 0;
}
