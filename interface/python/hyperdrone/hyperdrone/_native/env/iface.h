#pragma once
#include <cstdint>

// ABI boundary between hyperdrone.env and the JIT-compiled MultiEnvironment libraries.
// The library is a thin C shim over the rl_tools batch verbs (sample_initial_parameters /
// sample_initial_state / render / observe / step / reward / terminated / rotate_scene) on
// rl_tools::rl::environments::MultiEnvironment<hyperdrone::World>. Buffers are host
// float32/uint8 arrays sized by hyperdrone_env_config(). Bump
// HYPERDRONE_ENV_IFACE_VERSION on any change to this file.
#define HYPERDRONE_ENV_IFACE_VERSION 1

namespace hyperdrone::env {
    struct Config {
        uint32_t num_environments;
        uint32_t instances_per_environment;
        uint32_t total_instances;
        uint32_t n_agents;
        uint32_t cam_width;
        uint32_t cam_height;
        uint32_t image_channels;
        uint32_t observation_dim;
        uint32_t observation_dim_privileged;
        uint32_t action_dim;
        uint32_t episode_step_limit;
    };
}

extern "C" {
    int hyperdrone_env_iface_version();
    const char* hyperdrone_env_config_string();
    void* hyperdrone_env_create();
    void hyperdrone_env_destroy(void* handle);
    void hyperdrone_env_config(void* handle, hyperdrone::env::Config* config);
    // scene_directory: a directory of .glb scenes (rl_tools::...::datasets::Plain), at
    // least one scene per environment; partitions the sorted corpus across environments
    void hyperdrone_env_init(void* handle, const char* scene_directory, unsigned long long seed);
    // mask: total_instances uint8 flags; resamples parameters and states where set
    void hyperdrone_env_reset(void* handle, const uint8_t* mask);
    void hyperdrone_env_render(void* handle, const uint8_t* reset_mask);
    void hyperdrone_env_observe(void* handle, float* observations);                       // (total, observation_dim)
    void hyperdrone_env_observe_privileged(void* handle, float* observations);            // (total, observation_dim_privileged)
    // one verb sequence: step -> reward -> commit next state -> terminated; results are
    // read back via hyperdrone_env_rewards / hyperdrone_env_terminated
    void hyperdrone_env_step(void* handle, const float* actions);                         // (total, action_dim)
    void hyperdrone_env_rewards(void* handle, float* rewards);                            // (total,)
    void hyperdrone_env_terminated(void* handle, uint8_t* terminated_flags);              // (total,)
    // deterministic round-robin over each environment's scene partition; the caller must
    // reset all instances afterwards
    void hyperdrone_env_rotate_scene(void* handle);
}
