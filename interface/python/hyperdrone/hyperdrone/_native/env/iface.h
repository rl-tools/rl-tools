#pragma once
#include <cstdint>

// ABI boundary between hyperdrone.env and the JIT-compiled MultiEnvironment libraries.
// The library is a thin C shim over the rl_tools batch verbs (sample_initial_parameters /
// sample_initial_state / render / observe / step / reward / terminated / rotate_scene) on
// rl_tools::rl::environments::hyperdrone::MultiEnvironment<hyperdrone::World>. Buffers are host
// float32/uint8 arrays sized by hyperdrone_env_config(). Bump
// HYPERDRONE_ENV_IFACE_VERSION on any change to this file.
#define HYPERDRONE_ENV_IFACE_VERSION 5

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
        uint32_t observation_dim_imu;  // 0 when the World defines no IMU observation stream
        uint32_t frame_stride;         // env steps per rendered camera frame (1: every step)
        float dt;                      // step period [s] (the dynamics integration dt)
    };
}

extern "C" {
    int hyperdrone_env_iface_version();
    const char* hyperdrone_env_config_string();
    // named blocks of the observation vectors, line-oriented: "shape <d0> [<d1> <d2>]",
    // "axis channel|flat", then "block <name> <offset> <size>" entries in memory order —
    // offsets/sizes index the last (channel) axis for images and the flat vector otherwise.
    // which: 0 observation, 1 privileged, 2 IMU (empty string when observation_dim_imu == 0)
    const char* hyperdrone_env_observation_layout(int which);
    void* hyperdrone_env_create();
    void hyperdrone_env_destroy(void* handle);
    void hyperdrone_env_config(void* handle, hyperdrone::env::Config* config);
    // scenes: newline-separated dataset references (rl_tools::rendering::datasets::procthor::GLB).
    // A single entry naming an existing directory enumerates its .glb files as a sorted corpus
    // (the pre-v3 behavior); any other entries — .glb paths or "conta:HASH" references — form
    // the corpus verbatim, order preserved. At least one scene per environment; the corpus is
    // partitioned in contiguous blocks across environments.
    // drone_asset_path: body/prop_* GLB for SELF_VISIBLE specifications (empty to omit);
    // gate_asset_path: gate GLB for the moving_gate task (empty to omit); both also accept
    // "conta:HASH" references
    void hyperdrone_env_init(void* handle, const char* scenes, const char* drone_asset_path, const char* gate_asset_path, unsigned long long seed);
    // mask: total_instances uint8 flags; resamples parameters and states where set
    void hyperdrone_env_reset(void* handle, const uint8_t* mask);
    void hyperdrone_env_render(void* handle, const uint8_t* reset_mask);
    void hyperdrone_env_observe(void* handle, float* observations);                       // (total, observation_dim)
    void hyperdrone_env_observe_privileged(void* handle, float* observations);            // (total, observation_dim_privileged)
    // the IMU sample produced by the last step (Worlds with an ObservationIMU type, e.g. the
    // visual_inertial_localization task); a no-op when observation_dim_imu == 0
    void hyperdrone_env_observe_imu(void* handle, float* observations);                   // (total, observation_dim_imu)
    // one verb sequence: step -> reward -> commit next state -> terminated; results are
    // read back via hyperdrone_env_rewards / hyperdrone_env_terminated
    void hyperdrone_env_step(void* handle, const float* actions);                         // (total, action_dim)
    void hyperdrone_env_rewards(void* handle, float* rewards);                            // (total,)
    void hyperdrone_env_terminated(void* handle, uint8_t* terminated_flags);              // (total,)
    // deterministic round-robin over each environment's scene partition; the caller must
    // reset all instances afterwards
    void hyperdrone_env_rotate_scene(void* handle);
    // episode bookkeeping (hyperdrone::episodes, same-step autoreset): begin_step applies the
    // pending resets (terminated, time limit, forced) by resampling the due instances and
    // publishes the applied mask as the reset flag — render with it afterwards; end_step (after
    // hyperdrone_env_step) runs the terminal check, counters and truncation; force_reset marks
    // instances for the next begin_step; set_step_limit overrides the time limit (0: none)
    void hyperdrone_env_begin_step(void* handle);
    void hyperdrone_env_end_step(void* handle);
    void hyperdrone_env_force_reset(void* handle, const uint8_t* mask);                  // (total,)
    void hyperdrone_env_set_step_limit(void* handle, uint32_t step_limit);
    // (total,) each; end_reason: 0 none, 1 terminated, 2 time limit, 3 forced
    void hyperdrone_env_episode_flags(void* handle, uint8_t* terminated, uint8_t* truncated, uint8_t* reset, uint8_t* end_reason, uint32_t* episode_step);
}
