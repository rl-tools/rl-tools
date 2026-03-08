#pragma once

#include <rl_tools/rendering/raytracing/backends/optix/device.h>
#include <cstdint>

namespace rl_tools::rendering::raytracing::yaw_prediction {

    // Shared configuration constants
    static constexpr unsigned long SCENE_NUM_CAMERAS = 1024; // 2 * BATCH_SIZE
    static constexpr unsigned long SCENE_CAM_WIDTH = 64;
    static constexpr unsigned long SCENE_CAM_HEIGHT = 64;

    struct SceneHandle;

    SceneHandle* create_scene(const char* scene_path);
    void destroy_scene(SceneHandle* handle);

    // Sample a batch of camera pairs for yaw prediction training
    // cameras_out: array of SCENE_NUM_CAMERAS CameraData structs
    //   [0..batch_size-1] = image A cameras (base_yaw)
    //   [batch_size..2*batch_size-1] = image B cameras (base_yaw + delta_yaw)
    // delta_yaws_out: array of batch_size delta_yaw values (after swap augmentation)
    // targets_out: array of batch_size normalized displacement values (delta_yaw / half_hfov)
    // cos_fov_range: [min, max] range for random FOV (cos_fov parameter)
    void sample_camera_batch(
        SceneHandle* handle,
        CameraData* cameras_out,
        float* delta_yaws_out,
        float* targets_out,
        unsigned long batch_size,
        float max_angle,
        float cos_fov_min,
        float cos_fov_max
    );

    // Set cameras and render (blocking)
    void render_batch(SceneHandle* handle, const CameraData* cameras);

    // Get number of indoor initial states found
    unsigned long get_num_indoor_states(SceneHandle* handle);

    // Get device pointer to framebuffer (for direct GPU access from CUDA kernels)
    uint32_t* get_framebuffer_device_ptr(SceneHandle* handle);
}
