#include <rl_tools/operations/cpu_mux.h>

#include "../example/environment/environment.h"
#include "../example/environment/operations_cpu.h"

#include "scene.h"

#include <random>
#include <cmath>

namespace rlt = rl_tools;

namespace rl_tools::rendering::raytracing::yaw_prediction {

    struct SceneHandle {
        using T = float;
        using TI = unsigned long;

        using SPEC = rlt::rl::environments::raytracing_example::Specification<T, TI, SCENE_NUM_CAMERAS, SCENE_CAM_WIDTH, SCENE_CAM_HEIGHT, 64>;
        using ENV = rlt::rl::environments::raytracing_example::Environment<SPEC>;
        using DEVICE = rlt::devices::DEVICE_FACTORY<>;

        DEVICE device;
        ENV env;

        std::mt19937 data_rng;
        DEVICE::SPEC::RANDOM::ENGINE<> sampling_rng;
        rlt::rl::environments::raytracing_example::Parameters<SPEC> default_params;
    };

    SceneHandle* create_scene(const char* scene_path) {
        auto* handle = new SceneHandle();

        rlt::init(handle->device);
        handle->env.scene_path = scene_path;

        rlt::malloc(handle->device, handle->env);
        rlt::init(handle->device, handle->env);

        rlt::initial_parameters(handle->device, handle->env, handle->default_params);

        handle->data_rng.seed(42);
        rlt::malloc(handle->device, handle->sampling_rng);
        rlt::init(handle->device, handle->sampling_rng, 123);

        return handle;
    }

    void destroy_scene(SceneHandle* handle) {
        if (handle) {
            rlt::free(handle->device, handle->sampling_rng);
            rlt::free(handle->device, handle->env);
            delete handle;
        }
    }

    void sample_camera_batch(
        SceneHandle* handle,
        CameraData* cameras_out,
        float* delta_yaws_out,
        float* targets_out,
        unsigned long batch_size,
        float max_angle,
        float cos_fov_min,
        float cos_fov_max
    ) {
        using T = float;
        constexpr T PI = static_cast<T>(3.14159265358979323846);

        std::uniform_real_distribution<T> yaw_dist(0.0f, 2.0f * PI);
        std::uniform_real_distribution<T> unit_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<T> swap_dist(0.0f, 1.0f);
        std::uniform_real_distribution<T> fov_dist(cos_fov_min, cos_fov_max);

        const T aspect = static_cast<T>(SCENE_CAM_WIDTH) / static_cast<T>(SCENE_CAM_HEIGHT);

        for (unsigned long i = 0; i < batch_size; i++) {
            rlt::rl::environments::raytracing_example::State<SceneHandle::SPEC> state;
            auto params = handle->default_params;
            rlt::sample_initial_state(handle->device, handle->env, params, state, handle->sampling_rng);

            const T cos_fov = fov_dist(handle->data_rng);
            const T half_hfov = std::atan(static_cast<T>(0.5) * cos_fov * aspect);
            const T sample_max_angle = std::min(max_angle, half_hfov);

            const T base_yaw = yaw_dist(handle->data_rng);
            T delta_yaw = unit_dist(handle->data_rng) * sample_max_angle;

            const bool do_swap = swap_dist(handle->data_rng) < 0.5f;
            const T yaw_a = do_swap ? base_yaw + delta_yaw : base_yaw;
            const T yaw_b = do_swap ? base_yaw : base_yaw + delta_yaw;
            const T effective_delta = do_swap ? -delta_yaw : delta_yaw;

            // Build cameras with randomized FOV
            const owl::vec3f position(
                params.scene_translation[0] + state.position[0],
                params.scene_translation[1] + state.position[1] + handle->env.eye_height,
                params.scene_translation[2] + state.position[2]
            );
            const owl::vec3f up(0.f, 1.f, 0.f);

            auto make_cam = [&](T yaw) -> CameraData {
                const T cy = std::cos(yaw);
                const T sy = std::sin(yaw);
                const owl::vec3f look_at(
                    position.x + handle->env.look_ahead * cy,
                    position.y,
                    position.z + handle->env.look_ahead * sy
                );
                return rlt::make_camera_data(position, look_at, up, cos_fov, aspect);
            };

            cameras_out[i] = make_cam(yaw_a);
            cameras_out[batch_size + i] = make_cam(yaw_b);

            targets_out[i] = effective_delta / half_hfov;
            delta_yaws_out[i] = effective_delta;
        }
    }

    void render_batch(SceneHandle* handle, const CameraData* cameras) {
        rlt::set_cameras(handle->device, *handle->env.renderer, cameras, SCENE_NUM_CAMERAS);
        rlt::render_rgb_only(handle->device, *handle->env.renderer);
    }

    unsigned long get_num_indoor_states(SceneHandle* handle) {
        return handle->env.num_indoor_initial_states;
    }

    uint32_t* get_framebuffer_device_ptr(SceneHandle* handle) {
        return (uint32_t*)owlBufferGetPointer((OWLBuffer)handle->env.renderer->frame_buffer, 0);
    }
}
