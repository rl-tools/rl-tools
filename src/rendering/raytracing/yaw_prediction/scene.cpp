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
        float* sin_cos_targets_out,
        unsigned long batch_size,
        float max_angle
    ) {
        using T = float;
        constexpr T PI = static_cast<T>(3.14159265358979323846);

        std::uniform_real_distribution<T> yaw_dist(0.0f, 2.0f * PI);
        std::uniform_real_distribution<T> delta_dist(-max_angle, max_angle);
        std::uniform_real_distribution<T> swap_dist(0.0f, 1.0f);

        for (unsigned long i = 0; i < batch_size; i++) {
            rlt::rl::environments::raytracing_example::State<SceneHandle::SPEC> state;
            auto params = handle->default_params;
            rlt::sample_initial_state(handle->device, handle->env, params, state, handle->sampling_rng);

            const T base_yaw = yaw_dist(handle->data_rng);
            T delta_yaw = delta_dist(handle->data_rng);

            const bool do_swap = swap_dist(handle->data_rng) < 0.5f;
            const T yaw_a = do_swap ? base_yaw + delta_yaw : base_yaw;
            const T yaw_b = do_swap ? base_yaw : base_yaw + delta_yaw;
            const T effective_delta = do_swap ? -delta_yaw : delta_yaw;

            state.yaw = yaw_a;
            cameras_out[i] = rlt::make_camera_for_state(handle->env, params, state);

            state.yaw = yaw_b;
            cameras_out[batch_size + i] = rlt::make_camera_for_state(handle->env, params, state);

            sin_cos_targets_out[i * 2 + 0] = std::sin(effective_delta);
            sin_cos_targets_out[i * 2 + 1] = std::cos(effective_delta);
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
