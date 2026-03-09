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
            const T tan_half_hfov = static_cast<T>(0.5) * cos_fov * aspect;
            const T tan_half_vfov = static_cast<T>(0.5) * cos_fov;
            const T half_hfov = std::atan(tan_half_hfov);
            const T half_vfov = std::atan(tan_half_vfov);

            // Clamp displacement angles to FOV
            const T max_delta_yaw = std::min(max_angle, half_hfov);
            const T max_delta_pitch = std::min(max_angle, half_vfov);
            const T max_roll = max_angle;

            // Sample displacement angles
            const T delta_yaw = unit_dist(handle->data_rng) * max_delta_yaw;
            const T delta_pitch = unit_dist(handle->data_rng) * max_delta_pitch;
            const T roll = unit_dist(handle->data_rng) * max_roll;

            // Build rotation matrix R = R_roll * R_pitch * R_yaw
            // Camera local frame: x=right, y=up, z=forward
            // R_yaw: rotation around y-axis
            // R_pitch: rotation around x-axis
            // R_roll: rotation around z-axis
            const T cdy = std::cos(delta_yaw), sdy = std::sin(delta_yaw);
            const T cdp = std::cos(delta_pitch), sdp = std::sin(delta_pitch);
            const T cdr = std::cos(roll), sdr = std::sin(roll);

            // R = R_roll * R_pitch * R_yaw (computed analytically)
            T R[3][3] = {
                {cdr*cdy - sdr*sdp*sdy,   -sdr*cdp,   cdr*sdy + sdr*sdp*cdy},
                {sdr*cdy + cdr*sdp*sdy,    cdr*cdp,   sdr*sdy - cdr*sdp*cdy},
                {-cdp*sdy,                 sdp,        cdp*cdy              }
            };

            // Swap augmentation: randomly swap A and B
            const bool do_swap = swap_dist(handle->data_rng) < 0.5f;

            // Effective rotation for target computation (transposed if swapped)
            T eff[3][3];
            if (do_swap) {
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        eff[r][c] = R[c][r];
            } else {
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        eff[r][c] = R[r][c];
            }

            // Base camera orientation (random horizontal yaw)
            const T base_yaw = yaw_dist(handle->data_rng);
            const T cby = std::cos(base_yaw), sby = std::sin(base_yaw);

            // Camera A's local frame in world coords
            // forward = (cby, 0, sby), right = (-sby, 0, cby), up = (0, 1, 0)
            const owl::vec3f forward_a(cby, 0.0f, sby);
            const owl::vec3f right_a(-sby, 0.0f, cby);
            const owl::vec3f up_a(0.0f, 1.0f, 0.0f);

            const owl::vec3f position(
                params.scene_translation[0] + state.position[0],
                params.scene_translation[1] + state.position[1] + handle->env.eye_height,
                params.scene_translation[2] + state.position[2]
            );
            const owl::vec3f world_up(0.f, 1.f, 0.f);

            // Build base camera
            auto make_base_cam = [&]() -> CameraData {
                const owl::vec3f look_at(
                    position.x + handle->env.look_ahead * cby,
                    position.y,
                    position.z + handle->env.look_ahead * sby
                );
                return rlt::make_camera_data(position, look_at, world_up, cos_fov, aspect);
            };

            // Build rotated camera using R in camera A's local frame
            auto make_rotated_cam = [&]() -> CameraData {
                // Camera B's frame = Camera A's frame * R
                const owl::vec3f dir_b(
                    R[0][2]*right_a.x + R[1][2]*up_a.x + R[2][2]*forward_a.x,
                    R[0][2]*right_a.y + R[1][2]*up_a.y + R[2][2]*forward_a.y,
                    R[0][2]*right_a.z + R[1][2]*up_a.z + R[2][2]*forward_a.z
                );
                const owl::vec3f up_b(
                    R[0][1]*right_a.x + R[1][1]*up_a.x + R[2][1]*forward_a.x,
                    R[0][1]*right_a.y + R[1][1]*up_a.y + R[2][1]*forward_a.y,
                    R[0][1]*right_a.z + R[1][1]*up_a.z + R[2][1]*forward_a.z
                );
                const owl::vec3f look_at = position + handle->env.look_ahead * dir_b;
                return rlt::make_camera_data(position, look_at, up_b, cos_fov, aspect);
            };

            if (do_swap) {
                cameras_out[i] = make_rotated_cam();
                cameras_out[batch_size + i] = make_base_cam();
            } else {
                cameras_out[i] = make_base_cam();
                cameras_out[batch_size + i] = make_rotated_cam();
            }

            // Compute targets from effective rotation
            // z2 = third column of eff (camera B's optical axis in camera A's frame)
            const T z2x = eff[0][2];
            const T z2y = eff[1][2];
            const T z2z = eff[2][2];

            // FOV-normalized projection
            const T px = z2x / (z2z * tan_half_hfov);
            const T py = z2y / (z2z * tan_half_vfov);

            // Roll extraction via shortest-arc decomposition
            // Normalize z2 to get direction n
            const T z2_len = std::sqrt(z2x*z2x + z2y*z2y + z2z*z2z);
            const T nx = z2x / z2_len;
            const T ny = z2y / z2_len;
            const T nz = z2z / z2_len;

            // R_pan = shortest-arc rotation from e_z to n (Rodrigues formula)
            // R_pan^T row 0: [1 - nx²*k, -nx*ny*k, -nx]  where k = 1/(1+nz)
            // R_pan^T row 1: [-nx*ny*k, 1 - ny²*k, -ny]
            const T k = static_cast<T>(1) / (static_cast<T>(1) + nz);

            // R_roll_ext = R_pan^T * eff, extract angle from first column
            const T eff00 = eff[0][0], eff10 = eff[1][0], eff20 = eff[2][0];
            const T re00 = (static_cast<T>(1) - nx*nx*k)*eff00 + (-nx*ny*k)*eff10 + (-nx)*eff20;
            const T re10 = (-nx*ny*k)*eff00 + (static_cast<T>(1) - ny*ny*k)*eff10 + (-ny)*eff20;
            const T phi = std::atan2(re10, re00);

            targets_out[i * 3 + 0] = px;
            targets_out[i * 3 + 1] = py;
            targets_out[i * 3 + 2] = phi / PI;
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
