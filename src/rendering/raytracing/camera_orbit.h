#ifndef SRC_RENDERING_RAYTRACING_CAMERA_ORBIT_H
#define SRC_RENDERING_RAYTRACING_CAMERA_ORBIT_H

#include <rl_tools/rendering/camera.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>

// upper-hemisphere-biased golden-spiral orbit for viewing architectural scenes — demo/benchmark
// tooling, deliberately outside the library: the renderer only consumes camera tensors
namespace camera_orbit {
    template <typename T>
    struct Parameters {
        T inclination_range = 0.85;
        T floor_margin = 0.1;
        T relift_height = 0.3;
    };

    template <typename T, typename TI>
    void generate(rl_tools::rendering::Camera<T>* poses_out, TI num_cameras, const T center[3], T radius, const T up[3], T fov_degrees, T aspect, const Parameters<T>& parameters = {}){
        const T golden_ratio = (T{1} + sqrtf(T{5})) / T{2};
        for(TI i = 0; i < num_cameras; i++){
            T theta = T{2} * (T)M_PI * i / golden_ratio;
            T cos_inclination = (T{1} - T{2} * (i + T{0.5}) / num_cameras) * parameters.inclination_range;
            T sin_inclination = sqrtf(T{1} - cos_inclination * cos_inclination);
            T position[3] = {
                center[0] + radius * sin_inclination * cosf(theta),
                center[1] + radius * sin_inclination * sinf(theta),
                center[2] + radius * cos_inclination
            };
            if(position[2] < center[2] - radius * parameters.floor_margin){
                position[2] = center[2] + radius * parameters.relift_height;
            }
            poses_out[i] = rl_tools::make_camera_data(position, center, up, fov_degrees, aspect);
        }
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void write(DEVICE& device, rl_tools::rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const typename SPEC::T center[3], typename SPEC::T radius, const typename SPEC::T up[3], typename SPEC::T fov_degrees, const Parameters<typename SPEC::T>& parameters = {}){
        using T = typename SPEC::T;
        const T aspect = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
        rl_tools::Tensor<typename decltype(renderer.cameras)::SPEC> staging;
        rl_tools::malloc(device, staging);
        generate(rl_tools::data(staging), SPEC::NUM_CAMERAS, center, radius, up, fov_degrees, aspect, parameters);
        rl_tools::copy(device, renderer.device, staging, rl_tools::cameras(device, renderer));
        if constexpr (SPEC::HAS_CAMERA_PAIR){
            rl_tools::copy(device, renderer.device, staging, rl_tools::cameras_open(device, renderer));
        }
        rl_tools::free(device, staging);
    }
}

#endif
