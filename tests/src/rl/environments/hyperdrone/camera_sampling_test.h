#pragma once

namespace camera_sampling_test {
    namespace hd = rl_tools::rl::environments::hyperdrone;
    namespace out = rl_tools::rendering::raytracing::outputs;

    template <typename BASE_SPEC>
    struct Single: BASE_SPEC {
        using CAMERA_OUTPUTS = hd::CameraOutputs<out::List<out::RGB<out::format::Float32>>>;
        using OBSERVATION = hd::ImageObservation<0>;
        static constexpr typename BASE_SPEC::T CAMERA_FOV_RANDOMIZATION_RANGE = 10;
        static constexpr typename BASE_SPEC::T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = 0.2;
        static constexpr typename BASE_SPEC::T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = 0.3;
    };
    template <typename BASE_SPEC>
    struct Auxiliary: Single<BASE_SPEC> {
        using CAMERA_OUTPUTS = hd::CameraOutputs<out::List<out::RGB<>>, out::List<out::Depth<>>, out::List<out::RGB<out::format::Float32>>>;
        using OBSERVATION = hd::ImageObservation<2>;
    };

    template <typename BASE_SPEC, typename DEVICE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT bool unchanged_policy_sampling(DEVICE& device, RNG& rng) {
        using T = typename BASE_SPEC::T;
        using SINGLE = Single<BASE_SPEC>;
        using AUXILIARY = Auxiliary<BASE_SPEC>;
        hd::DroneParameters<SINGLE> single_drone;
        hd::DroneParameters<AUXILIARY> auxiliary_drone;
        hd::Parameters<SINGLE> single_parameters;
        hd::Parameters<AUXILIARY> auxiliary_parameters;
        single_drone.cameras[0].fov = 95;
        single_drone.cameras[0].mount.pose.position[0] = 0.4;
        auxiliary_drone.cameras[2] = single_drone.cameras[0];
        auto reference_rng = rng;
        auto auxiliary_rng = rng;
        hd::_sample_view_parameters<DEVICE, SINGLE>(device, single_drone.cameras[0], single_parameters.agents[0].cameras[0], reference_rng);
        hd::_sample_camera_parameters<DEVICE, AUXILIARY>(device, auxiliary_drone, auxiliary_parameters.agents[0], auxiliary_rng);
        hd::_sample_brightness<DEVICE, SINGLE>(device, single_parameters, reference_rng);
        hd::_sample_brightness<DEVICE, AUXILIARY>(device, auxiliary_parameters, auxiliary_rng);
        const auto& reference_camera = single_parameters.agents[0].cameras[0];
        const auto& auxiliary_camera = auxiliary_parameters.agents[0].cameras[2];
        bool equal = reference_camera.fov == auxiliary_camera.fov && single_parameters.brightness_scale == auxiliary_parameters.brightness_scale;
        for (unsigned int dim = 0; dim < 3; dim++) {
            equal = equal && reference_camera.mount.pose.position[dim] == auxiliary_camera.mount.pose.position[dim];
        }
        for (unsigned int dim = 0; dim < 4; dim++) {
            equal = equal && reference_camera.mount.pose.orientation[dim] == auxiliary_camera.mount.pose.orientation[dim];
        }
        const T reference_next = rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, reference_rng);
        const T auxiliary_next = rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, auxiliary_rng);
        return equal && reference_next == auxiliary_next;
    }
}
