#pragma once

namespace camera_sampling_test {
    namespace hd = rl_tools::rl::environments::hyperdrone;
    namespace out = rl_tools::rendering::raytracing::outputs;
    using rl_tools::get;
    using rl_tools::instance_rng;

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
    template <typename BASE_SPEC>
    struct Unobserved: Auxiliary<BASE_SPEC> {
        using OBSERVATION = hd::NoObservation;
    };

    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT bool equal_views(const hd::Camera<T>& a, const hd::Camera<T>& b) {
        bool equal = a.fov == b.fov;
        for (unsigned int dim = 0; dim < 3; dim++) {
            equal = equal && a.mount.pose.position[dim] == b.mount.pose.position[dim];
        }
        for (unsigned int dim = 0; dim < 4; dim++) {
            equal = equal && a.mount.pose.orientation[dim] == b.mount.pose.orientation[dim];
        }
        return equal;
    }

    // the policy stream continues as without auxiliary cameras, and the auxiliary cameras do not replay its continuation
    template <typename BASE_SPEC, typename DEVICE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT bool independent_auxiliary_sampling(DEVICE& device, RNG& rng) {
        using T = typename BASE_SPEC::T;
        using SINGLE = Single<BASE_SPEC>;
        using AUXILIARY = Auxiliary<BASE_SPEC>;
        hd::DroneParameters<SINGLE> single_drone;
        hd::DroneParameters<AUXILIARY> auxiliary_drone;
        hd::Parameters<SINGLE> single_parameters;
        hd::Parameters<AUXILIARY> auxiliary_parameters;
        single_drone.cameras[0].fov = 95;
        single_drone.cameras[0].mount.pose.position[0] = 0.4;
        for (unsigned int mount = 0; mount < AUXILIARY::CAMERA_OUTPUTS::COUNT; mount++) {
            auxiliary_drone.cameras[mount] = single_drone.cameras[0];
        }
        auto reference_rng = instance_rng<1>(rng, 0);
        hd::_sample_view_parameters<DEVICE, SINGLE>(device, single_drone.cameras[0], single_parameters.agents[0].cameras[0], reference_rng);
        hd::_sample_camera_parameters<DEVICE, AUXILIARY>(device, auxiliary_drone, auxiliary_parameters.agents[0], rng);
        bool equal = equal_views(single_parameters.agents[0].cameras[0], auxiliary_parameters.agents[0].cameras[2]);
        hd::Camera<T> continuation;
        auto continuation_rng = reference_rng;
        hd::_sample_view_parameters<DEVICE, SINGLE>(device, single_drone.cameras[0], continuation, continuation_rng);
        const bool independent = !equal_views(continuation, auxiliary_parameters.agents[0].cameras[0]) && !equal_views(continuation, auxiliary_parameters.agents[0].cameras[1]) && !equal_views(auxiliary_parameters.agents[0].cameras[0], auxiliary_parameters.agents[0].cameras[1]);
        hd::_sample_brightness<DEVICE, SINGLE>(device, single_parameters, reference_rng);
        hd::_sample_brightness<DEVICE, AUXILIARY>(device, auxiliary_parameters, rng);
        equal = equal && single_parameters.brightness_scale == auxiliary_parameters.brightness_scale;
        for (unsigned int draw = 0; draw < 8; draw++) {
            const T reference_next = rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, reference_rng);
            const T auxiliary_next = rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, rng);
            equal = equal && reference_next == auxiliary_next;
        }
        return equal && independent;
    }

    template <typename BASE_SPEC, typename DEVICE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT bool unobserved_preserves_sampling(DEVICE& device, RNG& rng) {
        using T = typename BASE_SPEC::T;
        using UNOBSERVED = Unobserved<BASE_SPEC>;
        hd::DroneParameters<UNOBSERVED> drone;
        hd::Parameters<UNOBSERVED> parameters;
        auto reference_rng = instance_rng<1>(rng, 0);
        hd::_sample_camera_parameters<DEVICE, UNOBSERVED>(device, drone, parameters.agents[0], rng);
        hd::Camera<T> continuation;
        auto continuation_rng = reference_rng;
        hd::_sample_view_parameters<DEVICE, UNOBSERVED>(device, drone.cameras[0], continuation, continuation_rng);
        bool independent = !equal_views(continuation, parameters.agents[0].cameras[0]);
        auto after_rng = instance_rng<1>(rng, 0);
        for (unsigned int draw = 0; draw < 8; draw++) {
            const T reference_next = rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, reference_rng);
            const T after_next = rl_tools::random::uniform_real_distribution(device.random, (T)0, (T)1, after_rng);
            independent = independent && reference_next == after_next;
        }
        return independent;
    }

    template <typename BASE_SPEC, typename DEVICE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT bool array_preserves_sampling(DEVICE& device, RNG& rng, RNG& reference_rng) {
        using TI = typename RNG::SPEC::TI;
        using SINGLE = Single<BASE_SPEC>;
        using AUXILIARY = Auxiliary<BASE_SPEC>;
        hd::DroneParameters<SINGLE> reference_drone;
        hd::Parameters<SINGLE> reference_parameters;
        hd::DroneParameters<AUXILIARY> drone;
        hd::Parameters<AUXILIARY> parameters;
        unsigned int before[RNG::NUM_RNGS];
        for (TI i = 0; i < RNG::NUM_RNGS; i++) {
            before[i] = get(rng.states, 0, i).state;
        }
        hd::_sample_camera_parameters<DEVICE, SINGLE>(device, reference_drone, reference_parameters.agents[0], reference_rng);
        hd::_sample_camera_parameters<DEVICE, AUXILIARY>(device, drone, parameters.agents[0], rng);
        before[0] = get(reference_rng.states, 0, 0).state;
        bool equal = equal_views(reference_parameters.agents[0].cameras[0], parameters.agents[0].cameras[2]);
        for (TI i = 0; i < RNG::NUM_RNGS; i++) {
            equal = equal && before[i] == get(rng.states, 0, i).state && get(reference_rng.states, 0, i).state == get(rng.states, 0, i).state;
        }
        equal = unobserved_preserves_sampling<BASE_SPEC>(device, rng) && equal;
        for (TI i = 0; i < RNG::NUM_RNGS; i++) {
            equal = equal && before[i] == get(rng.states, 0, i).state;
        }
        return equal;
    }
}
