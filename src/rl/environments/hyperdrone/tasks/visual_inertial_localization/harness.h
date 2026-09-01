#pragma once

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/autopilot.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/metrics.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/calibration.h>

#include "../../../../../nn_models/port_checkpoint/raptor/policy.h"

#include <chrono>
#include <string>

// shared RAPTOR-flown episode runner for the visual-inertial localization benchmark binaries
// (demo, EuRoC exporter, estimator baselines). The sink observes the causal sensor streams:
// sink.frame(...) on every fresh camera frame (before the step), sink.step(...) after every
// dynamics step with the IMU sample and the relative-to-start ground truth
namespace hyperdrone_vio {
    namespace rlt = rl_tools;
    namespace task = rlt::rl::environments::hyperdrone::tasks::visual_inertial_localization;

    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    using T = float;
    using TI = typename DEVICE::index_t;

    struct DefaultWorldSpec: rlt::rl::environments::hyperdrone::presets::X500FPVIMU<T, TI> {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
        static constexpr bool SELF_VISIBLE = false;
    };
    using DefaultBaseWorld = rlt::rl::environments::hyperdrone::World<DefaultWorldSpec>;
    using DefaultTaskSpec = task::Specification<DefaultBaseWorld>;
    using DefaultWorld = task::World<DefaultTaskSpec>;

    // baseline benchmark configuration: single instance, VIO-friendly resolution, and a
    // stationary hold so static visual-inertial initializers can bootstrap
    struct BaselineWorldSpec: rlt::rl::environments::hyperdrone::presets::X500FPVIMU<T, TI> {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 1;
        static constexpr bool SELF_VISIBLE = false;
        static constexpr TI CAM_WIDTH = 320;
        static constexpr TI CAM_HEIGHT = 240;
    };
    using BaselineBaseWorld = rlt::rl::environments::hyperdrone::World<BaselineWorldSpec>;
    struct BaselineTaskSpec: task::Specification<BaselineBaseWorld> {
        static constexpr TI INITIALIZATION_HOLD_STEPS = 500; // 2.5 s at 200 Hz
    };
    using BaselineWorld = task::World<BaselineTaskSpec>;

    constexpr TI RAPTOR_INPUT_DIM = rl_tools::checkpoint::actor::layer_0::INPUT_SHAPE::LAST;
    template <typename WORLD>
    using RaptorModel = rlt::nn_models::sequential::Build<rlt::nn::capability::Forward<true, false>, rl_tools::checkpoint::actor::TEMPLATE, rlt::tensor::Shape<TI, 1, WORLD::INSTANCES, RAPTOR_INPUT_DIM>>;

    template <typename WORLD>
    struct Tensors {
        static constexpr TI INSTANCES = WORLD::INSTANCES;
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
        rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ObservationIMU::DIM>>> observations_imu;
        rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> terminated_flags;
        void allocate(DEVICE& device){
            rlt::malloc(device, parameters);
            rlt::malloc(device, states);
            rlt::malloc(device, next_states);
            rlt::malloc(device, reset_mask);
            rlt::malloc(device, actions);
            rlt::malloc(device, observations);
            rlt::malloc(device, observations_imu);
            rlt::malloc(device, terminated_flags);
        }
        void deallocate(DEVICE& device){
            rlt::free(device, parameters);
            rlt::free(device, states);
            rlt::free(device, next_states);
            rlt::free(device, reset_mask);
            rlt::free(device, actions);
            rlt::free(device, observations);
            rlt::free(device, observations_imu);
            rlt::free(device, terminated_flags);
        }
    };

    template <typename WORLD>
    struct EpisodeInfo {
        static constexpr TI INSTANCES = WORLD::INSTANCES;
        static constexpr TI CAM_WIDTH = WORLD::SPEC::CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = WORLD::SPEC::CAM_HEIGHT;
        static constexpr TI FRAME_STRIDE = WORLD::FRAME_STRIDE;
        T dt;
        T gravity[3];
        task::PinholeIntrinsics<T> intrinsics;
        T camera_rotation_body[3][3]; // R_CtoI
        T camera_translation_body[3]; // p_CinI
        T imu_rate;
        T frame_rate;
        // continuous-time densities (Kalibr convention), derived from the per-sample parameters
        T gyro_noise_density;             // [rad/s/sqrt(Hz)]
        T gyro_random_walk;               // [rad/s^2/sqrt(Hz)]
        T accelerometer_noise_density;    // [m/s^2/sqrt(Hz)]
        T accelerometer_random_walk;      // [m/s^3/sqrt(Hz)]
    };

    struct EpisodeResult {
        double wall_clock_time = 0;
        bool any_terminated = false;
    };

    // SINK contract:
    //   void begin(const EpisodeInfo<WORLD>& info);
    //   void frame(double time, TI frame_index, const <image tensor>& observations);                 // fresh frames only, pre-step
    //   void step(double next_time, const <imu tensor>& observations_imu,
    //             const T relative_position[][3], const T relative_orientation[][4],            // per instance, relative to episode start
    //             Tensors<WORLD>& tensors);                                                     // post-step (next_states are current)
    template <typename WORLD, typename BASE_WORLD, typename SINK>
    EpisodeResult run_episode(const std::string& scene_path, TI seed, TI steps, SINK& sink){
        constexpr TI INSTANCES = WORLD::INSTANCES;
        DEVICE device;
        rlt::init(device);
        WORLD world;
        typename BASE_WORLD::SharedContext shared;
        rlt::malloc(device, shared.library);
        rlt::malloc(device, world);
        rlt::rendering::datasets::procthor::GLB dataset{{}, {scene_path}};
        typename rlt::rendering::datasets::procthor::GLB::Corpus corpus;
        rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
        rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);

        RNG rng;
        rlt::malloc(device, rng);
        rlt::init(device, rng, seed);

        task::Autopilot<WORLD, RaptorModel<WORLD>> autopilot;
        task::init(device, autopilot, rl_tools::checkpoint::actor::module, rng);

        Tensors<WORLD> tensors;
        tensors.allocate(device);
        rlt::set_all(device, tensors.reset_mask, true);
        rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
        rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);

        const typename WORLD::Parameters reference_parameters = rlt::get(device, tensors.parameters, (TI)0);
        EpisodeInfo<WORLD> info;
        info.dt = reference_parameters.dynamics.integration.dt;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            info.gravity[dim_i] = reference_parameters.dynamics.dynamics.gravity[dim_i];
        }
        info.intrinsics = task::pinhole_intrinsics(device, (TI)WORLD::SPEC::CAM_WIDTH, (TI)WORLD::SPEC::CAM_HEIGHT, reference_parameters.fov);
        task::camera_to_body_transform(device, reference_parameters.camera_mount, info.camera_rotation_body, info.camera_translation_body);
        info.imu_rate = (T)1 / info.dt;
        info.frame_rate = info.imu_rate / (T)WORLD::FRAME_STRIDE;
        {
            T sqrt_dt = rlt::math::sqrt(device.math, info.dt);
            info.gyro_noise_density = reference_parameters.dynamics.imu.gyro.error.noise.std * sqrt_dt;
            info.gyro_random_walk = reference_parameters.dynamics.imu.gyro.error.bias.sigma;
            info.accelerometer_noise_density = reference_parameters.dynamics.imu.accelerometer.error.noise.std * sqrt_dt;
            info.accelerometer_random_walk = reference_parameters.dynamics.imu.accelerometer.error.bias.sigma;
        }
        sink.begin(info);
        // exact step period for timestamping: the float dt (e.g. 0.005f) carries rounding that
        // would jitter nanosecond timestamps; the physical rate is an integer
        const double step_seconds = 1.0 / (double)((long long)((double)info.imu_rate + 0.5));

        T origin_positions[INSTANCES][3];
        T origin_orientations[INSTANCES][4];
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            typename WORLD::State state = rlt::get(device, tensors.states, instance_i);
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                origin_positions[instance_i][dim_i] = state.position[dim_i];
            }
            for(TI dim_i = 0; dim_i < 4; dim_i++){
                origin_orientations[instance_i][dim_i] = state.orientation[dim_i];
            }
        }

        const auto episode_start_time = std::chrono::steady_clock::now();
        TI frame_index = 0;
        for(TI step_i = 0; step_i < steps; step_i++){
            rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
            rlt::set_all(device, tensors.reset_mask, false);
            if(step_i % WORLD::FRAME_STRIDE == 0){
                rlt::observe(device, world, tensors.parameters, tensors.states, typename BASE_WORLD::Observation{}, tensors.observations, rng);
                sink.frame((double)step_i * step_seconds, frame_index, tensors.observations);
                frame_index++;
            }
            task::control(device, world, tensors.parameters, tensors.states, autopilot, tensors.actions, rng);
            rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
            rlt::observe(device, world, tensors.parameters, tensors.next_states, typename WORLD::ObservationIMU{}, tensors.observations_imu, rng);
            T relative_positions[INSTANCES][3];
            T relative_orientations[INSTANCES][4];
            T relative_velocities[INSTANCES][3];
            for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
                typename WORLD::State next_state = rlt::get(device, tensors.next_states, instance_i);
                task::relative_pose(device, origin_positions[instance_i], origin_orientations[instance_i], next_state.position, next_state.orientation, relative_positions[instance_i], relative_orientations[instance_i]);
                T conjugate_origin[4] = {origin_orientations[instance_i][0], -origin_orientations[instance_i][1], -origin_orientations[instance_i][2], -origin_orientations[instance_i][3]};
                rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(conjugate_origin, next_state.linear_velocity, relative_velocities[instance_i]);
            }
            sink.step((double)(step_i + 1) * step_seconds, tensors.observations_imu, relative_positions, relative_orientations, relative_velocities, tensors);
            rlt::copy(device, device, tensors.next_states, tensors.states);
        }
        rlt::terminated(device, world, tensors.parameters, tensors.states, tensors.terminated_flags, rng);
        EpisodeResult result;
        result.wall_clock_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - episode_start_time).count();
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            result.any_terminated = result.any_terminated || rlt::get(device, tensors.terminated_flags, instance_i);
        }
        tensors.deallocate(device);
        task::free(device, autopilot);
        rlt::free(device, world);
        rlt::free(device, shared.library);
        rlt::free(device, rng);
        return result;
    }
}
