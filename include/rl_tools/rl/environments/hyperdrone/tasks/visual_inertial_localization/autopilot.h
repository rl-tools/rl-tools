#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_AUTOPILOT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_AUTOPILOT_H

#include "visual_inertial_localization.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization {
    // waypoint-tracking autopilot around a recurrent state-feedback policy (e.g. RAPTOR): the
    // policy sees [clamped position error to the current waypoint | R | v | omega | last
    // action] through the privileged observation chain, so the benchmark's IMU noise does not
    // leak into the motion generation
    template <typename T_WORLD, typename T_MODEL>
    struct Autopilot {
        using WORLD = T_WORLD;
        using MODEL = T_MODEL;
        using T = typename WORLD::T;
        using TI = typename WORLD::TI;
        using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecificationPrivileged<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecificationPrivileged<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecificationPrivileged<T, TI,
                l2f::observation::ActionHistory<l2f::observation::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
        static constexpr TI OBSERVATION_DIM = OBSERVATION_TYPE::DIM;
        MODEL model;
        typename MODEL::template State<true> state;
        typename MODEL::template Buffer<true> buffer;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, WORLD::INSTANCES, OBSERVATION_DIM>>> observations;
    };
    template <typename DEVICE, typename WORLD, typename MODEL, typename SOURCE_MODULE, typename RNG>
    void init(DEVICE& device, Autopilot<WORLD, MODEL>& autopilot, const SOURCE_MODULE& source_module, RNG& rng){
        ::rl_tools::malloc(device, autopilot.model);
        ::rl_tools::malloc(device, autopilot.state);
        ::rl_tools::malloc(device, autopilot.buffer);
        ::rl_tools::malloc(device, autopilot.observations);
        ::rl_tools::copy(device, device, source_module, autopilot.model);
        ::rl_tools::reset(device, autopilot.model, autopilot.state, rng);
    }
    template <typename DEVICE, typename WORLD, typename MODEL>
    void free(DEVICE& device, Autopilot<WORLD, MODEL>& autopilot){
        ::rl_tools::free(device, autopilot.model);
        ::rl_tools::free(device, autopilot.state);
        ::rl_tools::free(device, autopilot.buffer);
        ::rl_tools::free(device, autopilot.observations);
    }
    // synchronized episodes: the recurrent state resets for all instances together
    template <typename DEVICE, typename WORLD, typename MODEL, typename RNG>
    void reset(DEVICE& device, Autopilot<WORLD, MODEL>& autopilot, RNG& rng){
        ::rl_tools::reset(device, autopilot.model, autopilot.state, rng);
    }
    template <typename DEVICE, typename TASK_SPEC, typename MODEL, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    void control(DEVICE& device, World<TASK_SPEC>& world, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Autopilot<World<TASK_SPEC>, MODEL>& autopilot, Tensor<ACTION_SPEC>& actions, RNG& rng){
        using WORLD = World<TASK_SPEC>;
        using T = typename WORLD::T;
        using TI = typename WORLD::TI;
        using AUTOPILOT = Autopilot<WORLD, MODEL>;
        static_assert(get<0>(typename ACTION_SPEC::SHAPE{}) == WORLD::INSTANCES);
        for (TI instance_i = 0; instance_i < WORLD::INSTANCES; instance_i++){
            auto& instance_parameters = get_ref(device, parameters, instance_i);
            const auto& state = get_ref(device, states, instance_i);
            Matrix<matrix::Specification<T, TI, 1, AUTOPILOT::OBSERVATION_DIM, true, matrix::layouts::RowMajorAlignment<TI, 1>>> observation_row;
            observation_row._data = data(autopilot.observations) + instance_i * AUTOPILOT::OBSERVATION_DIM;
            observe(device, world.dynamics, instance_parameters.dynamics, state, typename AUTOPILOT::OBSERVATION_TYPE{}, observation_row, rng);
            for (TI dim_i = 0; dim_i < 3; dim_i++){
                T error = state.position[dim_i] - instance_parameters.waypoints[state.current_waypoint][dim_i];
                error = math::clamp(device.math, error, -TASK_SPEC::TARGET_POSITION_ERROR_CLIP, TASK_SPEC::TARGET_POSITION_ERROR_CLIP);
                set(observation_row, 0, dim_i, error);
            }
        }
        Mode<nn::layers::gru::NoAutoResetMode<mode::Default<>>> no_auto_reset_mode;
        evaluate_step(device, autopilot.model, autopilot.observations, autopilot.state, actions, autopilot.buffer, rng, no_auto_reset_mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
