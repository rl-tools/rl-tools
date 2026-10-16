#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_RAPTOR_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_RAPTOR_OPERATIONS_GENERIC_H

#include "raptor.h"
#include "../../../l2f/quaternion_helper.h"
#include "../../../l2f/operations_generic.h"
#include "../../../../../nn/layers/dense/operations_generic.h"
#include "../../../../../nn/layers/gru/operations_generic.h"
#include "../../../../../nn_models/sequential/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller) {
        malloc(device, controller.model);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller) {
        free(device, controller.model);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state) {
        malloc(device, state.recurrent);
        malloc(device, state.previous_actions);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state) {
        free(device, state.previous_actions);
        free(device, state.recurrent);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::environments::hyperdrone::controllers::raptor::Buffer<SPEC>& buffer) {
        malloc(device, buffer.network);
        malloc(device, buffer.observations);
        malloc(device, buffer.expanded_mask);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::environments::hyperdrone::controllers::raptor::Buffer<SPEC>& buffer) {
        free(device, buffer.expanded_mask);
        free(device, buffer.observations);
        free(device, buffer.network);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const rl::environments::hyperdrone::controllers::raptor::Controller<SOURCE_SPEC>& source, rl::environments::hyperdrone::controllers::raptor::Controller<TARGET_SPEC>& target) {
        copy(source_device, target_device, source.model, target.model);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::environments::hyperdrone::controllers::raptor::State<SOURCE_SPEC>& source, rl::environments::hyperdrone::controllers::raptor::State<TARGET_SPEC>& target) {
        copy(source_device, target_device, source.recurrent, target.recurrent);
        copy(source_device, target_device, source.previous_actions, target.previous_actions);
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void reset(DEVICE& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, RNG& rng) {
        reset(device, controller.model, state.recurrent, rng);
        set_all(device, state.previous_actions, 0);
    }
    template <typename DEVICE, typename SPEC, typename MASK_SPEC>
    void reset_action_history(DEVICE& device, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, const Tensor<MASK_SPEC>& mask) {
        for (typename SPEC::TI row = 0; row < SPEC::INSTANCES; row++) {
            if (get(device, mask, row)) {
                for (typename SPEC::TI motor = 0; motor < SPEC::OUTPUT_DIM; motor++) {
                    set(device, state.previous_actions, 0, row, motor);
                }
            }
        }
    }
    template <typename DEVICE, typename SPEC, typename MASK_SPEC, typename RNG>
    void reset(DEVICE& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, const Tensor<MASK_SPEC>& mask, RNG& rng) {
        static_assert(length(typename MASK_SPEC::SHAPE{}) == 1 && get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::INSTANCES);
        auto mask_matrix = matrix_view(device, mask);
        Mode<mode::sequential::ResetMask<mode::Default<>, mode::sequential::ResetMaskSpecification<decltype(mask_matrix)>>> mode;
        mode.mask = mask_matrix;
        reset(device, controller.model, state.recurrent, rng, mode);
        reset_action_history(device, state, mask);
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void evaluate_step(DEVICE& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, rl::environments::hyperdrone::controllers::raptor::Buffer<SPEC>& buffer, RNG& rng) {
        Mode<nn::layers::gru::NoAutoResetMode<mode::Default<>>> mode;
        evaluate_step(device, controller.model, buffer.observations, state.recurrent, state.previous_actions, buffer.network, rng, mode);
    }

    template <typename DEVICE, typename SPEC, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT void update_yaw_target(DEVICE& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>&, const STATE& state, rl::environments::hyperdrone::controllers::raptor::Target<typename SPEC::T>& target, typename SPEC::T yaw_delta, typename SPEC::T dt, typename SPEC::T yaw_rate_limit = SPEC::YAW_RATE_LIMIT, typename SPEC::T yaw_error_limit = SPEC::YAW_ERROR_LIMIT) {
        using T = typename SPEC::T;
        const auto& q = state.orientation;
        const T yaw = math::atan2(device.math, (T)2 * (q[0] * q[3] + q[1] * q[2]), (T)1 - (T)2 * (q[2] * q[2] + q[3] * q[3]));
        const T max_delta = yaw_rate_limit * dt;
        const T requested_error = target.yaw + math::clamp(device.math, yaw_delta, -max_delta, max_delta) - yaw;
        const T error = math::atan2(device.math, math::sin(device.math, requested_error), math::cos(device.math, requested_error));
        const T limited_yaw = yaw + math::clamp(device.math, error, -yaw_error_limit, yaw_error_limit);
        target.yaw = math::atan2(device.math, math::sin(device.math, limited_yaw), math::cos(device.math, limited_yaw));
    }

    template <typename DEVICE, typename SPEC, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& policy, rl::environments::hyperdrone::controllers::raptor::Buffer<SPEC>& buffer, const STATE& state, const rl::environments::hyperdrone::controllers::raptor::Target<typename SPEC::T>& target, const typename SPEC::T previous_action[4], typename SPEC::TI instance_i) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using namespace rl::environments::l2f;
        T inverse_yaw[4] = {math::cos(device.math, target.yaw / (T)2), 0, 0, -math::sin(device.math, target.yaw / (T)2)};
        T position_error[3];
        T velocity_error[3];
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            position_error[axis_i] = state.position[axis_i] - target.position[axis_i];
            velocity_error[axis_i] = state.linear_velocity[axis_i] - target.linear_velocity[axis_i];
        }
        T rotated_position[3];
        T rotated_velocity[3];
        T orientation[4];
        T rotation[3][3];
        rotate_vector_by_quaternion<DEVICE, T>(inverse_yaw, position_error, rotated_position);
        rotate_vector_by_quaternion<DEVICE, T>(inverse_yaw, velocity_error, rotated_velocity);
        quaternion_multiplication<DEVICE, T>(inverse_yaw, state.orientation, orientation);
        quaternion_to_rotation_matrix<DEVICE, T>(orientation, rotation);
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            set(device, buffer.observations, math::clamp(device.math, rotated_position[axis_i], -SPEC::POSITION_CLIP, SPEC::POSITION_CLIP), instance_i, axis_i);
            set(device, buffer.observations, math::clamp(device.math, rotated_velocity[axis_i], -SPEC::VELOCITY_CLIP, SPEC::VELOCITY_CLIP), instance_i, 12 + axis_i);
            set(device, buffer.observations, state.angular_velocity[axis_i], instance_i, 15 + axis_i);
            for (TI column_i = 0; column_i < 3; column_i++) {
                set(device, buffer.observations, rotation[axis_i][column_i], instance_i, 3 + 3 * axis_i + column_i);
            }
        }
        for (TI action_i = 0; action_i < 4; action_i++) {
            set(device, buffer.observations, previous_action[action_i], instance_i, 18 + action_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename PHYSICAL_STATE>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, const rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, rl::environments::hyperdrone::controllers::raptor::Buffer<SPEC>& buffer, const PHYSICAL_STATE& physical, const rl::environments::hyperdrone::controllers::raptor::Target<typename SPEC::T>& target, typename SPEC::TI row) {
        typename SPEC::T previous[SPEC::OUTPUT_DIM];
        for (typename SPEC::TI motor = 0; motor < SPEC::OUTPUT_DIM; motor++) {
            previous[motor] = get(device, state.previous_actions, row, motor);
        }
        observe(device, controller, buffer, physical, target, previous, row);
    }
    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename PHYSICAL_STATE, typename COMMAND_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, const rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, rl::environments::hyperdrone::controllers::raptor::Buffer<SPEC>& buffer, const rl::environments::l2f::ParametersBase<PARAMETER_SPEC>&, const PHYSICAL_STATE& physical, const Tensor<COMMAND_SPEC>& command, typename SPEC::TI row, typename SPEC::TI) {
        static_assert(length(typename COMMAND_SPEC::SHAPE{}) == 1 && get<0>(typename COMMAND_SPEC::SHAPE{}) == 4);
        rl::environments::hyperdrone::controllers::raptor::Target<typename SPEC::T> target;
        for (typename SPEC::TI axis = 0; axis < 3; axis++) {
            target.position[axis] = get(device, command, axis);
        }
        target.yaw = get(device, command, 3);
        observe(device, controller, state, buffer, physical, target, row);
    }
    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename PHYSICAL_SPEC, typename COMMAND_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    auto control(DEVICE& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, rl::environments::hyperdrone::controllers::raptor::Buffer<SPEC>& buffer, const Tensor<PARAMETER_SPEC>& parameters, const Tensor<PHYSICAL_SPEC>& physical_states, const Tensor<COMMAND_SPEC>& commands, RNG& rng) {
        using TI = typename SPEC::TI;
        constexpr TI INSTANCES = get<0>(typename COMMAND_SPEC::SHAPE{});
        constexpr TI AGENTS = get<1>(typename COMMAND_SPEC::SHAPE{}) / 4;
        static_assert(length(typename COMMAND_SPEC::SHAPE{}) == 2 && get<1>(typename COMMAND_SPEC::SHAPE{}) % 4 == 0);
        static_assert(INSTANCES * AGENTS == SPEC::INSTANCES);
        static_assert(length(typename PARAMETER_SPEC::SHAPE{}) == 1 && get<0>(typename PARAMETER_SPEC::SHAPE{}) == INSTANCES);
        static_assert(length(typename PHYSICAL_SPEC::SHAPE{}) == 1 && get<0>(typename PHYSICAL_SPEC::SHAPE{}) == INSTANCES);
        for (TI instance = 0; instance < INSTANCES; instance++) {
            auto command = view(device, commands, instance);
            for (TI agent = 0; agent < AGENTS; agent++) {
                observe(device, controller, state, buffer, get_ref(device, parameters, instance), get_ref(device, physical_states, instance), command, instance * AGENTS + agent, agent);
            }
        }
        evaluate_step(device, controller, state, buffer, rng);
        return reshape_row_major(device, state.previous_actions, tensor::Shape<TI, INSTANCES, AGENTS * 4>{});
    }

    template <typename DEVICE, typename SPEC, typename MASK_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void reset(DEVICE& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, rl::environments::hyperdrone::controllers::raptor::State<SPEC>& state, rl::environments::hyperdrone::controllers::raptor::Buffer<SPEC>& buffer, const Tensor<MASK_SPEC>& mask, RNG& rng) {
        constexpr auto INSTANCES = get<0>(typename MASK_SPEC::SHAPE{});
        static_assert(length(typename MASK_SPEC::SHAPE{}) == 1 && SPEC::INSTANCES % INSTANCES == 0);
        if constexpr (INSTANCES == SPEC::INSTANCES) {
            reset(device, controller, state, mask, rng);
        }
        else {
            constexpr auto AGENTS = SPEC::INSTANCES / INSTANCES;
            for (typename SPEC::TI row = 0; row < SPEC::INSTANCES; row++) {
                set(device, buffer.expanded_mask, get(device, mask, row / AGENTS), row);
            }
            reset(device, controller, state, buffer.expanded_mask, rng);
        }
    }
    template <typename DEVICE, typename SPEC, typename T, typename TI, TI N>
    bool validate(DEVICE& device, const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>&, const rl::environments::l2f::parameters::Dynamics<T, TI, N>& dynamics) {
        return is_canonical_quadrotor(device, dynamics);
    }
    template <typename DEVICE, typename ACTION_SPEC, typename PHYSICAL_STATE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT auto transition_action(DEVICE&, rl::environments::hyperdrone::controllers::raptor::Config, const Matrix<ACTION_SPEC>&, const PHYSICAL_STATE& next_state, TI, TI component) {
        return next_state.last_action[component];
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
