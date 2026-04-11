#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_INFERENCE_APPLICATIONS_L2F_L2F_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_INFERENCE_APPLICATIONS_L2F_L2F_H

#include "../../executor/executor.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::inference::applications{
    namespace l2f{
        template <typename T_TYPE_POLICY, typename T_TI, typename T_TIMESTAMP, T_TI T_ACTION_HISTORY_LENGTH, T_TI T_OUTPUT_DIM, typename T_POLICY, T_TIMESTAMP T_CONTROL_INTERVAL_INTERMEDIATE_NS, T_TIMESTAMP T_CONTROL_INTERVAL_NATIVE_NS, bool T_FORCE_SYNC_INTERMEDIATE=false, T_TI T_FORCE_SYNC_NATIVE=0, bool T_FORCE_SYNC_NATIVE_RUNTIME=false, typename T_WARNING_LEVELS=executor::WarningLevelsDefault<T_TYPE_POLICY>, bool T_DYNAMIC_ALLOCATION=true, typename T_STATUS_SPEC=executor::StatusSpecification<T_TYPE_POLICY>>
        struct Specification{
            using TYPE_POLICY = T_TYPE_POLICY;
            using T = typename TYPE_POLICY::DEFAULT;
            using STATUS_SPEC = T_STATUS_SPEC;
            using TI = T_TI;
            using TIMESTAMP = T_TIMESTAMP;
            using POLICY = T_POLICY;
            static constexpr T_TI ACTION_HISTORY_LENGTH = T_ACTION_HISTORY_LENGTH;
            static constexpr T_TI OUTPUT_DIM = T_OUTPUT_DIM;
            static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
            using EXECUTOR_SPEC = executor::Specification<TYPE_POLICY, TI, TIMESTAMP, POLICY, T_CONTROL_INTERVAL_INTERMEDIATE_NS, T_CONTROL_INTERVAL_NATIVE_NS, T_FORCE_SYNC_INTERMEDIATE, T_FORCE_SYNC_NATIVE, T_FORCE_SYNC_NATIVE_RUNTIME, T_WARNING_LEVELS, T_DYNAMIC_ALLOCATION, T_STATUS_SPEC>;
        };
        enum class ObservationComponentType : unsigned char {
            POSITION = 0,
            ORIENTATION_ROTATION_MATRIX = 1,
            ORIENTATION_QUATERNION = 2,
            LINEAR_VELOCITY = 3,
            ANGULAR_VELOCITY = 4,
            ANGULAR_VELOCITY_DELAYED = 5,
            LINEAR_VELOCITY_DELAYED = 6,
            ACTION_HISTORY = 7,
            LINEAR_ACCELERATION_BODY_FRAME = 8,
            LINEAR_VELOCITY_BODY_FRAME = 9,
            ROTOR_SPEEDS = 10,
        };
        template <typename T_TI>
        struct ObservationComponent{
            using TI = T_TI;
            ObservationComponentType type;
            TI parameter;
            TI offset;
            TI dim;
        };
        template <typename T_TI>
        struct ObservationLayout{
            using TI = T_TI;
            static constexpr TI MAX_COMPONENTS = 16;
            ObservationComponent<TI> components[MAX_COMPONENTS];
            TI component_count = 0;
            TI total_dim = 0;
            TI action_history_length = 0;
        };
        template <typename SPEC>
        struct Observation{
            using T = typename SPEC::T;
            T position[3];
            T orientation[4]; // Quaternion: w, x, y, z
            T linear_velocity[3];
            T angular_velocity[3];
            T previous_action[4];
            bool position_set = false;
            bool orientation_set = false;
            bool linear_velocity_set = false;
            bool angular_velocity_set = false;
            bool previous_action_set = false;
        };
        template <typename SPEC>
        struct Action{
            float action[SPEC::OUTPUT_DIM];
        };
    }
    template <typename SPEC>
    struct L2F{
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using T = typename TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        using TIMESTAMP = typename SPEC::TIMESTAMP;
        T action_history[SPEC::ACTION_HISTORY_LENGTH][SPEC::OUTPUT_DIM];
        static constexpr TI INPUT_DIM = 18 + SPEC::OUTPUT_DIM * SPEC::ACTION_HISTORY_LENGTH;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1, INPUT_DIM>, SPEC::DYNAMIC_ALLOCATION>> input;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1, SPEC::OUTPUT_DIM>, SPEC::DYNAMIC_ALLOCATION>> output;
        Executor<typename SPEC::EXECUTOR_SPEC> executor;
        TI steps_since_original_control_step;
        l2f::ObservationLayout<TI> observation_layout;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
