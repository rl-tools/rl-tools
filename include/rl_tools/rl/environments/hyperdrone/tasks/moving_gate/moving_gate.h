#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_MOVING_GATE_MOVING_GATE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_MOVING_GATE_MOVING_GATE_H

#include "../../world.h"

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::moving_gate {

    template <typename T_T, typename T_TI, typename T_NEXT_COMPONENT>
    struct ComponentSpecification {
        using T = T_T;
        using TI = T_TI;
        using NEXT_COMPONENT = T_NEXT_COMPONENT;
    };
    // the gate trajectory: scene-frame center free-space-sampled at reset, fixed facing,
    // oscillation along the axis
    template <typename T_SPEC>
    struct ParametersMovingGate: T_SPEC::NEXT_COMPONENT {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        T gate_center[3] = {0, 0, 0};
        T gate_axis[3] = {0, 0, 1};
        T gate_orientation[4] = {1, 0, 0, 0};  // gate plane normal = orientation-rotated +X
        T gate_amplitude = 0;
        T gate_frequency = 0;
        T gate_aperture_radius = 0;
    };
    template <typename T_SPEC>
    struct StateMovingGate: T_SPEC::NEXT_COMPONENT {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using NEXT_COMPONENT = typename SPEC::NEXT_COMPONENT;
        static constexpr bool REQUIRES_INTEGRATION = false;
        static constexpr typename SPEC::TI DIM = 1 + NEXT_COMPONENT::DIM;
        T gate_phase = 0;
        bool gate_passed = false;
        bool gate_crashed = false;
    };

    template <typename DEVICE, typename PARAMETERS, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void gate_pose(DEVICE& device, const PARAMETERS& parameters, T phase, float out[12]){
        T position[3];
        T offset = parameters.gate_amplitude * math::sin(device.math, phase);
        position[0] = parameters.gate_center[0] + offset * parameters.gate_axis[0];
        position[1] = parameters.gate_center[1] + offset * parameters.gate_axis[1];
        position[2] = parameters.gate_center[2] + offset * parameters.gate_axis[2];
        T R[3][3];
        rl::environments::l2f::quaternion_to_rotation_matrix<DEVICE, T>(parameters.gate_orientation, R);
        for(unsigned row = 0; row < 3; row++){
            out[row * 4 + 0] = (float)R[row][0];
            out[row * 4 + 1] = (float)R[row][1];
            out[row * 4 + 2] = (float)R[row][2];
            out[row * 4 + 3] = (float)position[row];
        }
    }

    template <typename T_NEXT_WORLD>
    struct Specification {
        using NEXT_WORLD = T_NEXT_WORLD;
        using T = typename NEXT_WORLD::T;
        using TI = typename NEXT_WORLD::TI;
        // derive-and-shadow task knobs
        static constexpr T GATE_AMPLITUDE_MIN = 0.2;
        static constexpr T GATE_AMPLITUDE_MAX = 0.6;
        static constexpr T GATE_FREQUENCY_MIN = 0.1;
        static constexpr T GATE_FREQUENCY_MAX = 0.4;
        static constexpr T GATE_APERTURE_RADIUS = 0.4;
        static constexpr T GATE_PASS_REWARD = 10;
    };

    template <typename T_TASK_SPEC>
    struct World: T_TASK_SPEC::NEXT_WORLD {
        using TASK_SPEC = T_TASK_SPEC;
        using NEXT_WORLD = typename TASK_SPEC::NEXT_WORLD;
        using T = typename NEXT_WORLD::T;
        using TI = typename NEXT_WORLD::TI;
        static_assert(NEXT_WORLD::RENDERER_CONFIG::NUM_OVERLAYS > 0, "the moving-gate task needs entity slots: set MAX_ENTITY_SLOTS_PER_INSTANCE in the base Specification");

        using Parameters = ParametersMovingGate<ComponentSpecification<T, TI, typename NEXT_WORLD::Parameters>>;
        using State = StateMovingGate<ComponentSpecification<T, TI, typename NEXT_WORLD::State>>;
        // privileged contribution for the asymmetric critic: gate-relative position, gate
        // velocity, phase (sin/cos); the actor stays vision-only
        static constexpr TI GATE_STATE_DIM = 8;
        struct ObservationPrivileged {
            static constexpr TI DIM = NEXT_WORLD::OBSERVATION_DIM_PRIVILEGED + GATE_STATE_DIM;
            using SHAPE = tensor::Shape<TI, DIM>;
        };
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = ObservationPrivileged::DIM;

        std::string gate_asset_path;  // set before init
        TI entity_kind_index = 0;     // assigned at registration

        using GATE_POSE_STAGING_SPEC = tensor::Specification<float, TI, tensor::Shape<TI, NEXT_WORLD::INSTANCES, 2, 12>>;
        Tensor<GATE_POSE_STAGING_SPEC> gate_pose_staging;  // renderer-device shutter-open/close gate poses for the CUDA render path
        void* cuda_gate_pose_staging = nullptr;  // pinned host mirror
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
