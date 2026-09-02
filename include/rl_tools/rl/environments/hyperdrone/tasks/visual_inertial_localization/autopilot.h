#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_AUTOPILOT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_AUTOPILOT_H

#include "../../world.h"
#include "../../../../../numeric_types/policy.h"
#include "../../../../../nn/layers/dense/layer.h"
#include "../../../../../nn/layers/gru/layer.h"
#include "../../../../../nn_models/sequential/model.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::tasks::visual_inertial_localization {
    // the task's autopilot: RAPTOR, a recurrent state-feedback policy, flies the waypoint route
    // from [clamped position error to the current waypoint | R | v | omega | last action] read
    // through the privileged observation chain, so the benchmark's IMU noise never leaks into
    // the motion generation. The architecture is pinned here; the weights are loaded at init
    // from the checkpoint the task specification references
    namespace raptor {
        template <typename T_T, typename T_TI>
        struct Architecture {
            using T = T_T;
            using TI = T_TI;
            using TYPE_POLICY = numeric_types::Policy<T>;
            static constexpr TI INPUT_DIM = 22;
            static constexpr TI HIDDEN_DIM = 16;
            static constexpr TI OUTPUT_DIM = 4;
            using INPUT_LAYER = nn::layers::dense::BindConfiguration<nn::layers::dense::Configuration<TYPE_POLICY, TI, HIDDEN_DIM, nn::activation_functions::ActivationFunction::RELU, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Input>>;
            using RECURRENT_LAYER = nn::layers::gru::BindConfiguration<nn::layers::gru::Configuration<TYPE_POLICY, TI, HIDDEN_DIM, nn::parameters::groups::Normal, false>>;
            using OUTPUT_LAYER = nn::layers::dense::BindConfiguration<nn::layers::dense::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, nn::activation_functions::ActivationFunction::IDENTITY, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Output>>;
            using MODULE_CHAIN = nn_models::sequential::Module<INPUT_LAYER, RECURRENT_LAYER, OUTPUT_LAYER>;
        };
    }
    template <typename T_BASE_WORLD>
    struct Autopilot {
        using BASE_WORLD = T_BASE_WORLD;
        using T = typename BASE_WORLD::T;
        using TI = typename BASE_WORLD::TI;
        static constexpr TI INSTANCES = BASE_WORLD::INSTANCES;
        using ARCHITECTURE = raptor::Architecture<T, TI>;
        using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecificationPrivileged<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecificationPrivileged<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecificationPrivileged<T, TI,
                l2f::observation::ActionHistory<l2f::observation::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
        static constexpr TI OBSERVATION_DIM = OBSERVATION_TYPE::DIM;
        static constexpr TI ACTION_DIM = ARCHITECTURE::OUTPUT_DIM;
        static_assert(OBSERVATION_DIM == ARCHITECTURE::INPUT_DIM, "the autopilot observation chain must match the policy input");
        static_assert(ACTION_DIM == BASE_WORLD::ACTION_DIM, "the autopilot drives the base world's motors");
        using MODEL = nn_models::sequential::Build<nn::capability::Forward<true, false>, typename ARCHITECTURE::MODULE_CHAIN, tensor::Shape<TI, 1, INSTANCES, OBSERVATION_DIM>>;
        MODEL model;
        typename MODEL::template State<true> state;
        typename MODEL::template Buffer<true> buffer;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, INSTANCES, OBSERVATION_DIM>>> observations;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, INSTANCES, ACTION_DIM>>> actions;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
