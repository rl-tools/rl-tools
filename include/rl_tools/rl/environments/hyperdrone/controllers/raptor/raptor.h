#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_RAPTOR_RAPTOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_RAPTOR_RAPTOR_H

#include "../controllers.h"
#include "../../../l2f/multirotor.h"
#include "../../../../../numeric_types/policy.h"
#include "../../../../../nn/layers/dense/layer.h"
#include "../../../../../nn/layers/gru/layer.h"
#include "../../../../../nn_models/sequential/model.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::controllers::raptor {
    inline constexpr const char* CHECKPOINT = "conta:b6bd082c60fcb27f42a2b5f2f635587d2b60d170";

    template <typename T_TYPE_POLICY, typename T_TI, T_TI T_INSTANCES, bool T_DYNAMIC_ALLOCATION = true>
    struct Specification: controllers::Specification<T_TYPE_POLICY, T_TI, T_INSTANCES, T_DYNAMIC_ALLOCATION> {
        using TYPE_POLICY = T_TYPE_POLICY;
        using T = typename TYPE_POLICY::DEFAULT;
        using TI = T_TI;
        static constexpr TI INPUT_DIM = 22;
        static constexpr TI HIDDEN_DIM = 16;
        static constexpr TI OUTPUT_DIM = 4;
        static constexpr T POSITION_CLIP = 1;
        static constexpr T VELOCITY_CLIP = 2;
        static constexpr T YAW_ERROR_LIMIT = 0.25;
        static constexpr T YAW_RATE_LIMIT = 1;
        using INPUT_LAYER = nn::layers::dense::BindConfiguration<nn::layers::dense::Configuration<TYPE_POLICY, TI, HIDDEN_DIM, nn::activation_functions::ActivationFunction::RELU, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Input>>;
        using RECURRENT_LAYER = nn::layers::gru::BindConfiguration<nn::layers::gru::Configuration<TYPE_POLICY, TI, HIDDEN_DIM, nn::parameters::groups::Normal, false>>;
        using OUTPUT_LAYER = nn::layers::dense::BindConfiguration<nn::layers::dense::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, nn::activation_functions::ActivationFunction::IDENTITY, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Output>>;
        using MODULE_CHAIN = nn_models::sequential::Module<INPUT_LAYER, RECURRENT_LAYER, OUTPUT_LAYER>;
    };

    // Targets use the dynamics frame; yaw defines the policy's reference heading.
    template <typename T>
    struct Target {
        T position[3] = {};
        T linear_velocity[3] = {};
        T yaw = 0;
    };

    template <typename SPEC>
    struct State;
    template <typename SPEC>
    struct Buffer;

    template <typename T_SPEC>
    struct Controller {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Target = raptor::Target<T>;
        static constexpr const char* NAME = "position";
        static constexpr bool SUPPORTS_SCALAR = false;
        static constexpr TI INSTANCES = SPEC::INSTANCES;
        using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecificationPrivileged<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecificationPrivileged<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecificationPrivileged<T, TI,
                l2f::observation::ActionHistory<l2f::observation::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
        static constexpr TI OBSERVATION_DIM = OBSERVATION_TYPE::DIM;
        static constexpr TI COMMAND_DIM = 4;
        static constexpr TI ACTION_DIM = SPEC::OUTPUT_DIM;
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        using State = raptor::State<SPEC>;
        using Buffer = raptor::Buffer<SPEC>;
        static_assert(OBSERVATION_DIM == SPEC::INPUT_DIM);
        static_assert(INSTANCES > 0);
        using MODEL = nn_models::sequential::Build<nn::capability::Forward<SPEC::DYNAMIC_ALLOCATION, false>, typename SPEC::MODULE_CHAIN, tensor::Shape<TI, 1, INSTANCES, OBSERVATION_DIM>>;
        MODEL model;
    };
    template <typename SPEC>
    struct State {
        using CONTROLLER = Controller<SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        typename CONTROLLER::MODEL::template State<SPEC::DYNAMIC_ALLOCATION> recurrent;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SPEC::INSTANCES, SPEC::OUTPUT_DIM>, SPEC::DYNAMIC_ALLOCATION>> previous_actions;
    };
    template <typename SPEC>
    struct Buffer {
        using CONTROLLER = Controller<SPEC>;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        typename CONTROLLER::MODEL::template Buffer<SPEC::DYNAMIC_ALLOCATION> network;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SPEC::INSTANCES, SPEC::INPUT_DIM>, SPEC::DYNAMIC_ALLOCATION>> observations;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, SPEC::INSTANCES>, SPEC::DYNAMIC_ALLOCATION>> expanded_mask;
    };
    struct Config {
        template <typename TYPE_POLICY, typename TI, TI INSTANCES, bool DYNAMIC_ALLOCATION = true>
        using Bind = Controller<Specification<TYPE_POLICY, TI, INSTANCES, DYNAMIC_ALLOCATION>>;
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr unsigned int COMMAND_DIM = 4;
        template <typename BASE>
        struct DynamicsParameters: BASE {
            static constexpr auto ACTION_INTERFACE = Config::ACTION_INTERFACE;
        };
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
