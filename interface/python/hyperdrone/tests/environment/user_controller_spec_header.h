#pragma once
#include <rl_tools/rl/environments/hyperdrone/controllers/direct/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <cstddef>

namespace hyperdrone_user {
    namespace controllers = rl_tools::rl::environments::hyperdrone::controllers;
    template <typename SPEC>
    struct CollectiveBuffer: controllers::direct::Buffer<SPEC> {
        rl_tools::Tensor<rl_tools::tensor::Specification<typename SPEC::T, typename SPEC::TI,
            rl_tools::tensor::Shape<typename SPEC::TI, SPEC::INSTANCES, 4>, SPEC::DYNAMIC_ALLOCATION>> actuation;
    };
    template <typename SPEC>
    struct Collective: controllers::Direct<SPEC, controllers::direct::Config<>::ACTION_INTERFACE> {
        static constexpr typename SPEC::TI COMMAND_DIM = 1;
        static constexpr const char* NAME = "collective";
        static constexpr bool SUPPORTS_SCALAR = false;
        using Buffer = CollectiveBuffer<SPEC>;
        bool initialized = false;
    };
    struct CollectiveConfig: controllers::direct::Config<> {
        template <typename POLICY, typename INDEX, INDEX N, bool DA = true>
        using Bind = Collective<controllers::Specification<POLICY, INDEX, N, DA>>;
    };
    struct SPEC: rl_tools::rl::environments::hyperdrone::presets::X500FPV<float, std::size_t> {
        using CONTROLLER = CollectiveConfig;
        static constexpr std::size_t N_CAMERA_AGENTS = 1;
        static constexpr std::size_t INSTANCES_PER_WORLD = 2;
        static constexpr std::size_t MAX_DRONES = 3;
        static constexpr std::size_t CAM_WIDTH = 16;
        static constexpr std::size_t CAM_HEIGHT = 16;
        using SHADING = rl_tools::rendering::raytracing::Low;
    };
    using WORLD = rl_tools::rl::environments::hyperdrone::World<SPEC>;
}

namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    bool init(DEVICE&, hyperdrone_user::Collective<SPEC>& controller) {
        controller.initialized = true;
        return true;
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, hyperdrone_user::CollectiveBuffer<SPEC>& buffer) {
        malloc(device, buffer.actuation);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, hyperdrone_user::CollectiveBuffer<SPEC>& buffer) {
        free(device, buffer.actuation);
    }
    template <typename SPEC>
    constexpr hyperdrone_user::controllers::Bounds<typename SPEC::T> command_bounds(const hyperdrone_user::Collective<SPEC>&, typename SPEC::TI) {
        return {0, 1};
    }
    template <typename DEVICE, typename SPEC, typename PARAMETERS, typename STATES, typename COMMANDS, typename RNG>
    auto control(DEVICE& device, const hyperdrone_user::Collective<SPEC>& controller,
                 typename hyperdrone_user::Collective<SPEC>::State&,
                 hyperdrone_user::CollectiveBuffer<SPEC>& buffer,
                 const Tensor<PARAMETERS>&, const Tensor<STATES>&, const Tensor<COMMANDS>& commands, RNG&) {
        utils::assert_exit(device, controller.initialized, "custom controller init was not dispatched");
        for (typename SPEC::TI row = 0; row < SPEC::INSTANCES; row++) {
            for (typename SPEC::TI motor = 0; motor < 4; motor++) {
                set(device, buffer.actuation, 2 * get(device, commands, row, 0) - 1, row, motor);
            }
        }
        return view_range(device, buffer.actuation, tensor::ViewSpec<0, SPEC::INSTANCES>{});
    }
    template <typename DEVICE, typename ACTION, typename STATE, typename INDEX>
    auto transition_action(DEVICE&, hyperdrone_user::CollectiveConfig, const Matrix<ACTION>&, const STATE& next, INDEX, INDEX component) {
        return next.last_action[component];
    }
}
