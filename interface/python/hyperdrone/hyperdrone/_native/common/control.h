#pragma once

#include <rl_tools/rl/environments/hyperdrone/controllers/operations_cpu.h>
#include <array>

namespace hyperdrone {
    using rl_tools::command_bounds;
    template<int CONTROL>
    struct ControlConfig {
        static_assert(CONTROL >= 0 && CONTROL <= 2, "unknown control modality");
        using TYPE = rl_tools::utils::typing::conditional_t<CONTROL == 0,
            rl_tools::rl::environments::hyperdrone::controllers::direct::Config<>,
            rl_tools::utils::typing::conditional_t<CONTROL == 1,
                rl_tools::rl::environments::hyperdrone::controllers::ctbr::Config,
                rl_tools::rl::environments::hyperdrone::controllers::raptor::Config>>;
    };

    template<typename CONTROLLER>
    struct ActionBounds {
        static constexpr auto lower = []() {
            std::array<float, CONTROLLER::COMMAND_DIM> values{};
            for(unsigned int index = 0; index < CONTROLLER::COMMAND_DIM; index++) {
                values[index] = command_bounds(CONTROLLER{}, index).lower;
            }
            return values;
        }();
        static constexpr auto upper = []() {
            std::array<float, CONTROLLER::COMMAND_DIM> values{};
            for(unsigned int index = 0; index < CONTROLLER::COMMAND_DIM; index++) {
                values[index] = command_bounds(CONTROLLER{}, index).upper;
            }
            return values;
        }();
    };
}
