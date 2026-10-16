#pragma once

#include <rl_tools/rl/environments/hyperdrone/controllers/operations_cpu.h>
#include <array>

namespace hyperdrone {
    using rl_tools::command_bounds;
    namespace l2f = rl_tools::rl::environments::l2f;
    template<int CONTROL>
    struct ControlConfig {
        static_assert(CONTROL >= 0 && CONTROL <= 2, "unknown control modality");
        using TYPE = rl_tools::utils::typing::conditional_t<CONTROL == 0,
            rl_tools::rl::environments::hyperdrone::controllers::direct::Config<l2f::parameters::ActionInterface::DIRECT_MOTOR>,
            rl_tools::utils::typing::conditional_t<CONTROL == 1,
                rl_tools::rl::environments::hyperdrone::controllers::direct::Config<l2f::parameters::ActionInterface::CTBR>,
                rl_tools::rl::environments::hyperdrone::controllers::raptor::Config>>;
    };
    template<int ACTUATION>
    struct ActuationConfig {
        static_assert(ACTUATION >= 0 && ACTUATION <= 1, "unknown actuation model");
        template<typename BASE>
        using Apply = rl_tools::utils::typing::conditional_t<ACTUATION == 0, BASE, l2f::WithIdealRates<BASE>>;
    };
    template<typename ACTUATION>
    constexpr const char* actuation_name(){
        return rl_tools::utils::typing::is_same_v<ACTUATION, l2f::actuation::IdealRates> ? "ideal" : "rotors";
    }

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
