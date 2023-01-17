

#include "../../multirotor.h"
namespace layer_in_c::rl::environments::multirotor::parameters::reward_functions{
    template<typename DEVICE, typename SPEC, typename WEIGHTS>
    static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, const typename SPEC::T action[rl::environments::Multirotor<SPEC>::ACTION_DIM], const typename rl::environments::Multirotor<SPEC>::State& next_state) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        T quaternion_w = state.state[3];
        T orientation_cost = math::abs(2 * math::acos(typename DEVICE::SPEC::MATH(), quaternion_w));
        T position_cost = utils::vector_operations::norm<DEVICE, T, 3>(state.state);
        T linear_vel_cost = utils::vector_operations::norm<DEVICE, T, 3>(&state.state[3+4]);
        T angular_vel_cost = utils::vector_operations::norm<DEVICE, T, 3>(&state.state[3+4+3]);
        T action_diff[ACTION_DIM];
        utils::vector_operations::sub<DEVICE, T, ACTION_DIM>(action, utils::vector_operations::mean<DEVICE, T, ACTION_DIM>(action), action_diff);
        T action_cost = utils::vector_operations::norm<DEVICE, T, ACTION_DIM>(action_diff);
        T weighted_abs_cost = WEIGHTS::position * position_cost + WEIGHTS::orientation * orientation_cost + WEIGHTS::linear_vel * linear_vel_cost + WEIGHTS::angular_vel * angular_vel_cost + WEIGHTS::action * action_cost;
        T r = math::exp(typename DEVICE::SPEC::MATH(), -weighted_abs_cost);
        return r * 10;
//            return -weighted_abs_cost;
    }
}

