

#include "../../multirotor.h"
namespace layer_in_c::rl::environments::multirotor::parameters::reward_functions{
    template<typename DEVICE, typename SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, const typename SPEC::T action[rl::environments::Multirotor<SPEC>::ACTION_DIM], const typename rl::environments::Multirotor<SPEC>::State& next_state){
//        static_assert(utils::typing::is_same_v<typename SPEC::PARAMETERS::MDP::REWARD_FUNCTION, AbsExp<typename SPEC::T>>);
        constexpr auto STATE_DIM = rl::environments::Multirotor<SPEC>::STATE_DIM;
        constexpr auto ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        using T = typename SPEC::T;
        T acc = 0;
        for(typename DEVICE::index_t state_i = 0; state_i < STATE_DIM; state_i++){
            if(state_i < 3){
                acc += state.state[state_i] * state.state[state_i] * env.parameters.mdp.reward.position;
            }
            else{
                if(state_i < 3+4){
                    T v = state_i == 3 ? state.state[state_i] - 1 : state.state[state_i];
                    acc += v * v * env.parameters.mdp.reward.orientation;
                }
                else{
                    if(state_i < 3+4+3){
                        acc += state.state[state_i] * state.state[state_i] * env.parameters.mdp.reward.linear_velocity;
                    }
                    else{
                        acc += state.state[state_i] * state.state[state_i] * env.parameters.mdp.reward.angular_velocity;
                    }
                }
            }
        }
        for(typename DEVICE::index_t action_i = 0; action_i < ACTION_DIM; action_i++){
            T v = action[action_i] - env.parameters.mdp.reward.action_baseline;
            acc += v * v * env.parameters.mdp.reward.action;
        }
        T variance_position = env.parameters.mdp.init.max_position * env.parameters.mdp.init.max_position/(2*2) * env.parameters.mdp.reward.position;
        T variance_orientation = env.parameters.mdp.reward.orientation;
        T variance_linear_velocity = env.parameters.mdp.init.max_linear_velocity * env.parameters.mdp.init.max_linear_velocity/(2*2) * env.parameters.mdp.reward.linear_velocity;
        T variance_angular_velocity = env.parameters.mdp.init.max_angular_velocity * env.parameters.mdp.init.max_angular_velocity/(2*2) * env.parameters.mdp.reward.angular_velocity;
        T variance_action = env.parameters.mdp.reward.action;
        T standardization_factor = (variance_position * 3 + variance_orientation * 4 + variance_linear_velocity * 3 + variance_angular_velocity * 3 + variance_action * 4);
        standardization_factor *= 100;
        return math::exp(typename DEVICE::SPEC::MATH(), -acc/standardization_factor);
    }
}

