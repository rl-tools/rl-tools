#include "layer_in_c/rl/environments/environments.h"
#include <thread>

namespace layer_in_c{
    template<typename ENVIRONMENT, typename UI, typename POLICY, auto STEP_LIMIT, auto TIME_LAPSE=1>
    typename POLICY::T evaluate_visual(const ENVIRONMENT env, UI& ui, POLICY &policy, const typename ENVIRONMENT::State initial_state) {
        typedef typename POLICY::T T;
        typename ENVIRONMENT::State state;
        state = initial_state;
        T episode_return = 0;
        for (index_t i = 0; i < STEP_LIMIT; i++) {
            set_state(ui, state);
            T observation[ENVIRONMENT::OBSERVATION_DIM];
            observe(env, state, observation);
            T action[ENVIRONMENT::ACTION_DIM];
            evaluate(policy, observation, action);
            T action_clipped[ENVIRONMENT::ACTION_DIM];
            for(index_t action_i=0; action_i<ENVIRONMENT::ACTION_DIM; action_i++){
                action_clipped[action_i] = std::clamp<T>(action[action_i], -1, 1);
            }
            typename ENVIRONMENT::State next_state;
            T dt = step(env, state, action_clipped, next_state);
            std::this_thread::sleep_for(std::chrono::milliseconds((index_t)(dt*1000/TIME_LAPSE)));
            T r = reward(env, state, action_clipped, next_state);
            state = next_state;
            episode_return += r;
            bool terminated_flag = terminated(env, state);
            if (terminated_flag) {
                break;
            }
        }
        return episode_return;
    }
}
