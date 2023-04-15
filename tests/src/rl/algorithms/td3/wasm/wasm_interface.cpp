#include "full_training.h"
#include <emscripten.h>

using TRAINING_STATE = TrainingState<TrainingConfig>;
extern "C" {
    EMSCRIPTEN_KEEPALIVE
    TRAINING_STATE* proxy_create_training_state(){
        TRAINING_STATE* ts = new TRAINING_STATE{};
        training_init(*ts);
        return ts;
    }

    EMSCRIPTEN_KEEPALIVE
    int proxy_training_step(TRAINING_STATE* ts){
        return training_step(*ts);
    }

    EMSCRIPTEN_KEEPALIVE
    int proxy_get_step(TRAINING_STATE* ts){
        return ts->step;
    }

EMSCRIPTEN_KEEPALIVE
int proxy_get_state_dim(){
    return TRAINING_STATE::TRAINING_CONFIG::ENVIRONMENT::State::DIM;
}

EMSCRIPTEN_KEEPALIVE
double proxy_get_state_value(TRAINING_STATE* ts, int env_index, int state_index){
    static_assert(TRAINING_STATE::TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS == 1);
    if(env_index < TRAINING_STATE::TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS && state_index < TRAINING_STATE::TRAINING_CONFIG::ENVIRONMENT::State::DIM){
        auto& env = ts->off_policy_runner.envs[env_index];
        auto& state = lic::get(ts->off_policy_runner.states, 0, (decltype(ts->device)::index_t) env_index);
        return lic::get_serialized_state(ts->device, env, state, state_index);
    }
    else{
        return -1337;
    }
}

EMSCRIPTEN_KEEPALIVE
    int proxy_get_evaluation_count(){
#ifdef LAYER_IN_C_ENABLE_EVALUATION
        return TRAINING_STATE::N_EVALUATIONS;
#endif
        return 0;
    }

EMSCRIPTEN_KEEPALIVE
    double proxy_get_evaluation_return(TRAINING_STATE* ts, int index){
#ifdef LAYER_IN_C_ENABLE_EVALUATION
        return ts->evaluation_returns[index];
#endif
        return 0;
    }

EMSCRIPTEN_KEEPALIVE
    void proxy_destroy_training_state(TRAINING_STATE* ts){
        training_destroy(*ts);
        delete ts;
    }
}