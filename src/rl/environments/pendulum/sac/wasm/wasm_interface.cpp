#include "full_training.h"
#include <emscripten.h>

using TRAINING_STATE = TrainingState<TrainingConfig>;
DEVICE device;
extern "C" {
    EMSCRIPTEN_KEEPALIVE
    TRAINING_STATE* proxy_create_training_state(int seed){
        TRAINING_STATE* ts = new TRAINING_STATE{};
        training_init(device, *ts, seed);
        return ts;
    }

    EMSCRIPTEN_KEEPALIVE
    int proxy_training_step(TRAINING_STATE* ts){
        return training_step(device, *ts);
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
        auto& state = rlt::get(ts->off_policy_runner.states, 0, (decltype(device)::index_t) env_index);
        return rlt::get_serialized_state(device, env, state, state_index);
    }
    else{
        return -1337;
    }
}

EMSCRIPTEN_KEEPALIVE
int proxy_get_episode(TRAINING_STATE* ts, int env_index){
    static_assert(TRAINING_STATE::TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS == 1);
    if(env_index < TRAINING_STATE::TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS){
        auto episode = ts->off_policy_runner.episode_stats[env_index].next_episode_i;
        return episode;
    }
    else{
        return -1337;
    }
}

EMSCRIPTEN_KEEPALIVE
double proxy_get_episode_return(TRAINING_STATE* ts, int env_index, int episode_i){
    static_assert(TRAINING_STATE::TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS == 1);
    if(env_index < TRAINING_STATE::TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS && episode_i < TRAINING_STATE::TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::EPISODE_STATS_BUFFER_SIZE){
        return get(ts->off_policy_runner.episode_stats[env_index].returns, episode_i, 0);
    }
    else{
        return -1337;
    }
}

EMSCRIPTEN_KEEPALIVE
    int proxy_get_evaluation_count(){
#ifdef RL_TOOLS_ENABLE_EVALUATION
        return TRAINING_STATE::N_EVALUATIONS;
#endif
        return 0;
    }

EMSCRIPTEN_KEEPALIVE
    double proxy_get_evaluation_return(TRAINING_STATE* ts, int index){
#ifdef RL_TOOLS_ENABLE_EVALUATION
        return ts->evaluation_returns[index];
#endif
        return 0;
    }

EMSCRIPTEN_KEEPALIVE
    void proxy_destroy_training_state(TRAINING_STATE* ts){
        training_destroy(device, *ts);
        delete ts;
    }
}