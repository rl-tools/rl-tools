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