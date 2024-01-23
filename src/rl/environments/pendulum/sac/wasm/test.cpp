
#include "full_training.h"
int main(){
    TrainingState<TrainingConfig> ts;

    training_init(ts, 10);

    bool finished = false;
    while(!finished){
        finished = training_step(ts);
    }
    training_destroy(ts);
    return 0;
}