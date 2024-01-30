
#include "full_training.h"
int main(){
    DEVICE device;
    TrainingState<TrainingConfig> ts;

    training_init(device, ts, 10);

    bool finished = false;
    while(!finished){
        finished = training_step(device, ts);
    }
    training_destroy(device, ts);
    return 0;
}