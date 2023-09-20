
#include "training.h"

int main(){
    using namespace multirotor_training::config;

    using CONFIG = multirotor_training::config::Config;

    multirotor_training::operations::TrainingState ts;

    multirotor_training::operations::init(ts);
    for(Config::TI step_i=0; step_i < Config::STEP_LIMIT; step_i++){
        multirotor_training::operations::step(ts);
    }
    return 0;
}