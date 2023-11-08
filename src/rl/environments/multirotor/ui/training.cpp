
#include "training.h"


template <typename T_ABLATION_SPEC>
void run(){
    using namespace multirotor_training::config;

    using CONFIG = multirotor_training::config::Config<T_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;

    for (TI run_i = 0; run_i < 10; run_i++){
        std::cout << "Run " << run_i << "\n";
        multirotor_training::operations::TrainingState<CONFIG> ts;
        multirotor_training::operations::init(ts, run_i);

        for(TI step_i=0; step_i < CONFIG::STEP_LIMIT; step_i++){
            multirotor_training::operations::step(ts);
        }

        multirotor_training::operations::destroy(ts);
    }
}


int main(){
    run<multirotor_training::config::DEFAULT_ABLATION_SPEC>();
    return 0;
}