
#include "training.h"
#include <malloc.h>


template <typename T_ABLATION_SPEC>
void run(){
    using namespace multirotor_training::config;

    using CONFIG = multirotor_training::config::Config<T_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;

    for (TI run_i = 0; run_i < 10; run_i++){
        std::cout << "Run " << run_i << "\n";
        struct mallinfo mi = mallinfo();
        std::cout << "Total allocated space before declaration: " << mi.uordblks << " bytes\n";
        multirotor_training::operations::TrainingState<T_ABLATION_SPEC> ts;
        mi = mallinfo();
        std::cout << "Total allocated space after declaration: " << mi.uordblks << " bytes\n";
        multirotor_training::operations::init(ts, run_i);
        mi = mallinfo();
        std::cout << "Total allocated space after init: " << mi.uordblks << " bytes\n";

        for(TI step_i=0; step_i < CONFIG::STEP_LIMIT; step_i++){
            multirotor_training::operations::step(ts);
        }

        mi = mallinfo();
        std::cout << "Total allocated space after training: " << mi.uordblks << " bytes\n";
        multirotor_training::operations::destroy(ts);
        mi = mallinfo();
        std::cout << "Total allocated space after destruction: " << mi.uordblks << " bytes\n";
    }
}


int main(){
    run<multirotor_training::config::DEFAULT_ABLATION_SPEC>();
    return 0;
}