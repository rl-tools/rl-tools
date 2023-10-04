
#include "training.h"
#include <malloc.h>

int main(){
    using namespace multirotor_training::config;

    using CONFIG = multirotor_training::config::Config;

    for (CONFIG::TI run_i = 0; run_i < 10; run_i++){
        std::cout << "Run " << run_i << "\n";
        struct mallinfo mi = mallinfo();
        std::cout << "Total allocated space before declaration: " << mi.uordblks << " bytes\n";
        multirotor_training::operations::TrainingState ts;
        mi = mallinfo();
        std::cout << "Total allocated space after declaration: " << mi.uordblks << " bytes\n";
        multirotor_training::operations::init(ts, run_i);
        mi = mallinfo();
        std::cout << "Total allocated space after init: " << mi.uordblks << " bytes\n";

        for(Config::TI step_i=0; step_i < Config::STEP_LIMIT; step_i++){
            multirotor_training::operations::step(ts);
        }

        mi = mallinfo();
        std::cout << "Total allocated space after training: " << mi.uordblks << " bytes\n";
        multirotor_training::operations::destroy(ts);
        mi = mallinfo();
        std::cout << "Total allocated space after destruction: " << mi.uordblks << " bytes\n";
    }
    return 0;
}