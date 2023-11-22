//#define RL_TOOLS_DISABLE_EVALUATION
#include "training.h"
int main(int argc, char** argv){
    TI seed = 0;
    if(argc > 1){
        seed = std::atoi(argv[1]);
    }
    auto return_code = run(seed);
    return return_code < 4 ? 0 : return_code;
}