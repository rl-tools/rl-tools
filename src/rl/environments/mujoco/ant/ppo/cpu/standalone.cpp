#include "training.h"

int main(int argc, char** argv){
    if(argc > 1){
        int seed = std::stoi(argv[1]);
        run(seed);
    }
    else{
        std::cout << "Usage: " << argv[0] << " <seed>" << std::endl;
    }
    return 0;
}