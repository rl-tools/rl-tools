#define STDLIB
#include "interface.h"
#include <iostream>
int main(){
    State state(0);
    std::cout << state.get_parameters() << std::endl;

    return 0;
}

