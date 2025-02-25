#include "interface.h"
int main(){
    State state;
    init(&state, 0);
    set_action(&state, 0, 0);
    set_action(&state, 1, 0);
    set_action(&state, 2, 0);
    set_action(&state, 3, 0);
    step(&state);

    return 0;
}

