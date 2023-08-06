#include "training.h"
struct AblationSpec: parameters::DefaultAblationSpec{
    static constexpr bool DISTURBANCE = true;
    static constexpr bool OBSERVATION_NOISE = true;
    static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
    static constexpr bool ROTOR_DELAY = true;
    static constexpr bool ACTION_HISTORY = true;
    static constexpr bool ENABLE_CURRICULUM = true;
};


int main(){
    train<AblationSpec>();
    return 0;
}