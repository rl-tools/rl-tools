#include "training.h"
struct AblationSpecBase: parameters::DefaultAblationSpec{
    static constexpr bool DISTURBANCE = true;
    static constexpr bool OBSERVATION_NOISE = true;
    static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
    static constexpr bool ROTOR_DELAY = true;
    static constexpr bool ACTION_HISTORY = true;
    static constexpr bool ENABLE_CURRICULUM = true;
    static constexpr TI NUM_RUNS = 10;
};

struct AblationSpec_0: AblationSpecBase{
    static constexpr bool DISTURBANCE = false;
};

struct AblationSpec_1: AblationSpecBase{
    static constexpr bool OBSERVATION_NOISE = false;
};

struct AblationSpec_2: AblationSpecBase{
    static constexpr bool ASYMMETRIC_ACTOR_CRITIC = false;
};

struct AblationSpec_3: AblationSpecBase{
    static constexpr bool ROTOR_DELAY = false;
};

struct AblationSpec_4: AblationSpecBase{
    static constexpr bool ACTION_HISTORY = false;
};

struct AblationSpec_5: AblationSpecBase{
    static constexpr bool ENABLE_CURRICULUM = false;
};


int main(){
    train<AblationSpecBase>();
    train<AblationSpec_0>();
    train<AblationSpec_1>();
    train<AblationSpec_2>();
    train<AblationSpec_3>();
    train<AblationSpec_4>();
    train<AblationSpec_5>();
    return 0;
}