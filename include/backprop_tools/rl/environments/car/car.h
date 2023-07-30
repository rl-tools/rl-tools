#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_CAR_CAR_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_CAR_CAR_H

#include <backprop_tools/math/operations_generic.h>

namespace backprop_tools::rl::environments::car {
    template <typename T>
    struct Tire{
        T B;
        T C;
        T D;
    };
    template <typename T>
    struct Parameters{
        T g   = 9.81;
        T m   = 0.041;
        T I   = 27.8e-6;
        T lf  = 0.029;
        T lr  = 0.033;
        Tire<T> tf = {2.5790, 1.2000, 0.1920};
        Tire<T> tr = {3.3852, 1.2691, 0.1737};
        T cm  = 0.287;
        T cr0 = 0.0;
        T cr2 = 0.00035;
        T vt  = 0.01;
        T dt  = 0.01;
    };
    template <typename T_T, typename T_TI>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
    };

    template <typename T, typename TI>
    struct State{
        static constexpr TI DIM = 6;
        T x;
        T y;
        T mu;
        T vx;
        T vy;
        T omega;
    };

}

namespace backprop_tools::rl::environments{
    template <typename T_SPEC>
    struct Car{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using State = car::State<T, TI>;
        static constexpr TI OBSERVATION_DIM = 6;
        static constexpr TI OBSERVATION_DIM_PRIVILEGED = OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = 2;
        using PARAMETERS = car::Parameters<T>;

        PARAMETERS parameters;
    };
}







#endif
