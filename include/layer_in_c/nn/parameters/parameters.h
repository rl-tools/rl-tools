#ifndef LAYER_IN_C_NN_PARAMETERS_PARAMETERS_H
#define LAYER_IN_C_NN_PARAMETERS_PARAMETERS_H

namespace layer_in_c::nn::parameters{
    struct Plain{
        template <typename CONTAINER>
        struct instance{
            CONTAINER parameters;
        };
    };
    struct Gradient{
        template <typename CONTAINER>
        struct instance: Plain::instance<CONTAINER>{
            CONTAINER gradient;
        };
    };
}
#endif
