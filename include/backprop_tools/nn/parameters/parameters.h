#ifndef BACKPROP_TOOLS_NN_PARAMETERS_PARAMETERS_H
#define BACKPROP_TOOLS_NN_PARAMETERS_PARAMETERS_H

namespace backprop_tools::nn::parameters{
    struct Plain{
        // todo: evaluate replacing the instance mechanism with a tag similar to the container type tags
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
