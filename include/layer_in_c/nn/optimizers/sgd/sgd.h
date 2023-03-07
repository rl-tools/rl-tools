#ifndef LAYER_IN_C_NN_OPTIMIZERS_SGD_H
#define LAYER_IN_C_NN_OPTIMIZERS_SGD_H

#include <layer_in_c/nn/parameters/parameters.h>

namespace layer_in_c::nn::optimizers::sgd {
    template<typename T>
    struct DefaultParameters{
    public:
        static constexpr T ALPHA = 0.001;
    };

}
namespace layer_in_c::nn::parameters{
    struct SGD{
        template <typename CONTAINER>
        struct instance: Gradient::instance<CONTAINER>{};
    };
}

#endif
