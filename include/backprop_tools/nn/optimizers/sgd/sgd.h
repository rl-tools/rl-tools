#ifndef BACKPROP_TOOLS_NN_OPTIMIZERS_SGD_H
#define BACKPROP_TOOLS_NN_OPTIMIZERS_SGD_H

#include <backprop_tools/nn/parameters/parameters.h>

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::nn::optimizers::sgd {
    template<typename T>
    struct DefaultParameters{
    public:
        static constexpr T ALPHA = 0.001;
    };

}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::nn::parameters{
    struct SGD{
        template <typename CONTAINER>
        struct instance: Gradient::instance<CONTAINER>{};
    };
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif
