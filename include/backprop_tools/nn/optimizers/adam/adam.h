#ifndef BACKPROP_TOOLS_NN_OPTIMIZERS_ADAM_ADAM_H
#define BACKPROP_TOOLS_NN_OPTIMIZERS_ADAM_ADAM_H

#include <backprop_tools/nn/parameters/parameters.h>

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::nn::optimizers{
    namespace adam{
        template<typename T_T, typename T_TI>
        struct DefaultParametersTF {
            using T = T_T;
            using TI = T_TI;
            static constexpr T ALPHA = 0.001;
            static constexpr T BETA_1 = 0.9;
            static constexpr T BETA_2 = 0.999;
            static constexpr T EPSILON = 1e-7;

        };
        template<typename T_T, typename T_TI>
        struct DefaultParametersTorch {
            using T = T_T;
            using TI = T_TI;
            static constexpr T ALPHA = 0.001;
            static constexpr T BETA_1 = 0.9;
            static constexpr T BETA_2 = 0.999;
            static constexpr T EPSILON = 1e-8;

        };
    }
    template<typename T_PARAMETERS>
    struct Adam{
        using PARAMETERS = T_PARAMETERS;
        using T = typename PARAMETERS::T;
        using TI = typename PARAMETERS::TI;
        typename PARAMETERS::T first_order_moment_bias_correction;
        typename PARAMETERS::T second_order_moment_bias_correction;
        T alpha = PARAMETERS::ALPHA;
        TI age = 1;
    };


}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::nn::parameters {
    struct Adam{
        template <typename CONTAINER>
        struct instance: Gradient::instance<CONTAINER>{
            CONTAINER gradient_first_order_moment;
            CONTAINER gradient_second_order_moment;
        };
    };
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif