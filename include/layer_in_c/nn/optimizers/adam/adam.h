#ifndef LAYER_IN_C_NN_OPTIMIZERS_ADAM
#define LAYER_IN_C_NN_OPTIMIZERS_ADAM

#include <layer_in_c/nn/parameters/parameters.h>

namespace layer_in_c::nn::optimizers{
    namespace adam{
        template<typename T_T>
        struct DefaultParametersTF {
            using T = T_T;
            static constexpr T ALPHA = 0.001;
            static constexpr T BETA_1 = 0.9;
            static constexpr T BETA_2 = 0.999;
            static constexpr T EPSILON = 1e-7;

        };
        template<typename T_T>
        struct DefaultParametersTorch {
            using T = T_T;
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
        typename PARAMETERS::T first_order_moment_bias_correction;
        typename PARAMETERS::T second_order_moment_bias_correction;
        T alpha = PARAMETERS::ALPHA;
    };


}
namespace layer_in_c::nn::parameters {
    struct Adam{
        template <typename CONTAINER>
        struct instance: Gradient::instance<CONTAINER>{
            CONTAINER gradient_first_order_moment;
            CONTAINER gradient_second_order_moment;
        };
    };
}

#endif