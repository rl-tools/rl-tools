#ifndef LAYER_IN_C_NN_OPTIMIZERS_ADAM
#define LAYER_IN_C_NN_OPTIMIZERS_ADAM

namespace layer_in_c::nn::optimizers::adam {
    template<typename T>
    struct DefaultParametersTF {
    public:
        static constexpr T ALPHA = 0.001;
        static constexpr T BETA_1 = 0.9;
        static constexpr T BETA_2 = 0.999;
        static constexpr T EPSILON = 1e-7;

    };
    template<typename T>
    struct DefaultParametersTorch {
    public:
        static constexpr T ALPHA = 0.001;
        static constexpr T BETA_1 = 0.9;
        static constexpr T BETA_2 = 0.999;
        static constexpr T EPSILON = 1e-8;

    };

}

#endif