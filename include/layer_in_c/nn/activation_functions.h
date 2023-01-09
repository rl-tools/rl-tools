#ifndef LAYER_IN_C_NN_ACTIVATION_FUNCTIONS
#define LAYER_IN_C_NN_ACTIVATION_FUNCTIONS
namespace layer_in_c::nn::activation_functions {
    enum ActivationFunction{
        IDENTITY,
        RELU,
        GELU,
        TANH,
        SIGMOID,
        SIGMOID_STRETCHED,
    };
    template<enum ActivationFunction F>
    constexpr bool check_activation_function = F == IDENTITY || F == RELU || F == GELU || F == TANH || F == SIGMOID || F == SIGMOID_STRETCHED;

    template<typename DEVICE, typename T, ActivationFunction F>
    T activation(T x){
        static_assert(DEVICE::DOMAIN == devices::Domain::math, "DEVICE is not a math device");
        static_assert(check_activation_function<F>, "Invalid activation function");
        if (F == IDENTITY){
            return x;
        }
        else if (F == RELU){
            return x > 0 ? x : 0;
        }
        else if (F == GELU){
            constexpr T a = math::FRAC_2_SQRTPI<T> * math::SQRT1_2<T> * (T)0.5;
            return (T)0.5 * (x + x * math::tanh(DEVICE(), a * ((T)0.044715f * x * x * x + x)));
        }
        else if (F == TANH){
            return math::tanh(DEVICE(), x);
        }
        else if (F == SIGMOID){
            return (T)1 / ((T)1 + math::exp(DEVICE(), -x));
        }
        else if (F == SIGMOID_STRETCHED){
            return activation<DEVICE, T, SIGMOID>(x) * (T)2 - (T)1;
        }
    }

    template<typename DEVICE, typename T, ActivationFunction F>
    T d_activation_d_x(T x){
        static_assert(DEVICE::DOMAIN == devices::Domain::math, "DEVICE is not a math device");
        static_assert(check_activation_function<F>, "Invalid activation function");
        if (F == IDENTITY){
            return 1;
        }
        else if (F == RELU){
            return x > 0 ? 1 : 0;
        }
        else if (F == GELU){
            constexpr T a = math::FRAC_2_SQRTPI<T> * math::SQRT1_2<T> * (T)0.5;
            constexpr T b = 0.044715f;
            T tanh_term = math::tanh(DEVICE(), a * (b * x * x * x + x));
            return (T)0.5*((T)1 + tanh_term) + (T)0.5 * x * ((T)1 - tanh_term * tanh_term) * a * ((T)3 * b * x * x + (T)1);
        }
        else if (F == TANH){
            T a = math::tanh(DEVICE(), x);
            return (T)1 - a * a;
        }
        else if (F == SIGMOID){
            return activation<DEVICE, T, SIGMOID>(x) * (1 - activation<DEVICE, T, SIGMOID>(x));
        }
        else if (F == SIGMOID_STRETCHED){
            return d_activation_d_x<DEVICE, T, SIGMOID>(x) * (T)2;
        }
    }
}


#endif