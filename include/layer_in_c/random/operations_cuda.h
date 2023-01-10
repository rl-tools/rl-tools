#ifndef LAYER_IN_C_UTILS_RANDOM_OPERATIONS_CUDA_H
#define LAYER_IN_C_UTILS_RANDOM_OPERATIONS_CUDA_H


#include <layer_in_c/utils/generic/typing.h>

#include <curand_kernel.h>

namespace layer_in_c::random{
    curandState default_engine(const devices::random::CUDA& dev){
        return curandState();
    };

    template<typename T, typename RNG>
    FUNCTION_PLACEMENT T uniform_real_distribution(const devices::random::CUDA& dev, T low, T high, RNG& rng){
        if constexpr(utils::typing::is_same_v<T, float>){
            return curand_uniform(&rng) * (high - low) + low;
        }
        else{
            if constexpr(utils::typing::is_same_v<T, double>){
                return curand_uniform_double(&rng) * (high - low) + low;
            }
            else{
                return 0;
            }
        }
        return 0;
    }
    template<typename T, typename RNG>
    FUNCTION_PLACEMENT T uniform_int_distribution(const devices::random::CUDA& dev, T low, T high, RNG& rng){
        auto r = uniform_real_distribution(dev, low, high, rng);
        return (T)r;
    }
    template<typename T, typename RNG>
    FUNCTION_PLACEMENT T normal_distribution(const devices::random::CUDA& dev, T mean, T std, RNG& rng){
        if constexpr(utils::typing::is_same_v<T, float>){
            return curand_normal(&rng) * std + mean;
        }
        else{
            if constexpr(utils::typing::is_same_v<T, double>){
                return curand_normal_double(&rng) * std + mean;
            }
            else{
                return 0;
            }
        }
        return 0;
    }
}

#endif
