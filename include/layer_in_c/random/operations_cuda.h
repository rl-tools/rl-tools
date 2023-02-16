#ifndef LAYER_IN_C_UTILS_RANDOM_OPERATIONS_CUDA_H
#define LAYER_IN_C_UTILS_RANDOM_OPERATIONS_CUDA_H


#include <layer_in_c/utils/generic/typing.h>

#include <curand_kernel.h>

namespace layer_in_c::random{
    namespace cuda{
        using RNG = unsigned int; // actually the seed
    }
    cuda::RNG default_engine(const devices::random::CUDA& dev){
        return 1337;
    };

    cuda::RNG next(const devices::random::CUDA& dev, cuda::RNG& rng){
        return rng + 1;
    };


    template<typename T, typename RNG>
    LAYER_IN_C_FUNCTION_PLACEMENT T uniform_real_distribution(const devices::random::CUDA& dev, T low, T high, RNG& rng){
        static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>, "Only float and double are supported");
        if constexpr(utils::typing::is_same_v<T, float>){
            return curand_uniform(&rng) * (high - low) + low;
        }
        else{
            if constexpr(utils::typing::is_same_v<T, double>){
                return curand_uniform_double(&rng) * (high - low) + low;
            }
        }
        return 0;
    }
    template<typename T, typename RNG>
    LAYER_IN_C_FUNCTION_PLACEMENT T uniform_int_distribution(const devices::random::CUDA& dev, T low, T high, RNG& rng){
        auto r = uniform_real_distribution(dev, low, high, rng);
        return (T)r;
    }
    template<typename T, typename RNG>
    LAYER_IN_C_FUNCTION_PLACEMENT T normal_distribution(const devices::random::CUDA& dev, T mean, T std, RNG& rng){
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
