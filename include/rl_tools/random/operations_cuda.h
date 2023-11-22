#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_RANDOM_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_RANDOM_OPERATIONS_CUDA_H


#include "../utils/generic/typing.h"
#include "operations_generic.h"

#include <curand_kernel.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::random{
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
    RL_TOOLS_FUNCTION_PLACEMENT T uniform_real_distribution(const devices::random::CUDA& dev, T low, T high, RNG& rng){
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
    RL_TOOLS_FUNCTION_PLACEMENT T uniform_int_distribution(const devices::random::CUDA& dev, T low, T high, RNG& rng){
        auto r = uniform_real_distribution(dev, low, high, rng);
        return (T)r;
    }
    namespace normal_distribution{
        template<typename T, typename RNG>
        RL_TOOLS_FUNCTION_PLACEMENT T sample(const devices::random::CUDA& dev, T mean, T std, RNG& rng){
            static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>);
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
        }
        template<typename DEVICE, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T log_prob(const devices::random::CUDA& dev, T mean, T log_std, T value){
            static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>);
            T neg_log_sqrt_pi = -0.5 * math::log(typename DEVICE::SPEC::MATH{}, 2 * math::PI<T>);
            T diff = (value - mean);
            T std = math::exp(typename DEVICE::SPEC::MATH{}, log_std);
            T pre_square = diff/std;
            return neg_log_sqrt_pi - log_std - 0.5 * pre_square * pre_square;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
