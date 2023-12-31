#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_RANDOM_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_RANDOM_OPERATIONS_CPU_H


#include "../utils/generic/typing.h"
#include "operations_generic.h"

#include <random>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::random{
    auto default_engine(const devices::random::CPU& dev, devices::random::CPU::index_t seed = 0){
        return std::default_random_engine(seed+1);
    };

    template<typename T, typename RNG>
    T uniform_int_distribution(const devices::random::CPU& dev, T low, T high, RNG& rng){
        return std::uniform_int_distribution<T>(low, high)(rng);
    }
    template<typename T, typename RNG>
    T uniform_real_distribution(const devices::random::CPU& dev, T low, T high, RNG& rng){
        static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>);
        return std::uniform_real_distribution<T>(low, high)(rng);
    }
//    template<typename T, typename RNG>
//    const std::normal_distribution<T> standard_normal_distribution(0, 1);
    namespace normal_distribution{
        template<typename T, typename RNG>
        T sample(const devices::random::CPU& dev, T mean, T std, RNG& rng){
            static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>);
//        return standard_normal_distribution<T, RNG>(rng) * std + mean;
            return std::normal_distribution<T>(mean, std)(rng);
        }
        template<typename DEVICE, typename T>
        T log_prob(const devices::random::CPU& dev, T mean, T log_std, T value){
            static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>);
            T neg_log_sqrt_pi = -0.5 * math::log(typename DEVICE::SPEC::MATH{}, 2 * math::PI<T>);
            T diff = (value - mean);
            T std = math::exp(typename DEVICE::SPEC::MATH{}, log_std);
            T pre_square = diff/std;
            return neg_log_sqrt_pi - log_std - 0.5 * pre_square * pre_square;
        }
        template<typename DEVICE, typename T>
        T d_log_prob_d_mean(const devices::random::CPU& dev, T mean, T log_std, T value){
            T diff = (value - mean);
            T std = math::exp(typename DEVICE::SPEC::MATH{}, log_std);
            T pre_square = diff/std;
            return pre_square / std;
        }
        template<typename DEVICE, typename T>
        T d_log_prob_d_log_std(const devices::random::CPU& dev, T mean, T log_std, T value){
            T diff = (value - mean);
            T std = math::exp(typename DEVICE::SPEC::MATH{}, log_std);
            T pre_square = diff/std;
            return - 1 + pre_square * pre_square;
        }
        template<typename DEVICE, typename T>
        T d_log_prob_d_sample(const devices::random::CPU& dev, T mean, T log_std, T value){
            T diff = (value - mean);
            T std = math::exp(typename DEVICE::SPEC::MATH{}, log_std);
            T pre_square = diff/std;
            return - pre_square / std;
        }

    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
