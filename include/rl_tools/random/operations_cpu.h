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
            return std::normal_distribution<T>(mean, std)(rng);
        }
        template<typename DEVICE, typename T>
        T log_prob(const devices::random::CPU& dev, T mean, T log_std, T value){
            static_assert(utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>);
            return log_prob(devices::random::Generic<devices::math::CPU>{}, mean, log_std, value);
        }
        template<typename DEVICE, typename T>
        T d_log_prob_d_mean(const devices::random::CPU& dev, T mean, T log_std, T value){
            return d_log_prob_d_mean(devices::random::Generic<devices::math::CPU>{}, mean, log_std, value);
        }
        template<typename DEVICE, typename T>
        T d_log_prob_d_log_std(const devices::random::CPU& dev, T mean, T log_std, T value){
            return d_log_prob_d_log_std(devices::random::Generic<devices::math::CPU>{}, mean, log_std, value);
        }
        template<typename DEVICE, typename T>
        T d_log_prob_d_sample(const devices::random::CPU& dev, T mean, T log_std, T value){
            return d_log_prob_d_sample(devices::random::Generic<devices::math::CPU>{}, mean, log_std, value);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
