#ifndef LAYER_IN_C_UTILS_RANDOM_OPERATIONS_ARM_H
#define LAYER_IN_C_UTILS_RANDOM_OPERATIONS_ARM_H


#include <layer_in_c/utils/generic/typing.h>

 namespace layer_in_c::random{
     devices::random::ARM::index_t default_engine(const devices::random::ARM& dev){
         return 0;
     };

     template<typename T, typename RNG>
     T uniform_int_distribution(const devices::random::ARM& dev, T low, T high, RNG& rng){
//         return std::uniform_int_distribution<T>(low, high)(rng);
        return 0;
     }
     template<typename T, typename RNG>
     T uniform_real_distribution(const devices::random::ARM& dev, T low, T high, RNG& rng){
//         return std::uniform_real_distribution<T>(low, high)(rng);
        return 0;
     }
 //    template<typename T, typename RNG>
 //    const std::normal_distribution<T> standard_normal_distribution(0, 1);
     template<typename T, typename RNG>
     T normal_distribution(const devices::random::ARM& dev, T mean, T std, RNG& rng){
 //        return standard_normal_distribution<T, RNG>(rng) * std + mean;
//         return std::normal_distribution<T>(mean, std)(rng);
        return 0;
     }
 }

#endif
