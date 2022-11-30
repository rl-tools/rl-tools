#include <random>
namespace layer_in_c::utils::random::stdlib{
    template <typename T, typename RNG>
    T uniform(T min, T max, RNG& rng){
        std::uniform_real_distribution<T> dist(min, max);
        return dist(rng);
    }
}

