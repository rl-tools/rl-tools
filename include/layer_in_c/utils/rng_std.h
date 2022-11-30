#include <random>
template <typename T, typename RNG>
T random_uniform_std(T min, T max, RNG& rng){
    std::uniform_real_distribution<T> dist(min, max);
    return dist(rng);
}

