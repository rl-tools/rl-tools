#ifndef TESTS_HYPERLUX_GOLDEN_METRICS_H
#define TESTS_HYPERLUX_GOLDEN_METRICS_H

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace golden {
    namespace detail {
        inline bool within_neighborhood_range(const uint32_t* image, int width, int height, int x, int y, const uint8_t* color, int slack){
            int low[3] = {255, 255, 255};
            int high[3] = {0, 0, 0};
            for(int dy = -1; dy <= 1; dy++){
                const int neighbor_y = std::clamp(y + dy, 0, height - 1);
                for(int dx = -1; dx <= 1; dx++){
                    const int neighbor_x = std::clamp(x + dx, 0, width - 1);
                    const auto* neighbor = reinterpret_cast<const uint8_t*>(image + (size_t)neighbor_y * width + neighbor_x);
                    for(int channel = 0; channel < 3; channel++){
                        low[channel] = std::min(low[channel], (int)neighbor[channel]);
                        high[channel] = std::max(high[channel], (int)neighbor[channel]);
                    }
                }
            }
            for(int channel = 0; channel < 3; channel++){
                if((int)color[channel] < low[channel] - slack || (int)color[channel] > high[channel] + slack){
                    return false;
                }
            }
            return true;
        }
    }

    // A differing pixel whose color lies within the per-channel range of the other image's 3x3
    // neighborhood (plus slack), in either direction, is a silhouette flip or an anti-aliasing
    // blend; a pixel that fails both tests is a surface shaded differently.
    inline bool neighborhood_explains(const uint32_t* a, const uint32_t* b, int width, int height, int x, int y, int slack){
        const size_t index = (size_t)y * width + x;
        return detail::within_neighborhood_range(a, width, height, x, y, reinterpret_cast<const uint8_t*>(b + index), slack)
            || detail::within_neighborhood_range(b, width, height, x, y, reinterpret_cast<const uint8_t*>(a + index), slack);
    }
}

#endif
