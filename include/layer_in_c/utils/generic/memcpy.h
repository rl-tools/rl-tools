#ifndef LAYER_IN_C_UTILS_GENERIC_MEMCPY_H
#define LAYER_IN_C_UTILS_GENERIC_MEMCPY_H

namespace layer_in_c::utils{
    template<typename T>
    void memcpy(T* target, const T* source, const index_t size) {
        for(index_t i = 0; i < size; i++) {
            target[i] = source[i];
        }
    }
}
#endif
