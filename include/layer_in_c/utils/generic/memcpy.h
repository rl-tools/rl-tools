#ifndef LAYER_IN_C_UTILS_GENERIC_MEMCPY_H
#define LAYER_IN_C_UTILS_GENERIC_MEMCPY_H

namespace layer_in_c::utils{
    template<typename T, typename ST>
    void memcpy(T* target, const T* source, ST size) {
        for(ST i = 0; i < size; i++) {
            target[i] = source[i];
        }
    }
}
#endif
