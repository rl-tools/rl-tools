#ifndef BACKPROP_TOOLS_UTILS_GENERIC_MEMCPY_H
#define BACKPROP_TOOLS_UTILS_GENERIC_MEMCPY_H

namespace backprop_tools::utils{
    template<typename T, typename ST>
    void memcpy(T* target, const T* source, ST size) {
        for(ST i = 0; i < size; i++) {
            target[i] = source[i];
        }
    }
}
#endif
