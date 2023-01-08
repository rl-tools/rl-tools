#ifndef LAYER_IN_C_UTILS_LOGGING_OPERATIONS_CPU_H
#define LAYER_IN_C_UTILS_LOGGING_OPERATIONS_CPU_H

#include <iostream>

namespace layer_in_c::logging{
    template <typename A>
    void text(const A a){
        std::cout << a << std::endl;
    }
    template <typename A, typename B>
    void text(const A a, const B b){
        std::cout << a << b << std::endl;
    }
    template <typename A, typename B, typename C, typename D>
    void text(const A a, const B b, const C c, const D d){
        std::cout << a << b << c << d << std::endl;
    }
}
#endif
