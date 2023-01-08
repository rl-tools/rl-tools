#ifndef LAYER_IN_C_UTILS_LOGGING_OPERATIONS_DUMMY_H
#define LAYER_IN_C_UTILS_LOGGING_OPERATIONS_DUMMY_H

namespace layer_in_c::logging{
    template <typename A>
    void text(const devices::logging::Dummy& dev, const A a){
    }
    template <typename A, typename B>
    void text(const devices::logging::Dummy& dev, const A a, const B b){
    }
    template <typename A, typename B, typename C, typename D>
    void text(const devices::logging::Dummy& dev, const A a, const B b, const C c, const D d){
    }
}
#endif
