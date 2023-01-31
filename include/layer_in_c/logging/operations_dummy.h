#ifndef LAYER_IN_C_LOGGING_OPERATIONS_DUMMY_H
#define LAYER_IN_C_LOGGING_OPERATIONS_DUMMY_H

namespace layer_in_c{
    namespace logging{
        template <typename A>
        void text(devices::logging::Dummy& dev, const A a){
        }
        template <typename A, typename B>
        void text(devices::logging::Dummy& dev, const A a, const B b){
        }
        template <typename A, typename B, typename C, typename D>
        void text(devices::logging::Dummy& dev, const A a, const B b, const C c, const D d){
        }
    }
    void add_scalar(devices::logging::Dummy& dev, const char* key, const float value, const typename devices::logging::Dummy::index_t cadence = 1){
        //noop
    }
}
#endif
