#ifndef LAYER_IN_C_LOGGING_OPERATIONS_CPU_TENSORBOARD_H
#define LAYER_IN_C_LOGGING_OPERATIONS_CPU_TENSORBOARD_H

namespace layer_in_c::logging{
    template <typename T>
    void add_scalar(devices::logging::CPU_TENSORBOARD& dev, const char* key, const T value, const typename devices::logging::CPU_TENSORBOARD::index_t cadence = 1){
        if(dev.step % cadence == 0){
            dev.tb->add_scalar(key, value, dev.step);
        }
    }
}
#endif
